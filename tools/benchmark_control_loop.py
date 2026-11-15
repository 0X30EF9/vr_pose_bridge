"""Scheduled live-callback benchmark with real Pinocchio and fake ROS/SDK.

No ROS runtime, vendor SDK or network connection is used. The existing test
fixture supplies in-memory publishers; a 50 ms sleeping sidecar represents SDK
blocking time. Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 set before
starting Python. Results describe this software setup, not real-time hardware.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "app/src"), str(root / "app/src/tests")]
import test_bridge_boundaries as boundaries
from teleop_direct_retarget import DirectRetargetConfig
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
from teleop_skeleton_retarget import SkeletonRetargeter
from teleop_startup_pose import NaturalDownStartupConfig, NaturalDownStartupGate
from teleop_hand_ik import HandIkFrameResult, LatestHandIkWorker
from teleop_control_safety import joint_velocity_limits_from_urdf

spec = importlib.util.spec_from_file_location("mapper_benchmark", root / "tools/benchmark_ik.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), **{key: float(np.percentile(values, percentile)) for key, percentile in
                                   (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}} if len(values) else {"count": 0}


class SlowSidecar:
    def __init__(self):
        import threading
        self.stop = threading.Event()
        self.starts, self.durations, self.completions = [], [], []

    def solve_replay_frame(self, frame):
        began = time.monotonic()
        receipt = frame["receive_monotonic"]
        self.starts.append((began, receipt))
        self.stop.wait(0.05)
        self.durations.append((receipt, (time.monotonic() - began) * 1000))
        self.completions.append((time.monotonic(), receipt))
        phase = frame["packet_timestamp_us"] * 1e-6
        positions = tuple(0.5 + 0.15 * np.sin(phase + index * 0.1) for index in range(12))
        return HandIkFrameResult(len(self.starts), int(frame["packet_timestamp_us"] // 1000), positions,
                                 True, True, 50.0, 50.0, "calibrated", "calibrated")

    def abort(self):
        self.stop.set()

    def close(self):
        self.stop.set()


class CapturePublisher:
    def __init__(self, hook=None):
        self.times, self.hook = [], hook

    def publish(self, message):
        self.times.append(time.monotonic())
        if self.hook is not None:
            self.hook(message)


def profile_case(mode, count=600):
    fixture = boundaries.BridgeBoundariesTest(methodName="runTest")
    fixture.setUp()
    node, module = fixture.node, fixture.module
    module.time = time
    module.CONTROL_MODE = "skeleton"
    module.CONTROL_RATE_HZ = module.SAFE_COMMAND_RATE_HZ = 100.0
    module.MAX_JOINT_VELOCITY = 0.48
    module.MAX_JOINT_DELTA_PER_STEP = 0.06
    module.MAX_COMMAND_DELTA_FROM_BASELINE = 1.0
    module.HAND_CONTROL_RATE_HZ, module.MAX_HAND_VELOCITY = 20.0, 1.0
    module.LOG_RATE_HZ, module.LOG_EVERY_N_FRAMES = 1.0, 100
    module.HAND_CONTROL_CONFIG = replace(module.HAND_CONTROL_CONFIG, publish_enabled=True,
                                        max_step=0.05, thumb_rotation_mode="mapped")
    config = DirectRetargetConfig(joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
                                  max_joint_delta_per_step=0.0048, max_command_delta_from_baseline=1.0)
    mapper = SkeletonRetargeter(config)
    home, frames = benchmark.trajectory_frames(mapper.solver, count, "reachable")
    node.skeleton_retargeter = mapper
    node.ik_solver = None
    node.log_frame_index = 0
    node.startup_pose_locked = False
    node.startup_pose_gate = NaturalDownStartupGate(NaturalDownStartupConfig(hold_seconds=2.0))
    node.safe_baseline_q = None
    node.latest_mocap_data = None
    node.last_hand_submission = 0
    node.hand_feedback = {side: (0.0,) * 6 for side in ("left", "right")}
    node.hand_feedback_time = {side: None for side in ("left", "right")}
    node.last_hand_command = {side: None for side in ("left", "right")}
    node.last_hand_publish_time = {side: None for side in ("left", "right")}
    node._warned_hand_state = {side: False for side in ("left", "right")}
    node.local_logger = SimpleNamespace(info=lambda text: None)
    node.should_log = module.MocapBridgeNode.should_log.__get__(node)
    node.command_joint_limits = list(mapper.solver.joint_limits)
    node.command_velocity_limits = np.asarray(joint_velocity_limits_from_urdf(
        root / "app/src/models/walker_tienkung_ei_description/walker_tienkung_ei.urdf", ARM_JOINT_NAMES))
    violations = {"measured_step": 0.0, "submitted_velocity": 0.0, "hard_limit": 0.0}
    bounds = np.asarray(node.command_joint_limits)
    last_published = None
    last_published_at = None

    def check_arm(message):
        nonlocal last_published, last_published_at
        q = np.asarray([row["pos"] for row in message["cmds"]])
        now = time.monotonic()
        violations["measured_step"] = max(violations["measured_step"], float(np.max(abs(q - node.current_lr_arm_motor_q) - .0048)))
        violations["hard_limit"] = max(violations["hard_limit"], float(np.max(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]))))
        if last_published is not None:
            violations["submitted_velocity"] = max(violations["submitted_velocity"], float(np.max(abs(q - last_published) - .48 * min(.01, now - last_published_at))))
        last_published, last_published_at = q.copy(), now

    node.arm_cmd_pub = CapturePublisher(check_arm)
    node.hand_cmd_pubs = {side: CapturePublisher() for side in ("left", "right")}
    sdk = SlowSidecar()
    node.hand_ik_sidecar = LatestHandIkWorker(sdk, rate_hz=20)
    ik_times = []
    original_command = mapper.command_from_frame

    def timed_command(*args, **kwargs):
        began = time.perf_counter_ns()
        result = original_command(*args, **kwargs)
        ik_times.append((time.perf_counter_ns() - began) * 1e-6)
        return result
    mapper.command_from_frame = timed_command
    fresh_hand_results = []
    profile_start = None
    original_hand_publish = node.publish_hand_result
    def timed_hand_publish(result, received_at):
        if profile_start is not None and received_at >= profile_start:
            fresh_hand_results.append((time.monotonic(), received_at))
        original_hand_publish(result, received_at)
    node.publish_hand_result = timed_hand_publish

    sequence = 0
    def feed(frame):
        nonlocal sequence
        sequence += 1
        now = time.monotonic()
        if mode == "follow" and node.last_command_q is not None:
            node.current_lr_arm_motor_q = node.last_command_q.copy()
        elif mode == "stagnant":
            node.current_lr_arm_motor_q = np.zeros(14)
        node.last_arm_state_time = now
        for side in ("left", "right"):
            if node.last_hand_command[side] is not None:
                node.hand_feedback[side] = node.last_hand_command[side]
            node.hand_feedback_time[side] = now
        data = dict(frame)
        data.update(timestamp=time.time(), receive_time=sequence * .01,
                    packet_timestamp_us=sequence * 10000, receive_monotonic=now)
        node.latest_mocap_data = data
        began = time.perf_counter_ns()
        node.control_loop()
        return (time.perf_counter_ns() - began) * 1e-6

    try:
        warmup_start = time.monotonic()
        for index in range(215):
            remaining = warmup_start + index * .01 - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            feed(home)
        if not node.startup_pose_locked:
            raise RuntimeError("benchmark startup alignment failed")
        ik_times.clear()
        node.arm_cmd_pub.times.clear()
        for pub in node.hand_cmd_pubs.values():
            pub.times.clear()
        fresh_hand_results.clear()
        # Separate steady state from startup batches and their in-flight SDK
        # results, and avoid an extra callback immediately after warmup.
        node.hand_ik_sidecar.invalidate()
        began = time.monotonic() + .01
        profile_start = began
        costs, lateness = [], []
        for index, frame in enumerate(frames):
            deadline = began + index * .01
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            lateness.append(max(0., time.monotonic() - deadline) * 1000)
            costs.append(feed(frame))
        ended = time.monotonic()
        starts = [now for now, receipt in sdk.starts if receipt >= began and now <= ended]
        completions = [now for now, receipt in sdk.completions if receipt >= began and now <= ended]
        pubs = node.arm_cmd_pub.times
        return {
            "mode": mode, "frames": count, "control_rate": 100, "feedback_rate": 100,
            "input_timestamp_keys_unique": sequence, "periodic_json_log_rate_hz": 1,
            "callback_ms": stats(costs), "ik_ms": stats(ik_times),
            "callback_overruns_10ms": sum(value > 10 for value in costs),
            "dispatch_lateness_ms": stats(lateness), "arm_sends": len(pubs),
            "arm_send_interval_ms": stats(np.diff(pubs) * 1000),
            "arm_send_hz": (len(pubs)-1)/(pubs[-1]-pubs[0]) if len(pubs)>1 else 0,
            "hand_sdk_calls": len(starts), "hand_sdk_completed": len(completions),
            "hand_sdk_ms": stats([cost for receipt, cost in sdk.durations if receipt >= began]),
            "hand_fresh_results": len(fresh_hand_results),
            "hand_fresh_completed_ratio": len(fresh_hand_results) / len(completions) if completions else None,
            "hand_result_age_ms": stats([(now-received)*1000 for now, received in fresh_hand_results]),
            "hand_sends": {side: len(pub.times) for side, pub in node.hand_cmd_pubs.items()},
            "command_faulted": node.command_faulted, "constraint_max_violation_rad": violations,
            "scope": "real Pinocchio mapper and live callback; fake ROS transport and measured plant; 50ms sleeping hand SDK; no hardware",
        }
    finally:
        node.hand_ik_sidecar.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600, help="每场景动态控制帧数，至少 100")
    parser.add_argument("--output", type=Path, default=root / "app/data/diagnostics/control-loop-profile.json")
    args = parser.parse_args()
    if args.frames < 100:
        parser.error("--frames must be at least 100 for a continuous dynamic trajectory")
    import pinocchio
    import scipy
    source_paths = [
        "tools/benchmark_control_loop.py", "tools/benchmark_ik.py",
        "app/src/mocap_ros_2_bridge_node.py", "app/src/teleop_ik_core.py",
        "app/src/teleop_skeleton_retarget.py", "app/src/teleop_direct_retarget.py",
        "app/src/teleop_hand_ik.py", "app/src/teleop_control_safety.py",
        "app/src/tests/test_bridge_boundaries.py",
        "app/src/models/walker_tienkung_ei_description/walker_tienkung_ei.urdf",
    ]
    results = {
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "pinocchio": pinocchio.__version__,
                        "threads": {name: os.environ.get(name) for name in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_paths},
        "startup_excluded": "215 natural-down frames with a real 2 second gate; in-flight hand results invalidated before measurement",
        "scope": "software callback, real kinematics, in-memory ROS publishers and fake 50ms SDK; no network, vendor .so or physical robot",
        "cases": [profile_case(mode, args.frames) for mode in ("follow", "stagnant")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
