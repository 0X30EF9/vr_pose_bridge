"""Deterministic, hardware-free retargeting latency and accuracy benchmark.

Run with the same Python/BLAS environment for both source snapshots. Frame
generation, model loading and the simple lagged feedback plant are outside the
timed region. This measures the mapper, not ROS, transport or hardware latency.
"""
from __future__ import annotations

import argparse
import cProfile
import contextlib
import ctypes
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import time

import numpy as np
import pinocchio as pin
import scipy
from scipy.spatial.transform import Rotation

def make_mapper(side="both"):
    from teleop_direct_retarget import DirectRetargetConfig
    from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
    from teleop_skeleton_retarget import SkeletonRetargeter
    return SkeletonRetargeter(DirectRetargetConfig(
        joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
        control_side=side, calibration_window_frames=3,
    ))


def desired_joint_trajectory(phase, case):
    """Known joint-space trajectory, also usable by independent plant tests."""
    if case == "wrist_twist":
        q = np.zeros(14)
        q[[5, 12]] = 0.30 * np.sin(phase)
        q[[6, 13]] = 0.35 * np.sin(phase * 2.0)
        return q
    offsets = np.arange(14) * 0.17
    amplitudes = np.array([0.08, 0.12, 0.12, 0.18, 0.15, 0.16, 0.18] * 2)
    q = amplitudes * (np.sin(phase + offsets) - np.sin(offsets))
    if case == "single_arm":
        q[7:] = 0.0
    return q


def robot_targets(solver, q):
    """Targets for the segment-direction API; all generation is untimed."""
    poses = solver.end_effector_poses(q)
    directions = {side: solver.segment_directions(q, side) for side in ("left", "right")}
    return directions, {side: pose[:3, :3] for side, pose in zip(("left", "right"), poses)}


def skeleton_for_targets(directions, rotations, robot_home_rotations):
    def pose(position, rotation=np.eye(3)):
        return {"pos": np.asarray(position).tolist(), "quat": Rotation.from_matrix(rotation).as_quat().tolist(), "tracked": True}

    frame = {"skeleton": {"torso": {
        "left_hip": pose([0, .15, 1]), "right_hip": pose([0, -.15, 1]),
        "waist": pose([0, 0, 1]), "chest": pose([0, 0, 1.4]),
    }}}
    for side, sign in (("left", 1), ("right", -1)):
        shoulder = np.array([0, sign * .2, 1.5])
        elbow = shoulder + .3 * directions[side][0]
        wrist = elbow + .3 * directions[side][1]
        hand_rotation = rotations[side] @ robot_home_rotations[side].T
        frame["skeleton"][side] = {
            "shoulder": pose(shoulder), "elbow": pose(elbow),
            "wrist": pose(wrist, hand_rotation), "hand": pose(wrist, hand_rotation),
        }
    return frame


def trajectory_frames(solver, count, case):
    home_directions, home_rotations = robot_targets(solver, np.zeros(14))
    home = skeleton_for_targets(home_directions, home_rotations, home_rotations)
    frames = []
    for index in range(count):
        phase = 4 * np.pi * index / max(1, count - 1)
        if case == "unreachable":
            directions = {
                side: (
                    Rotation.from_euler("x", 1.5 * np.sin(phase)).apply(home_directions[side][0]),
                    Rotation.from_euler("y", 1.8 * np.sin(phase)).apply(home_directions[side][1]),
                ) for side in ("left", "right")
            }
            rotations = {side: Rotation.from_euler("z", 1.8 * np.sin(phase)).as_matrix() @ home_rotations[side] for side in ("left", "right")}
            # Start from the same natural-down source reference and make the
            # first motion continuous, even though the targets become infeasible.
            if index == 0:
                frames.append(home)
                continue
        else:
            directions, rotations = robot_targets(solver, desired_joint_trajectory(phase, case))
        frame = skeleton_for_targets(directions, rotations, home_rotations)
        frame["receive_time"] = index * .01
        frames.append(frame)
    return home, frames


def measure_case(case, count, profiler=None):
    mapper = make_mapper("left" if case == "single_arm" else "both")
    home, frames = trajectory_frames(mapper.solver, count, case)
    mapper.align([home] * 3, np.zeros(14))
    for _ in range(30):
        mapper.command_from_frame(home, np.zeros(14), None)
    current = np.zeros(14)
    last = None
    durations, residuals, iterations, commands = [], [], [], []
    maximum_violation = 0.0
    constraint_violations = {name: {"maximum_rad": 0.0, "frames": 0} for name in (
        "hard_limits", "feedback_step", "last_command_step", "baseline_envelope", "inactive_arm",
    )}
    bounds = np.asarray(mapper.solver.joint_limits)
    for frame in frames:
        if last is not None:
            current = current + .8 * (last - current)
        if profiler is not None:
            profiler.enable()
        start = time.perf_counter_ns()
        command = mapper.command_from_frame(frame, current, last)
        durations.append((time.perf_counter_ns() - start) * 1e-6)
        if profiler is not None:
            profiler.disable()
        q = command.positions
        constraints = {
            "hard_limits": np.maximum(bounds[:, 0] - q, q - bounds[:, 1]),
            "feedback_step": abs(q-current)-.04, "baseline_envelope": abs(q)-.9,
        }
        if last is not None:
            constraints["last_command_step"] = abs(q-last)-.04
        if case == "single_arm":
            constraints["inactive_arm"] = abs(q[7:] - current[7:])
        for name, values in constraints.items():
            violation = max(0.0, float(np.max(values)))
            maximum_violation = max(maximum_violation, violation)
            constraint_violations[name]["maximum_rad"] = max(constraint_violations[name]["maximum_rad"], violation)
            constraint_violations[name]["frames"] += int(violation > 1e-12)
        residuals.extend(list(value.values()) for value in command.meta["final_residual_by_side"].values())
        iterations.append(command.meta["solver_meta"]["iterations"])
        commands.append(q.tolist())
        last = q.copy()
    latency = dict(zip(("p50_ms", "p95_ms", "p99_ms"), np.percentile(durations, [50, 95, 99]).tolist()))
    errors = np.asarray(residuals)
    return {
        **latency, "max_ms": max(durations), "over_10ms": sum(v > 10 for v in durations),
        "frames": count, "mean_iterations": float(np.mean(iterations)),
        "direction_rms_rad": float(np.sqrt(np.mean(errors[:, :2] ** 2))),
        "orientation_rms_rad": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "direction_max_rad": float(np.max(errors[:, :2])), "orientation_max_rad": float(np.max(errors[:, 2])),
        "maximum_constraint_violation_rad": max(0.0, maximum_violation),
        "constraint_violations": constraint_violations,
        "all_commands_finite": bool(np.all(np.isfinite(commands))),
        "commands": commands,
    }


def load_source(source_dir):
    default_source = Path(__file__).resolve().parents[1] / "app" / "src"
    # A four-file baseline backup can reuse unchanged schema and URDF assets
    # from this checkout. A complete worktree uses its own source and assets.
    sys.path.insert(0, str(default_source))
    sys.path.insert(0, str(source_dir))
    core = importlib.import_module("teleop_ik_core")
    asset_directory = source_dir / "models" / "walker_tienkung_ei_description"
    if not (asset_directory / "walker_tienkung_ei.urdf").is_file():
        asset_directory = default_source / "models" / "walker_tienkung_ei_description"
    if not (asset_directory / "walker_tienkung_ei.urdf").is_file():
        raise FileNotFoundError("walker URDF is missing from source and current checkout")
    core._MODEL_DIR = asset_directory
    core._URDF_PATH = asset_directory / "walker_tienkung_ei.urdf"
    for name in ("teleop_skeleton_retarget", "teleop_direct_retarget", "teleop_ik_types", "teleop_joint_schema"):
        importlib.import_module(name)
    return core._URDF_PATH


def environment(source_dir, urdf_path):
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
    libraries = []
    if Path("/proc/self/maps").exists():
        paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(name in line.lower() for name in ("libblas", "libblis", "openblas", "liblapack", "libmkl"))})
        for path in paths:
            library = ctypes.CDLL(path)
            thread_counts = {}
            for symbol in ("openblas_get_num_threads", "bli_thread_get_num_threads", "MKL_Get_Max_Threads"):
                if hasattr(library, symbol):
                    thread_counts[symbol] = int(getattr(library, symbol)())
            libraries.append({"path": path, "thread_counts": thread_counts})
    hashes = {}
    for name in ("teleop_ik_core", "teleop_skeleton_retarget", "teleop_direct_retarget", "teleop_ik_types", "teleop_joint_schema"):
        path = Path(sys.modules[name].__file__).resolve()
        hashes[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "platform": platform.platform(), "pinocchio": pin.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
        "loaded_blas_libraries": libraries,
        "source_dir": str(source_dir), "source_hashes": hashes,
        "urdf": {"path": str(urdf_path), "sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest()},
        "load_average_at_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "numpy_config": config.getvalue(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / "app" / "src")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.frames < 200:
        parser.error("--frames must be at least 200 to keep skeleton motion continuous")
    profiler = None if args.profile is None else cProfile.Profile()
    source_dir = args.source_dir.resolve()
    if not (source_dir / "teleop_ik_core.py").is_file():
        parser.error("--source-dir must contain teleop_ik_core.py")
    urdf_path = load_source(source_dir)
    results = {"environment": environment(source_dir, urdf_path), "profiled": profiler is not None, "cases": {}}
    for case in ("reachable", "wrist_twist", "single_arm", "unreachable"):
        results["cases"][case] = measure_case(case, args.frames, profiler)
        print(case, json.dumps({k: v for k, v in results["cases"][case].items() if k != "commands"}))
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if profiler is not None:
        with args.profile.open("w", encoding="utf-8") as output:
            pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(35)


if __name__ == "__main__":
    main()
