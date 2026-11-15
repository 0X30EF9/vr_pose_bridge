"""Offline 6DoF Cartesian IK benchmark; no controllers, ROS or vendor SDK.

Targets come from known reachable URDF joint trajectories, plus an explicitly
unreachable case. FK target generation, loading and warmup are untimed. The
timed solve includes strict input checks, bounded output and exact residuals.
The lagged feedback plant is synthetic, not a physics or hardware validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pinocchio as pin
import scipy


def load_source(path):
    sys.path.insert(0, str(path))
    core = importlib.import_module("teleop_ik_core")
    if not hasattr(core.TeleopArmIK, "solve_cartesian"):
        raise ValueError("--source-dir does not provide TeleopArmIK.solve_cartesian")
    return core


def make_solver(case, smoothing_alpha=1.0):
    from teleop_ik_core import TeleopArmIK
    from teleop_ik_types import TeleopIKConfig
    from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
    return TeleopArmIK(TeleopIKConfig(
        joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
        task_mode="position_only" if case == "position_only" else "pose",
        max_iterations=6, control_dt=.01, damping=.08,
        max_joint_delta_per_step=.0048, smoothing_alpha=smoothing_alpha,
        enable_local_log=False, log_to_console=False,
        show_model_load_progress=False, log_ik_result=False,
    ))


def trajectory(solver, count, case):
    home = solver.end_effector_poses(np.zeros(14))
    bounds = np.asarray(solver.joint_limits)
    targets = []
    offsets = np.arange(14)*.17
    amplitudes = np.array([.06, .08, .08, .10, .09, .12, .12]*2)
    for index in range(count):
        phase = 2*np.pi*index/max(1, count-1)
        desired = amplitudes*(np.sin(phase+offsets)-np.sin(offsets))
        if case == "wrist_twist":
            desired[:] = 0.
            desired[[5, 12]] = .25*np.sin(phase)
            desired[[6, 13]] = .25*(np.sin(phase+.3)-np.sin(.3))
        if case == "left_only":
            desired[7:] = 0.
        desired = np.clip(desired, bounds[:, 0], bounds[:, 1])
        poses = solver.end_effector_poses(desired)
        if case == "unreachable":
            poses[0][:3, 3] = [1.5, .5, 1.8]
            poses[1][:3, 3] = [1.5, -.5, 1.8]
        targets.append(poses)
    return home, targets


def measure_case(case, count, smoothing_alpha=1.0):
    from teleop_ik_types import TeleopIKInput
    solver = make_solver(case, smoothing_alpha)
    side = "left" if case == "left_only" else "both"
    active = np.arange(7) if side == "left" else np.arange(14)
    home, targets = trajectory(solver, count, case)
    for _ in range(30):
        solver.solve_cartesian(TeleopIKInput(0., side, home[0], home[1], np.zeros(14)))
    current = np.zeros(14)
    previous = None
    bounds = np.asarray(solver.joint_limits)
    violations = {key: {"frames": 0, "maximum_rad": 0.} for key in (
        "hard_limits", "feedback_step", "last_command_step", "inactive_arm",
    )}
    times, positions, orientations, iterations, commands = [], [], [], [], []
    waist_commands = 0
    for index, poses in enumerate(targets):
        if previous is not None:
            current += .8*(previous-current)
        request = TeleopIKInput(index*.01, side, poses[0], poses[1], current, previous)
        start = time.perf_counter_ns()
        result = solver.solve_cartesian(request)
        times.append((time.perf_counter_ns()-start)*1e-6)
        q = result.command.positions
        constraints = {
            "hard_limits": np.maximum(bounds[:, 0]-q, q-bounds[:, 1]),
            "feedback_step": abs(q[active]-current[active])-.0048,
        }
        if previous is not None:
            constraints["last_command_step"] = abs(q[active]-previous[active])-.0048
        if side == "left":
            constraints["inactive_arm"] = abs(q[7:]-current[7:])
        for key, values in constraints.items():
            amount = max(0., float(np.max(values)))
            violations[key]["maximum_rad"] = max(violations[key]["maximum_rad"], amount)
            violations[key]["frames"] += int(amount > 1e-12)
        for residual in result.command.meta["residual_by_side"].values():
            positions.append(residual["position_error_m"])
            orientations.append(residual["orientation_error_rad"])
        iterations.append(result.iteration_count)
        waist_commands += int(result.command.waist_position is not None)
        commands.append(q.tolist())
        previous = q.copy()
    percentiles = np.percentile(times, [50, 95, 99])
    return {
        "frames": count, "task_mode": solver.config.task_mode,
        "p50_ms": float(percentiles[0]), "p95_ms": float(percentiles[1]), "p99_ms": float(percentiles[2]),
        "max_ms": max(times), "over_10ms": sum(value > 10 for value in times),
        "mean_iterations": float(np.mean(iterations)), "max_iterations": max(iterations),
        "position_rms_m": float(np.sqrt(np.mean(np.square(positions)))),
        "position_p95_m": float(np.percentile(positions, 95)), "position_max_m": max(positions),
        "orientation_rms_rad": float(np.sqrt(np.mean(np.square(orientations)))),
        "orientation_p95_rad": float(np.percentile(orientations, 95)), "orientation_max_rad": max(orientations),
        "orientation_is_controlled": case != "position_only",
        "constraint_violations": violations, "waist_command_count": waist_commands,
        "all_commands_finite": bool(np.all(np.isfinite(commands))), "commands": commands,
    }


def environment(source):
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    hashes = {}
    for name in ("teleop_ik_core", "teleop_ik_types", "teleop_joint_schema"):
        path = source/(name+".py")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    urdf = source/"models"/"walker_tienkung_ei_description"/"walker_tienkung_ei.urdf"
    return {
        "cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "platform": platform.platform(), "pinocchio": pin.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "threadpools": pools, "thread_environment": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
        "source_dir": str(source), "source_sha256": hashes, "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1]/"app"/"src")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0,
                        help="Production controller mapper default is 1.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 200:
        parser.error("--frames must be at least 200")
    if not np.isfinite(args.smoothing_alpha) or not 0 <= args.smoothing_alpha <= 1:
        parser.error("--smoothing-alpha must be finite and within [0, 1]")
    source = args.source_dir.resolve()
    if not (source/"teleop_ik_core.py").is_file():
        parser.error("--source-dir must contain teleop_ik_core.py")
    load_source(source)
    cases = {}
    for case in ("reachable", "wrist_twist", "left_only", "position_only", "unreachable"):
        cases[case] = measure_case(case, args.frames, args.smoothing_alpha)
        print(case, json.dumps({key: value for key, value in cases[case].items() if key != "commands"}))
    report = {
        "scope": "Cartesian IK only; synthetic lagged feedback; excludes VR mapping/transport/ROS/SDK/physics",
        "environment": environment(source),
        "config": {"control_hz": 100, "max_iterations": 6, "max_joint_step_rad": .0048,
                   "damping": .08, "smoothing_alpha": args.smoothing_alpha, "position_weight": 1., "orientation_weight": .25},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
