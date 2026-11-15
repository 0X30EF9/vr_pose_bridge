"""Raw Unity/OpenXR controller packets to production Cartesian IK and Bullet.

Device-side fixtures are prepared before timing. Every measured control tick
still parses the raw JSON, applies the real clutch mapper and publication gate,
then advances independent physics. No robot, ROS or vendor SDK is connected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from teleop_joint_schema import ARM_JOINT_NAMES
from teleop_simulation import (BulletWorld, SOURCE, distribution,
                               initial_configuration, rotation_error, runtime_metadata,
                               validate_dynamics, validate_fk, write_viewer)

SCENARIOS = ("translation_xyz", "wrist_pose", "left_only", "near_limits", "unreachable",
             "tracking_loss", "clutch_reanchor", "invalid_input")
# Independent fixture inverse transforms; do not reuse the mapper's basis and
# thereby let an identical coordinate error cancel itself during validation.
BASES = {"unity": np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]]),
         "openxr": np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]),
         "robot": np.eye(3)}


def build_controller_solver(control_hz=100., max_joint_velocity=.48):
    """Use the exact production constructor, including step and smoothing."""
    from teleop_vr_controller import VrControllerConfig, VrControllerRetargeter
    return VrControllerRetargeter(VrControllerConfig(control_rate_hz=control_hz,
                  max_joint_velocity=max_joint_velocity, task_mode="pose")).solver


def active_sides(control_side):
    return ("left", "right") if control_side == "both" else (control_side,)


def indices(side):
    return slice(0, 7) if side == "left" else slice(7, 14)


def packet_from_robot_poses(poses, *, seq, timestamp, coordinate_system, enabled, tracked, activation_ids):
    from scipy.spatial.transform import Rotation
    basis = BASES[coordinate_system]
    packet = {"version": 1, "type": "vr_controller", "sequence": seq,
              "timestamp": timestamp, "coordinate_system": coordinate_system}
    for side, pose in poses.items():
        rotation = basis.T @ pose[:3, :3] @ basis
        packet[side] = {"position": (basis.T @ pose[:3, 3]).tolist(),
                        "rotation_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
                        "tracked": bool(tracked[side]), "enabled": bool(enabled[side]),
                        "activation_id": int(activation_ids[side])}
    return packet


def fixture_joint_pose(scene, phase, baseline, limits):
    progress = max(0., min(1., (phase-.12)/.55))
    blend = .5-.5*math.cos(math.pi*progress)
    q = baseline.copy()
    if scene == "wrist_pose":
        q[[5, 6, 12, 13]] += blend*np.array([.2, .35, -.2, -.35])
    elif scene == "near_limits":
        q[[1, 6, 8, 13]] += blend*np.array([.34, .85, -.50, -.85])
    else:
        q += blend*np.array([-.12, .09, .12, -.25, -.10, .12, .16, -.10, -.08, -.12, -.22, .10, -.12, -.16])
    if scene in {"tracking_loss", "clutch_reanchor", "invalid_input"}:
        resumed = max(0., min(1., (phase-.70)/.20))
        q[[0, 6, 7, 13]] += resumed*np.array([-.03, .10, -.03, -.10])
    if scene == "left_only":
        q[7:] = baseline[7:]
    return np.clip(q, np.maximum(limits[:, 0], baseline-.9)+1e-5,
                   np.minimum(limits[:, 1], baseline+.9)-1e-5)


def axis_reference_configuration(solver, q, baseline, neutral, phase, limits):
    """Offline fixture construction: sequential X, Y, Z translations.

    A small independent Jacobian solve finds a legal q. Only that q's actual
    URDF FK becomes controller input; this never updates physical state or
    precomputes any clutch-mapper output.
    """
    import pinocchio as pin
    blend = lambda begin, end: .5-.5*math.cos(math.pi*max(0., min(1., (phase-begin)/(end-begin))))
    displacement = np.array([-.02*blend(.12, .30), .015*blend(.32, .50), .015*blend(.52, .70)])
    targets = [pose.copy() for pose in neutral]
    for pose in targets:
        pose[:3, 3] += displacement
    for _ in range(3):
        pin.forwardKinematics(solver.model, solver.data, q)
        pin.updateFramePlacements(solver.model, solver.data)
        for offset, frame_id, target in ((0, solver.left_frame_id, targets[0]), (7, solver.right_frame_id, targets[1])):
            actual = solver.data.oMf[frame_id]
            error = np.concatenate((target[:3, 3]-actual.translation, pin.log3(target[:3, :3] @ actual.rotation.T)))
            jacobian = pin.computeFrameJacobian(solver.model, solver.data, q, frame_id,
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, offset:offset+7]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+1e-4*np.eye(6), error)
            step *= min(1., .02/max(1e-12, float(np.max(np.abs(step)))))
            q[offset:offset+7] += step
        q = np.clip(q, np.maximum(limits[:, 0], baseline-.8), np.minimum(limits[:, 1], baseline+.8))
    actual = solver.end_effector_poses(q)
    error = max(float(np.linalg.norm(pose[:3, 3]-target[:3, 3])) for pose, target in zip(actual, targets))
    return q, error


def make_fixtures(solver, baseline, limits, name, count, dt, coordinate_system):
    from scipy.spatial.transform import Rotation
    fixtures, costs = [], []
    axis_q = baseline.copy()
    neutral = solver.end_effector_poses(baseline)
    for index in range(count):
        began = time.perf_counter()
        phase = index/(count-1)
        stamp = 1000.+index*dt
        q = fixture_joint_pose(name, phase, baseline, limits)
        axis_error = 0.
        if name == "translation_xyz":
            axis_q, axis_error = axis_reference_configuration(solver, axis_q, baseline, neutral, phase, limits)
            q = axis_q
        poses = {side: pose.copy() for side, pose in zip(("left", "right"), solver.end_effector_poses(q))}
        enabled = {side: phase >= .08 for side in poses}
        tracked = {side: True for side in poses}
        ids = {side: int(enabled[side]) for side in poses}
        invalid = False
        if name == "left_only":
            enabled["right"], ids["right"] = False, 0
        if name == "tracking_loss":
            if .32 <= phase < .48:
                tracked = {side: False for side in poses}
            if .58 <= phase < .64:
                enabled = {side: False for side in poses}
            if phase >= .64:
                ids = {side: 2 for side in poses}
        if name == "clutch_reanchor":
            if .32 <= phase < .56:
                enabled = {side: False for side in poses}
            if phase >= .32:
                offset = min(1., (phase-.32)/.20)
                for pose in poses.values():
                    pose[:3, 3] += offset*np.array([.25, -.10, .12])
                    pose[:3, :3] = Rotation.from_euler("z", .35*offset).as_matrix() @ pose[:3, :3]
            if phase >= .56:
                ids = {side: 2 for side in poses}
        if name == "invalid_input":
            invalid = .30 <= phase < .50
            if .60 <= phase < .65:
                enabled = {side: False for side in poses}
            if phase >= .65:
                ids = {side: 2 for side in poses}
        if name == "unreachable":
            for pose in poses.values():
                pose[:3, 3] += (.5-.5*math.cos(math.pi*max(0., min(1., (phase-.12)/.55))))*np.array([.8, 0, .5])
        packet = packet_from_robot_poses(poses, seq=index, timestamp=stamp, coordinate_system=coordinate_system,
                                         enabled=enabled, tracked=tracked, activation_ids=ids)
        if invalid:
            kind = index % 5
            if kind == 0:
                packet["coordinate_system"] = "unknown_axes"
            elif kind == 1:
                packet["left"]["rotation_xyzw"] = [0, 0, 0, 0]
            elif kind == 2:
                packet["right"]["position"][0] = float("nan")
            elif kind == 3:
                packet["version"] = 999
            else:
                packet = None
        raw = b'{"broken"' if packet is None else json.dumps(packet, separators=(",", ":")).encode()
        fixtures.append({"raw": raw, "poses": poses, "enabled": enabled, "tracked": tracked,
                         "activation_ids": ids, "invalid": invalid, "timestamp": stamp,
                         "axis_reference_error_m": axis_error})
        costs.append((time.perf_counter()-began)*1000)
    return fixtures, distribution(costs)


class AnchorOracle:
    """Independent expected relative-pose contract, driven by fixture truth."""
    def __init__(self):
        self.anchors = {side: None for side in ("left", "right")}
        self.blocked = {side: False for side in self.anchors}

    def update(self, fixture, measured_poses):
        targets, anchored = {}, {}
        if fixture["invalid"]:
            for side in self.anchors:
                self.anchors[side], self.blocked[side] = None, True
            return targets, anchored
        for side in self.anchors:
            if not fixture["tracked"][side]:
                self.anchors[side], self.blocked[side] = None, True
                continue
            if not fixture["enabled"][side]:
                self.anchors[side], self.blocked[side] = None, False
                continue
            if self.blocked[side]:
                continue
            source = fixture["poses"][side]
            activation = fixture["activation_ids"][side]
            anchor = self.anchors[side]
            anchored[side] = anchor is None or anchor[2] != activation
            if anchored[side]:
                anchor = (source.copy(), measured_poses[side].copy(), activation)
                self.anchors[side] = anchor
            target = anchor[1].copy()
            target[:3, 3] += source[:3, 3]-anchor[0][:3, 3]
            target[:3, :3] = source[:3, :3] @ anchor[0][:3, :3].T @ anchor[1][:3, :3]
            targets[side] = target
        return targets, anchored


class ControllerWorld(BulletWorld):
    def reset(self, q):
        super().reset(q)
        self.motor_targets = np.asarray(q).copy()

    def submit(self, command):
        for side in active_sides(command.control_side):
            for local in range(7):
                arm_index = local+(0 if side == "left" else 7)
                index = self.indices[arm_index]
                info = self.all_info[index]
                value = float(command.positions[arm_index])
                self.p.setJointMotorControl2(self.body, index, self.p.POSITION_CONTROL,
                    targetPosition=value, targetVelocity=0, force=info[10],
                    maxVelocity=.98*min(info[11], self.max_joint_velocity),
                    positionGain=.18, velocityGain=1, physicsClientId=self.client)
                self.motor_targets[arm_index] = value


def run_controller_scene(world, solver, name, seconds=4., realtime=False, coordinate_system="unity", initial_q=None):
    from teleop_vr_controller import VrControllerConfig, VrControllerRetargeter, parse_controller_packet
    from vr_controller_bridge import ControllerCommandGate
    dt = world.dt
    world.reset(initial_configuration(world.limits) if initial_q is None else initial_q)
    for _ in range(50):
        world.step()
    baseline = world.measured()[0]
    solver.set_home_q(baseline)
    mapper = VrControllerRetargeter(VrControllerConfig(control_side="both", control_rate_hz=1/dt,
        max_joint_velocity=world.max_joint_velocity, max_joint_delta_per_step=.06,
        max_command_delta_from_baseline=1., position_scale=1., orientation_scale=1., task_mode="pose"), solver=solver)
    gate = ControllerCommandGate(baseline, control_rate=1/dt, max_joint_velocity=world.max_joint_velocity,
                                 max_step=.06, max_from_baseline=1.)
    count = max(20, round(seconds/dt))
    fixtures, fixture_cost = make_fixtures(solver, baseline, world.limits, name, count, dt, coordinate_system)
    oracle = AnchorOracle()
    timing = {key: [] for key in ("solver_ms", "mapper_ms", "packet_decode_ms", "control_compute_ms", "physics_ms",
               "validation_observer_ms", "pipeline_ms", "wake_lateness_ms")}
    records, position_errors, orientation_errors = [], [], []
    mapping_positions, mapping_angles, anchor_positions, anchor_joints = [], [], [], []
    actual_committed = {side: None for side in ("left", "right")}
    inactive_origin = {side: None for side in actual_committed}
    inactive_motion = 0.
    inactive_target_changes = invalid_submissions = history_violations = hard_violations = velocity_violations = 0
    unexpected_rejections = input_rejections = anchor_frames = deadline_misses = 0
    peak_velocity = np.zeros(14)
    wall_periods, last_begin = [], None
    started = time.perf_counter()
    for index, fixture in enumerate(fixtures):
        scheduled = started+index*dt
        if realtime and scheduled > time.perf_counter():
            time.sleep(scheduled-time.perf_counter())
        began = time.perf_counter()
        if last_begin is not None:
            wall_periods.append((began-last_begin)*1000)
        last_begin = began
        timing["wake_lateness_ms"].append(max(0., began-scheduled)*1000 if realtime else 0.)
        now = fixture["timestamp"]
        measured, _ = world.measured()
        oracle_begin = time.perf_counter()
        before = {side: pose for side, pose in zip(("left", "right"), world.effector_poses(solver))}
        expected, anchored = oracle.update(fixture, before)
        oracle_ms = (time.perf_counter()-oracle_begin)*1000
        reference = gate.reference(measured)
        for side, committed in actual_committed.items():
            if committed is not None and np.max(np.abs(reference[indices(side)]-committed)) > 1e-10:
                history_violations += 1
        previous_targets = world.motor_targets.copy()
        submitted, command, status = (), None, "握把已松开 · 不发送新目标"
        decode_ms = mapper_ms = solve_ms = 0.
        original_solve = solver.solve_cartesian
        inner_cost = [0.]
        def timed_solve(*args, **kwargs):
            inner = time.perf_counter()
            result = original_solve(*args, **kwargs)
            inner_cost[0] += (time.perf_counter()-inner)*1000
            return result
        solver.solve_cartesian = timed_solve
        try:
            decode_start = time.perf_counter()
            frame = parse_controller_packet(fixture["raw"], received_at=now)
            decode_ms = (time.perf_counter()-decode_start)*1000
            map_start = time.perf_counter()
            command = mapper.update(frame, current_q=measured, last_q_cmd=reference, now=now)
            mapper_ms = (time.perf_counter()-map_start)*1000
            solve_ms = inner_cost[0]
            if command is not None:
                submitted = active_sides(command.control_side)
                meta_targets = command.meta.get("target_pose_by_side", {})
                for side in submitted:
                    if side not in expected:
                        invalid_submissions += 1
                        continue
                    target = np.asarray(meta_targets[side])
                    mapping_positions.append(float(np.linalg.norm(target[:3, 3]-expected[side][:3, 3])))
                    mapping_angles.append(rotation_error(target[:3, :3], expected[side][:3, :3]))
                    if anchored.get(side):
                        anchor_frames += 1
                        anchor_positions.append(float(np.linalg.norm(target[:3, 3]-before[side][:3, 3])))
                        anchor_joints.append(float(np.max(np.abs(command.positions[indices(side)]-measured[indices(side)]))))
                guarded = gate.prepare(command, measured, now, now, now)
                world.submit(guarded)
                gate.commit(guarded, now)
                for side in submitted:
                    actual_committed[side] = np.asarray(guarded.positions[indices(side)]).copy()
                status = "重新对齐 · 当帧零位移" if any(anchored.values()) else "握把有效 · 空间位置与姿态跟随"
            elif expected and not any(anchored.values()):
                unexpected_rejections += 1
                status = "有效输入未生成目标 · 需要检查"
            elif not all(fixture["tracked"].values()):
                status = "跟踪失效 · 等待松手重握"
            elif any(fixture["enabled"].values()):
                status = "控制已暂停 · 请先松开握把再按下"
        except (ValueError, TypeError, RuntimeError) as exc:
            input_rejections += 1
            mapper.invalidate("simulation_invalid_input")
            if not fixture["invalid"]:
                unexpected_rejections += 1
            submitted, command = (), None
            status = "输入被拒绝 · 不发送新目标："+str(exc)
        finally:
            solver.solve_cartesian = original_solve
        for side in ("left", "right"):
            if side not in submitted and np.max(np.abs(world.motor_targets[indices(side)]-previous_targets[indices(side)])) > 1e-12:
                inactive_target_changes += 1
        if fixture["invalid"] and submitted:
            invalid_submissions += len(submitted)
        compute_end = time.perf_counter()
        timing["control_compute_ms"].append((compute_end-began)*1000-oracle_ms)
        timing["solver_ms"].append(solve_ms)
        timing["mapper_ms"].append(mapper_ms)
        timing["packet_decode_ms"].append(decode_ms)
        world.step()
        physics_end = time.perf_counter()
        timing["physics_ms"].append((physics_end-compute_end)*1000)
        measured_after, _ = world.measured()
        after = dict(zip(("left", "right"), world.effector_poses(solver)))
        for side in ("left", "right"):
            if side not in expected:
                if inactive_origin[side] is None:
                    inactive_origin[side] = measured_after[indices(side)].copy()
                inactive_motion = max(inactive_motion, float(np.max(np.abs(measured_after[indices(side)]-inactive_origin[side]))))
            else:
                inactive_origin[side] = None
        pos = max((float(np.linalg.norm(after[s][:3, 3]-target[:3, 3])) for s, target in expected.items()), default=0.)
        angle = max((rotation_error(after[s][:3, :3], target[:3, :3]) for s, target in expected.items()), default=0.)
        position_errors.append(pos)
        orientation_errors.append(angle)
        hard_violations += world.last_step_hard_violations
        velocity_violations += world.last_step_velocity_violations
        peak_velocity = np.maximum(peak_velocity, world.last_step_peak_velocity)
        if index % max(1, round(1/dt/20)) == 0 or index == count-1:
            records.append({"scene": name, "time": index*dt, "q": measured_after.tolist(), "command": world.motor_targets.tolist(),
                "points": world.skeleton(), "target": [expected.get(s, after[s]).tolist() for s in ("left", "right")],
                "actual": [after[s].tolist() for s in ("left", "right")], "position_error_m": pos, "orientation_error_rad": angle,
                "fault": fixture["invalid"] or not all(fixture["tracked"].values()), "command_applied": bool(submitted), "status": status})
        pipeline_ms = (time.perf_counter()-began)*1000
        timing["validation_observer_ms"].append((time.perf_counter()-physics_end)*1000+oracle_ms)
        timing["pipeline_ms"].append(pipeline_ms)
        deadline_misses += int(pipeline_ms > dt*1000)
    steady = slice(int(count*.8), None)
    reachable = name != "unreachable"
    position_threshold = .05 if name == "near_limits" else .02
    angle_threshold = .25 if name == "near_limits" else .10
    functional = {"hard_limits": hard_violations == 0, "velocity_limits": velocity_violations == 0,
        "independent_xyz_fixture": name != "translation_xyz" or max(f["axis_reference_error_m"] for f in fixtures) < 1e-4,
        "clutch_anchor_pose_zero": max(anchor_positions, default=0.) < 1e-6,
        "clutch_anchor_joint_zero": max(anchor_joints, default=0.) < 1e-10,
        "mapping_position_contract": max(mapping_positions, default=0.) < 1e-6,
        "mapping_orientation_contract": max(mapping_angles, default=0.) < 2e-5,
        "inactive_targets_unchanged": inactive_target_changes == 0, "inactive_settling_bounded": inactive_motion < .01,
        "invalid_input_stops": invalid_submissions == 0, "committed_side_history": history_violations == 0,
        "valid_input_accepted": unexpected_rejections == 0,
        "reachable_position": not reachable or max(position_errors[steady]) < position_threshold,
        "reachable_orientation": not reachable or max(orientation_errors[steady]) < angle_threshold}
    compute_ok = distribution(timing["control_compute_ms"])["p99"] < dt*1000
    simulation_ok = distribution(timing["pipeline_ms"])["p99"] < dt*1000
    pacing_ok = distribution(wall_periods)["p99"] <= dt*1000*1.1 and distribution(timing["wake_lateness_ms"])["p99"] <= dt*1000*.1
    checks = {**functional, "control_compute_p99_under_deadline": compute_ok, "pipeline_p99_under_deadline": simulation_ok,
              "realtime_pacing": not realtime or pacing_ok}
    return {"name": name, "frames": count, "coordinate_system": coordinate_system, "scope": "raw packet + actual clutch mapper + pose IK + production gate + physics",
        "passed": all(checks.values()), "checks": checks, "accuracy_safety_passed": all(functional.values()),
        "control_compute_passed": compute_ok, "simulation_budget_passed": simulation_ok,
        "realtime_pacing_passed": pacing_ok if realtime else None, "fixture_preparation_ms": fixture_cost,
        "axis_isolated_translation_fixture": name == "translation_xyz",
        "axis_reference_error_max_m": max(f["axis_reference_error_m"] for f in fixtures),
        "hard_limit_violation_frames": hard_violations, "velocity_violation_frames": velocity_violations,
        "compute_budget_misses": deadline_misses, "wall_period_overruns": sum(p > dt*1050 for p in wall_periods) if realtime else None,
        "wall_period_ms": distribution(wall_periods), "anchor_side_frames": anchor_frames,
        "anchor_target_position_max_m": max(anchor_positions, default=0.), "anchor_joint_delta_max_rad": max(anchor_joints, default=0.),
        "mapping_position_max_m": max(mapping_positions, default=0.), "mapping_rotation_max_rad": max(mapping_angles, default=0.),
        "inactive_motor_target_changes": inactive_target_changes, "inactive_settling_max_rad": inactive_motion,
        "invalid_input_submissions": invalid_submissions, "input_rejections": input_rejections, "unexpected_rejections": unexpected_rejections,
        "committed_history_violations": history_violations, "peak_measured_velocity_rad_s": peak_velocity.tolist(),
        "position_error_m": distribution(position_errors), "orientation_error_rad": distribution(orientation_errors),
        "steady_position_max_m": max(position_errors[steady]), "steady_orientation_max_rad": max(orientation_errors[steady]),
        "timing": {key: distribution(value) for key, value in timing.items()}}, records


def run_controller_simulation(output, control_hz=100., physics_substeps=4, duration=4., realtime=False,
                              max_joint_velocity=.48, scenarios=None, coordinate_system="mixed"):
    from threadpoolctl import threadpool_limits, threadpool_info
    if not math.isfinite(control_hz) or control_hz <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("control_hz and duration must be positive and finite")
    if coordinate_system not in {"mixed", "unity", "openxr"}:
        raise ValueError("coordinate_system must be mixed, unity or openxr")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    solver = build_controller_solver(control_hz, max_joint_velocity)
    with threadpool_limits(limits=1):
        world = ControllerWorld(output, dt=1/control_hz, substeps=physics_substeps, max_joint_velocity=max_joint_velocity)
        try:
            fk, dynamics = validate_fk(world, solver), validate_dynamics(world)
            scenes, frames = [], []
            for index, name in enumerate(SCENARIOS if scenarios is None else scenarios):
                source = ("unity" if index % 2 == 0 else "openxr") if coordinate_system == "mixed" else coordinate_system
                scene, recorded = run_controller_scene(world, solver, name, duration, realtime, source)
                scenes.append(scene)
                frames.extend(recorded)
            metadata = runtime_metadata()
            metadata["source_sha256"] = {f"app/src/{name}": hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                for name in ("teleop_controller_simulation.py", "teleop_vr_controller.py", "teleop_ik_core.py", "vr_controller_bridge.py", "teleop_simulation.py")}
            report = {"schema_version": 1, "engine": "PyBullet DIRECT", "control_hz": control_hz,
                "physics_hz": control_hz*physics_substeps, "viewer_hz": control_hz/max(1, round(control_hz/20)),
                "realtime_pacing": realtime, "max_joint_velocity_rad_s": max_joint_velocity,
                "runtime_metadata": metadata, "blas_threadpools": threadpool_info(), "geometry": world.manifest,
                "solver_config": {"iterations": solver.config.max_iterations, "damping": solver.config.damping,
                                  "max_joint_step_rad": solver.config.max_joint_delta_per_step,
                                  "smoothing_alpha": solver.config.smoothing_alpha},
                "fk_validation": fk, "dynamics_validation": dynamics, "scenarios": scenes,
                "accuracy_safety_passed": fk["passed"] and dynamics["passed"] and all(s["accuracy_safety_passed"] for s in scenes),
                "control_compute_passed": all(s["control_compute_passed"] for s in scenes),
                "simulation_budget_passed": all(s["simulation_budget_passed"] for s in scenes),
                "realtime_pacing_passed": all(s["realtime_pacing_passed"] for s in scenes) if realtime else None,
                "scope": "synthetic raw Unity/OpenXR poses, actual parser/clutch/Cartesian IK/publication gate and independent physics; no network/SDK",
                "limitations": ["fixed base; gravity -9.81; original URDF inertia/axes/limits retained",
                    "missing mesh spheres and self-collision disabled; not collision-safety evidence",
                    "simulated POSITION_CONTROL gains .18/1, motor speed 98% of operator/URDF intersection",
                    "device-side raw fixtures and oracle reference poses are prepared before timing; mapper outputs are never precomputed",
                    "finite poses mislabeled with the wrong supported coordinate system cannot be detected automatically"]}
            report["passed"] = bool(report["accuracy_safety_passed"] and all(s["passed"] for s in scenes))
            (output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            (output/"frames.json").write_text(json.dumps(frames, separators=(",", ":"), allow_nan=False), encoding="utf-8")
            write_viewer(output/"simulation.html", frames, world.edges, report)
            world.render(output/"preview.png")
            return report
        finally:
            world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE.parent/"data/controller-simulation")
    parser.add_argument("--control-hz", type=float, default=100.)
    parser.add_argument("--physics-substeps", type=int, default=4)
    parser.add_argument("--duration", type=float, default=4.)
    parser.add_argument("--max-joint-velocity", type=float, default=.48)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--coordinate-system", choices=("mixed", "unity", "openxr"), default="mixed")
    args = parser.parse_args()
    try:
        report = run_controller_simulation(args.output, args.control_hz, args.physics_substeps, args.duration,
            args.realtime, args.max_joint_velocity, args.scenario, args.coordinate_system)
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"CONTROLLER_SIMULATION_ERROR {exc}\n")
    print(json.dumps({"passed": report["passed"], "accuracy_safety_passed": report["accuracy_safety_passed"],
                      "control_compute_passed": report["control_compute_passed"], "output": str(args.output.resolve())}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
