"""Independent, fixed-base PyBullet closed loop for the Pinocchio arm solver.

No ROS, robot connection, native vendor solver or hand SDK is imported here.
Only missing mesh geometry is substituted; kinematics and inertias stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import struct
import time
from typing import Any
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS

SOURCE = Path(__file__).resolve().parent
URDF = SOURCE / "models/walker_tienkung_ei_description/walker_tienkung_ei.urdf"
SCENARIOS = ("reachable", "wrist_twist", "left_only", "near_limits", "unreachable", "tracking_loss", "limit_guard", "mocap_end_to_end")
PROFILES = ("skeleton-current", "skeleton-100hz")


def runtime_metadata():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    versions = {}
    for package in ("numpy", "scipy", "pin", "pybullet", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_model": cpu,
            "logical_cpu_count": os.cpu_count(), "package_versions": versions,
            "source_sha256": {f"app/src/{name}": hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                              for name in ("teleop_simulation.py", "teleop_ik_core.py", "teleop_skeleton_retarget.py")}}


def model_signature(root: ET.Element) -> str:
    """Signature deliberately excludes only visual/collision geometry."""
    data = [ET.tostring(j, encoding="unicode") for j in root.findall("joint")]
    data += [(link.get("name", "") + ET.tostring(link.find("inertial"), encoding="unicode"))
             for link in root.findall("link") if link.find("inertial") is not None]
    return hashlib.sha256("\n".join(data).encode()).hexdigest()


def prepare_urdf(source: Path, destination: Path) -> dict[str, Any]:
    tree = ET.parse(source)
    root = tree.getroot()
    before = model_signature(root)
    replacements = []
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for item in link.findall(kind):
                geometry = item.find("geometry")
                if geometry is None:
                    continue
                mesh = geometry.find("mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename", "")
                original = (source.parent / filename).resolve()
                if original.is_file() and not filename.startswith("package://"):
                    mesh.set("filename", str(original))
                else:
                    geometry.remove(mesh)
                    ET.SubElement(geometry, "sphere", {"radius": "0.035"})
                    replacements.append({"link": link.get("name"), "kind": kind, "missing_mesh": filename})
    after = model_signature(root)
    if before != after:
        raise RuntimeError("URDF joint/inertia preservation check failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "kinematics_inertia_sha256": before, "preserved": before == after,
            "missing_mesh_replacements": replacements, "collision_safety_validated": False}


def transform(position, quaternion) -> np.ndarray:
    x, y, z, w = quaternion
    squared = x*x+y*y+z*z+w*w
    if squared <= 1e-16:
        raise ValueError("invalid physics quaternion")
    scale = 2./squared
    xx, yy, zz = x*x*scale, y*y*scale, z*z*scale
    xy, xz, yz = x*y*scale, x*z*scale, y*z*scale
    wx, wy, wz = w*x*scale, w*y*scale, w*z*scale
    result = np.eye(4)
    result[:3, :3] = ((1-yy-zz, xy-wz, xz+wy), (xy+wz, 1-xx-zz, yz-wx), (xz-wy, yz+wx, 1-xx-yy))
    result[:3, 3] = position
    return result


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(max(-1., min(1., float((np.trace(a @ b.T) - 1.0) * .5))))


def distribution(values) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {key: 0.0 for key in ("p50", "p95", "p99", "max")}
    return {"p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)), "max": float(np.max(data))}


def command_is_valid(candidate, previous, measured, limits, velocity_limits, dt, max_step=.06) -> bool:
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (14,) or not np.all(np.isfinite(candidate)):
        return False
    lower, upper = np.asarray(limits).T
    if np.any(candidate < lower - 1e-9) or np.any(candidate > upper + 1e-9):
        return False
    delta = np.minimum(max_step, np.asarray(velocity_limits) * dt)
    return bool(np.all(np.abs(candidate - previous) <= delta + 1e-8)
                and np.all(np.abs(candidate - measured) <= delta + 1e-8))


def clamp_operator_command(candidate, previous, measured, baseline, limits, velocity_limits, dt,
                           max_step=.06, max_baseline_delta=1.):
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (14,) or not np.all(np.isfinite(candidate)):
        return None
    lower, upper = np.asarray(limits).T
    delta = np.minimum(max_step, np.asarray(velocity_limits)*dt)
    lower = np.maximum.reduce([lower, baseline-max_baseline_delta, measured-delta, previous-delta])
    upper = np.minimum.reduce([upper, baseline+max_baseline_delta, measured+delta, previous+delta])
    if np.any(lower > upper):
        return None
    return np.clip(candidate, lower, upper)


class BulletWorld:
    def __init__(self, output: Path, *, dt=.01, substeps=4, source=URDF, max_joint_velocity=.48):
        import pybullet as bullet
        self.p = bullet
        self.client = bullet.connect(bullet.DIRECT)
        if self.client < 0:
            raise RuntimeError("PyBullet DIRECT connection failed")
        self.dt, self.substeps = float(dt), int(substeps)
        self.max_joint_velocity = float(max_joint_velocity)
        if self.dt <= 0 or self.substeps < 1 or not math.isfinite(self.max_joint_velocity) or self.max_joint_velocity <= 0:
            self.close()
            raise ValueError("dt must be positive and substeps >= 1")
        path = output / "simulation_geometry.urdf"
        self.manifest = prepare_urdf(source, path)
        bullet.setPhysicsEngineParameter(fixedTimeStep=self.dt / self.substeps, numSolverIterations=80,
                                        enableFileCaching=0, physicsClientId=self.client)
        bullet.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.body = bullet.loadURDF(str(path), useFixedBase=True,
                                   flags=bullet.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.client)
        self.joints = {}
        self.links = {}
        self.edges = []
        self.all_info = []
        root = ET.parse(source).getroot()
        self.root_name = next(link.get("name") for link in root.findall("link")
                              if link.get("name") not in {j.find("child").get("link") for j in root.findall("joint")})
        for index in range(bullet.getNumJoints(self.body, physicsClientId=self.client)):
            info = bullet.getJointInfo(self.body, index, physicsClientId=self.client)
            self.all_info.append(info)
            self.joints[info[1].decode()] = index
            self.links[info[12].decode()] = index
            self.edges.append([int(info[16]) + 1, index + 1])
            if info[2] != bullet.JOINT_FIXED:
                bullet.setJointMotorControl2(self.body, index, bullet.POSITION_CONTROL, targetPosition=0,
                                            force=max(1.0, info[10]), maxVelocity=max(.1, info[11]),
                                            physicsClientId=self.client)
        self.indices = [self.joints[name] for name in ARM_JOINT_NAMES]
        self.urdf_velocity_limits = np.array([self.all_info[i][11] for i in self.indices])
        self.velocity_limits = np.minimum(self.urdf_velocity_limits, self.max_joint_velocity)
        self.limits = np.array([[self.all_info[i][8], self.all_info[i][9]] for i in self.indices])
        self._link_states = None
        base_position, base_quaternion = bullet.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        base_inertia = bullet.getDynamicsInfo(self.body, -1, physicsClientId=self.client)
        self.base_link_position = (transform(base_position, base_quaternion)
                                   @ np.linalg.inv(transform(base_inertia[3], base_inertia[4])))[:3, 3].tolist()
        self.frames = []

    def close(self):
        if getattr(self, "client", -1) >= 0:
            self.p.disconnect(physicsClientId=self.client)
            self.client = -1

    def reset(self, q):
        for info in self.all_info:
            if info[2] != self.p.JOINT_FIXED:
                self.p.resetJointState(self.body, info[0], 0, 0, physicsClientId=self.client)
        for index, value in zip(self.indices, q):
            self.p.resetJointState(self.body, index, float(value), 0, physicsClientId=self.client)
        self._link_states = None
        self.command(q)

    def measured(self):
        states = self.p.getJointStates(self.body, self.indices, physicsClientId=self.client)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def command(self, q):
        for index, value in zip(self.indices, q):
            info = self.all_info[index]
            self.p.setJointMotorControl2(self.body, index, self.p.POSITION_CONTROL,
                                        targetPosition=float(value), targetVelocity=0, force=info[10],
                                        maxVelocity=.98*min(info[11], self.max_joint_velocity), positionGain=.18, velocityGain=1,
                                        physicsClientId=self.client)

    def step(self):
        self.last_step_hard_violations = 0
        self.last_step_velocity_violations = 0
        self.last_step_peak_velocity = np.zeros(14)
        for _ in range(self.substeps):
            self.p.stepSimulation(physicsClientId=self.client)
            q, velocity = self.measured()
            self.last_step_peak_velocity = np.maximum(self.last_step_peak_velocity, np.abs(velocity))
            self.last_step_hard_violations += int(np.any(q < self.limits[:, 0]-1e-4) or np.any(q > self.limits[:, 1]+1e-4))
            self.last_step_velocity_violations += int(np.any(np.abs(velocity) > self.velocity_limits+1e-3))
        self._link_states = None

    def link_states(self):
        if self._link_states is None:
            self._link_states = self.p.getLinkStates(self.body, list(range(len(self.all_info))),
                                                    computeForwardKinematics=1, physicsClientId=self.client)
        return self._link_states

    def link_pose(self, index):
        result = self.link_states()[index]
        return transform(result[4], result[5])

    def effector_poses(self, solver):
        result = []
        for frame_id in (solver.left_frame_id, solver.right_frame_id):
            frame = solver.model.frames[frame_id]
            name = solver.model.names[frame.parentJoint]
            result.append(self.link_pose(self.joints[name]) @ frame.placement.homogeneous)
        return result

    def segment_directions(self, side):
        suffix = "l" if side == "left" else "r"
        points = [np.asarray(self.link_states()[self.joints[f"{part}_{suffix}_joint"]][4])
                  for part in ("shoulder_pitch", "elbow_pitch", "wrist_pitch")]
        return tuple(vector / np.linalg.norm(vector) for vector in (points[1]-points[0], points[2]-points[1]))

    def skeleton(self):
        return [self.base_link_position] + [list(state[4]) for state in self.link_states()]

    def render(self, path):
        p = self.p
        view = p.computeViewMatrixFromYawPitchRoll([0, 0, -.1], 2.5, 45, -15, 0, 2)
        projection = p.computeProjectionMatrixFOV(48, 1.0, .05, 10)
        rgba = np.asarray(p.getCameraImage(640, 640, view, projection, renderer=p.ER_TINY_RENDERER,
                                         physicsClientId=self.client)[2], dtype=np.uint8).reshape(640, 640, 4)
        write_png(path, rgba)


def build_solver(profile, dt):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    from teleop_ik_core import TeleopArmIK
    from teleop_ik_types import TeleopIKConfig
    return TeleopArmIK(TeleopIKConfig(joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
                       control_dt=dt, max_iterations=6, damping=.08, max_joint_delta_per_step=.06,
                       smoothing_alpha=.65, orientation_weight=.25, nullspace_gain=.015,
                       enable_local_log=False, log_to_console=False, show_model_load_progress=False,
                       log_ik_result=False))


def validate_fk(world, solver, samples=8):
    random = np.random.default_rng(7301)
    position_errors, angles, direction_angles = [], [], []
    for _ in range(samples):
        q = random.uniform(world.limits[:, 0] * .7, world.limits[:, 1] * .7)
        world.reset(q)  # Reset is only used for FK validation / scenario initialization.
        for a, b in zip(solver.end_effector_poses(q), world.effector_poses(solver)):
            position_errors.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
            angles.append(rotation_error(a[:3, :3], b[:3, :3]))
        for side in ("left", "right"):
            for a, b in zip(solver.segment_directions(q, side), world.segment_directions(side)):
                direction_angles.append(math.acos(float(np.clip(np.dot(a, b), -1, 1))))
    result = {"samples": samples, "position_max_m": max(position_errors), "rotation_max_rad": max(angles),
              "segment_direction_max_rad": max(direction_angles)}
    result["passed"] = result["position_max_m"] < 2e-6 and max(angles + direction_angles) < 2e-5
    return result


def validate_dynamics(world, source=URDF):
    """Check that the engine actually retained moving-link mass/inertia."""
    masses, inertias = [], []
    for link in ET.parse(source).getroot().findall("link"):
        if link.get("name") not in world.links:  # The base is deliberately static.
            continue
        inertia = link.find("inertial/inertia")
        mass = link.find("inertial/mass")
        if inertia is None or mass is None:
            continue
        a = {key: float(value) for key, value in inertia.attrib.items()}
        matrix = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]],
                           [a["ixz"], a["iyz"], a["izz"]]])
        actual = world.p.getDynamicsInfo(world.body, world.links[link.get("name")], physicsClientId=world.client)
        masses.append(abs(actual[0]-float(mass.get("value"))))
        inertias.append(float(np.max(np.abs(np.sort(actual[2])-np.linalg.eigvalsh(matrix)))))
    return {"moving_links_checked": len(masses), "mass_max_error_kg": max(masses, default=0.),
            "principal_inertia_max_error_kg_m2": max(inertias, default=0.), "fixed_base_excluded": True,
            "passed": bool(masses and max(masses) < 1e-8 and max(inertias) < 1e-7)}


def initial_configuration(limits):
    q = np.array([-.12, .15, 0, -.45, -.25, 0, 0, -.12, -.15, 0, -.45, .25, 0, 0])
    return np.clip(q, limits[:, 0], limits[:, 1])


def desired_configuration(scene, phase, start, limits):
    # Smooth rise followed by a stationary final third, allowing settling metrics.
    rise = .5 - .5 * math.cos(math.pi * min(1., phase / .65))
    q = start.copy()
    if scene == "wrist_twist":
        q[[5, 6, 12, 13]] += rise * np.array([.24, .45, -.24, -.45])
    elif scene == "near_limits":
        q[[1, 6, 8, 13]] = (1-rise)*start[[1, 6, 8, 13]] + rise*np.array([
            limits[1, 1]-.04, limits[6, 1]-.04, limits[8, 0]+.04, limits[13, 0]+.04])
    else:
        q += rise*np.array([-.12, .09, .12, -.25, -.12, .12, .16, -.10, -.08, -.12, -.22, .12, -.12, -.16])
        if scene == "left_only":
            q[7:] = start[7:]
    return np.clip(q, limits[:, 0]+1e-5, limits[:, 1]-1e-5)


def fixture_targets(solver, world, name, phase, start):
    limits = np.column_stack((np.maximum(world.limits[:, 0], start-1.), np.minimum(world.limits[:, 1], start+1.)))
    desired = desired_configuration(name, phase, start, limits)
    targets = solver.end_effector_poses(desired)
    directions = {side: solver.segment_directions(desired, side) for side in ("left", "right")}
    orientations = {side: targets[i][:3, :3].copy() for i, side in enumerate(("left", "right"))}
    if name == "unreachable":
        directions = {side: (np.array([0., 0., 1.]), np.array([0., 0., 1.])) for side in ("left", "right")}
        for i in range(2):
            targets[i][:3, 3] = [2., .6 if i == 0 else -.6, 2.]
    return targets, directions, orientations


def canonical_mocap_frame(directions, poses, timestamp, phase=0., return_packet=False):
    """Synthetic upper-body fixture, including real 64-node encode/decode.

    Unused legs/fingers remain explicitly untracked; this is not hand SDK data.
    """
    from scipy.spatial.transform import Rotation
    from teleop_mocap_protocol import (JointPose, MOCAP_JOINT_NAMES, MocapFrame,
                                       encode_mocap_g2_frame, parse_mocap_packet)
    positions = {"chest": [0, 0, 1.3], "waist": [0, 0, 1.0], "neck": [0, 0, 1.55],
                 "head": [0, 0, 1.7], "spine1": [0, 0, 1.2], "spine2": [0, 0, 1.1],
                 "left_hip": [0, .14, .95], "right_hip": [0, -.14, .95]}
    yaw = .35*math.sin(math.pi*phase)
    body = Rotation.from_euler("z", yaw).as_matrix()
    offset = np.array([.15*math.sin(math.pi*phase), .1*math.sin(math.pi*phase), 0.])
    hand_rotations = {}
    for i, side in enumerate(("left", "right")):
        shoulder = np.array([0., .22 if side == "left" else -.22, 1.45])
        elbow = shoulder+.33*directions[side][0]
        wrist = elbow+.34*directions[side][1]
        positions.update({f"{side}_shoulder": shoulder, f"{side}_elbow": elbow,
                          f"{side}_wrist": wrist, f"{side}_hand": wrist})
        hand_rotations[f"{side}_hand"] = poses[i][:3, :3]
    joints = []
    for name in MOCAP_JOINT_NAMES:
        if name not in positions:
            joints.append(JointPose((0., 0., 0.), (0., 0., 0., 0.), tracked=False))
        else:
            rotation = body @ hand_rotations.get(name, np.eye(3))
            joints.append(JointPose(tuple(body @ np.asarray(positions[name])+offset),
                                    tuple(Rotation.from_matrix(rotation).as_quat()), tracked=True))
    frame = MocapFrame(timestamp_s=timestamp, timestamp_us=round(timestamp*1e6), receive_time_s=timestamp,
                       joints=tuple(joints), source="simulation_synthetic_upper_body")
    packet = encode_mocap_g2_frame(frame)
    return packet if return_packet else parse_mocap_packet(packet, timestamp, device_mode="glove")


def align_mocap_mapper(world, solver, start):
    from teleop_direct_retarget import DirectRetargetConfig
    from teleop_skeleton_retarget import SkeletonRetargeter
    from teleop_startup_pose import NaturalDownStartupConfig, NaturalDownStartupGate
    mapper = SkeletonRetargeter(DirectRetargetConfig(joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
                 control_side="both", max_joint_delta_per_step=.06, max_command_delta_from_baseline=1.,
                 direction_iterations=6, direction_damping=.08), solver=solver, orientation_scale=1.)
    gate = NaturalDownStartupGate(NaturalDownStartupConfig(hold_seconds=2.))
    targets, directions, _ = fixture_targets(solver, world, "mocap_end_to_end", 0., start)
    for index in range(math.ceil(2.1/world.dt)+1):
        frame = canonical_mocap_frame(directions, targets, index*world.dt)
        measured, _ = world.measured()
        status = gate.observe(index*world.dt, frame, measured,
                              np.asarray(frame.joint("left_wrist").position), np.asarray(frame.joint("right_wrist").position))
        if status.ready:
            mapper.align(status.window_frames, status.robot_baseline_q)
            return mapper, {"passed": True, "held_seconds": status.held_seconds,
                            "window_frames": len(status.window_frames), "metrics": status.metrics}
        world.step()
    raise ValueError(f"Synthetic upper-body fixture cannot pass actual startup gate: {status.reasons}")


def run_scene(world, solver, name, seconds, realtime):
    dt = world.dt
    full_mapper = name == "mocap_end_to_end"
    start = initial_configuration(world.limits)
    if full_mapper:
        start = np.zeros(14)
        start[[0, 7]], start[[3, 10]] = -.06, -.18
    world.reset(start)
    solver.set_home_q(start)
    previous = start.copy()
    # Physics settling/warm-up is excluded from deadline statistics.
    for _ in range(50):
        world.step()
    mapper, startup = align_mocap_mapper(world, solver, start) if full_mapper else (None, None)
    count = max(10, round(seconds / dt))
    records = []
    timings = {key: [] for key in ("solver_ms", "mapper_ms", "packet_decode_ms", "control_compute_ms", "physics_ms",
                                  "validation_observer_ms", "pipeline_ms", "pipeline_full_mapper_ms", "target_generation_ms", "wake_lateness_ms")}
    prepared = []
    preparation_ms = []
    for index in range(count):
        begin_preparation = time.perf_counter()
        fixture = fixture_targets(solver, world, name, index/(count-1), start)
        packet = canonical_mocap_frame(fixture[1], fixture[0], 3.+index*dt, index/(count-1), return_packet=True) if full_mapper else None
        prepared.append((fixture, packet))
        preparation_ms.append((time.perf_counter()-begin_preparation)*1000)
    viewer_stride = max(1, round(1/dt/20))
    position_errors, orientation_errors, direction_errors, joint_errors = [], [], [], []
    hard_violations = velocity_violations = command_rejections = fault_commands = deadline_missed = 0
    fault_frames = 0
    raw_solver_hard_violations = 0
    peak_velocity = np.zeros(14)
    uncontrolled_command_change = uncontrolled_measured_drift = 0.
    wall_start = time.perf_counter()
    wall_periods = []
    last_begin = None
    for index in range(count):
        scheduled = wall_start + index*dt
        if realtime:
            delay = scheduled-time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        begin = time.perf_counter()
        if last_begin is not None:
            wall_periods.append((begin-last_begin)*1000)
        last_begin = begin
        timings["wake_lateness_ms"].append(max(0., begin-scheduled)*1000 if realtime else 0.)
        measured, velocity = world.measured()
        phase = index/(count-1)
        generation_start = time.perf_counter()
        targets, directions, orientations = prepared[index][0]
        if full_mapper:
            from teleop_mocap_protocol import parse_mocap_packet
            mocap = parse_mocap_packet(prepared[index][1], 3.+index*dt, device_mode="glove").to_replay_dict()
        else:
            mocap = None
        timings["packet_decode_ms"].append((time.perf_counter()-generation_start)*1000 if full_mapper else 0.)
        active = ("left",) if name == "left_only" else ("left", "right")
        fault = name == "tracking_loss" and .3 <= phase <= .65
        timings["target_generation_ms"].append(preparation_ms[index])
        solve_ms = 0.
        mapper_ms = 0.
        applied = False
        if fault:
            fault_frames += 1
        else:
            solve_start = time.perf_counter()
            if full_mapper:
                measured_solve_ms = [0.]
                original_solve = solver.solve_segment_directions
                def measured_solve(**kwargs):
                    inner = time.perf_counter()
                    result = original_solve(**kwargs)
                    measured_solve_ms[0] += (time.perf_counter()-inner)*1000
                    return result
                solver.solve_segment_directions = measured_solve
                try:
                    command = mapper.command_from_frame(mocap, measured, previous, timestamp=3.+index*dt)
                    candidate = np.asarray(command.positions).copy()
                finally:
                    solver.solve_segment_directions = original_solve
                mapper_ms = (time.perf_counter()-solve_start)*1000
                solve_ms = measured_solve_ms[0]
            else:
                candidate, _ = solver.solve_segment_directions(
                    timestamp=index*dt, control_side="left" if name == "left_only" else "both",
                    target_directions={side: directions[side] for side in active},
                    target_orientations={side: orientations[side] for side in active},
                    current_q=measured, last_q_cmd=previous, upper_weight=1., forearm_weight=.85,
                    max_iterations=6, damping=.08, orientation_weight=.25)
                solve_ms = (time.perf_counter()-solve_start)*1000
            raw_solver_hard_violations += int(not np.all(np.isfinite(candidate)) or np.any(candidate < world.limits[:, 0]-1e-8)
                                              or np.any(candidate > world.limits[:, 1]+1e-8))
            if name == "left_only":
                # Match controlled-side publication: no new right-arm target.
                candidate[7:] = previous[7:]
            if name == "limit_guard" and .35 <= phase <= .65:
                candidate = world.limits[:, 1]+.2  # Deliberate invalid candidate; never sent to physics.
            else:
                candidate = clamp_operator_command(candidate, previous, measured, start, world.limits, world.velocity_limits, dt)
            if candidate is not None and command_is_valid(candidate, previous, measured, world.limits, world.velocity_limits, dt):
                world.command(candidate)
                previous = candidate.copy()
                applied = True
            else:
                command_rejections += 1
        timings["solver_ms"].append(solve_ms)
        timings["mapper_ms"].append(mapper_ms)
        if fault and applied:
            fault_commands += 1
        control_end = time.perf_counter()
        timings["control_compute_ms"].append((control_end-begin)*1000)
        world.step()
        physics_end = time.perf_counter()
        timings["physics_ms"].append((physics_end-control_end)*1000)
        actual_q, actual_velocity = world.measured()
        actual = world.effector_poses(solver)
        pos = max(float(np.linalg.norm(actual[i][:3, 3]-targets[i][:3, 3])) for i in range(1 if name == "left_only" else 2))
        angle = max(rotation_error(actual[i][:3, :3], targets[i][:3, :3]) for i in range(1 if name == "left_only" else 2))
        direction = max(math.acos(float(np.clip(np.dot(a, b), -1, 1))) for side in active
                        for a, b in zip(world.segment_directions(side), directions[side]))
        position_errors.append(pos)
        orientation_errors.append(angle)
        direction_errors.append(direction)
        joint_errors.append(float(np.max(np.abs(actual_q-previous))))
        hard_violations += world.last_step_hard_violations
        velocity_violations += world.last_step_velocity_violations
        peak_velocity = np.maximum(peak_velocity, world.last_step_peak_velocity)
        if name == "left_only":
            uncontrolled_command_change = max(uncontrolled_command_change, float(np.max(np.abs(previous[7:]-start[7:]))))
            uncontrolled_measured_drift = max(uncontrolled_measured_drift, float(np.max(np.abs(actual_q[7:]-start[7:]))))
        if index % viewer_stride == 0 or index == count-1:
            records.append({"time": index*dt, "scene": name, "q": actual_q.tolist(), "command": previous.tolist(),
                            "points": world.skeleton(), "target": [p.tolist() for p in targets],
                            "actual": [p.tolist() for p in actual], "fault": fault, "command_applied": applied,
                            "position_error_m": pos, "orientation_error_rad": angle})
        pipeline_ms = (time.perf_counter()-begin)*1000
        timings["validation_observer_ms"].append((time.perf_counter()-physics_end)*1000)
        timings["pipeline_ms"].append(pipeline_ms)
        if full_mapper:
            timings["pipeline_full_mapper_ms"].append(pipeline_ms)
        deadline_missed += int(pipeline_ms > dt*1000)
    steady = slice(int(count*.8), None)
    expected_reachable = name not in {"unreachable", "limit_guard"}
    checks = {"hard_limits": hard_violations == 0, "velocity_limits": velocity_violations == 0,
              "raw_solver_finite_and_in_limits": raw_solver_hard_violations == 0,
              "fault_stops_new_commands": fault_commands == 0,
              "limit_guard_rejects": name != "limit_guard" or command_rejections > 0,
              "uncontrolled_arm_holds": name != "left_only" or (uncontrolled_command_change < 1e-9 and uncontrolled_measured_drift < .01),
              "reachable_direction": not expected_reachable or max(direction_errors[steady]) < .12,
              "reachable_position": not expected_reachable or max(position_errors[steady]) < .045,
              "reachable_orientation": not expected_reachable or max(orientation_errors[steady]) < .22,
              "control_compute_p99_under_deadline": distribution(timings["control_compute_ms"])["p99"] < dt*1000,
              "pipeline_p99_under_deadline": distribution(timings["pipeline_ms"])["p99"] < dt*1000,
              "realtime_period_p99": not realtime or distribution(wall_periods)["p99"] <= dt*1000*1.1,
              "realtime_wake_p99": not realtime or distribution(timings["wake_lateness_ms"])["p99"] <= dt*1000*.1}
    timing_checks = {"control_compute_p99_under_deadline", "pipeline_p99_under_deadline", "realtime_period_p99", "realtime_wake_p99"}
    return {"name": name, "frames": count, "passed": all(checks.values()), "checks": checks,
            "accuracy_safety_passed": all(value for key, value in checks.items() if key not in timing_checks),
            "control_compute_passed": checks["control_compute_p99_under_deadline"],
            "simulation_budget_passed": checks["pipeline_p99_under_deadline"],
            "realtime_pacing_passed": checks["realtime_period_p99"] and checks["realtime_wake_p99"] if realtime else None,
            "scope": "64-node encode/decode + startup gate + mapper + IK + physics" if full_mapper else "solver-target + physics",
            "startup": startup, "target_generation_in_pipeline": False, "packet_decode_in_pipeline": full_mapper,
            "viewer_hz": 1/dt/viewer_stride,
            "deadline_missed": deadline_missed, "hard_limit_violation_frames": hard_violations,
            "compute_budget_misses": deadline_missed,
            "wall_period_ms": distribution(wall_periods),
            "wall_period_overruns": sum(period > dt*1000*1.05 for period in wall_periods) if realtime else None,
            "wall_period_overrun_threshold_ms": dt*1000*1.05 if realtime else None,
            "velocity_violation_frames": velocity_violations, "command_rejections": command_rejections,
            "raw_solver_invalid_candidates": raw_solver_hard_violations,
            "peak_measured_velocity_rad_s": peak_velocity.tolist(),
            "violation_sample_rate_hz": 1/dt*world.substeps,
            "uncontrolled_command_change_max_rad": uncontrolled_command_change,
            "uncontrolled_measured_drift_max_rad": uncontrolled_measured_drift,
            "fault_frames": fault_frames, "commands_during_fault": fault_commands,
            "position_error_m": distribution(position_errors), "orientation_error_rad": distribution(orientation_errors),
            "direction_error_rad": distribution(direction_errors), "joint_tracking_error_rad": distribution(joint_errors),
            "steady_position_max_m": max(position_errors[steady]), "steady_orientation_max_rad": max(orientation_errors[steady]),
            "timing": {key: distribution(values) for key, values in timings.items()}}, records


def write_png(path, rgba):
    height, width, _ = rgba.shape
    def chunk(kind, value):
        return struct.pack(">I", len(value))+kind+value+struct.pack(">I", zlib.crc32(kind+value)&0xffffffff)
    rows = b"".join(b"\x00"+row.tobytes() for row in rgba)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0))
                     +chunk(b"IDAT", zlib.compress(rows))+chunk(b"IEND", b""))


def write_viewer(path, frames, edges, report):
    template = (SOURCE / "simulation_viewer.html").read_text(encoding="utf-8")
    payload = json.dumps({"frames": frames, "edges": edges, "report": report}, separators=(",", ":"), allow_nan=False)
    path.write_text(template.replace("__SIMULATION_DATA__", payload.replace("</", "<\\/")), encoding="utf-8")


def run_simulation(output, control_hz=100., physics_substeps=4, profile="skeleton-100hz", seconds_per_scene=4., realtime=False,
                   max_joint_velocity=.48, scenarios=None):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(control_hz) or control_hz <= 0 or not math.isfinite(seconds_per_scene) or seconds_per_scene <= 0:
        raise ValueError("control_hz and seconds_per_scene must be positive")
    from threadpoolctl import threadpool_limits, threadpool_info
    solver = build_solver(profile, 1/control_hz)
    with threadpool_limits(limits=1):
        world = BulletWorld(output, dt=1/control_hz, substeps=physics_substeps, max_joint_velocity=max_joint_velocity)
        try:
            fk = validate_fk(world, solver)
            dynamics = validate_dynamics(world)
            scenes, frames = [], []
            for name in (SCENARIOS if scenarios is None else scenarios):
                summary, recorded = run_scene(world, solver, name, seconds_per_scene, realtime)
                scenes.append(summary)
                frames.extend(recorded)
            report = {"schema_version": 1, "engine": "PyBullet DIRECT", "control_hz": control_hz,
                      "runtime_metadata": runtime_metadata(),
                      "physics_hz": control_hz*physics_substeps, "physics_substeps": physics_substeps,
                      "viewer_hz": control_hz/max(1, round(control_hz/20)),
                      "max_joint_velocity_rad_s": max_joint_velocity, "max_joint_step_rad": .06,
                      "max_baseline_delta_rad": 1.,
                      "realtime_pacing": realtime, "profile": profile,
                      "fixed_base": True, "gravity_m_s2": [0, 0, -9.81],
                      "actuator_model": {"mode": "POSITION_CONTROL", "position_gain": .18, "velocity_gain": 1.,
                                         "force_limits": "URDF effort", "max_velocity": "0.98 * min(URDF velocity, operator velocity)",
                                         "numerical_velocity_margin_ratio": .98,
                                         "motor_velocity_caps_rad_s": (.98*world.velocity_limits).tolist()},
                      "profile_note": "current and 100hz use identical solver parameters; algorithm benchmarking is separate",
                      "blas_threadpools": threadpool_info(), "fk_validation": fk, "geometry": world.manifest,
                      "dynamics_validation": dynamics,
                      "passed": bool(fk["passed"] and dynamics["passed"] and all(scene["passed"] for scene in scenes)), "scenarios": scenes,
                      "accuracy_safety_passed": bool(fk["passed"] and dynamics["passed"] and all(s["accuracy_safety_passed"] for s in scenes)),
                      "control_compute_passed": all(s["control_compute_passed"] for s in scenes),
                      "simulation_budget_passed": all(s["simulation_budget_passed"] for s in scenes),
                      "realtime_pacing_passed": all(s["realtime_pacing_passed"] for s in scenes) if realtime else None,
                      "scope": "7 solver-target scenes and 1 synthetic 64-node protocol/startup/full-mapper scene; excludes ROS/network/SDK",
                      "mocap_fixture": {"human_upper_arm_m": .33, "human_forearm_m": .34,
                                        "unused_finger_leg_nodes": "zero position/quaternion, explicitly untracked",
                                        "body_motion": "synthetic translation and yaw removed by actual body-relative mapper"},
                      "limitations": ["missing meshes replaced by 35mm spheres", "self-collision disabled; no collision-safety claim",
                                      "URDF inertia/effort/velocity retained; motor gains are simulated, not identified hardware",
                                      "timings exclude startup/rendering and synthetic device input preparation; protocol decode/body mapping/IK/physics stay in full-mapper pipeline",
                                      "tracking loss is simulation input fault injection, not a ROS bridge integration test",
                                      "full mapper fixture validates upper body only, with unused nodes marked untracked, not a hand SDK/full-body test"]}
            (output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            (output/"frames.json").write_text(json.dumps(frames, separators=(",", ":"), allow_nan=False), encoding="utf-8")
            write_viewer(output/"simulation.html", frames, world.edges, report)
            world.render(output/"preview.png")
            return report
        finally:
            world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE.parent/"data/simulation")
    parser.add_argument("--control-hz", type=float, default=100.)
    parser.add_argument("--physics-substeps", type=int, default=4)
    parser.add_argument("--profile", choices=PROFILES, default="skeleton-100hz")
    parser.add_argument("--max-joint-velocity", type=float, default=.48, help="Operator velocity cap in rad/s, intersected with URDF")
    parser.add_argument("--duration", "--seconds-per-scene", type=float, default=4., help="Simulated seconds per scenario")
    parser.add_argument("--realtime", action="store_true", help="Pace control ticks against absolute monotonic deadlines")
    parser.add_argument("--scenario", choices=SCENARIOS, action="append", help="Run only a selected scenario; repeat for multiple")
    args = parser.parse_args()
    try:
        report = run_simulation(args.output, args.control_hz, args.physics_substeps, args.profile,
                                args.duration, args.realtime, args.max_joint_velocity, args.scenario)
    except (ImportError, ValueError, OSError) as exc:
        parser.exit(1, f"SIMULATION_ERROR {exc}\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve()),
                      "scenarios": [{"name": s["name"], "passed": s["passed"]} for s in report["scenarios"]]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
