"""Versioned controller poses and relative Cartesian IK; no transport or ROS.

Clutch anchors are per arm. The robot baseline belongs to the whole session,
and proposed commands are never recorded as if they had been published.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from teleop_control_safety import feedback_is_fresh, joint_velocity_limits_from_urdf
from teleop_ik_types import IKCommand, TeleopIKConfig, TeleopIKInput
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS

MAX_CONTROLLER_PACKET_BYTES = 16384
_BASES = {
    "unity": np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]]),
    "openxr": np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]),
    "robot": np.eye(3),
}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return tuple(_number(item, label) for item in value)


@dataclass(frozen=True)
class ControllerPose:
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    tracked: bool
    enabled: bool
    activation_id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        quaternion = _vector(self.rotation_xyzw, 4, "rotation_xyzw")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
            raise ValueError("rotation_xyzw must be a unit quaternion (within 1%)")
        object.__setattr__(self, "rotation_xyzw", tuple(value / norm for value in quaternion))
        if type(self.tracked) is not bool or type(self.enabled) is not bool:
            raise ValueError("tracked and enabled must be booleans")
        if self.activation_id is not None:
            _integer(self.activation_id, "activation_id")


@dataclass(frozen=True)
class ControllerFrame:
    sequence: int
    timestamp: float
    coordinate_system: str
    left: ControllerPose
    right: ControllerPose
    received_at: float

    def __post_init__(self):
        _integer(self.sequence, "sequence")
        for name in ("timestamp", "received_at"):
            value = _number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if not isinstance(self.coordinate_system, str) or self.coordinate_system not in _BASES:
            raise ValueError("coordinate_system must be unity, openxr or robot")
        if not isinstance(self.left, ControllerPose) or not isinstance(self.right, ControllerPose):
            raise ValueError("both controller poses are required")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_controller_packet(data: bytes | str | Mapping, received_at: float) -> ControllerFrame:
    """Parse JSON only. Never trust a sender-supplied host receipt timestamp."""
    if isinstance(data, bytes):
        if len(data) > MAX_CONTROLLER_PACKET_BYTES:
            raise ValueError("controller packet exceeds 16 KiB")
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("controller packet must be UTF-8 JSON") from exc
    if isinstance(data, str):
        if len(data.encode("utf-8")) > MAX_CONTROLLER_PACKET_BYTES:
            raise ValueError("controller packet exceeds 16 KiB")
        try:
            data = json.loads(data, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("controller packet is not valid JSON") from exc
    if not isinstance(data, Mapping) or data.get("type") != "vr_controller":
        raise ValueError("expected a vr_controller JSON object; binary C8FF is not supported")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("unsupported controller protocol version")
    try:
        def pose(side):
            raw = data[side]
            if not isinstance(raw, Mapping):
                raise ValueError(f"{side} must be a pose object")
            return ControllerPose(raw["position"], raw["rotation_xyzw"], raw["tracked"], raw["enabled"], raw.get("activation_id"))
        return ControllerFrame(data["sequence"], data["timestamp"], data["coordinate_system"],
                               pose("left"), pose("right"), received_at)
    except KeyError as exc:
        raise ValueError(f"controller packet is missing {exc.args[0]}") from exc


def controller_basis(coordinate_system: str) -> np.ndarray:
    try:
        return _BASES[coordinate_system].copy()
    except KeyError as exc:
        raise ValueError("coordinate_system must be unity, openxr or robot") from exc


def pose_in_robot(pose: ControllerPose, coordinate_system: str) -> np.ndarray:
    basis = controller_basis(coordinate_system)
    result = np.eye(4)
    result[:3, 3] = basis @ np.asarray(pose.position)
    result[:3, :3] = basis @ Rotation.from_quat(pose.rotation_xyzw).as_matrix() @ basis.T
    return result


@dataclass(frozen=True)
class VrControllerConfig:
    control_side: str = "both"
    control_rate_hz: float = 100.0
    max_frame_age_seconds: float = 0.3
    max_joint_velocity: float = 0.48
    max_joint_delta_per_step: float = 0.06
    max_command_delta_from_baseline: float = 1.0
    position_scale: float = 1.0
    orientation_scale: float = 1.0
    task_mode: str = "pose"

    def __post_init__(self):
        if self.control_side not in {"left", "right", "both"}:
            raise ValueError("control_side must be left, right or both")
        if self.task_mode not in {"pose", "position_only"}:
            raise ValueError("task_mode must be pose or position_only")
        for name in ("control_rate_hz", "max_frame_age_seconds", "max_joint_velocity",
                     "max_joint_delta_per_step", "max_command_delta_from_baseline"):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("position_scale", "orientation_scale"):
            if not 0 <= _number(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass
class _Clutch:
    enabled: bool = False
    blocked: bool = False
    activation_id: int | None = None
    pending_anchor: bool = False
    controller_anchor: np.ndarray | None = None
    robot_anchor: np.ndarray | None = None
    previous_controller_rotation: np.ndarray | None = None
    scaled_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))


def _indices(side):
    return np.arange(0, 7) if side == "left" else np.arange(7, 14)


class VrControllerRetargeter:
    def __init__(self, config: VrControllerConfig = VrControllerConfig(), solver=None):
        self.config = config
        if solver is None:
            from teleop_ik_core import TeleopArmIK
            solver = TeleopArmIK(TeleopIKConfig(
                joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
                task_mode=config.task_mode, control_dt=1.0 / config.control_rate_hz,
                max_iterations=6, damping=0.08,
                max_joint_delta_per_step=min(config.max_joint_delta_per_step, config.max_joint_velocity / config.control_rate_hz),
                smoothing_alpha=1.0, enable_local_log=False, log_to_console=False,
                show_model_load_progress=False, log_ik_result=False,
            ))
        self.solver = solver
        self.sides = ("left", "right") if config.control_side == "both" else (config.control_side,)
        self._clutches = {side: _Clutch() for side in self.sides}
        self._limits = np.asarray(solver.joint_limits, dtype=float)
        if self._limits.shape != (14, 2) or not np.all(np.isfinite(self._limits)) or np.any(self._limits[:, 0] > self._limits[:, 1]):
            raise ValueError("solver must provide finite 14-axis hard joint limits")
        urdf = Path(__file__).resolve().parent / "models/walker_tienkung_ei_description/walker_tienkung_ei.urdf"
        velocities = np.asarray(joint_velocity_limits_from_urdf(urdf, ARM_JOINT_NAMES))
        self._step = np.minimum(config.max_joint_delta_per_step, np.minimum(config.max_joint_velocity, velocities) / config.control_rate_hz)
        self._baseline_q = None
        self.latest_frame: ControllerFrame | None = None
        self._last_sequence = None
        self._last_timestamp = None
        self._last_received = None
        self.last_invalidation_reason = None

    @property
    def session_baseline_q(self):
        return None if self._baseline_q is None else self._baseline_q.copy()

    def invalidate(self, reason, sides=None):
        affected = self.sides if sides is None else tuple(sides)
        if any(side not in self._clutches for side in affected):
            raise ValueError("cannot invalidate an unconfigured controller side")
        for side in affected:
            state = self._clutches[side]
            state.blocked, state.pending_anchor = True, False
            state.controller_anchor = state.robot_anchor = None
            state.previous_controller_rotation = None
            state.scaled_rotation = np.eye(3)
        if sides is None:
            self.latest_frame = None
        self.last_invalidation_reason = str(reason)

    def ingest(self, frame: ControllerFrame, now: float) -> bool:
        """Observe all packet edges in order; batching must not drop releases."""
        if not isinstance(frame, ControllerFrame):
            raise ValueError("frame must be a parsed ControllerFrame")
        if self._last_sequence is not None and frame.sequence <= self._last_sequence:
            return False
        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp:
            return False
        if self._last_received is not None and frame.received_at < self._last_received:
            return False
        if not feedback_is_fresh(frame.received_at, now, self.config.max_frame_age_seconds):
            self.invalidate("controller_input_stale")
            return False
        if self._last_received is not None and frame.received_at - self._last_received > self.config.max_frame_age_seconds:
            self.invalidate("controller_input_gap")
        if self.latest_frame is not None and frame.coordinate_system != self.latest_frame.coordinate_system:
            self.invalidate("controller_coordinate_system_changed")
        for side in self.sides:
            pose, state = getattr(frame, side), self._clutches[side]
            generation_changed = pose.activation_id is not None and pose.activation_id != state.activation_id
            if not pose.tracked:
                self.invalidate(f"{side}_tracking_lost", sides=(side,))
            elif not pose.enabled:
                state.blocked, state.pending_anchor = False, False
                state.controller_anchor = state.robot_anchor = None
                state.previous_controller_rotation = None
                state.scaled_rotation = np.eye(3)
            elif not state.blocked and (not state.enabled or generation_changed):
                if state.activation_id is not None and pose.activation_id is not None and pose.activation_id < state.activation_id:
                    self.invalidate(f"{side}_activation_counter_regressed", sides=(side,))
                    state.enabled, state.activation_id = pose.enabled, pose.activation_id
                    continue
                state.pending_anchor = True
                state.controller_anchor = state.robot_anchor = None
                state.previous_controller_rotation = None
                state.scaled_rotation = np.eye(3)
            elif not state.blocked and state.previous_controller_rotation is not None:
                rotation = pose_in_robot(pose, frame.coordinate_system)[:3, :3]
                if self.config.orientation_scale != 1.0:
                    increment = Rotation.from_matrix(rotation @ state.previous_controller_rotation.T)
                    scaled = Rotation.from_rotvec(increment.as_rotvec() * self.config.orientation_scale).as_matrix()
                    state.scaled_rotation = Rotation.from_matrix(scaled @ state.scaled_rotation).as_matrix()
                state.previous_controller_rotation = rotation
            state.enabled = pose.enabled
            state.activation_id = pose.activation_id
        self._last_sequence, self._last_timestamp, self._last_received = frame.sequence, frame.timestamp, frame.received_at
        self.latest_frame = frame
        return True

    def _q(self, values, label):
        raw = np.asarray(values)
        q = np.asarray(values, dtype=float)
        if raw.dtype == bool or any(isinstance(value, (bool, np.bool_)) for value in np.asarray(values, dtype=object).flat) or q.shape != (14,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{label} must contain 14 finite joint positions")
        if np.any(q < self._limits[:, 0] - 1e-9) or np.any(q > self._limits[:, 1] + 1e-9):
            raise ValueError(f"{label} exceeds hard joint limits")
        return q.copy()

    def update(self, frame: ControllerFrame, current_q, last_q_cmd=None, now=None) -> IKCommand | None:
        now = time.monotonic() if now is None else now
        self.ingest(frame, now)
        accepted = self.latest_frame
        if accepted is None:
            return None
        if not feedback_is_fresh(accepted.received_at, now, self.config.max_frame_age_seconds):
            self.invalidate("controller_input_stale")
            return None
        try:
            current = self._q(current_q, "current_q")
            previous = current if last_q_cmd is None else self._q(last_q_cmd, "last_q_cmd")
            if self._baseline_q is None:
                self.solver.set_home_q(current)
                self._baseline_q = current.copy()
            active = tuple(side for side in self.sides if getattr(accepted, side).tracked and
                           self._clutches[side].enabled and not self._clutches[side].blocked)
            if not active:
                return None
            actual_poses = self.solver.end_effector_poses(current)
            if len(actual_poses) != 2:
                raise ValueError("solver must provide left and right measured TCP poses")
            actual = {side: np.asarray(pose, dtype=float).copy() for side, pose in zip(("left", "right"), actual_poses)}
            for pose in actual.values():
                if (pose.shape != (4, 4) or not np.all(np.isfinite(pose)) or
                    not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-8) or
                    not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6) or
                    abs(float(np.linalg.det(pose[:3, :3])) - 1.) > 1e-6):
                    raise ValueError("solver measured TCP must be a proper finite SE(3) pose")
            targets, reanchored = {}, {side: False for side in self.sides}
            for side in active:
                state = self._clutches[side]
                pose = pose_in_robot(getattr(accepted, side), accepted.coordinate_system)
                if state.pending_anchor:
                    state.controller_anchor, state.robot_anchor = pose.copy(), actual[side].copy()
                    state.pending_anchor = False
                    state.previous_controller_rotation = pose[:3, :3].copy()
                    state.scaled_rotation = np.eye(3)
                    reanchored[side] = True
                if state.controller_anchor is None or state.robot_anchor is None:
                    raise ValueError("active controller has no clutch anchor")
                target = state.robot_anchor.copy()
                target[:3, 3] += self.config.position_scale * (pose[:3, 3] - state.controller_anchor[:3, 3])
                delta = pose[:3, :3] @ state.controller_anchor[:3, :3].T if self.config.orientation_scale == 1.0 else state.scaled_rotation
                target[:3, :3] = delta @ state.robot_anchor[:3, :3]
                targets[side] = target
            continuous = tuple(side for side in active if not reanchored[side])
            proposed, solver_meta = current.copy(), {}
            if continuous:
                solve_side = "both" if len(continuous) == 2 else continuous[0]
                result = self.solver.solve_cartesian(TeleopIKInput(
                    timestamp=accepted.timestamp, control_side=solve_side,
                    target_pose_l=targets.get("left", actual["left"]),
                    target_pose_r=targets.get("right", actual["right"]),
                    current_q=current, last_q_cmd=previous,
                ))
                if result.reason not in {"ok", "limited", "iteration_budget"} or result.command.waist_position is not None:
                    raise ValueError(f"Cartesian IK rejected controller frame: {result.reason}")
                if (tuple(result.command.motor_ids) != ARM_MOTOR_IDS or tuple(result.command.joint_names) != ARM_JOINT_NAMES or
                    result.command.control_side != solve_side):
                    raise ValueError("Cartesian IK returned an incompatible arm schema")
                proposed = self._q(result.command.positions, "IK command")
                solver_meta = {**result.command.meta, "reason": result.reason,
                               "converged": bool(getattr(result, "converged", False))}
            lower = np.maximum.reduce((self._limits[:, 0], current - self._step, previous - self._step,
                                       self._baseline_q - self.config.max_command_delta_from_baseline))
            upper = np.minimum.reduce((self._limits[:, 1], current + self._step, previous + self._step,
                                       self._baseline_q + self.config.max_command_delta_from_baseline))
            safe = current.copy()
            for side in active:
                indices = _indices(side)
                if np.any(lower[indices] > upper[indices] + 1e-12):
                    raise ValueError("current feedback and committed target/baseline bounds are incompatible")
                if reanchored[side]:
                    if np.any(current[indices] < lower[indices] - 1e-9) or np.any(current[indices] > upper[indices] + 1e-9):
                        raise ValueError("zero-motion clutch frame is outside the session safety envelope")
                else:
                    safe[indices] = np.clip(proposed[indices], lower[indices], upper[indices])
            return IKCommand(
                timestamp=accepted.timestamp, control_side="both" if len(active) == 2 else active[0],
                joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS), positions=safe,
                meta={"source": "vr_controller_relative_cartesian", "sequence": accepted.sequence,
                      "input_received_at": accepted.received_at, "active_sides": active,
                      "target_pose_by_side": targets, "reanchored_by_side": reanchored,
                      "solver_meta": solver_meta},
            )
        except (ValueError, RuntimeError):
            self.invalidate("controller_mapping_invalid")
            raise
