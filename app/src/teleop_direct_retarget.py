from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from teleop_ik_core import TeleopArmIK, apply_joint_safety, clip_to_limits
from teleop_ik_types import ControlSide, IKCommand, TeleopIKConfig

ARM_JOINTS_PER_SIDE = 7
ARM_JOINT_COUNT = 14
SKELETON_RETARGET_SOURCE = "skeleton_segment_direction_dls"


@dataclass(frozen=True)
class DirectRetargetConfig:
    joint_names: list[str]
    motor_ids: list[int]
    control_side: ControlSide = "both"
    # 兼容原 CLI 名称；现在表示人体相对骨段转角的缩放，而不是手部位置增益。
    position_scale: float = 1.0
    max_joint_delta_per_step: float = 0.04
    max_command_delta_from_baseline: float = 0.90
    smoothing_alpha: float = 0.65
    calibration_window_frames: int = 45
    calibration_max_direction_rms_rad: float = 0.055
    calibration_max_length_cv: float = 0.025
    # 标定窗口不仅要稳定，还必须接近“立正”：双臂向下且肘部基本伸直。
    # 否则稳定的遮挡/错骨架会被永久写成相对重定向零姿态。
    calibration_max_down_angle_rad: float = 0.70
    calibration_max_elbow_angle_rad: float = 0.45
    min_segment_length_m: float = 0.12
    max_segment_length_m: float = 0.55
    max_segment_length_ratio: float = 1.18
    max_joint_position_step_m: float = 0.10
    upper_direction_weight: float = 1.0
    forearm_direction_weight: float = 0.85
    direction_damping: float = 0.08
    direction_iterations: int = 6
    per_joint_envelope: tuple[float, ...] = (0.70, 0.45, 0.55, 0.80, 0.55, 0.12, 0.12)

    def __post_init__(self) -> None:
        nonnegative_fields = (
            "position_scale", "max_joint_delta_per_step", "max_command_delta_from_baseline",
            "calibration_max_direction_rms_rad", "calibration_max_length_cv",
            "calibration_max_down_angle_rad", "calibration_max_elbow_angle_rad",
            "max_joint_position_step_m", "upper_direction_weight", "forearm_direction_weight",
            "direction_damping",
        )
        for name in nonnegative_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须为有限非负数")
        if not np.isfinite(self.smoothing_alpha) or not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha 必须为 [0, 1] 内的有限数")
        envelope = np.asarray(self.per_joint_envelope, dtype=float)
        if envelope.shape != (ARM_JOINTS_PER_SIDE,) or not np.all(np.isfinite(envelope)) or np.any(envelope < 0.0):
            raise ValueError("per_joint_envelope 必须包含 7 个有限非负数")
        lengths = np.asarray([self.min_segment_length_m, self.max_segment_length_m, self.max_segment_length_ratio], dtype=float)
        if not np.all(np.isfinite(lengths)) or not 0.0 < lengths[0] <= lengths[1] or lengths[2] < 1.0:
            raise ValueError("骨段长度范围及比例非法")
        for name, minimum in (("calibration_window_frames", 3), ("direction_iterations", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} 必须为不小于 {minimum} 的整数")


@dataclass(frozen=True)
class BodyFrame:
    origin: np.ndarray
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray

    @property
    def world_to_body(self) -> np.ndarray:
        # 行向量顺序严格对应 Formotion: X=前、Y=左、Z=上。
        return np.vstack((self.forward, self.left, self.up))

    def vector_to_local(self, vector_world: np.ndarray) -> np.ndarray:
        return self.world_to_body @ np.asarray(vector_world, dtype=float)


@dataclass(frozen=True)
class ArmSegments:
    upper_direction_local: np.ndarray
    forearm_direction_local: np.ndarray
    upper_length_m: float
    forearm_length_m: float
    shoulder_world: np.ndarray
    elbow_world: np.ndarray
    wrist_world: np.ndarray


@dataclass(frozen=True)
class SkeletonObservation:
    body_frame: BodyFrame
    arms: dict[str, ArmSegments]


@dataclass
class DirectRetargetState:
    baseline_q: np.ndarray | None = None
    last_q_cmd: np.ndarray | None = None
    calibration_samples: deque[SkeletonObservation] = field(default_factory=deque)
    home_arms: dict[str, ArmSegments] = field(default_factory=dict)
    robot_home_directions: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    last_observation: SkeletonObservation | None = None
    calibrated: bool = False


class DirectRetargeter:
    """以人体骨段方向驱动机器人 URDF 骨段方向的关节空间重定向器。"""

    def __init__(self, config: DirectRetargetConfig, solver: TeleopArmIK | None = None):
        if len(config.joint_names) != ARM_JOINT_COUNT or len(config.motor_ids) != ARM_JOINT_COUNT:
            raise ValueError("skeleton retarget 需要完整 14 维手臂关节定义")
        if config.control_side not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported control_side: {config.control_side}")
        if config.calibration_window_frames < 3:
            raise ValueError("calibration_window_frames 必须 >= 3")
        self.config = config
        self.state = DirectRetargetState(
            calibration_samples=deque(maxlen=config.calibration_window_frames)
        )
        self.solver = solver or TeleopArmIK(
            TeleopIKConfig(
                joint_names=config.joint_names,
                motor_ids=config.motor_ids,
                max_iterations=config.direction_iterations,
                damping=config.direction_damping,
                max_joint_delta_per_step=config.max_joint_delta_per_step,
                smoothing_alpha=config.smoothing_alpha,
                enable_local_log=False,
                log_to_console=False,
                show_model_load_progress=True,
                log_ik_result=False,
            )
        )

    def reset(self) -> None:
        self.state = DirectRetargetState(
            calibration_samples=deque(maxlen=self.config.calibration_window_frames)
        )

    def command_from_frame(
        self,
        frame: dict[str, Any],
        baseline_q: np.ndarray,
        timestamp: float | None = None,
    ) -> IKCommand:
        baseline = ensure_arm_q(baseline_q, "baseline_q")
        bounded_baseline = clip_to_limits(baseline, self.solver.joint_limits)
        if not np.allclose(baseline, bounded_baseline, atol=1e-8, rtol=0.0):
            raise ValueError("baseline_q 超出 URDF 关节硬限位，请检查模型与反馈")
        baseline = bounded_baseline
        if self.state.baseline_q is None:
            self.state.baseline_q = baseline.copy()
            self.solver.set_home_q(baseline)
            self.state.robot_home_directions = {
                side: self.solver.segment_directions(baseline, side)
                for side in ("left", "right")
            }
        elif not np.allclose(self.state.baseline_q, baseline, atol=1e-8):
            raise ValueError("DirectRetargeter 运行期间 baseline_q 不允许变化；请 reset 后重新标定")

        observation, quality_reason = skeleton_observation(frame, self.config)
        if observation is None:
            return self._hold_command(frame, timestamp, f"invalid_skeleton:{quality_reason}")

        if not self.state.calibrated:
            calibrated, calibration_meta = self._update_calibration(observation)
            if not calibrated:
                return self._hold_command(frame, timestamp, "calibrating", calibration_meta)

        valid, reason = observation_is_safe(
            observation,
            self.state.home_arms,
            self.state.last_observation,
            self.config,
        )
        if not valid:
            return self._hold_command(frame, timestamp, f"quality_hold:{reason}")

        target_directions = {
            side: (
                relative_direction_target(
                    self.state.home_arms[side].upper_direction_local,
                    observation.arms[side].upper_direction_local,
                    self.state.robot_home_directions[side][0],
                    self.config.position_scale,
                ),
                relative_direction_target(
                    self.state.home_arms[side].forearm_direction_local,
                    observation.arms[side].forearm_direction_local,
                    self.state.robot_home_directions[side][1],
                    self.config.position_scale,
                ),
            )
            for side in selected_sides(self.config.control_side)
        }
        reference = self.state.baseline_q if self.state.last_q_cmd is None else self.state.last_q_cmd
        solved_q, solver_meta = self.solver.solve_segment_directions(
            timestamp=frame_timestamp(frame, timestamp),
            control_side=self.config.control_side,
            target_directions=target_directions,
            current_q=reference,
            last_q_cmd=self.state.last_q_cmd,
            upper_weight=self.config.upper_direction_weight,
            forearm_weight=self.config.forearm_direction_weight,
            max_iterations=self.config.direction_iterations,
            damping=self.config.direction_damping,
        )
        safe_q, envelope_limited = apply_retarget_safety(
            solved_q,
            reference,
            self.state.baseline_q,
            self.solver.joint_limits,
            self.config,
        )
        actual_directions = {
            side: self.solver.segment_directions(safe_q, side)
            for side in selected_sides(self.config.control_side)
        }
        final_residual_by_side = {
            side: {
                "upper_angle_rad": direction_angle(actual_directions[side][0], target_directions[side][0]),
                "forearm_angle_rad": direction_angle(actual_directions[side][1], target_directions[side][1]),
            }
            for side in selected_sides(self.config.control_side)
        }
        self.state.last_q_cmd = safe_q.copy()
        self.state.last_observation = observation
        return IKCommand(
            timestamp=frame_timestamp(frame, timestamp),
            control_side=self.config.control_side,
            joint_names=self.config.joint_names,
            positions=safe_q,
            motor_ids=self.config.motor_ids,
            meta={
                "solver": SKELETON_RETARGET_SOURCE,
                "source_by_side": {side: SKELETON_RETARGET_SOURCE for side in selected_sides(self.config.control_side)},
                "calibrated": True,
                "limited": bool(envelope_limited or solver_meta["limited"]),
                "solver_meta": solver_meta,
                "target_directions": {
                    side: {"upper": target[0], "forearm": target[1]}
                    for side, target in target_directions.items()
                },
                "actual_directions": {
                    side: {"upper": actual[0], "forearm": actual[1]}
                    for side, actual in actual_directions.items()
                },
                "final_residual_by_side": final_residual_by_side,
                "quality": "ok",
                "safety": "urdf_limits+step+baseline_envelope",
            },
        )

    def _update_calibration(self, observation: SkeletonObservation) -> tuple[bool, dict[str, Any]]:
        self.state.calibration_samples.append(observation)
        sample_count = len(self.state.calibration_samples)
        required = self.config.calibration_window_frames
        meta: dict[str, Any] = {"samples": sample_count, "required": required}
        if sample_count < required:
            return False, meta

        stable, metrics = calibration_window_is_stable(list(self.state.calibration_samples), self.config)
        meta.update(metrics)
        if not stable:
            return False, meta

        self.state.home_arms = median_home_arms(list(self.state.calibration_samples))
        self.state.calibrated = True
        self.state.last_observation = observation
        meta["calibrated"] = True
        return True, meta

    def _hold_command(
        self,
        frame: dict[str, Any],
        timestamp: float | None,
        reason: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> IKCommand:
        if self.state.baseline_q is None:
            raise RuntimeError("baseline_q 尚未初始化")
        held_q = self.state.baseline_q if self.state.last_q_cmd is None else self.state.last_q_cmd
        held_q = clip_to_limits(held_q, self.solver.joint_limits)
        meta: dict[str, Any] = {
            "solver": SKELETON_RETARGET_SOURCE,
            "source_by_side": {side: reason for side in selected_sides(self.config.control_side)},
            "calibrated": self.state.calibrated,
            "limited": False,
            "quality": reason,
            "held": True,
        }
        if extra_meta:
            meta["calibration"] = extra_meta
        return IKCommand(
            timestamp=frame_timestamp(frame, timestamp),
            control_side=self.config.control_side,
            joint_names=self.config.joint_names,
            positions=held_q.copy(),
            motor_ids=self.config.motor_ids,
            meta=meta,
        )


def selected_sides(control_side: str) -> tuple[str, ...]:
    if control_side == "left":
        return ("left",)
    if control_side == "right":
        return ("right",)
    if control_side == "both":
        return ("left", "right")
    raise ValueError(f"Unsupported control_side: {control_side}")


def frame_timestamp(frame: dict[str, Any], fallback: float | None) -> float:
    return float(frame.get("receive_time", 0.0 if fallback is None else fallback))


def ensure_arm_q(values: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(values, dtype=float).reshape(-1)
    if q.shape != (ARM_JOINT_COUNT,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{name} 必须是 14 维有限关节角")
    return q


def normalize_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须是 3 维有限向量")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} 长度过小")
    return vector / norm


def direction_angle(first: np.ndarray, second: np.ndarray) -> float:
    a = normalize_vector(first, "first direction")
    b = normalize_vector(second, "second direction")
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def _joint_entry(frame: dict[str, Any], group: str, joint: str) -> dict[str, Any] | None:
    skeleton = frame.get("skeleton")
    if not isinstance(skeleton, dict):
        return None
    group_data = skeleton.get(group)
    if not isinstance(group_data, dict):
        return None
    entry = group_data.get(joint)
    return entry if isinstance(entry, dict) else None


def read_skeleton_joint(frame: dict[str, Any], group: str, joint: str) -> np.ndarray | None:
    entry = _joint_entry(frame, group, joint)
    if entry is None or entry.get("tracked", True) is False or "pos" not in entry:
        return None
    position = np.asarray(entry["pos"], dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return None
    return position


def read_side_joint(frame: dict[str, Any], side: str, joint: str) -> np.ndarray | None:
    return read_skeleton_joint(frame, side, joint)


def read_torso_joint(frame: dict[str, Any], joint: str) -> np.ndarray | None:
    return read_skeleton_joint(frame, "torso", joint)


def first_available(*values: np.ndarray | None) -> np.ndarray | None:
    return next((value for value in values if value is not None), None)


def body_frame_from_skeleton(frame: dict[str, Any]) -> BodyFrame | None:
    left_shoulder = first_available(read_side_joint(frame, "left", "shoulder"), read_torso_joint(frame, "left_shoulder"))
    right_shoulder = first_available(read_side_joint(frame, "right", "shoulder"), read_torso_joint(frame, "right_shoulder"))
    if left_shoulder is None or right_shoulder is None:
        return None
    try:
        # 协议 +Y 指向人体左侧，因此横轴必须是 right -> left。
        left = normalize_vector(left_shoulder - right_shoulder, "body left")
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        left_hip = read_torso_joint(frame, "left_hip")
        right_hip = read_torso_joint(frame, "right_hip")
        hip_center = None if left_hip is None or right_hip is None else (left_hip + right_hip) * 0.5
        waist = read_torso_joint(frame, "waist")
        chest = read_torso_joint(frame, "chest")
        neck = read_torso_joint(frame, "neck")
        up_raw = first_available(
            None if hip_center is None else shoulder_center - hip_center,
            None if waist is None or chest is None else chest - waist,
            None if waist is None or neck is None else neck - waist,
        )
        if up_raw is None:
            return None
        up = normalize_vector(up_raw - left * float(np.dot(up_raw, left)), "body up")
        # +Y(left) × +Z(up) = +X(forward)。
        forward = normalize_vector(np.cross(left, up), "body forward")
        left = normalize_vector(np.cross(up, forward), "body left orthogonal")
        return BodyFrame(origin=shoulder_center, forward=forward, left=left, up=up)
    except ValueError:
        return None


def skeleton_observation(
    frame: dict[str, Any],
    config: DirectRetargetConfig,
) -> tuple[SkeletonObservation | None, str]:
    body = body_frame_from_skeleton(frame)
    if body is None:
        return None, "missing_body_frame"
    arms: dict[str, ArmSegments] = {}
    for side in ("left", "right"):
        shoulder = read_side_joint(frame, side, "shoulder")
        elbow = read_side_joint(frame, side, "elbow")
        wrist = read_side_joint(frame, side, "wrist")
        if shoulder is None or elbow is None or wrist is None:
            return None, f"missing_{side}_arm_joint"
        upper_vector = elbow - shoulder
        forearm_vector = wrist - elbow
        upper_length = float(np.linalg.norm(upper_vector))
        forearm_length = float(np.linalg.norm(forearm_vector))
        if not config.min_segment_length_m <= upper_length <= config.max_segment_length_m:
            return None, f"invalid_{side}_upper_length"
        if not config.min_segment_length_m <= forearm_length <= config.max_segment_length_m:
            return None, f"invalid_{side}_forearm_length"
        arms[side] = ArmSegments(
            upper_direction_local=normalize_vector(body.vector_to_local(upper_vector), f"{side} upper"),
            forearm_direction_local=normalize_vector(body.vector_to_local(forearm_vector), f"{side} forearm"),
            upper_length_m=upper_length,
            forearm_length_m=forearm_length,
            shoulder_world=shoulder,
            elbow_world=elbow,
            wrist_world=wrist,
        )
    return SkeletonObservation(body_frame=body, arms=arms), "ok"


def direction_rms(directions: list[np.ndarray]) -> float:
    center = normalize_vector(np.median(np.asarray(directions), axis=0), "median direction")
    angles = [math.acos(float(np.clip(np.dot(center, direction), -1.0, 1.0))) for direction in directions]
    return float(math.sqrt(np.mean(np.square(angles))))


def coefficient_of_variation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    return float(np.std(array) / mean) if mean > 1e-9 else float("inf")


def calibration_window_is_stable(
    samples: list[SkeletonObservation],
    config: DirectRetargetConfig,
) -> tuple[bool, dict[str, Any]]:
    direction_metrics: list[float] = []
    length_metrics: list[float] = []
    for side in ("left", "right"):
        direction_metrics.extend(
            (
                direction_rms([sample.arms[side].upper_direction_local for sample in samples]),
                direction_rms([sample.arms[side].forearm_direction_local for sample in samples]),
            )
        )
        length_metrics.extend(
            (
                coefficient_of_variation([sample.arms[side].upper_length_m for sample in samples]),
                coefficient_of_variation([sample.arms[side].forearm_length_m for sample in samples]),
            )
        )
    max_direction_rms = max(direction_metrics)
    max_length_cv = max(length_metrics)
    median_directions: list[tuple[np.ndarray, np.ndarray]] = []
    for side in ("left", "right"):
        median_directions.append(
            (
                normalize_vector(
                    np.median([sample.arms[side].upper_direction_local for sample in samples], axis=0),
                    f"{side} calibration upper",
                ),
                normalize_vector(
                    np.median([sample.arms[side].forearm_direction_local for sample in samples], axis=0),
                    f"{side} calibration forearm",
                ),
            )
        )
    down = np.asarray([0.0, 0.0, -1.0])
    max_down_angle = max(
        math.acos(float(np.clip(np.dot(direction, down), -1.0, 1.0)))
        for directions in median_directions
        for direction in directions
    )
    max_elbow_angle = max(
        math.acos(float(np.clip(np.dot(upper, forearm), -1.0, 1.0)))
        for upper, forearm in median_directions
    )
    stable = (
        max_direction_rms <= config.calibration_max_direction_rms_rad
        and max_length_cv <= config.calibration_max_length_cv
        and max_down_angle <= config.calibration_max_down_angle_rad
        and max_elbow_angle <= config.calibration_max_elbow_angle_rad
    )
    return stable, {
        "stable": stable,
        "variation_stable": (
            max_direction_rms <= config.calibration_max_direction_rms_rad
            and max_length_cv <= config.calibration_max_length_cv
        ),
        "stand_pose_valid": (
            max_down_angle <= config.calibration_max_down_angle_rad
            and max_elbow_angle <= config.calibration_max_elbow_angle_rad
        ),
        "max_direction_rms_rad": max_direction_rms,
        "max_length_cv": max_length_cv,
        "max_down_angle_rad": max_down_angle,
        "max_elbow_angle_rad": max_elbow_angle,
    }


def median_home_arms(samples: list[SkeletonObservation]) -> dict[str, ArmSegments]:
    result: dict[str, ArmSegments] = {}
    for side in ("left", "right"):
        source = samples[-1].arms[side]
        result[side] = ArmSegments(
            upper_direction_local=normalize_vector(
                np.median([sample.arms[side].upper_direction_local for sample in samples], axis=0),
                f"{side} median upper",
            ),
            forearm_direction_local=normalize_vector(
                np.median([sample.arms[side].forearm_direction_local for sample in samples], axis=0),
                f"{side} median forearm",
            ),
            upper_length_m=float(np.median([sample.arms[side].upper_length_m for sample in samples])),
            forearm_length_m=float(np.median([sample.arms[side].forearm_length_m for sample in samples])),
            shoulder_world=source.shoulder_world.copy(),
            elbow_world=source.elbow_world.copy(),
            wrist_world=source.wrist_world.copy(),
        )
    return result


def observation_is_safe(
    observation: SkeletonObservation,
    home_arms: dict[str, ArmSegments],
    previous: SkeletonObservation | None,
    config: DirectRetargetConfig,
) -> tuple[bool, str]:
    for side in ("left", "right"):
        current = observation.arms[side]
        home = home_arms[side]
        for segment, current_length, home_length in (
            ("upper", current.upper_length_m, home.upper_length_m),
            ("forearm", current.forearm_length_m, home.forearm_length_m),
        ):
            ratio = max(current_length / home_length, home_length / current_length)
            if ratio > config.max_segment_length_ratio:
                return False, f"{side}_{segment}_length_jump"
        if previous is not None:
            old = previous.arms[side]
            max_step = max(
                float(np.linalg.norm(current.shoulder_world - old.shoulder_world)),
                float(np.linalg.norm(current.elbow_world - old.elbow_world)),
                float(np.linalg.norm(current.wrist_world - old.wrist_world)),
            )
            if max_step > config.max_joint_position_step_m:
                return False, f"{side}_joint_position_jump"
    return True, "ok"


def relative_direction_target(
    home_human: np.ndarray,
    current_human: np.ndarray,
    robot_home: np.ndarray,
    scale: float,
) -> np.ndarray:
    a = normalize_vector(home_human, "home human direction")
    b = normalize_vector(current_human, "current human direction")
    robot = normalize_vector(robot_home, "robot home direction")
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine <= 1e-9:
        if cosine > 0.0:
            return robot
        axis_seed = np.asarray([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
        axis = normalize_vector(np.cross(a, axis_seed), "opposite direction axis")
        angle = math.pi
    else:
        axis = cross / sine
        angle = math.atan2(sine, cosine)
    scaled_angle = angle * max(0.0, float(scale))
    return normalize_vector(rodrigues_rotate(robot, axis, scaled_angle), "scaled robot target")


def rodrigues_rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis_unit = normalize_vector(axis, "rotation axis")
    value = np.asarray(vector, dtype=float)
    return (
        value * math.cos(angle)
        + np.cross(axis_unit, value) * math.sin(angle)
        + axis_unit * float(np.dot(axis_unit, value)) * (1.0 - math.cos(angle))
    )


def apply_retarget_safety(
    solved_q: np.ndarray,
    reference_q: np.ndarray,
    baseline_q: np.ndarray,
    joint_limits: list[tuple[float, float]],
    config: DirectRetargetConfig,
) -> tuple[np.ndarray, bool]:
    target = ensure_arm_q(solved_q, "solved_q").copy()
    reference = ensure_arm_q(reference_q, "reference_q")
    baseline = ensure_arm_q(baseline_q, "baseline_q")
    envelope_single = np.minimum(
        np.abs(np.asarray(config.per_joint_envelope, dtype=float)),
        abs(float(config.max_command_delta_from_baseline)),
    )
    if envelope_single.shape != (ARM_JOINTS_PER_SIDE,):
        raise ValueError("per_joint_envelope 必须为单臂 7 维")
    envelope = np.concatenate((envelope_single, envelope_single))
    lower = baseline - envelope
    upper = baseline + envelope
    controlled = controlled_indices(config.control_side)
    before = target.copy()
    uncontrolled = sorted(set(range(ARM_JOINT_COUNT)) - set(controlled))
    target[uncontrolled] = baseline[uncontrolled]
    target[controlled] = np.clip(target[controlled], lower[controlled], upper[controlled])
    safe, step_limited = apply_joint_safety(
        q_next=target,
        q_ref=reference,
        limits=limits_with_envelope(joint_limits, lower, upper),
        max_step=config.max_joint_delta_per_step,
        smoothing_alpha=config.smoothing_alpha,
    )
    return safe, bool(step_limited or not np.allclose(before, safe, atol=1e-9))


def limits_with_envelope(
    joint_limits: list[tuple[float, float]],
    envelope_lower: np.ndarray,
    envelope_upper: np.ndarray,
) -> list[tuple[float, float]]:
    if len(joint_limits) != ARM_JOINT_COUNT:
        raise ValueError("joint_limits 必须为 14 维")
    limits: list[tuple[float, float]] = []
    for (base_lower, base_upper), lower, upper in zip(joint_limits, envelope_lower, envelope_upper):
        combined_lower = max(float(base_lower), float(lower))
        combined_upper = min(float(base_upper), float(upper))
        if combined_lower > combined_upper:
            # 基础限位与当前姿态包络没有交集时，退化到包络中点，避免无意义的硬裁剪。
            boundary = float(np.clip((lower + upper) * 0.5, base_lower, base_upper))
            combined_lower = combined_upper = boundary
        limits.append((combined_lower, combined_upper))
    return limits


def add_direct_retarget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-frames", type=int, default=45)
    parser.add_argument("--calibration-max-direction-rms", type=float, default=0.055)
    parser.add_argument("--calibration-max-length-cv", type=float, default=0.025)
    parser.add_argument("--max-segment-length-ratio", type=float, default=1.18)
    parser.add_argument("--max-skeleton-joint-step", type=float, default=0.10)
    parser.add_argument("--upper-direction-weight", type=float, default=1.0)
    parser.add_argument("--forearm-direction-weight", type=float, default=0.85)
    parser.add_argument("--direction-damping", type=float, default=0.08)
    parser.add_argument("--direction-iterations", type=int, default=6)


def direct_retarget_config_from_args(
    args: argparse.Namespace,
    *,
    joint_names: list[str],
    motor_ids: list[int],
) -> DirectRetargetConfig:
    return DirectRetargetConfig(
        joint_names=joint_names,
        motor_ids=motor_ids,
        control_side=args.control_side,
        position_scale=float(args.position_scale),
        max_joint_delta_per_step=float(args.max_joint_delta_per_step),
        max_command_delta_from_baseline=float(args.max_command_delta_from_baseline),
        calibration_window_frames=int(args.calibration_frames),
        calibration_max_direction_rms_rad=float(args.calibration_max_direction_rms),
        calibration_max_length_cv=float(args.calibration_max_length_cv),
        max_segment_length_ratio=float(args.max_segment_length_ratio),
        max_joint_position_step_m=float(args.max_skeleton_joint_step),
        upper_direction_weight=float(args.upper_direction_weight),
        forearm_direction_weight=float(args.forearm_direction_weight),
        direction_damping=float(args.direction_damping),
        direction_iterations=int(args.direction_iterations),
    )


def controlled_indices(control_side: str) -> list[int]:
    if control_side == "left":
        return list(range(ARM_JOINTS_PER_SIDE))
    if control_side == "right":
        return list(range(ARM_JOINTS_PER_SIDE, ARM_JOINT_COUNT))
    return list(range(ARM_JOINT_COUNT))
