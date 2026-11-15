from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from teleop_mocap_protocol import MocapFrame


ARM_JOINT_COUNT = 14


@dataclass(frozen=True)
class NaturalDownStartupConfig:
    hold_seconds: float = 2.0
    human_vertical_cosine: float = 0.78
    human_min_wrist_drop_m: float = 0.30
    human_max_lateral_offset_m: float = 0.16
    max_home_rms_m: float = 0.025
    max_robot_joint_span_rad: float = 0.02

    def __post_init__(self) -> None:
        if not math.isfinite(self.hold_seconds) or self.hold_seconds < 0.0:
            raise ValueError("hold_seconds 不能为负数")
        if not 0.0 < self.human_vertical_cosine <= 1.0:
            raise ValueError("human_vertical_cosine 必须位于 (0, 1]")
        if not math.isfinite(self.human_min_wrist_drop_m) or self.human_min_wrist_drop_m <= 0.0:
            raise ValueError("human_min_wrist_drop_m 必须大于 0")
        if not math.isfinite(self.human_max_lateral_offset_m) or self.human_max_lateral_offset_m <= 0.0:
            raise ValueError("human_max_lateral_offset_m 必须大于 0")
        if not math.isfinite(self.max_home_rms_m) or self.max_home_rms_m <= 0.0:
            raise ValueError("max_home_rms_m 必须大于 0")
        if not math.isfinite(self.max_robot_joint_span_rad) or self.max_robot_joint_span_rad <= 0.0:
            raise ValueError("max_robot_joint_span_rad 必须为有限正数")


@dataclass(frozen=True)
class NaturalDownStartupStatus:
    ready: bool
    held_seconds: float
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    left_home_position: np.ndarray | None = None
    right_home_position: np.ndarray | None = None
    window_frames: tuple[dict[str, Any], ...] = ()
    robot_baseline_q: np.ndarray | None = None


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.all(np.isfinite(value)) or norm <= 1e-9:
        raise ValueError(f"{label} 无法形成有效方向")
    return value / norm


def _joint_entry(frame: MocapFrame | dict[str, Any], name: str) -> dict[str, Any]:
    if isinstance(frame, MocapFrame):
        joint = frame.joint(name)
        return {"pos": joint.position, "quat": joint.quaternion_xyzw, "tracked": joint.tracked}
    skeleton = frame.get("skeleton", {})
    if name.startswith(("left_", "right_")):
        side, joint_name = name.split("_", 1)
        return skeleton.get(side, {}).get(joint_name, skeleton.get("torso", {}).get(name, {}))
    return skeleton.get("torso", {}).get(name, {})


def _joint_position(frame: MocapFrame | dict[str, Any], name: str) -> np.ndarray:
    joint = _joint_entry(frame, name)
    if joint.get("tracked") is not True:
        raise ValueError(f"{name} 未跟踪")
    value = np.asarray(joint.get("pos"), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 位置无效")
    return value


def check_human_natural_down(
    frame: MocapFrame | dict[str, Any],
    config: NaturalDownStartupConfig,
) -> tuple[bool, tuple[str, ...], dict[str, float]]:
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    try:
        left_shoulder = _joint_position(frame, "left_shoulder")
        right_shoulder = _joint_position(frame, "right_shoulder")
        left_hip = _joint_position(frame, "left_hip")
        right_hip = _joint_position(frame, "right_hip")
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        hip_center = (left_hip + right_hip) * 0.5
        body_down = -_unit(shoulder_center - hip_center, "躯干竖直轴")
        shoulder_axis = _unit(right_shoulder - left_shoulder, "肩部横向轴")
        body_lateral = _unit(
            shoulder_axis - np.dot(shoulder_axis, body_down) * body_down,
            "躯干横向轴",
        )

        for side, shoulder in (("left", left_shoulder), ("right", right_shoulder)):
            elbow = _joint_position(frame, f"{side}_elbow")
            wrist = _joint_position(frame, f"{side}_wrist")
            _joint_position(frame, f"{side}_hand")
            hand_quat = np.asarray(_joint_entry(frame, f"{side}_hand").get("quat"), dtype=float)
            if hand_quat.shape != (4,) or not np.all(np.isfinite(hand_quat)) or np.linalg.norm(hand_quat) <= 1e-8:
                raise ValueError(f"{side}_hand 姿态无效")
            upper_cosine = float(np.dot(_unit(elbow - shoulder, f"{side} 上臂"), body_down))
            forearm_cosine = float(np.dot(_unit(wrist - elbow, f"{side} 前臂"), body_down))
            wrist_drop = float(np.dot(wrist - shoulder, body_down))
            lateral_offset = abs(float(np.dot(wrist - shoulder, body_lateral)))
            metrics[f"{side}_upper_vertical_cosine"] = upper_cosine
            metrics[f"{side}_forearm_vertical_cosine"] = forearm_cosine
            metrics[f"{side}_wrist_drop_m"] = wrist_drop
            metrics[f"{side}_lateral_offset_m"] = lateral_offset
            if upper_cosine < config.human_vertical_cosine:
                reasons.append(f"{side}_upper_arm_not_down")
            if forearm_cosine < config.human_vertical_cosine:
                reasons.append(f"{side}_forearm_not_down")
            if wrist_drop < config.human_min_wrist_drop_m:
                reasons.append(f"{side}_wrist_not_below_shoulder")
            if lateral_offset > config.human_max_lateral_offset_m:
                reasons.append(f"{side}_wrist_too_far_lateral")
    except (KeyError, ValueError) as exc:
        reasons.append(str(exc))
    return not reasons, tuple(reasons), metrics


def check_robot_arm_state(
    current_q: np.ndarray,
) -> tuple[bool, tuple[str, ...], dict[str, float]]:
    q = np.asarray(current_q, dtype=float)
    if q.shape != (ARM_JOINT_COUNT,) or not np.all(np.isfinite(q)):
        return False, ("robot_arm_state_invalid",), {}
    return True, (), {}


@dataclass
class NaturalDownStartupGate:
    config: NaturalDownStartupConfig
    valid_since: float | None = None
    left_samples: list[np.ndarray] = field(default_factory=list)
    right_samples: list[np.ndarray] = field(default_factory=list)
    robot_samples: list[np.ndarray] = field(default_factory=list)
    frame_samples: list[dict[str, Any]] = field(default_factory=list)
    last_observation_time: float | None = None
    ready: bool = False

    def reset(self) -> None:
        self.valid_since = None
        self.left_samples.clear()
        self.right_samples.clear()
        self.robot_samples.clear()
        self.frame_samples.clear()
        self.last_observation_time = None
        self.ready = False

    def observe(
        self,
        now: float,
        frame: MocapFrame | dict[str, Any],
        current_q: np.ndarray,
        left_home_position: np.ndarray,
        right_home_position: np.ndarray,
    ) -> NaturalDownStartupStatus:
        if not math.isfinite(now) or (self.last_observation_time is not None and now < self.last_observation_time):
            self.reset()
            return NaturalDownStartupStatus(False, 0.0, ("startup_time_invalid",), {})
        if now == self.last_observation_time:
            return NaturalDownStartupStatus(False, 0.0, ("startup_duplicate_frame",), {})
        human_ok, human_reasons, human_metrics = check_human_natural_down(frame, self.config)
        robot_ok, robot_reasons, robot_metrics = check_robot_arm_state(current_q)
        metrics = {**human_metrics, **robot_metrics}
        reasons = human_reasons + robot_reasons
        if not human_ok or not robot_ok:
            self.reset()
            return NaturalDownStartupStatus(False, 0.0, reasons, metrics)

        left = np.asarray(left_home_position, dtype=float)
        right = np.asarray(right_home_position, dtype=float)
        if left.shape != (3,) or right.shape != (3,) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            self.reset()
            return NaturalDownStartupStatus(False, 0.0, ("runtime_home_position_invalid",), metrics)

        if self.valid_since is None:
            self.valid_since = float(now)
        self.last_observation_time = float(now)
        self.left_samples.append(left.copy())
        self.right_samples.append(right.copy())
        self.robot_samples.append(np.asarray(current_q, dtype=float).copy())
        self.frame_samples.append(frame.to_replay_dict() if isinstance(frame, MocapFrame) else dict(frame))
        if len(self.frame_samples) > 4096:
            self.reset()
            return NaturalDownStartupStatus(False, 0.0, ("startup_window_overflow",), metrics)
        held_seconds = max(0.0, float(now) - self.valid_since)
        left_stack = np.stack(self.left_samples)
        right_stack = np.stack(self.right_samples)
        left_home = np.median(left_stack, axis=0)
        right_home = np.median(right_stack, axis=0)
        left_rms = float(np.sqrt(np.mean(np.sum((left_stack - left_home) ** 2, axis=1))))
        right_rms = float(np.sqrt(np.mean(np.sum((right_stack - right_home) ** 2, axis=1))))
        metrics["left_home_rms_m"] = left_rms
        metrics["right_home_rms_m"] = right_rms
        robot_stack = np.stack(self.robot_samples)
        robot_span = float(np.max(np.ptp(robot_stack, axis=0)))
        metrics["robot_joint_span_rad"] = robot_span
        robot_stable = robot_span <= self.config.max_robot_joint_span_rad
        stable = max(left_rms, right_rms) <= self.config.max_home_rms_m and robot_stable
        enough_samples = len(self.left_samples) >= 3
        self.ready = bool(stable and enough_samples and held_seconds >= self.config.hold_seconds)
        if not stable:
            self.valid_since = float(now)
            self.left_samples[:] = [left.copy()]
            self.right_samples[:] = [right.copy()]
            self.robot_samples[:] = [np.asarray(current_q, dtype=float).copy()]
            self.frame_samples[:] = [frame.to_replay_dict() if isinstance(frame, MocapFrame) else dict(frame)]
            reasons = ("natural_down_pose_unstable" if robot_stable else "robot_arm_pose_unstable",)
            held_seconds = 0.0
        elif not self.ready:
            reasons = ("holding_natural_down",)
        else:
            reasons = ()
        return NaturalDownStartupStatus(
            self.ready,
            held_seconds,
            reasons,
            metrics,
            left_home.copy() if self.ready else None,
            right_home.copy() if self.ready else None,
            tuple(self.frame_samples) if self.ready else (),
            np.median(robot_stack, axis=0) if self.ready else None,
        )
