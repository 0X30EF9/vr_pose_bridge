from __future__ import annotations

from dataclasses import dataclass, field
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

    def __post_init__(self) -> None:
        if self.hold_seconds < 0.0:
            raise ValueError("hold_seconds 不能为负数")
        if not 0.0 < self.human_vertical_cosine <= 1.0:
            raise ValueError("human_vertical_cosine 必须位于 (0, 1]")
        if self.human_min_wrist_drop_m <= 0.0:
            raise ValueError("human_min_wrist_drop_m 必须大于 0")
        if self.human_max_lateral_offset_m <= 0.0:
            raise ValueError("human_max_lateral_offset_m 必须大于 0")
        if self.max_home_rms_m <= 0.0:
            raise ValueError("max_home_rms_m 必须大于 0")


@dataclass(frozen=True)
class NaturalDownStartupStatus:
    ready: bool
    held_seconds: float
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    left_home_position: np.ndarray | None = None
    right_home_position: np.ndarray | None = None


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.all(np.isfinite(value)) or norm <= 1e-9:
        raise ValueError(f"{label} 无法形成有效方向")
    return value / norm


def _joint_position(frame: MocapFrame, name: str) -> np.ndarray:
    joint = frame.joint(name)
    if not joint.tracked:
        raise ValueError(f"{name} 未跟踪")
    value = np.asarray(joint.position, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 位置无效")
    return value


def check_human_natural_down(
    frame: MocapFrame,
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
    ready: bool = False

    def reset(self) -> None:
        self.valid_since = None
        self.left_samples.clear()
        self.right_samples.clear()
        self.ready = False

    def observe(
        self,
        now: float,
        frame: MocapFrame,
        current_q: np.ndarray,
        left_home_position: np.ndarray,
        right_home_position: np.ndarray,
    ) -> NaturalDownStartupStatus:
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
        self.left_samples.append(left.copy())
        self.right_samples.append(right.copy())
        held_seconds = max(0.0, float(now) - self.valid_since)
        left_stack = np.stack(self.left_samples)
        right_stack = np.stack(self.right_samples)
        left_home = np.median(left_stack, axis=0)
        right_home = np.median(right_stack, axis=0)
        left_rms = float(np.sqrt(np.mean(np.sum((left_stack - left_home) ** 2, axis=1))))
        right_rms = float(np.sqrt(np.mean(np.sum((right_stack - right_home) ** 2, axis=1))))
        metrics["left_home_rms_m"] = left_rms
        metrics["right_home_rms_m"] = right_rms
        stable = max(left_rms, right_rms) <= self.config.max_home_rms_m
        enough_samples = len(self.left_samples) >= 2
        self.ready = bool(stable and enough_samples and held_seconds >= self.config.hold_seconds)
        if not stable:
            self.valid_since = float(now)
            self.left_samples[:] = [left.copy()]
            self.right_samples[:] = [right.copy()]
            reasons = ("natural_down_pose_unstable",)
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
        )