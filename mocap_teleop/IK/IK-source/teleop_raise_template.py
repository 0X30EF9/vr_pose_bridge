from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from teleop_ik_types import IKCommand

RaisePose = Literal["stand", "side_raise", "front_raise"]


@dataclass(frozen=True)
class RaiseTemplateConfig:
    joint_names: list[str]
    motor_ids: list[int]
    control_side: Literal["left", "right", "both"] = "both"
    max_joint_delta_per_step: float = 0.04
    smoothing_alpha: float = 1.0
    stand_up_threshold: float = 0.08
    full_raise_up_delta: float = 0.30
    side_score_scale: float = 0.28
    front_score_scale: float = 0.25
    score_sharpness: float = 3.0
    side_raise_roll_delta: float = 1.15
    front_raise_pitch_delta: float = -0.76
    shoulder_pitch_assist_delta: float = -0.12
    front_raise_roll_assist_delta: float = 0.18
    side_raise_elbow_bend_delta: float = -0.08
    front_raise_elbow_bend_delta: float = -0.42
    front_raise_wrist_pitch_delta: float = 0.26
    front_raise_yaw_delta: float = 0.16
    elbow_yaw_follow_delta: float = 0.10
    mocap_variation_scale: float = 0.18


@dataclass
class RaiseTemplateState:
    home_features: dict[str, dict[str, float]] = field(default_factory=dict)
    last_q_cmd: np.ndarray | None = None


class RaiseTemplateRetargeter:
    """固定动作集模板映射：立正 / 侧平举 / 前平举。

    这个映射器不做通用 IK。它只从人体上肢骨架判断当前动作阶段，
    再把阶段权重映射到一组安全、可解释的机器人关节模板。
    """

    def __init__(self, config: RaiseTemplateConfig):
        self.config = config
        self.state = RaiseTemplateState()

    def command_from_frame(
        self,
        frame: dict[str, Any],
        baseline_q: np.ndarray,
        timestamp: float | None = None,
    ) -> IKCommand:
        baseline = ensure_q(baseline_q)
        features = raise_features(frame)
        if not self.state.home_features:
            self.state.home_features = features

        weights, feature_delta = classify_raise_weights(features, self.state.home_features, self.config)
        template_q = blend_templates(baseline, weights, self.config)
        template_q = apply_mocap_variation(template_q, baseline, weights, feature_delta, self.config)
        template_q = apply_control_side(template_q, baseline, self.config.control_side)

        reference_q = baseline if self.state.last_q_cmd is None else self.state.last_q_cmd
        safe_q, limited = apply_template_safety(
            target_q=template_q,
            reference_q=reference_q,
            max_step=self.config.max_joint_delta_per_step,
            smoothing_alpha=self.config.smoothing_alpha,
        )
        self.state.last_q_cmd = safe_q.copy()

        pose = dominant_pose(weights)
        return IKCommand(
            timestamp=float(frame.get("receive_time", timestamp if timestamp is not None else 0.0)),
            control_side=self.config.control_side,
            joint_names=self.config.joint_names,
            positions=safe_q,
            motor_ids=self.config.motor_ids,
            meta={
                "solver": "raise_template",
                "pose": pose,
                "weights": weights,
                "feature_delta": feature_delta,
                "limited": limited,
            },
        )


def ensure_q(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=float).reshape(-1)
    if q.shape[0] != 14:
        raise ValueError(f"baseline_q 维度错误: {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("baseline_q 包含 NaN/Inf")
    return q


def raise_features(frame: dict[str, Any]) -> dict[str, dict[str, float]]:
    skeleton = frame.get("skeleton")
    if not isinstance(skeleton, dict):
        raise ValueError("raise_template 需要 CSV/UDP 上肢骨架字段")

    left_shoulder = joint_pos(skeleton, "left", "shoulder")
    right_shoulder = joint_pos(skeleton, "right", "shoulder")
    chest = joint_pos(skeleton, "torso", "chest")
    waist = joint_pos(skeleton, "torso", "waist")

    right_axis = unit(right_shoulder - left_shoulder)
    up_axis = unit(chest - waist)
    forward_axis = unit(np.cross(right_axis, up_axis))

    return {
        "left": side_features(skeleton, "left", left_shoulder, -1.0, right_axis, up_axis, forward_axis),
        "right": side_features(skeleton, "right", right_shoulder, 1.0, right_axis, up_axis, forward_axis),
    }


def joint_pos(skeleton: dict[str, Any], group: str, name: str) -> np.ndarray:
    try:
        return np.asarray(skeleton[group][name]["pos"], dtype=float)
    except Exception as exc:
        raise ValueError(f"缺少骨架点: {group}.{name}.pos") from exc


def side_features(
    skeleton: dict[str, Any],
    side: Literal["left", "right"],
    shoulder: np.ndarray,
    outward_sign: float,
    right_axis: np.ndarray,
    up_axis: np.ndarray,
    forward_axis: np.ndarray,
) -> dict[str, float]:
    hand = joint_pos(skeleton, side, "hand")
    rel = hand - shoulder
    return {
        "outward": float(outward_sign * np.dot(rel, right_axis)),
        "up": float(np.dot(rel, up_axis)),
        "front": float(np.dot(rel, forward_axis)),
    }


def classify_raise_weights(
    features: dict[str, dict[str, float]],
    home: dict[str, dict[str, float]],
    config: RaiseTemplateConfig,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    delta = {
        side: {key: features[side][key] - home[side][key] for key in ("outward", "up", "front")}
        for side in ("left", "right")
    }

    up_score = max(delta["left"]["up"], delta["right"]["up"])
    raise_amount = clamp01((up_score - config.stand_up_threshold) / max(1e-6, config.full_raise_up_delta - config.stand_up_threshold))
    if raise_amount <= 1e-6:
        return {"stand": 1.0, "side_raise": 0.0, "front_raise": 0.0}, delta

    side_signal = max(0.0, 0.5 * (delta["left"]["outward"] + delta["right"]["outward"]))
    # 当前动捕样本中前平举最稳定的特征是左手 front 负向大幅变化；右手 front 变化较小，作为辅助。
    front_signal = max(0.0, -delta["left"]["front"], -0.7 * delta["right"]["front"])

    side_score = clamp01(side_signal / max(1e-6, config.side_score_scale))
    front_score = clamp01(front_signal / max(1e-6, config.front_score_scale))

    # 固定动作集不需要连续混合太多自由度。分数锐化后，侧平举不会因为手臂略带前后摆动而被稀释。
    sharpness = max(1.0, float(config.score_sharpness))
    side_vote = side_score**sharpness
    front_vote = front_score**sharpness
    total = side_vote + front_vote
    if total <= 1e-6:
        return {"stand": 1.0 - raise_amount, "side_raise": raise_amount, "front_raise": 0.0}, delta

    side_weight = raise_amount * side_vote / total
    front_weight = raise_amount * front_vote / total
    stand_weight = max(0.0, 1.0 - side_weight - front_weight)
    return {"stand": stand_weight, "side_raise": side_weight, "front_raise": front_weight}, delta


def blend_templates(baseline_q: np.ndarray, weights: dict[str, float], config: RaiseTemplateConfig) -> np.ndarray:
    stand = baseline_q.copy()
    side = side_raise_template(baseline_q, config)
    front = front_raise_template(baseline_q, config)
    return stand * weights["stand"] + side * weights["side_raise"] + front * weights["front_raise"]


def side_raise_template(baseline_q: np.ndarray, config: RaiseTemplateConfig) -> np.ndarray:
    q = baseline_q.copy()
    q[0] = relative_target(baseline_q, 0, config.shoulder_pitch_assist_delta)
    q[7] = relative_target(baseline_q, 7, config.shoulder_pitch_assist_delta)
    q[1] = relative_target(baseline_q, 1, config.side_raise_roll_delta)
    q[8] = relative_target(baseline_q, 8, -config.side_raise_roll_delta)
    q[2] = soft_relative(baseline_q, 2, 0.06)
    q[9] = soft_relative(baseline_q, 9, -0.06)
    q[3] = relative_target(baseline_q, 3, config.side_raise_elbow_bend_delta)
    q[10] = relative_target(baseline_q, 10, config.side_raise_elbow_bend_delta)
    q[4] = soft_relative(baseline_q, 4, config.elbow_yaw_follow_delta * 0.35)
    q[11] = soft_relative(baseline_q, 11, -config.elbow_yaw_follow_delta * 0.35)
    keep_wrists_near_home(q, baseline_q)
    return q


def front_raise_template(baseline_q: np.ndarray, config: RaiseTemplateConfig) -> np.ndarray:
    q = baseline_q.copy()
    q[0] = relative_target(baseline_q, 0, config.front_raise_pitch_delta)
    q[7] = relative_target(baseline_q, 7, config.front_raise_pitch_delta)
    q[1] = relative_target(baseline_q, 1, config.front_raise_roll_assist_delta)
    q[8] = relative_target(baseline_q, 8, -config.front_raise_roll_assist_delta)
    q[2] = relative_target(baseline_q, 2, config.front_raise_yaw_delta)
    q[9] = relative_target(baseline_q, 9, -config.front_raise_yaw_delta)
    q[3] = relative_target(baseline_q, 3, config.front_raise_elbow_bend_delta)
    q[10] = relative_target(baseline_q, 10, config.front_raise_elbow_bend_delta)
    q[4] = relative_target(baseline_q, 4, config.elbow_yaw_follow_delta)
    q[11] = relative_target(baseline_q, 11, -config.elbow_yaw_follow_delta)
    q[5] = relative_target(baseline_q, 5, config.front_raise_wrist_pitch_delta)
    q[12] = relative_target(baseline_q, 12, config.front_raise_wrist_pitch_delta)
    q[6] = soft_relative(baseline_q, 6, 0.05)
    q[13] = soft_relative(baseline_q, 13, -0.05)
    return q


def apply_mocap_variation(
    q: np.ndarray,
    baseline_q: np.ndarray,
    weights: dict[str, float],
    feature_delta: dict[str, dict[str, float]],
    config: RaiseTemplateConfig,
) -> np.ndarray:
    front_weight = float(weights.get("front_raise", 0.0))
    side_weight = float(weights.get("side_raise", 0.0))
    if front_weight <= 1e-6 and side_weight <= 1e-6:
        return q

    left_front = clamp01(-feature_delta["left"]["front"] / max(1e-6, config.front_score_scale))
    right_front = clamp01(-0.7 * feature_delta["right"]["front"] / max(1e-6, config.front_score_scale))
    front_asym = float(np.clip(left_front - right_front, -1.0, 1.0))
    left_out = clamp01(feature_delta["left"]["outward"] / max(1e-6, config.side_score_scale))
    right_out = clamp01(feature_delta["right"]["outward"] / max(1e-6, config.side_score_scale))
    side_asym = float(np.clip(left_out - right_out, -1.0, 1.0))
    scale = float(config.mocap_variation_scale)

    varied = q.copy()
    varied[2] = relative_target(varied, 2, front_weight * scale * front_asym)
    varied[9] = relative_target(varied, 9, -front_weight * scale * front_asym)
    varied[4] = relative_target(varied, 4, front_weight * 0.5 * scale * front_asym)
    varied[11] = relative_target(varied, 11, -front_weight * 0.5 * scale * front_asym)
    varied[5] = relative_target(varied, 5, front_weight * 0.35 * scale * (left_front - 0.5))
    varied[12] = relative_target(varied, 12, front_weight * 0.35 * scale * (right_front - 0.5))
    varied[0] = relative_target(varied, 0, -side_weight * 0.25 * scale * max(left_out, right_out))
    varied[7] = relative_target(varied, 7, -side_weight * 0.25 * scale * max(left_out, right_out))
    varied[1] = relative_target(varied, 1, side_weight * 0.25 * scale * side_asym)
    varied[8] = relative_target(varied, 8, -side_weight * 0.25 * scale * side_asym)
    return varied


def keep_wrists_near_home(q: np.ndarray, baseline_q: np.ndarray) -> None:
    for index in (5, 6, 12, 13):
        q[index] = float(baseline_q[index])


def relative_target(base_q: np.ndarray, index: int, delta: float) -> float:
    return float(base_q[index]) + float(delta)


def soft_relative(base_q: np.ndarray, index: int, delta: float) -> float:
    return relative_target(base_q, index, delta)


def apply_control_side(q: np.ndarray, baseline_q: np.ndarray, control_side: str) -> np.ndarray:
    if control_side == "both":
        return q
    masked = baseline_q.copy()
    if control_side == "left":
        masked[:7] = q[:7]
    elif control_side == "right":
        masked[7:] = q[7:]
    else:
        raise ValueError(f"Unsupported control_side: {control_side}")
    return masked


def apply_template_safety(
    target_q: np.ndarray,
    reference_q: np.ndarray,
    max_step: float,
    smoothing_alpha: float,
) -> tuple[np.ndarray, bool]:
    target = np.asarray(target_q, dtype=float)
    reference = np.asarray(reference_q, dtype=float)
    step_limited = np.clip(target, reference - max_step, reference + max_step)
    limited = not np.allclose(step_limited, target, atol=1e-8)
    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
    safe_q = alpha * step_limited + (1.0 - alpha) * reference
    return safe_q, limited


def dominant_pose(weights: dict[str, float]) -> RaisePose:
    return max(weights, key=lambda key: weights[key])  # type: ignore[return-value]


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("骨架参考轴长度过小")
    return value / norm


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))