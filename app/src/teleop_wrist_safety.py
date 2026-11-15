from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

SIDES = ("left", "right")
SIDE_OFFSETS = {"left": 0, "right": 7}
WRIST_LOCAL_INDICES = (5, 6)
ELBOW_LOCAL_INDICES = (3, 4)


@dataclass(frozen=True)
class WristBehaviorLimits:
    max_wrist_step_rad: float
    max_elbow_step_rad: float
    soft_limit_margin_rad: float = 0.01
    max_wrist_position_error_m: float = 0.010


def controlled_sides(control_side: str) -> tuple[str, ...]:
    if control_side == "both":
        return SIDES
    if control_side not in SIDES:
        raise ValueError(f"不支持的 control_side: {control_side}")
    return (control_side,)


def wrist_position_errors(
    actual_poses: Sequence[np.ndarray],
    target_poses: Mapping[str, np.ndarray],
    control_side: str,
) -> dict[str, float]:
    if len(actual_poses) != 2:
        raise ValueError("actual_poses 必须包含左右两个 wrist pose")
    result: dict[str, float] = {}
    for side, actual in zip(SIDES, actual_poses):
        if side not in controlled_sides(control_side):
            continue
        target = np.asarray(target_poses[side], dtype=float)
        actual_pose = np.asarray(actual, dtype=float)
        if target.shape != (4, 4) or actual_pose.shape != (4, 4):
            raise ValueError(f"{side} wrist pose 必须是 4x4")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual_pose)):
            raise ValueError(f"{side} wrist pose 包含 NaN/Inf")
        result[side] = float(np.linalg.norm(actual_pose[:3, 3] - target[:3, 3]))
    return result


def validate_wrist_behavior_candidate(
    candidate_q: Sequence[float],
    reference_q: Sequence[float],
    joint_limits: Sequence[tuple[float, float]],
    control_side: str,
    wrist_position_error_by_side: Mapping[str, float],
    limits: WristBehaviorLimits,
) -> dict[str, tuple[str, ...]]:
    candidate = np.asarray(candidate_q, dtype=float).reshape(-1)
    reference = np.asarray(reference_q, dtype=float).reshape(-1)
    if candidate.shape != (14,) or reference.shape != (14,):
        raise ValueError("腕部行为门禁要求 14 维 candidate_q/reference_q")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        raise ValueError("腕部行为门禁输入包含 NaN/Inf")
    if len(joint_limits) != 14:
        raise ValueError("腕部行为门禁要求 14 组关节限位")

    reasons: dict[str, tuple[str, ...]] = {}
    for side in controlled_sides(control_side):
        offset = SIDE_OFFSETS[side]
        wrist_indices = tuple(offset + index for index in WRIST_LOCAL_INDICES)
        elbow_indices = tuple(offset + index for index in ELBOW_LOCAL_INDICES)
        side_reasons: list[str] = []

        wrist_step = float(np.max(np.abs(candidate[list(wrist_indices)] - reference[list(wrist_indices)])))
        if wrist_step > limits.max_wrist_step_rad:
            side_reasons.append(
                f"wrist_step={wrist_step:.6f}>{limits.max_wrist_step_rad:.6f}rad"
            )
        elbow_step = float(np.max(np.abs(candidate[list(elbow_indices)] - reference[list(elbow_indices)])))
        if elbow_step > limits.max_elbow_step_rad:
            side_reasons.append(
                f"elbow_step={elbow_step:.6f}>{limits.max_elbow_step_rad:.6f}rad"
            )

        for index in wrist_indices:
            lower, upper = (float(value) for value in joint_limits[index])
            margin = min(limits.soft_limit_margin_rad, max(0.0, (upper - lower) * 0.25))
            soft_lower = lower + margin
            soft_upper = upper - margin
            moving_deeper_below = candidate[index] < soft_lower and candidate[index] < reference[index]
            moving_deeper_above = candidate[index] > soft_upper and candidate[index] > reference[index]
            if moving_deeper_below or moving_deeper_above:
                side_reasons.append(
                    f"joint[{index}]={candidate[index]:.6f} outside_soft_limit="
                    f"[{soft_lower:.6f},{soft_upper:.6f}]"
                )

        position_error = wrist_position_error_by_side.get(side)
        if position_error is None or not np.isfinite(position_error):
            side_reasons.append("wrist_position_error=missing_or_invalid")
        elif float(position_error) > limits.max_wrist_position_error_m:
            side_reasons.append(
                f"wrist_position_error={float(position_error):.6f}>"
                f"{limits.max_wrist_position_error_m:.6f}m"
            )

        if side_reasons:
            reasons[side] = tuple(side_reasons)
    return reasons