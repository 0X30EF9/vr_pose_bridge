from __future__ import annotations

from typing import Sequence

# 这些顺序是机器人命令与可视化之间的协议契约，不属于某个具体求解器。
ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)

ARM_MOTOR_IDS: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27)

HAND_JOINT_ORDER: tuple[str, ...] = (
    "left_thumb_flex",
    "left_thumb_abduct",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb_flex",
    "right_thumb_abduct",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)

_MOTOR_TO_JOINT = dict(zip(ARM_MOTOR_IDS, ARM_JOINT_NAMES))


def arm_joint_names(control_side: str) -> tuple[str, ...]:
    if control_side == "left":
        return ARM_JOINT_NAMES[:7]
    if control_side == "right":
        return ARM_JOINT_NAMES[7:]
    if control_side == "both":
        return ARM_JOINT_NAMES
    raise ValueError(f"Unsupported control_side: {control_side}")


def joint_names_from_motor_ids(motor_ids: Sequence[int]) -> tuple[str, ...]:
    try:
        return tuple(_MOTOR_TO_JOINT[int(motor_id)] for motor_id in motor_ids)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"未知手臂电机 ID: {motor_ids}") from exc