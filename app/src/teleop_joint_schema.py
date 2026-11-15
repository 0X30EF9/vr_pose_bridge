from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

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


def arm_positions_from_status(status: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
    """Read a complete arm sample, preserving only the unambiguous unnamed legacy layout."""
    if not status or not all(isinstance(item, Mapping) for item in status):
        raise ValueError("手臂反馈必须包含完整电机状态")
    named = ["name" in item for item in status]
    if any(named) and not all(named):
        raise ValueError("手臂反馈不能混合命名和未命名电机")
    if not any(named) and len(status) != len(ARM_MOTOR_IDS):
        raise ValueError("旧版未命名手臂反馈必须恰好包含 14 轴")

    positions = []
    by_id = {}
    for item in status:
        try:
            if isinstance(item.get("pos"), bool):
                raise ValueError("电机位置必须为数值")
            position = float(item["pos"])
            error = float(item.get("error", 0))
            if not math.isfinite(position) or not math.isfinite(error) or error != 0:
                raise ValueError("电机反馈非有限或带有告警")
            if all(named):
                motor_id = int(item["name"])
                if isinstance(item["name"], bool) or float(item["name"]) != motor_id:
                    raise ValueError("电机 ID 必须为整数")
                if motor_id in by_id:
                    raise ValueError(f"电机 ID 重复: {motor_id}")
                by_id[motor_id] = position
            positions.append(position)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"手臂电机反馈无效: {exc}") from exc

    if all(named):
        if not all(motor_id in by_id for motor_id in ARM_MOTOR_IDS):
            raise ValueError("手臂反馈缺少预期电机 ID")
        return tuple(by_id[motor_id] for motor_id in ARM_MOTOR_IDS)
    return tuple(positions)


def arm_joint_limits_from_urdf(path: str | Path) -> list[tuple[float, float]]:
    """Read the command-order mechanical limits without loading native model libraries."""
    joints = {joint.attrib["name"]: joint for joint in ET.parse(path).getroot().findall("joint")}
    limits = []
    for name in ARM_JOINT_NAMES:
        try:
            limit = joints[name].find("limit")
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"URDF 缺少 {name} 的有效机械限位") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"URDF 的 {name} 机械限位无效")
        limits.append((lower, upper))
    return limits
