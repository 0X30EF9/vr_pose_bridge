from __future__ import annotations

import copy
from typing import Any

import numpy as np

from teleop_ik_types import IKCommand


def select_command_slice(command: IKCommand, send_scope: str = "controlled_side"):
    """把统一 14 维命令切成当前后端需要的范围。

    仿真和真机可能只接收单侧 7 轴，也可能接收完整 14 轴。切片规则必须集中在适配层，
    否则桥接节点会重新产生一套业务分支。
    """
    positions = np.asarray(command.positions, dtype=float)
    if command.control_side not in {"left", "right", "both"}:
        raise ValueError("关节命令控制侧无效")
    if send_scope not in {"controlled_side", "full"}:
        raise ValueError("关节命令发送范围无效")
    if positions.shape != (14,) or not np.all(np.isfinite(positions)):
        raise ValueError("关节命令必须包含有限的 14 维位置")
    if len(command.joint_names) != 14 or len(set(command.joint_names)) != 14:
        raise ValueError("关节命令名称缺失或重复")
    if len(command.motor_ids) != 14 or len(set(command.motor_ids)) != 14:
        raise ValueError("关节命令电机 ID 缺失或重复")
    if send_scope == "full" or command.control_side == "both":
        return command.joint_names, positions, command.motor_ids
    if command.control_side == "left":
        return command.joint_names[:7], positions[:7], command.motor_ids[:7]
    return command.joint_names[7:], positions[7:], command.motor_ids[7:]


def build_http_payload(
    command: IKCommand,
    payload_format: str = "joint_positions",
    send_scope: str = "controlled_side",
) -> dict[str, Any]:
    """将统一 IKCommand 转成仿真 HTTP 负载。

    HTTP 层只关心可序列化数据；不要在这里再做 IK、限幅或状态回退。
    """

    names, positions, motor_ids = select_command_slice(command, send_scope)
    if payload_format == "motor_commands":
        return {
            "timestamp": command.timestamp,
            "control_side": command.control_side,
            "format": payload_format,
            "cmds": [
                {"name": int(motor_id), "pos": float(position)}
                for motor_id, position in zip(motor_ids, positions)
            ],
            "meta": command.meta,
        }

    return {
        "timestamp": command.timestamp,
        "control_side": command.control_side,
        "format": payload_format,
        "joint_names": names,
        "positions": [float(value) for value in positions],
        "meta": command.meta,
    }


def build_rosbridge_motor_message(
    command: IKCommand,
    motor_templates: list[dict[str, Any]],
    send_scope: str = "controlled_side",
) -> dict[str, Any]:
    """将 IKCommand 转成 /arm/cmd_pos 的 rosbridge 消息。

    `motor_templates` 是真机 `/arm/cmd_pos` 的固定消息模板，适配器只填 pos，避免复制底层消息结构知识。
    """

    _, positions, motor_ids = select_command_slice(command, send_scope)
    command_motors = copy.deepcopy(motor_templates)
    template_ids = [int(motor["name"]) for motor in command_motors]
    if len(set(template_ids)) != len(template_ids) or set(template_ids) != set(motor_ids):
        raise ValueError("电机模板必须与本次命令的电机 ID 完整匹配")
    by_id = dict(zip(motor_ids, positions))
    for motor, motor_id in zip(command_motors, template_ids):
        motor["pos"] = float(by_id[motor_id])

    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": command.frame_id},
        "cmds": command_motors,
    }


def build_rosbridge_waist_message(
    command: IKCommand,
    motor_template: dict[str, Any],
) -> dict[str, Any]:
    """将官方 15 轴解中的腰部角度转成 `/waist/cmd_pos` 消息。"""

    if command.waist_position is None:
        raise ValueError("当前 IKCommand 不包含腰部位置")
    if not np.isfinite(command.waist_position):
        raise ValueError("腰部位置必须有限")
    waist_motor = copy.deepcopy(motor_template)
    waist_motor["pos"] = float(command.waist_position)
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": command.frame_id},
        "cmds": [waist_motor],
    }
