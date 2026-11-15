from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from teleop_joint_schema import HAND_JOINT_ORDER
from teleop_robot_constants import HAND_CHANNEL_ORDER, HAND_MOTOR_NAMES

_HAND_SIDES = ("left", "right")
_ROBOT_CHANNEL_TO_SDK_SUFFIX = {
    "little": "pinky",
    "ring": "ring",
    "middle": "middle",
    "fore": "index",
    "thumb_bend": "thumb_flex",
    "thumb_rotation": "thumb_abduct",
}
_DEFAULT_CLOSED_RAD = {
    "little": 1.41,
    "ring": 1.41,
    "middle": 1.41,
    "fore": 1.41,
    "thumb_bend": 1.03,
    "thumb_rotation": 1.57,
}


class HandControlError(ValueError):
    pass


@dataclass(frozen=True)
class HandChannelCalibration:
    open_rad: float
    closed_rad: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.open_rad) or not math.isfinite(self.closed_rad):
            raise HandControlError("手部标定范围必须是有限弧度值")
        if abs(self.closed_rad - self.open_rad) < 1e-9:
            raise HandControlError("手部标定 open_rad 与 closed_rad 不能相同")


@dataclass(frozen=True)
class HandControlConfig:
    publish_enabled: bool
    state_timeout_s: float
    max_step: float
    thumb_rotation_mode: str
    left: tuple[HandChannelCalibration, ...]
    right: tuple[HandChannelCalibration, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.state_timeout_s) or self.state_timeout_s <= 0:
            raise HandControlError("手部状态超时必须大于 0")
        if not 0 < self.max_step <= 1:
            raise HandControlError("手部单帧最大变化必须在 (0, 1] 内")
        if self.thumb_rotation_mode not in {"hold_feedback", "mapped"}:
            raise HandControlError("thumb_rotation_mode 必须是 hold_feedback 或 mapped")
        expected = len(HAND_CHANNEL_ORDER)
        if len(self.left) != expected or len(self.right) != expected:
            raise HandControlError(f"每只手必须配置 {expected} 路标定")

    def calibrations(self, side: str) -> tuple[HandChannelCalibration, ...]:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise HandControlError(f"未知手部侧别: {side}")


def default_hand_control_mapping() -> dict[str, Any]:
    calibration = {
        channel: {
            "open_rad": 0.0,
            "closed_rad": _DEFAULT_CLOSED_RAD[channel],
            "enabled": True,
        }
        for channel in HAND_CHANNEL_ORDER
    }
    return {
        "publish": False,
        "state_timeout": 0.5,
        "max_step": 0.05,
        "thumb_rotation_mode": "hold_feedback",
        "calibration": {
            "left": {name: dict(values) for name, values in calibration.items()},
            "right": {name: dict(values) for name, values in calibration.items()},
        },
    }


def _channel_calibration(raw: Mapping[str, Any], side: str, channel: str) -> HandChannelCalibration:
    try:
        values = raw["calibration"][side][channel]
        return HandChannelCalibration(
            open_rad=float(values["open_rad"]),
            closed_rad=float(values["closed_rad"]),
            enabled=bool(values.get("enabled", True)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HandControlError(f"手部标定配置无效: {side}.{channel}") from exc


def hand_control_config_from_mapping(raw: Mapping[str, Any] | None) -> HandControlConfig:
    defaults = default_hand_control_mapping()
    source = defaults if raw is None else raw
    try:
        calibrations = {
            side: tuple(_channel_calibration(source, side, channel) for channel in HAND_CHANNEL_ORDER)
            for side in _HAND_SIDES
        }
        return HandControlConfig(
            publish_enabled=bool(source.get("publish", defaults["publish"])),
            state_timeout_s=float(source.get("state_timeout", defaults["state_timeout"])),
            max_step=float(source.get("max_step", defaults["max_step"])),
            thumb_rotation_mode=str(source.get("thumb_rotation_mode", defaults["thumb_rotation_mode"])),
            left=calibrations["left"],
            right=calibrations["right"],
        )
    except (TypeError, ValueError) as exc:
        raise HandControlError("手部控制配置无效") from exc


def parse_hand_state_message(message: Mapping[str, Any]) -> tuple[float, ...]:
    names = message.get("name", ())
    positions = message.get("position", ())
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise HandControlError("手部状态 name 必须是数组")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        raise HandControlError("手部状态 position 必须是数组")
    if len(names) != len(positions):
        raise HandControlError("手部状态 name 与 position 长度不一致")

    by_name: dict[str, float] = {}
    for raw_name, raw_position in zip(names, positions):
        name = str(raw_name)
        if name in by_name:
            raise HandControlError(f"手部状态包含重复 ID: {name}")
        try:
            position = float(raw_position)
        except (TypeError, ValueError) as exc:
            raise HandControlError(f"手部状态 ID={name} 位置不是数值") from exc
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise HandControlError(f"手部状态 ID={name} 位置必须在 0..1 内")
        by_name[name] = position

    missing = [name for name in HAND_MOTOR_NAMES if name not in by_name]
    if missing:
        raise HandControlError(f"手部状态缺少 ID: {missing}")
    return tuple(by_name[name] for name in HAND_MOTOR_NAMES)


def _named_sdk_radians(joint_positions: Sequence[float]) -> dict[str, float]:
    if len(joint_positions) != len(HAND_JOINT_ORDER):
        raise HandControlError(f"Revo2 手部输出必须是 {len(HAND_JOINT_ORDER)} 维")
    try:
        named = {name: float(value) for name, value in zip(HAND_JOINT_ORDER, joint_positions)}
    except (TypeError, ValueError) as exc:
        raise HandControlError("Revo2 手部输出包含非数值") from exc
    if not all(math.isfinite(value) for value in named.values()):
        raise HandControlError("Revo2 手部输出包含 NaN/Inf")
    return named


def map_hand_radians(
    joint_positions: Sequence[float],
    side: str,
    config: HandControlConfig,
) -> tuple[float, ...]:
    named = _named_sdk_radians(joint_positions)
    calibrations = config.calibrations(side)
    ratios: list[float] = []
    for channel, calibration in zip(HAND_CHANNEL_ORDER, calibrations):
        sdk_name = f"{side}_{_ROBOT_CHANNEL_TO_SDK_SUFFIX[channel]}"
        radians = named[sdk_name]
        ratio = (calibration.closed_rad - radians) / (calibration.closed_rad - calibration.open_rad)
        ratios.append(min(1.0, max(0.0, ratio)))
    return tuple(ratios)


def apply_hand_command_safety(
    target: Sequence[float],
    feedback: Sequence[float],
    last_command: Sequence[float] | None,
    side: str,
    config: HandControlConfig,
) -> tuple[float, ...]:
    if side not in _HAND_SIDES:
        raise HandControlError(f"未知手部侧别: {side}")
    expected = len(HAND_CHANNEL_ORDER)
    if len(target) != expected or len(feedback) != expected:
        raise HandControlError(f"手部目标和反馈必须是 {expected} 维")
    if last_command is not None and len(last_command) != expected:
        raise HandControlError(f"手部上一帧命令必须是 {expected} 维")

    target_values = tuple(float(value) for value in target)
    feedback_values = tuple(float(value) for value in feedback)
    reference_values = feedback_values if last_command is None else tuple(float(value) for value in last_command)
    if not all(math.isfinite(value) for value in target_values + feedback_values + reference_values):
        raise HandControlError("手部安全层输入包含 NaN/Inf")

    calibrations = config.calibrations(side)
    safe: list[float] = []
    for index, (value, feedback_value, reference, calibration) in enumerate(
        zip(target_values, feedback_values, reference_values, calibrations)
    ):
        desired = min(1.0, max(0.0, value))
        if not calibration.enabled:
            desired = feedback_value
        if HAND_CHANNEL_ORDER[index] == "thumb_rotation" and config.thumb_rotation_mode == "hold_feedback":
            desired = feedback_value
        desired = min(reference + config.max_step, max(reference - config.max_step, desired))
        safe.append(min(1.0, max(0.0, desired)))
    return tuple(safe)


def build_hand_joint_state_message(positions: Sequence[float]) -> dict[str, Any]:
    if len(positions) != len(HAND_MOTOR_NAMES):
        raise HandControlError(f"手部命令必须是 {len(HAND_MOTOR_NAMES)} 维")
    values = [float(value) for value in positions]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise HandControlError("手部命令位置必须是 0..1 内的有限值")
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
        "name": list(HAND_MOTOR_NAMES),
        "position": values,
        "velocity": [],
        "effort": [],
    }
