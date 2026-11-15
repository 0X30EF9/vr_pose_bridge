from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


def feedback_is_fresh(received_at: float | None, now: float, timeout: float) -> bool:
    """Check local monotonic receipt times, including clock and configuration validity."""
    if received_at is None:
        return False
    try:
        received_at, now, timeout = float(received_at), float(now), float(timeout)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in (received_at, now, timeout))
        and timeout > 0.0
        and 0.0 <= now - received_at <= timeout
    )


def require_fresh_feedback(
    received_at: float | None, now: float, timeout: float, label: str = "feedback"
) -> None:
    if not feedback_is_fresh(received_at, now, timeout):
        raise RuntimeError(f"{label} 缺失、无效或已过期，停止发送控制命令")


def command_time_step(rate: float, now: float, last_sent_at: float | None = None) -> float:
    """Never accumulate a larger motion allowance across missed control periods."""
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(now):
        raise ValueError("control rate and monotonic time must be finite and valid")
    nominal = 1.0 / rate
    if last_sent_at is None:
        return nominal
    if not math.isfinite(last_sent_at) or now < last_sent_at:
        raise ValueError("last command time is invalid")
    return min(nominal, now - last_sent_at)


def joint_velocity_limits_from_urdf(path: str | Path, joint_names) -> tuple[float, ...]:
    joints = {joint.attrib.get("name"): joint for joint in ET.parse(path).getroot().findall("joint")}
    result = []
    for name in joint_names:
        joint = joints.get(name)
        limit = None if joint is None else joint.find("limit")
        value = float("nan") if limit is None else float(limit.attrib.get("velocity", "nan"))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"URDF joint {name} must provide a positive finite velocity limit")
        result.append(value)
    return tuple(result)
