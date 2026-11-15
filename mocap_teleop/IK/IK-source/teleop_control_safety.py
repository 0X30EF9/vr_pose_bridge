from __future__ import annotations

import math


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
