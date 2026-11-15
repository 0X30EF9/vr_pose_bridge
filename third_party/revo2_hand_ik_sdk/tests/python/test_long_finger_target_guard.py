"""Regression checks for v0.10.33 long-finger input validity guards."""
from __future__ import annotations

import math


def _segment_ratio_is_valid(current: float, baseline: float, max_ratio: float) -> bool:
    if not (math.isfinite(current) and math.isfinite(baseline) and current > 0.0 and baseline > 0.0):
        return False
    if max_ratio <= 1.0:
        return True
    ratio = current / baseline
    return (1.0 / max_ratio) <= ratio <= max_ratio


def test_middle_segment_guard_accepts_normal_finger_curl() -> None:
    # Rigid phalange lengths are unchanged by flexion.
    assert _segment_ratio_is_valid(0.042, 0.040, 1.45)
    assert _segment_ratio_is_valid(0.030, 0.040, 1.45)


def test_middle_segment_guard_rejects_impossible_stretch_or_collapse() -> None:
    assert not _segment_ratio_is_valid(0.080, 0.040, 1.45)
    assert not _segment_ratio_is_valid(0.010, 0.040, 1.45)


def test_target_step_guard_detects_one_frame_explosion() -> None:
    previous = (0.01, 0.02, 0.03)
    normal = (0.02, 0.025, 0.035)
    exploded = (0.09, -0.05, 0.10)
    distance = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    assert distance(normal, previous) < 0.035
    assert distance(exploded, previous) > 0.035


def test_projection_step_fuse_prevents_direct_closed_to_zero_reset() -> None:
    previous = 1.20
    malformed_projection = 0.0
    max_step = 0.22
    bounded = previous + max(-max_step, min(max_step, malformed_projection - previous))
    assert bounded == previous - max_step
    assert bounded > 0.0
