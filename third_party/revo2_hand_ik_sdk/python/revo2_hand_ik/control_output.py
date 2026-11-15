"""Direct joint-radian output helpers for the Revo2 Python SDK.

The Python SDK public output is intentionally **not normalized**.  The native
C ABI still exposes both ``position_norm_0_100`` and ``position_rad`` for C/C++
compatibility, but this module only builds the customer-facing Python output
from ``position_rad``.

Fixed 12-channel order:
    [left thumb flex, left thumb abduct, left index, left middle, left ring,
     left pinky, right thumb flex, right thumb abduct, right index,
     right middle, right ring, right pinky]

Unit: radian.
"""
from __future__ import annotations

import json
from typing import Any, Final

DIRECT_JOINT_OUTPUT_ORDER: Final[tuple[str, ...]] = (
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

# Compatibility alias for older in-tree tests/imports.  The semantic unit is now
# radians, not the old 0..100 finger_data_set convention.
FINGER_DATA_SET_ORDER: Final[tuple[str, ...]] = DIRECT_JOINT_OUTPUT_ORDER


def _named_rad_controls(result: dict[str, Any], side: str) -> dict[str, float]:
    """Read six named native SDK radian outputs without depending on array order."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    prefix = f"{side}_"
    named = {
        str(item.get("name", "")): float(item.get("rad", 0.0))
        for item in result.get("joints", [])
    }
    required = {
        "thumb_flex": prefix + "thumb_flex_joint",
        "thumb_abduct": prefix + "thumb_abduct_joint",
        "index": prefix + "index_joint",
        "middle": prefix + "middle_joint",
        "ring": prefix + "ring_joint",
        "pinky": prefix + "pinky_joint",
    }
    missing = [joint for joint in required.values() if joint not in named]
    if missing:
        raise ValueError("SDK did not return required 6-DOF controls: " + ", ".join(missing))
    return {key: named[joint] for key, joint in required.items()}


def joint_position_rad_from_sdk(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
) -> tuple[float, ...]:
    """Return the 12-value joint-position vector in radians."""
    left = _named_rad_controls(left_result, "left")
    right = _named_rad_controls(right_result, "right")
    return (
        left["thumb_flex"],
        left["thumb_abduct"],
        left["index"],
        left["middle"],
        left["ring"],
        left["pinky"],
        right["thumb_flex"],
        right["thumb_abduct"],
        right["index"],
        right["middle"],
        right["ring"],
        right["pinky"],
    )


def joint_position_rad_named(values: tuple[float, ...] | list[float]) -> dict[str, float]:
    """Expose a named view for logs and integrations that do not use indexes."""
    if len(values) != len(DIRECT_JOINT_OUTPUT_ORDER):
        raise ValueError("joint position output requires exactly 12 values")
    return {name: float(value) for name, value in zip(DIRECT_JOINT_OUTPUT_ORDER, values)}


def make_joint_position_rad_payload(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    *,
    sequence: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    """Build the Python SDK public output dictionary.

    This function performs no 0..100 normalization and no clamping.  Values are
    exactly the native SDK ``position_rad`` values selected by joint name.
    """
    values = joint_position_rad_from_sdk(left_result, right_result)
    return {
        "protocol": "revo2_dual_hand_ik_sdk_v2",
        "sequence": int(sequence),
        "timestamp_ms": int(timestamp_ms),
        "joint_unit": "rad",
        "joint_order": list(DIRECT_JOINT_OUTPUT_ORDER),
        "joint_positions": list(values),
        "joint_positions_named": joint_position_rad_named(values),
        "left_solve_time_ms": float(left_result.get("solve_time_ms", 0.0)),
        "right_solve_time_ms": float(right_result.get("solve_time_ms", 0.0)),
    }


def make_joint_position_rad_json_payload(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    *,
    sequence: int,
    timestamp_ms: int,
) -> bytes:
    """Encode the direct interface result as JSON bytes when a caller wants JSON."""
    return json.dumps(
        make_joint_position_rad_payload(
            left_result,
            right_result,
            sequence=sequence,
            timestamp_ms=timestamp_ms,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


# Backward-compatible function names.  They now return radians rather than
# normalized 0..100 values.  New integrations should use the joint_* names above.
finger_data_set_from_sdk = joint_position_rad_from_sdk
finger_data_set_named = joint_position_rad_named


def make_finger_data_set_json_payload(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    *,
    sequence: int,
    timestamp_ms: int,
    role_name: str | None = None,
) -> bytes:
    """Compatibility wrapper for older UDP bridge call sites.

    The old function name is preserved so legacy scripts still import, but the
    payload is the new public SDK contract: ``joint_positions`` in radians.
    ``role_name`` is accepted and ignored because the direct SDK result is not a
    robot-control protocol packet.
    """
    del role_name
    return make_joint_position_rad_json_payload(
        left_result,
        right_result,
        sequence=sequence,
        timestamp_ms=timestamp_ms,
    )
