from __future__ import annotations

from pathlib import Path

from revo2_hand_ik.api import load_dual_sdk_config
from revo2_hand_ik.control_output import (
    DIRECT_JOINT_OUTPUT_ORDER,
    joint_position_rad_from_sdk,
    make_joint_position_rad_payload,
)


def _result(side: str, base: float) -> dict:
    names = (
        "thumb_flex_joint",
        "thumb_abduct_joint",
        "index_joint",
        "middle_joint",
        "ring_joint",
        "pinky_joint",
    )
    return {
        "joints": [
            {"name": f"{side}_{name}", "norm_0_100": 1000.0 + i, "rad": base + i}
            for i, name in enumerate(names)
        ],
        "solve_time_ms": 0.1,
    }


def test_direct_radian_vector_order_without_normalization() -> None:
    values = joint_position_rad_from_sdk(_result("left", -10.0), _result("right", 20.0))
    assert values == (-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0)
    assert len(DIRECT_JOINT_OUTPUT_ORDER) == 12


def test_direct_payload_has_rad_unit_and_no_finger_data_set() -> None:
    payload = make_joint_position_rad_payload(
        _result("left", 1.0), _result("right", 11.0),
        sequence=7, timestamp_ms=99,
    )
    assert payload["protocol"] == "revo2_dual_hand_ik_sdk_v2"
    assert payload["joint_unit"] == "rad"
    assert tuple(payload["joint_order"]) == DIRECT_JOINT_OUTPUT_ORDER
    assert payload["joint_positions"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    assert "finger_data_set" not in payload


def test_direct_sdk_config_has_no_udp_sections() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json"
    cfg = load_dual_sdk_config(path)
    assert cfg.left_sdk_config.name == "mocap_ik_revo2_left_hand_sdk.json"
    assert cfg.right_sdk_config.name == "mocap_ik_revo2_right_hand_sdk.json"
    text = path.read_text(encoding="utf-8")
    assert "input_udp" not in text
    assert "output_udp" not in text
