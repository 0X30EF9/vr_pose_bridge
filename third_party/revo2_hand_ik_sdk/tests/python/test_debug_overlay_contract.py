from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from revo2_hand_ik.udp_bridge import make_debug_overlay_payload  # noqa: E402


def _point(x: float, y: float, z: float) -> dict[str, float | bool]:
    return {"x": x, "y": y, "z": z, "valid": True}


def test_overlay_keeps_source_skeleton_separate_from_exact_tasks() -> None:
    debug = {
        "frame_id": "left_base_link",
        "calibrated": True,
        "points": {"thumb_near": _point(0.0, 0.0, 0.0), "thumb_middle": _point(0.01, 0.0, 0.0)},
        "thumb_target_middle": _point(0.02, 0.0, 0.0),
        "thumb_target_far": _point(0.03, 0.0, 0.0),
        "thumb_robot_middle": _point(0.021, 0.0, 0.0),
        "thumb_robot_far": _point(0.031, 0.0, 0.0),
        "long_finger_tasks": [{
            "name": "index", "valid": True,
            "target_middle": _point(0.0, 0.02, 0.0), "target_far": _point(0.0, 0.04, 0.0),
            "robot_middle": _point(0.0, 0.021, 0.0), "robot_far": _point(0.0, 0.041, 0.0),
        }],
    }
    raw = {"hand": (1.0, 2.0, 3.0), "thumb_middle": (1.01, 2.0, 3.0)}
    payload = json.loads(make_debug_overlay_payload(
        debug, debug, sequence=7, timestamp_ms=12,
        left_raw_points=raw, right_raw_points=raw,
    ).decode("utf-8"))
    hand = payload["hands"][0]
    source_middle = next(item["point"] for item in hand["points"] if item["name"] == "thumb_middle")
    assert source_middle["x"] == 0.01
    assert hand["thumb_task"]["target_middle"]["x"] == 0.02
    assert hand["long_finger_tasks"][0]["name"] == "index"
    assert hand["long_finger_tasks"][0]["robot_far"]["y"] == 0.041
    assert hand["raw_frame_id"] == "world"
    raw_middle = next(item["point"] for item in hand["raw_points"] if item["name"] == "thumb_middle")
    assert raw_middle["x"] == 1.01
