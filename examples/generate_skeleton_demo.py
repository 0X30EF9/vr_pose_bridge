"""Generate a synthetic 64-node CSV for offline arm-mapping previews only."""
from pathlib import Path
import csv
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "src"))
from mocap_csv_format import CSV_FIELDNAMES, mocap_frame_to_csv_row
from teleop_mocap_protocol import JointPose, MOCAP_JOINT_NAMES, MocapFrame


def demo_frame(index: int) -> MocapFrame:
    bend = max(0, index - 44) * 0.006
    positions = {
        "left_shoulder": (0.0, 0.22, 1.45), "right_shoulder": (0.0, -0.22, 1.45),
        "left_hip": (0.0, 0.14, 0.95), "right_hip": (0.0, -0.14, 0.95),
        "waist": (0.0, 0.0, 0.95), "chest": (0.0, 0.0, 1.30),
        "neck": (0.0, 0.0, 1.55), "head": (0.0, 0.0, 1.70),
    }
    for side, lateral in (("left", 0.22), ("right", -0.22)):
        positions[f"{side}_elbow"] = (0.0, lateral, 1.12)
        positions[f"{side}_wrist"] = (0.34 * math.sin(bend), lateral, 1.12 - 0.34 * math.cos(bend))
        positions[f"{side}_hand"] = (0.40 * math.sin(bend), lateral, 1.12 - 0.40 * math.cos(bend))
    joints = []
    for name in MOCAP_JOINT_NAMES:
        quaternion = (0.0, 0.0, 0.0, 1.0)
        if name in {"left_hand", "right_hand", "left_wrist", "right_wrist"}:
            quaternion = (0.0, math.sin(bend / 2), 0.0, math.cos(bend / 2))
        joints.append(JointPose(positions.get(name, (0.0, 0.0, 0.0)), quaternion))
    # Use an unambiguous microsecond epoch; small historical CSV timestamps
    # are intentionally interpreted as milliseconds by the compatibility reader.
    timestamp_us = 1_700_000_000_000_000 + index * 50_000
    return MocapFrame(timestamp_us / 1e6, timestamp_us, timestamp_us / 1e6,
                      tuple(joints), "synthetic_offline_example")


if __name__ == "__main__":
    output = Path(__file__).with_name("skeleton_demo.csv")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(mocap_frame_to_csv_row(demo_frame(index)) for index in range(60))
    print(output)
