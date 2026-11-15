#!/usr/bin/env python3
"""Native SDK smoke test executed after build_python_runtime.sh.

This test intentionally checks the Python ctypes ABI and the agreed external
6-DOF result contract without requiring ROS. It uses a synthetic open-hand
frame only to prove the compiled SDK can initialize, solve, and return the
six named controls in radians.
"""
from __future__ import annotations

import os
from pathlib import Path
import unittest

from revo2_hand_ik import Revo2HandIkSdk


EXPECTED_LEFT_OUTPUTS = [
    "left_thumb_flex_joint",
    "left_thumb_abduct_joint",
    "left_index_joint",
    "left_middle_joint",
    "left_ring_joint",
    "left_pinky_joint",
]


def open_left_hand_points() -> dict[str, tuple[float, float, float]]:
    """Small non-degenerate synthetic hand pose in metres.

    The SDK calibrates the first frame as the open-hand baseline. The exact
    geometry is not intended to match a wearer; it only guarantees all three
    points per finger are present and non-collinear enough for the IK path.
    """
    return {
        "hand": (0.0, 0.0, 0.0),
        "thumb_near": (-0.020, -0.020, 0.000),
        "thumb_middle": (-0.040, -0.035, 0.000),
        "thumb_far": (-0.055, -0.050, 0.000),
        "index_near": (0.020, 0.000, 0.000),
        "index_middle": (0.020, 0.030, 0.000),
        "index_far": (0.020, 0.060, 0.000),
        "middle_near": (0.000, 0.000, 0.000),
        "middle_middle": (0.000, 0.035, 0.000),
        "middle_far": (0.000, 0.070, 0.000),
        "ring_near": (-0.020, 0.000, 0.000),
        "ring_middle": (-0.020, 0.032, 0.000),
        "ring_far": (-0.020, 0.064, 0.000),
        "little_near": (-0.038, 0.000, 0.000),
        "little_middle": (-0.038, 0.027, 0.000),
        "little_far": (-0.038, 0.054, 0.000),
    }


class NativeSdkSmokeTest(unittest.TestCase):
    def test_left_hand_sdk_contract(self) -> None:
        runtime_root = Path(__file__).resolve().parents[2]
        config = runtime_root / "bin/conf/mocap_ik/mocap_ik_revo2_left_hand_sdk.json"
        self.assertTrue(config.is_file(), f"missing runtime config: {config}")
        library = runtime_root / "lib/librevo2_ik_sdk.so"
        if not library.is_file():
            self.skipTest("SDK .so is not built in the source archive")

        with Revo2HandIkSdk(config, library_path=library) as sdk:
            result = sdk.solve(
                open_left_hand_points(),
                quaternions_wxyz={"thumb_far": (1.0, 0.0, 0.0, 0.0)},
                sequence=1,
                timestamp_ms=1,
            )

        self.assertEqual(result["joint_count"], 6)
        self.assertEqual([item["name"] for item in result["joints"]], EXPECTED_LEFT_OUTPUTS)
        for item in result["joints"]:
            self.assertIn("rad", item)
        self.assertGreaterEqual(result["solve_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
