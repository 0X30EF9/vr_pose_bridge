from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import Revo2DualHandIkSdk


def _open_hand_points(side: str) -> dict[str, tuple[float, float, float]]:
    """Synthetic open-hand frame used only for smoke testing the interface."""
    sign = 1.0 if side == "left" else -1.0
    return {
        "hand": (0.0, 0.0, 0.0),
        "thumb_near": (-0.020 * sign, -0.020, 0.000),
        "thumb_middle": (-0.040 * sign, -0.035, 0.000),
        "thumb_far": (-0.055 * sign, -0.050, 0.000),
        "index_near": (0.020 * sign, 0.000, 0.000),
        "index_middle": (0.020 * sign, 0.030, 0.000),
        "index_far": (0.020 * sign, 0.060, 0.000),
        "middle_near": (0.000, 0.000, 0.000),
        "middle_middle": (0.000, 0.035, 0.000),
        "middle_far": (0.000, 0.070, 0.000),
        "ring_near": (-0.020 * sign, 0.000, 0.000),
        "ring_middle": (-0.020 * sign, 0.032, 0.000),
        "ring_far": (-0.020 * sign, 0.064, 0.000),
        "little_near": (-0.038 * sign, 0.000, 0.000),
        "little_middle": (-0.038 * sign, 0.027, 0.000),
        "little_far": (-0.038 * sign, 0.054, 0.000),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Revo2 hand IK direct-call Python SDK smoke demo. No UDP socket is used."
    )
    parser.add_argument("config", help="dual-hand direct API JSON")
    parser.add_argument("--library", default=None, help="optional librevo2_ik_sdk.so path")
    parser.add_argument("--frames", type=int, default=50, help="number of synthetic open-hand frames to solve")
    args = parser.parse_args()

    with Revo2DualHandIkSdk(Path(args.config), library_path=args.library) as sdk:
        result = None
        for i in range(max(1, args.frames)):
            result = sdk.solve(
                left_points=_open_hand_points("left"),
                right_points=_open_hand_points("right"),
                sequence=i,
                timestamp_ms=i * 10,
            )
    assert result is not None
    print(json.dumps({
        "protocol": result["protocol"],
        "joint_unit": result["joint_unit"],
        "joint_order": result["joint_order"],
        "joint_positions": result["joint_positions"],
        "left_solve_time_ms": result["left_solve_time_ms"],
        "right_solve_time_ms": result["right_solve_time_ms"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
