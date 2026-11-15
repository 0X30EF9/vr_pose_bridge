from __future__ import annotations

import unittest

from teleop_hand_ik import HAND_JOINT_ORDER, hand_ik_request, parse_hand_ik_result
from teleop_mocap_protocol import MOCAP_G2_JOINT_COUNT, frame_from_joint_rows


class HandIkAdapterTest(unittest.TestCase):
    def test_canonical_frame_maps_names_and_xyzw_to_wxyz(self) -> None:
        rows = [
            (float(index), float(index + 1), float(index + 2), 0.1, 0.2, 0.3, 0.9)
            for index in range(MOCAP_G2_JOINT_COUNT)
        ]
        frame = frame_from_joint_rows(rows, timestamp_s=12.345, timestamp_us=12_345_000)
        request = hand_ik_request(frame, sequence=7)

        self.assertEqual(request["sequence"], 7)
        self.assertEqual(request["timestamp_ms"], 12_345)
        self.assertEqual(request["left_points"]["hand"], (10.0, 11.0, 12.0))
        self.assertEqual(request["left_points"]["thumb_middle"], (16.0, 17.0, 18.0))
        self.assertEqual(request["left_points"]["little_far"], (33.0, 34.0, 35.0))
        self.assertEqual(request["right_points"]["hand"], (39.0, 40.0, 41.0))
        left_hand_quat = request["left_quaternions_wxyz"]["hand"]
        self.assertAlmostEqual(left_hand_quat[0], 0.9 / (0.95 ** 0.5))
        self.assertAlmostEqual(left_hand_quat[1], 0.1 / (0.95 ** 0.5))

    def test_worker_result_contract_is_exactly_twelve_radians(self) -> None:
        values = [index * 0.1 for index in range(12)]
        payload = {
            "type": "result",
            "sequence": 3,
            "result": {
                "timestamp_ms": 456,
                "joint_order": list(HAND_JOINT_ORDER),
                "joint_positions": values,
                "left_solve_time_ms": 1.2,
                "right_solve_time_ms": 1.3,
                "left": {"debug": {"calibrated": True, "calibration_status": "done"}},
                "right": {"debug": {"calibrated": False, "calibration_status": "collecting"}},
            },
        }
        result = parse_hand_ik_result(payload, expected_sequence=3)
        self.assertEqual(result.joint_positions, tuple(values))
        self.assertTrue(result.left_calibrated)
        self.assertFalse(result.right_calibrated)
        self.assertEqual(result.to_summary()["published"], False)
        self.assertEqual(result.to_summary()["transport"], "local_subprocess_pipe")


if __name__ == "__main__":
    unittest.main()