from __future__ import annotations

import math
import unittest

from teleop_hand_control import (
    HandControlError,
    apply_hand_command_safety,
    build_hand_joint_state_message,
    default_hand_control_mapping,
    hand_control_config_from_mapping,
    map_hand_radians,
    parse_hand_state_message,
)


class HandControlMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = hand_control_config_from_mapping(default_hand_control_mapping())

    def test_sdk_order_is_reordered_to_robot_ids(self) -> None:
        radians = (
            1.03, 1.57, 0.0, 1.41, 0.705, 1.41,
            0.0, 0.0, 1.41, 0.0, 1.41, 0.0,
        )
        self.assertEqual(map_hand_radians(radians, "left", self.config), (0.0, 0.5, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(map_hand_radians(radians, "right", self.config), (1.0, 0.0, 1.0, 0.0, 1.0, 1.0))

    def test_reversed_calibration_and_clamping(self) -> None:
        raw = default_hand_control_mapping()
        raw["calibration"]["left"]["fore"] = {
            "open_rad": 1.0,
            "closed_rad": -1.0,
            "enabled": True,
        }
        config = hand_control_config_from_mapping(raw)
        radians = [0.0] * 12
        radians[2] = 2.0
        mapped = map_hand_radians(radians, "left", config)
        self.assertEqual(mapped[3], 1.0)

    def test_safety_limits_step_and_holds_thumb_rotation(self) -> None:
        feedback = (1.0,) * 6
        target = (0.0,) * 6
        safe = apply_hand_command_safety(target, feedback, None, "left", self.config)
        self.assertEqual(safe[:5], (0.95,) * 5)
        self.assertEqual(safe[5], 1.0)

    def test_disabled_channel_holds_feedback(self) -> None:
        raw = default_hand_control_mapping()
        raw["calibration"]["right"]["middle"]["enabled"] = False
        config = hand_control_config_from_mapping(raw)
        safe = apply_hand_command_safety(
            target=(0.0,) * 6,
            feedback=(0.8,) * 6,
            last_command=(0.7,) * 6,
            side="right",
            config=config,
        )
        self.assertAlmostEqual(safe[2], 0.75)

    def test_non_finite_sdk_value_is_rejected(self) -> None:
        with self.assertRaises(HandControlError):
            map_hand_radians([0.0] * 11 + [math.nan], "right", self.config)


class HandJointStateContractTest(unittest.TestCase):
    def test_state_is_parsed_by_name_not_array_order(self) -> None:
        state = {
            "name": ["6", "2", "4", "1", "5", "3"],
            "position": [0.6, 0.2, 0.4, 0.1, 0.5, 0.3],
        }
        self.assertEqual(parse_hand_state_message(state), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    def test_missing_or_out_of_range_state_is_rejected(self) -> None:
        with self.assertRaises(HandControlError):
            parse_hand_state_message({"name": ["1"], "position": [0.5]})
        with self.assertRaises(HandControlError):
            parse_hand_state_message({"name": ["1", "2", "3", "4", "5", "6"], "position": [0, 0, 0, 0, 0, 1.1]})

    def test_command_message_uses_ids_one_through_six(self) -> None:
        message = build_hand_joint_state_message((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self.assertEqual(message["name"], ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(message["position"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(message["velocity"], [])
        self.assertEqual(message["effort"], [])


if __name__ == "__main__":
    unittest.main()