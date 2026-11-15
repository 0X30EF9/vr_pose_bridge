from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

import test_bridge_boundaries as fixtures


class LiveRateSafetyTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BridgeBoundariesTest(methodName="runTest")
        self.fixture.setUp()
        self.node, self.module, self.clock = self.fixture.node, self.fixture.module, self.fixture.clock
        self.module.CONTROL_RATE_HZ = 100.0
        self.module.SAFE_COMMAND_RATE_HZ = 100.0
        self.module.MAX_JOINT_VELOCITY = 0.48
        self.module.MAX_JOINT_DELTA_PER_STEP = 0.06
        self.module.MAX_COMMAND_DELTA_FROM_BASELINE = 2.0

    def test_actual_live_send_gate_allows_100_bounded_commands_in_one_second(self):
        for index in range(100):
            self.clock.now = 100.0 + index / 100.0
            current = np.zeros(14) if self.node.last_command_q is None else self.node.last_command_q.copy()
            self.node.current_lr_arm_motor_q = current
            self.node.last_arm_state_time = self.clock.now
            safe = self.node.apply_safety(np.ones(14), current)
            self.assertIsNotNone(safe)
            self.node.publish_to_robot(self.fixture.command(safe), mocap_received_at=self.clock.now)
            self.assertLessEqual(np.max(safe - current), 0.0048 + 1e-9)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 100)
        np.testing.assert_allclose(self.node.last_command_q, 0.48, atol=1e-9)

    def test_live_final_gate_rejects_speedup_even_with_legacy_large_step_cap(self):
        self.node.publish_to_robot(self.fixture.command(np.full(14, 0.06)), mocap_received_at=self.clock.now)
        self.assertEqual(self.node.arm_cmd_pub.messages, [])

    def test_urdf_velocity_is_an_additional_cap(self):
        self.node.command_velocity_limits = np.full(14, 0.1)
        np.testing.assert_allclose(self.node.apply_safety(np.ones(14), np.zeros(14)), 0.001)

    def test_old_hand_result_is_not_refreshed_with_a_new_control_frame(self):
        self.node.hand_ik_sidecar = SimpleNamespace(
            submit=lambda *args: True,
            poll=lambda: SimpleNamespace(submission=1, received_at=99.0, result=object()),
        )
        self.node.publish_hand_result = Mock()
        self.node.process_hand_ik({"receive_monotonic": 100.0})
        self.node.publish_hand_result.assert_not_called()
        self.assertIsNone(getattr(self.node, "latest_hand_ik_result", None))

    def test_hand_worker_failure_latches_session_fault_before_any_later_send(self):
        from teleop_hand_ik import HandIkError
        def fail():
            raise HandIkError("fake SDK worker failure")
        self.node.hand_ik_sidecar = SimpleNamespace(submit=lambda *args: True, poll=fail)
        self.node.process_hand_ik({"receive_monotonic": self.clock.now})
        self.assertTrue(self.node.command_faulted)
        self.node.publish_to_robot(self.fixture.command(np.full(14, 0.001)), mocap_received_at=self.clock.now)
        self.assertEqual(self.node.arm_cmd_pub.messages, [])

    def test_already_faulted_sdk_is_checked_before_arm_without_polling_results(self):
        from teleop_hand_ik import HandIkError
        self.module.CONTROL_MODE = "skeleton"
        self.node.startup_pose_locked = True
        self.node.startup_pose_gate = SimpleNamespace(reset=Mock())
        self.node.hand_ik_sidecar = SimpleNamespace(
            check_health=Mock(side_effect=HandIkError("already faulted")), poll=Mock(),
        )
        self.node.process_mocap_data = Mock()
        self.node.process_hand_ik = Mock()
        self.node.control_loop()
        self.assertTrue(self.node.command_faulted)
        self.assertFalse(self.node.startup_pose_locked)
        self.node.process_mocap_data.assert_not_called()
        self.node.process_hand_ik.assert_not_called()
        self.node.hand_ik_sidecar.poll.assert_not_called()
        self.assertEqual(self.node.arm_cmd_pub.messages, [])

    def test_live_runtime_rate_and_log_parameters_validate_and_update_timer_period(self):
        parsed, _ = self.module.parse_runtime_args(["--control-rate", "100", "--max-joint-velocity", "0.48",
                                                  "--hand-control-rate", "20", "--max-hand-velocity", "1",
                                                  "--safe-rate", "100", "--log-rate", "1"])
        self.module.apply_runtime_args(parsed)
        self.assertEqual(self.module.CONTROL_RATE_HZ, 100)
        self.assertEqual(self.module.LOG_EVERY_N_FRAMES, 100)
        self.assertEqual(self.module.TIME_WARNING_THRESHOLD_MS, 8.0)
        for name, value in (("--control-rate", "0"), ("--max-joint-velocity", "nan"),
                            ("--hand-control-rate", "-1"), ("--max-hand-velocity", "0"),
                            ("--log-rate", "nan")):
            with self.subTest(name=name):
                parsed, _ = self.module.parse_runtime_args([name, value])
                with self.assertRaises(ValueError):
                    self.module.apply_runtime_args(parsed)


if __name__ == "__main__":
    unittest.main()
