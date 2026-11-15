from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

import test_replay_safety as fixtures
from teleop_control_safety import command_time_step

replay = fixtures.replay


class ReplayRateSafetyTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReplaySafetyTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = self.fixture.args
        self.args.max_joint_velocity = 0.48
        self.args.max_joint_delta_per_step = 0.06
        self.args.max_command_delta_from_baseline = 2.0

    def test_one_second_motion_is_rate_independent_at_8_20_and_100_hz(self):
        for rate in (8, 20, 100):
            with self.subTest(rate=rate):
                self.args.rate = rate
                gate = self.fixture.gate()
                current = np.zeros(14)
                for index in range(rate):
                    self.fixture.clock.now = 100.0 + index / rate
                    self.fixture.arm_feedback(current)
                    command = self.fixture.publish(self.fixture.command(1.0), gate)
                    self.assertLessEqual(np.max(command.positions - current), 0.48 / rate + 1e-9)
                    current = command.positions
                np.testing.assert_allclose(current, 0.48, atol=1e-9)

    def test_urdf_velocity_and_short_real_elapsed_time_are_both_enforced(self):
        self.args.rate = 100
        gate = self.fixture.gate()
        gate.joint_velocity_limits = np.full(14, 0.1)
        first = self.fixture.publish(self.fixture.command(1.0), gate)
        np.testing.assert_allclose(first.positions, 0.001)
        self.fixture.clock.now += 0.002
        self.fixture.arm_feedback(first.positions)
        second = self.fixture.publish(self.fixture.command(1.0), gate)
        np.testing.assert_allclose(second.positions - first.positions, 0.0002)

    def test_long_gap_does_not_accumulate_a_large_step(self):
        self.args.rate = 100
        gate = self.fixture.gate()
        first = self.fixture.publish(self.fixture.command(1.0), gate)
        self.fixture.clock.now += 10.0
        self.fixture.arm_feedback(first.positions)
        second = self.fixture.publish(self.fixture.command(1.0), gate)
        np.testing.assert_allclose(second.positions - first.positions, 0.0048)

    def test_slow_feedback_cannot_accumulate_unconfirmed_motion(self):
        self.args.rate = 100
        gate = self.fixture.gate()
        measured = np.zeros(14)
        command = None
        for index in range(100):
            self.fixture.clock.now = 100.0 + index / 100
            if index % 5 == 0:
                if command is not None:
                    measured = command.positions.copy()
                self.fixture.arm_feedback(measured)
            command = self.fixture.publish(self.fixture.command(1.0), gate)
            self.assertLessEqual(np.max(command.positions - measured), 0.0048 + 1e-9)
        self.assertLess(float(command.positions[0]), 0.11)

    def test_hand_speed_uses_seconds_even_if_planned_outputs_publish_at_100_hz(self):
        self.args.rate = 100
        self.args.control_side = "left"
        config = replace(self.fixture.hand_config, publish_enabled=True, thumb_rotation_mode="mapped")
        gate = replay.ReplayFeedbackGate(self.args, self.fixture.client, None, None,
                                       {"left": self.fixture.hand}, config)
        topic = fixtures.FakeTopic(self.fixture.client, "hand", "fake")
        last = {"left": None}
        previous = np.full(6, 0.4)
        for index in range(20):
            self.fixture.clock.now = 100.0 + index / 100
            self.fixture.hand_feedback(previous)
            replay.publish_hand_output({"left": (0.9,) * 6}, {"left": topic},
                                       {"left": self.fixture.hand}, last, config, safety=gate)
            sent = np.asarray(topic.messages[-1]["position"])
            if index:
                self.assertLessEqual(np.max(sent - previous), 0.01 + 1e-9)
            previous = sent

    def test_command_dt_rejects_bad_clock_and_never_banks_missed_time(self):
        self.assertEqual(command_time_step(100, 10, None), 0.01)
        self.assertEqual(command_time_step(100, 10, 0), 0.01)
        self.assertAlmostEqual(command_time_step(100, 10, 9.998), 0.002)
        for rate, now, previous in ((0, 10, None), (float("nan"), 10, None), (100, 10, 11)):
            with self.assertRaises(ValueError):
                command_time_step(rate, now, previous)


if __name__ == "__main__":
    unittest.main()
