from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import test_bridge_boundaries as boundaries
from test_startup_pose import make_frame
from teleop_direct_retarget import DirectRetargetConfig
from teleop_ik_types import IKCommand
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
from teleop_startup_pose import NaturalDownStartupConfig, NaturalDownStartupGate


class MappingStub:
    def __init__(self):
        self.config = DirectRetargetConfig(joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS))
        self.solver = SimpleNamespace(joint_limits=[(-1.0, 1.0)] * 14)
        self.alignments = []
        self.requests = []

    def align(self, frames, baseline_q):
        self.alignments.append((frames, baseline_q.copy()))

    def command_from_frame(self, frame, current_q, last_q_cmd=None, timestamp=None):
        self.requests.append((current_q.copy(), last_q_cmd))
        return IKCommand(timestamp, "both", list(ARM_JOINT_NAMES),
                         current_q + 0.02, list(ARM_MOTOR_IDS))


class SkeletonBridgeIntegrationTest(unittest.TestCase):
    def setUp(self):
        fixture = boundaries.BridgeBoundariesTest()
        fixture.setUp()
        self.node, self.module, self.clock = fixture.node, fixture.module, fixture.clock
        self.module.CONTROL_MODE = "skeleton"
        from dataclasses import replace
        self.module.HAND_CONTROL_CONFIG = replace(self.module.HAND_CONTROL_CONFIG, publish_enabled=False)
        self.mapper = MappingStub()
        self.node.skeleton_retargeter = self.mapper
        self.node.ik_solver = None
        self.node.log_frame_index = 0
        self.node.startup_pose_locked = False
        self.node.startup_pose_gate = NaturalDownStartupGate(NaturalDownStartupConfig(hold_seconds=0.2))
        self.node.safe_baseline_q = None
        self.hand_calls = []
        self.node.process_hand_ik = self.hand_calls.append

    def feed(self, now, frame=None, *, feedback=True):
        self.clock.now = now
        data = make_frame().to_replay_dict() if frame is None else frame
        data.update(timestamp=self.clock.wall, receive_monotonic=now)
        self.node.latest_mocap_data = data
        self.node.last_arm_state_time = now if feedback else now - 1
        self.node.control_loop()

    def align(self):
        for now in (100.0, 100.1, 100.21):
            self.feed(now)

    def test_one_automatic_window_gates_arm_and_hand_without_prepose(self):
        for now in (100.0, 100.1):
            self.feed(now)
            self.assertFalse(self.node.arm_cmd_pub.messages)
            self.assertFalse(self.hand_calls)
        self.feed(100.21)
        self.assertTrue(self.node.startup_pose_locked)
        self.assertEqual(len(self.mapper.alignments), 1)
        self.assertEqual(len(self.mapper.alignments[0][0]), 3)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 1)
        self.assertEqual(len(self.hand_calls), 1)
        self.assertFalse(self.node.waist_cmd_pub.messages)
        np.testing.assert_allclose(self.mapper.requests[0][0], self.node.current_lr_arm_motor_q)

    def test_untracked_frame_stops_hands_and_requires_a_new_window(self):
        self.align()
        sent = len(self.node.arm_cmd_pub.messages)
        frame = make_frame().to_replay_dict()
        frame["skeleton"]["left"]["wrist"]["tracked"] = False
        self.feed(100.4, frame)
        self.assertFalse(self.node.startup_pose_locked)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), sent)
        self.assertEqual(len(self.hand_calls), 1)
        self.feed(100.5)
        self.feed(100.6)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), sent)
        self.feed(100.71)
        self.assertEqual(len(self.mapper.alignments), 2)

    def test_stale_robot_feedback_revokes_alignment_even_if_fallback_is_disabled(self):
        self.align()
        self.module.REQUIRE_ARM_STATE_BEFORE_IK = False
        sent = len(self.node.arm_cmd_pub.messages)
        self.feed(100.4, feedback=False)
        self.assertFalse(self.node.startup_pose_locked)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), sent)
        self.assertEqual(len(self.hand_calls), 1)

    def test_failed_send_does_not_commit_the_generated_target(self):
        self.node.arm_cmd_pub.error = OSError("fake transport failure")
        self.align()
        self.assertTrue(self.node.command_faulted)
        self.assertIsNone(self.node.last_command_q)

    def test_displaced_feedback_after_tracking_loss_requires_a_new_session(self):
        self.align()
        self.feed(100.4, feedback=False)
        self.node.current_lr_arm_motor_q = np.full(14, 0.3)
        for now in (100.5, 100.6, 100.71):
            self.feed(now)
        self.assertTrue(self.node.command_faulted)
        self.assertFalse(self.node.startup_pose_locked)
        self.assertEqual(len(self.mapper.alignments), 1)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 1)
        np.testing.assert_allclose(self.node.last_command_q, 0.02)

    def test_arm_feedback_expiring_during_solve_also_stops_hands(self):
        original = self.mapper.command_from_frame

        def slow_feedback(*args, **kwargs):
            command = original(*args, **kwargs)
            self.node.last_arm_state_time = self.clock.now - 1
            return command

        self.mapper.command_from_frame = slow_feedback
        self.align()
        self.assertFalse(self.node.startup_pose_locked)
        self.assertFalse(self.node.arm_cmd_pub.messages)
        self.assertFalse(self.hand_calls)

    def test_skeleton_cli_rejects_manual_calibration_and_legacy_solver(self):
        parsed, _ = self.module.parse_runtime_args(["--wrist-calibration", "missing.json"])
        with self.assertRaises(ValueError):
            self.module.apply_runtime_args(parsed)
        parsed, _ = self.module.parse_runtime_args(["--ik-backend", "tiangong2pro_official"])
        with self.assertRaises(ValueError):
            self.module.apply_runtime_args(parsed)


if __name__ == "__main__":
    unittest.main()
