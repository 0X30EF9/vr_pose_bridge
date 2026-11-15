from __future__ import annotations

import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import test_replay_safety as fixtures
from test_startup_pose import make_frame

replay = fixtures.replay


class FakeRetargeter:
    instances = []
    solve_hook = None

    def __init__(self, config, solver=None, orientation_scale=1.0):
        self.config = config
        self.orientation_scale = orientation_scale
        self.alignments = []
        self.requests = []
        self.instances.append(self)

    def align(self, frames, baseline_q):
        self.alignments.append((tuple(frames), baseline_q.copy()))

    def command_from_frame(self, frame, current_q, last_q_cmd=None, timestamp=None):
        self.requests.append((current_q.copy(), None if last_q_cmd is None else last_q_cmd.copy()))
        if FakeRetargeter.solve_hook is not None:
            FakeRetargeter.solve_hook(frame)
        if frame.get("bad_tracking"):
            raise ValueError("tracking lost")
        return fixtures.IKCommand(
            timestamp=timestamp, control_side=self.config.control_side,
            joint_names=replay.IK_ARM_JOINT_NAMES, motor_ids=replay.ARM_MOTOR_IDS,
            positions=np.full(14, frame.get("target", 0.5)),
            waist_position=frame.get("waist_target"),
        )


class SkeletonReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReplaySafetyTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = self.fixture.args
        self.args.control_mode = "skeleton"
        self.args.startup_hold_seconds = 0.05
        self.args.orientation_scale = 1.0
        self.args.prepose = "natural_down"
        self.clock = self.fixture.clock
        self.client = self.fixture.client
        self.stack = self.fixture.stack
        module = types.ModuleType("teleop_skeleton_retarget")
        module.SkeletonRetargeter = FakeRetargeter
        self.stack.enter_context(fixtures.stub_modules({"teleop_skeleton_retarget": module}))
        self.stack.enter_context(patch.object(replay, "direct_retarget_config_from_args", side_effect=lambda args, **kwargs: args))
        self.prepose = self.stack.enter_context(patch.object(replay, "prepose_to_baseline", side_effect=AssertionError("no skeleton prepose")))
        self.stack.enter_context(patch.object(replay, "load_workspace_calibration", side_effect=AssertionError("no manual calibration")))
        self.stack.enter_context(patch.object(replay, "load_wrist_calibration", side_effect=AssertionError("no manual calibration")))
        FakeRetargeter.instances = []
        FakeRetargeter.solve_hook = None
        self.frames = []
        for index in range(6):
            frame = make_frame().to_replay_dict()
            frame.pop("mocap_frame", None)
            frame["receive_time"] = index * 0.05
            self.frames.append(frame)

    def follow_published_arm(self, name, message):
        if name == replay.ARM_CMD_TOPIC:
            positions = self.fixture.arm.latest_complete()[1]
            for row in message["cmds"]:
                positions[replay.ARM_MOTOR_IDS.index(row["name"])] = row["pos"]
            self.fixture.arm_feedback(positions)

    def execute(self):
        return replay.execute_skeleton_replay(
            self.args, self.client, self.fixture.arm, self.frames, np.zeros(14),
            {}, {}, self.fixture.hand_config,
        )

    def test_parser_defaults_to_skeleton_and_preserves_advanced_orientation_default(self):
        with patch.object(sys, "argv", ["replay", "unused.jsonl"]):
            parsed = replay.parse_args()
        self.assertEqual(parsed.control_mode, "skeleton")
        self.assertEqual(parsed.orientation_scale, 1.0)
        with patch.object(sys, "argv", ["replay", "unused.jsonl", "--control-mode", "ik"]):
            self.assertEqual(replay.parse_args().orientation_scale, 0.25)

    def test_preview_aligns_once_from_the_startup_window_without_manual_files(self):
        commands, summary = replay.plan_replay_commands(self.frames, self.args, np.zeros(14))
        retargeter = FakeRetargeter.instances[0]
        self.assertEqual(len(retargeter.alignments), 1)
        self.assertEqual(len(retargeter.alignments[0][0]), 3)
        self.assertEqual(summary["startup_frame_count"], 3)
        self.assertEqual(summary["command_count"], 4)
        self.assertEqual(summary["feedback_source"], "simulated_preview")
        self.assertTrue(all(command.waist_position is None for command in commands))
        self.prepose.assert_not_called()

    def test_missing_startup_tracking_never_generates_or_publishes_commands(self):
        for frame in self.frames:
            frame["skeleton"]["left"]["hand"]["tracked"] = False
        with self.assertRaisesRegex(ValueError, "稳定"):
            replay.plan_replay_commands(self.frames, self.args, np.zeros(14))
        self.assertEqual(self.execute(), 14)
        self.assertEqual(fixtures.FakeTopic.topics, [])
        self.assertTrue(all(not instance.alignments for instance in FakeRetargeter.instances))

    def test_explicit_manual_files_are_rejected_before_solver_or_motion(self):
        for attribute in ("workspace_calibration", "wrist_calibration"):
            with self.subTest(attribute=attribute):
                setattr(self.args, attribute, Path("unused-calibration.json"))
                with self.assertRaisesRegex(ValueError, "人工标定"):
                    replay.plan_replay_commands(self.frames, self.args, np.zeros(14))
                with self.assertRaisesRegex(ValueError, "人工标定"):
                    self.execute()
                setattr(self.args, attribute, None)
        self.assertEqual(FakeRetargeter.instances, [])
        self.assertEqual(fixtures.FakeTopic.topics, [])

    def test_unsupported_solver_options_are_rejected(self):
        for attribute, value in (("ik_backend", "tiangong2pro_official"), ("ik_task", "position_only")):
            with self.subTest(attribute=attribute):
                previous = getattr(self.args, attribute)
                setattr(self.args, attribute, value)
                with self.assertRaisesRegex(ValueError, "walker"):
                    replay.plan_replay_commands(self.frames, self.args, np.zeros(14))
                setattr(self.args, attribute, previous)
        self.assertEqual(FakeRetargeter.instances, [])

    def test_hand_sidecar_warms_up_in_same_window_without_publishing_early(self):
        self.args.control_side = "left"
        self.args.hand_ik_compute = True
        config = replace(self.fixture.hand_config, publish_enabled=True, thumb_rotation_mode="mapped")
        # Measured hand feedback keeps arriving even when the asynchronous SDK
        # has not produced a command yet.
        self.clock.on_sleep = lambda: self.fixture.hand_feedback(self.fixture.hand.latest_valid()[1])
        sidecar = types.SimpleNamespace(
            solve_replay_frame=lambda frame: types.SimpleNamespace(
                left_calibrated=True, right_calibrated=False, joint_positions=np.zeros(12),
            ), close=lambda: None,
        )
        with patch.object(replay, "Revo2HandIkSidecar", return_value=sidecar), patch.object(
            replay, "hand_ik_config_from_args", return_value=None
        ), patch.object(replay, "map_hand_radians", return_value=(0.44,) * 6):
            result = replay.execute_skeleton_replay(
                self.args, self.client, self.fixture.arm, self.frames[:2], np.zeros(14),
                {"left": self.fixture.hand}, {"left": (0.4,) * 6}, config,
            )
        self.assertEqual(result, 14)
        self.assertEqual(fixtures.FakeTopic.topics, [])

        first_publish_at = []
        def follow(name, message):
            first_publish_at.append(self.clock.now)
            self.follow_published_arm(name, message)
            if "position" in message:
                self.fixture.hand_feedback(message["position"])
        fixtures.FakeTopic.on_publish = follow
        startup_at = self.clock.now
        with patch.object(replay, "Revo2HandIkSidecar", return_value=sidecar), patch.object(
            replay, "hand_ik_config_from_args", return_value=None
        ), patch.object(replay, "map_hand_radians", return_value=(0.44,) * 6):
            result = replay.execute_skeleton_replay(
                self.args, self.client, self.fixture.arm, self.frames, np.zeros(14),
                {"left": self.fixture.hand}, {"left": (0.4,) * 6}, config,
            )
        self.assertEqual(result, 0)
        self.assertGreaterEqual(min(first_publish_at), startup_at + 0.09)
        self.assertEqual(len(fixtures.FakeTopic.topics), 2)
        np.testing.assert_allclose(self.fixture.hand.latest_valid()[1], 0.4)

    def test_execution_uses_actual_feedback_and_actual_clamped_published_state(self):
        self.fixture.arm_feedback(np.full(14, 0.1))
        fixtures.FakeTopic.on_publish = self.follow_published_arm
        self.assertEqual(self.execute(), 0)
        retargeter = FakeRetargeter.instances[0]
        self.assertEqual(len(retargeter.alignments), 1)
        self.assertEqual(len(retargeter.alignments[0][0]), 3)
        np.testing.assert_allclose(retargeter.alignments[0][1], 0.1)
        np.testing.assert_allclose(retargeter.requests[0][0], 0.1)
        np.testing.assert_allclose(retargeter.requests[1][0], 0.14)
        np.testing.assert_allclose(retargeter.requests[1][1], 0.14)
        self.assertEqual(len(fixtures.FakeTopic.topics), 1)
        self.assertEqual(fixtures.FakeTopic.topics[0].name, replay.ARM_CMD_TOPIC)
        np.testing.assert_allclose(self.fixture.arm.latest_complete()[1], 0.1, atol=1e-3)
        self.prepose.assert_not_called()

    def test_moving_robot_startup_does_not_align(self):
        self.args.startup_hold_seconds = 0.2
        self.clock.on_sleep = lambda: self.fixture.arm_feedback(np.full(14, self.clock.now - 100.0))
        self.assertEqual(self.execute(), 14)
        self.assertEqual(FakeRetargeter.instances[0].alignments, [])
        self.assertEqual(fixtures.FakeTopic.topics, [])

    def test_20hz_recording_keeps_its_duration_and_real_two_second_startup_at_100hz(self):
        self.args.rate = 100
        self.args.startup_hold_seconds = 2.0
        self.frames = []
        for index in range(60):
            frame = make_frame().to_replay_dict()
            frame.pop("mocap_frame", None)
            frame.update(receive_time=index * .05, packet_timestamp_us=index * 50000, target=0.0)
            self.frames.append(frame)
        original_times = [frame["receive_time"] for frame in self.frames]
        self.clock.on_sleep = lambda: self.fixture.arm_feedback(np.zeros(14))
        began = self.clock.now
        self.assertEqual(self.execute(), 0)
        self.assertAlmostEqual(self.clock.now - began, 2.96, places=7)
        self.assertEqual(len(FakeRetargeter.instances[0].alignments), 1)
        self.assertEqual(len(FakeRetargeter.instances[0].alignments[0][0]), 41)
        self.assertEqual([frame["receive_time"] for frame in self.frames], original_times)

    def test_replay_hold_does_not_refresh_source_receipt_or_choose_future_samples(self):
        frames = [{"receive_time": value} for value in (0., .05, .1)]
        records = []
        for index, frame, received_at in replay.replay_control_ticks(frames, 100):
            records.append((self.clock.now, index, received_at))
        self.assertEqual(len(records), 11)
        self.assertEqual(len({record[2] for record in records}), 3)
        for now, index, received_at in records:
            self.assertLessEqual(100.0 + frames[index]["receive_time"], now + 1e-9)
            self.assertAlmostEqual(received_at, 100.0 + frames[index]["receive_time"])

    def test_large_recording_gap_expires_instead_of_refreshing_a_held_source(self):
        self.args.rate = 100
        self.args.startup_hold_seconds = 2.0
        self.frames = [self.frames[0], self.frames[-1]]
        self.frames[0]["receive_time"], self.frames[-1]["receive_time"] = 0., 1.
        self.clock.on_sleep = lambda: self.fixture.arm_feedback(np.zeros(14))
        self.assertEqual(self.execute(), 14)
        self.assertLess(self.clock.now, 100.32)
        self.assertEqual(fixtures.FakeTopic.topics, [])

    def test_tracking_loss_after_alignment_aborts_without_baseline_commands(self):
        self.frames[3]["bad_tracking"] = True
        self.assertEqual(self.execute(), 14)
        messages = fixtures.FakeTopic.topics[0].messages
        self.assertEqual(len(messages), 1)
        np.testing.assert_allclose([row["pos"] for row in messages[0]["cmds"]], 0.04)

    def test_slow_result_is_not_published_even_when_feedback_was_refreshed(self):
        def slow(frame):
            self.clock.now += 1.0
            self.fixture.arm_feedback(np.zeros(14))
        FakeRetargeter.solve_hook = slow
        self.assertEqual(self.execute(), 14)
        self.assertEqual(fixtures.FakeTopic.topics[0].messages, [])

    def test_waist_output_is_rejected_before_arm_publication(self):
        self.frames[2]["waist_target"] = 0.1
        self.assertEqual(self.execute(), 14)
        self.assertEqual(fixtures.FakeTopic.topics[0].messages, [])

    def test_main_routes_normal_and_streaming_skeleton_execute_to_live_solver(self):
        self.args.hand_control_json = "{}"
        self.args.input = Path("unused.jsonl")
        self.args.max_frames = None
        self.args.stride = 1
        self.args.max_position_step = 0.15
        self.args.max_angular_step = 0.8
        self.args.use_robot_status = False
        self.args.execute = True
        self.args.host = "localhost"
        self.args.port = 0
        self.args.connect_timeout = 1.0
        self.args.status_timeout = 1.0
        self.args.yes = True

        class StatusTopic(fixtures.FakeTopic):
            def subscribe(topic_self, callback):
                callback({"status": [{"name": motor_id, "pos": 0.0} for motor_id in replay.ARM_MOTOR_IDS]})
            def unsubscribe(topic_self):
                pass

        for streaming in (False, True):
            with self.subTest(streaming=streaming), patch.object(replay, "parse_args", return_value=self.args), patch.object(
                replay, "resolve_input_path", return_value=self.args.input
            ), patch.object(replay, "load_frames", return_value=self.frames), patch.object(
                replay, "connect_rosbridge", return_value=self.client
            ), patch.object(replay.roslibpy, "Topic", StatusTopic), patch.object(
                replay, "execute_skeleton_replay", return_value=42
            ) as execute, patch.object(replay, "plan_replay_commands", side_effect=AssertionError("no second alignment")):
                self.args.streaming = streaming
                self.assertEqual(replay.main(), 42)
                execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
