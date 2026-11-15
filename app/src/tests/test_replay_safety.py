from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


@contextmanager
def stub_modules(modules):
    """Restore only stubbed keys so lazy native imports remain loaded."""
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
retarget_stub = types.ModuleType("teleop_direct_retarget")
retarget_stub.DirectRetargeter = object
retarget_stub.add_direct_retarget_arguments = lambda parser: None
retarget_stub.direct_retarget_config_from_args = lambda args: None
retarget_stub.skeleton_observation = lambda *args, **kwargs: None
spec = importlib.util.spec_from_file_location(
    "replay_safety_under_test", Path(__file__).resolve().parents[1] / "mocap_replay_to_robot.py"
)
replay = importlib.util.module_from_spec(spec)
# Keep native IK libraries and all network transports outside these control tests.
with stub_modules({
    "teleop_direct_retarget": retarget_stub,
    "roslibpy": types.ModuleType("roslibpy"),
}):
    spec.loader.exec_module(replay)
from teleop_hand_control import hand_control_config_from_mapping
from teleop_ik_types import IKCommand
from teleop_robot_constants import HAND_MOTOR_NAMES


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def time(self):
        return 1700000000.0 + self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class FakeTopic:
    topics = []
    on_publish = None

    def __init__(self, client, name, message_type):
        self.name = name
        self.messages = []
        self.advertised = False
        self.topics.append(self)

    def advertise(self):
        self.advertised = True

    def unadvertise(self):
        self.advertised = False

    def publish(self, message):
        self.messages.append(copy.deepcopy(message))
        if FakeTopic.on_publish is not None:
            FakeTopic.on_publish(self.name, message)


class ReplaySafetyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.clock = FakeClock()
        self.stack.enter_context(patch.object(replay, "time", self.clock))
        self.stack.enter_context(patch.object(replay.roslibpy, "Topic", FakeTopic, create=True))
        self.stack.enter_context(patch.object(replay, "arm_joint_limits_from_urdf", return_value=[(-2.0, 2.0)] * 14))
        self.args = types.SimpleNamespace(
            control_side="both", control_mode="direct", ik_backend="teleop_dls",
            rate=20.0, max_joint_delta_per_step=0.04,
            max_joint_velocity=100.0, max_hand_velocity=1.0, hand_control_rate=20.0,
            max_command_delta_from_baseline=0.5, restore_seconds=1.0,
            prepose="none", prepose_seconds=1.0, prepose_steps=30,
            spd=0.7, cur=2.0, allow_risk=False, ik_task="pose",
            position_scale=0.45, orientation_scale=0.25,
            workspace_calibration=None, wrist_calibration=None,
            hand_ik_compute=False, hand_only=False,
        )
        self.client = types.SimpleNamespace(is_connected=True)
        self.arm = replay.ArmStatusBuffer()
        self.waist = replay.WaistStatusBuffer()
        self.hand = replay.HandStatusBuffer("left")
        self.hand_config = hand_control_config_from_mapping(None)
        self.arm_feedback(np.zeros(14))
        self.waist_feedback(0.0)
        self.hand_feedback((0.4,) * 6)
        FakeTopic.topics = []
        FakeTopic.on_publish = None

    def arm_feedback(self, positions):
        self.arm.callback({"status": [
            {"name": motor_id, "pos": float(value), "error": 0}
            for motor_id, value in zip(replay.ARM_MOTOR_IDS, positions)
        ]})

    def waist_feedback(self, position):
        self.waist.callback({"status": [{"name": replay.WAIST_MOTOR_ID, "pos": position, "error": 0}]})

    def hand_feedback(self, positions):
        self.hand.callback({"name": list(HAND_MOTOR_NAMES), "position": list(positions)})

    def command(self, value, waist=None):
        return IKCommand(
            timestamp=0.0, control_side=self.args.control_side,
            joint_names=list(replay.IK_ARM_JOINT_NAMES),
            positions=np.full(14, value), motor_ids=list(replay.ARM_MOTOR_IDS),
            waist_position=waist,
        )

    def gate(self, *, hands=False, waist=False, baseline=None):
        if waist:
            self.args.control_mode = "ik"
            self.args.ik_backend = "tiangong2pro_official"
        config = replace(self.hand_config, publish_enabled=hands)
        hand_buffers = {"left": self.hand} if hands else {}
        if hands:
            self.args.control_side = "left"
        return replay.ReplayFeedbackGate(
            self.args, self.client, self.arm, self.waist if waist else None,
            hand_buffers, config, np.zeros(14) if baseline is None else baseline,
        )

    def publish(self, command, gate, arm_topic=None, waist_topic=None):
        arm_topic = arm_topic or FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
        return replay.publish_command(
            arm_topic, command,
            replay.make_motor_templates(self.args.control_side, self.args.spd, self.args.cur),
            waist_topic, safety=gate,
        )

    def test_bad_named_arm_sample_invalidates_previous_complete_feedback(self):
        valid = [{"name": motor_id, "pos": 0.0} for motor_id in replay.ARM_MOTOR_IDS]
        mutations = (
            lambda rows: rows[0].update(name=999),
            lambda rows: rows[0].update(name=rows[1]["name"]),
            lambda rows: rows[0].update(pos=float("nan")),
            lambda rows: rows[0].update(error=1),
            lambda rows: rows.pop(),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.arm.callback({"status": valid})
                self.assertIsNotNone(self.arm.latest_complete())
                bad = copy.deepcopy(valid)
                mutate(bad)
                self.arm.callback({"status": bad})
                self.assertIsNone(self.arm.latest_complete())

    def test_arm_named_order_and_legacy_unnamed_order_are_supported(self):
        rows = [{"name": motor_id, "pos": index * 0.01} for index, motor_id in enumerate(replay.ARM_MOTOR_IDS)]
        self.arm.callback({"status": list(reversed(rows))})
        np.testing.assert_allclose(self.arm.latest_complete()[1], np.arange(14) * 0.01)
        self.arm.callback({"status": [{"pos": row["pos"]} for row in rows]})
        np.testing.assert_allclose(self.arm.latest_complete()[1], np.arange(14) * 0.01)

    def test_bad_waist_or_hand_sample_clears_previous_feedback(self):
        for message in (
            {"status": [{"name": 999, "pos": 0.0}]},
            {"status": [{"name": 31, "pos": 0.0}, {"name": 31, "pos": 0.1}]},
            {"status": [{"name": 31, "pos": float("nan")}]},
            {"status": [{"name": 31, "pos": 0.0, "error": 0.5}]},
        ):
            self.waist_feedback(0.0)
            self.waist.callback(message)
            self.assertIsNone(self.waist.latest_valid())
        for message in (None, {"name": list(HAND_MOTOR_NAMES), "position": [float("nan")] * 6}):
            self.hand_feedback((0.4,) * 6)
            self.hand.callback(message)
            self.assertIsNone(self.hand.latest_valid())

    def test_publication_uses_measured_positions_and_limits_each_step(self):
        gate = self.gate()
        first = self.publish(self.command(1.0), gate)
        np.testing.assert_allclose(first.positions, 0.04)
        second = self.publish(self.command(1.0), gate)
        np.testing.assert_allclose(second.positions, 0.04)
        self.arm_feedback(np.full(14, 0.04))
        self.clock.now += 1.0 / self.args.rate
        third = self.publish(self.command(1.0), gate)
        np.testing.assert_allclose(third.positions, 0.08)

    def test_invalid_command_ids_names_or_nan_never_publish(self):
        for changes in (
            {"motor_ids": [999] + replay.ARM_MOTOR_IDS[1:]},
            {"joint_names": ["wrong"] + replay.IK_ARM_JOINT_NAMES[1:]},
            {"positions": np.full(14, float("nan"))},
        ):
            topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
            with self.assertRaises((RuntimeError, ValueError)):
                self.publish(replace(self.command(0.02), **changes), self.gate(), topic)
            self.assertEqual(topic.messages, [])

    def test_stale_future_or_disconnected_feedback_blocks_publication(self):
        gate = self.gate()
        for received_at in (self.clock.now - 1.0, self.clock.now + 1.0):
            self.arm.sample = (received_at, np.zeros(14))
            topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
            with self.assertRaises(RuntimeError):
                self.publish(self.command(0.02), gate, topic)
            self.assertEqual(topic.messages, [])
        self.arm_feedback(np.zeros(14))
        self.client.is_connected = False
        with self.assertRaises(RuntimeError):
            self.publish(self.command(0.02), gate)

    def test_required_hand_or_waist_freshness_blocks_arm_command(self):
        for gate in (self.gate(hands=True), self.gate(waist=True)):
            self.clock.now += 1.0
            self.arm_feedback(np.zeros(14))
            topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
            with self.assertRaises(RuntimeError):
                self.publish(self.command(0.02, 0.02 if gate.waist_limit else None), gate, topic)
            self.assertEqual(topic.messages, [])

    def test_feedback_is_rechecked_between_arm_and_waist_topics(self):
        gate = self.gate(waist=True)
        waist_topic = FakeTopic(self.client, replay.WAIST_CMD_TOPIC, "fake")
        FakeTopic.on_publish = lambda name, message: setattr(self.clock, "now", self.clock.now + 1.0)
        with self.assertRaises(RuntimeError):
            self.publish(self.command(0.02, 0.02), gate, waist_topic=waist_topic)
        self.assertEqual(waist_topic.messages, [])

    def test_recovery_moves_in_bounded_steps_and_confirms_feedback(self):
        self.arm_feedback(np.full(14, 0.3))
        gate = self.gate()
        topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
        observed = np.full(14, 0.3)

        def follow(name, message):
            nonlocal observed
            target = np.asarray([row["pos"] for row in message["cmds"]])
            self.assertLessEqual(np.max(np.abs(target - observed)), 0.04 + 1e-9)
            observed = target
            self.arm_feedback(target)

        FakeTopic.on_publish = follow
        replay.restore_baselines(self.args, gate, {}, {}, {}, topic,
                                 replay.make_motor_templates("both", 0.7, 2.0), np.zeros(14))
        np.testing.assert_allclose(observed, 0.0, atol=1e-3)
        self.assertGreater(len(topic.messages), 1)

    def test_recovery_cannot_integrate_commands_when_feedback_does_not_move(self):
        self.arm_feedback(np.full(14, 0.3))
        gate = self.gate()
        topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
        self.clock.on_sleep = lambda: self.arm_feedback(np.full(14, 0.3))
        with self.assertRaises(RuntimeError):
            replay.restore_baselines(self.args, gate, {}, {}, {}, topic,
                                     replay.make_motor_templates("both", 0.7, 2.0), np.zeros(14))
        self.assertLessEqual(len(topic.messages), 21)
        for message in topic.messages:
            np.testing.assert_allclose([row["pos"] for row in message["cmds"]], 0.26)

    def test_recovery_stops_when_feedback_expires(self):
        self.arm_feedback(np.full(14, 0.3))
        topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
        with self.assertRaises(RuntimeError):
            replay.restore_baselines(self.args, self.gate(), {}, {}, {}, topic,
                                     replay.make_motor_templates("both", 0.7, 2.0), np.zeros(14))
        self.assertLessEqual(len(topic.messages), 11)
        self.assertLess(self.clock.now, 101.0)

    def test_regular_replay_aborts_without_restoring_after_status_loss(self):
        self.arm.sample = (self.clock.now - 1.0, np.zeros(14))
        result = replay.execute_replay(self.args, self.client, self.arm, [self.command(0.02)],
                                      np.zeros(14), 0.0, [], {}, {}, self.hand_config)
        self.assertEqual(result, 14)
        self.assertTrue(all(not topic.messages for topic in FakeTopic.topics))

    def test_regular_replay_cancels_outstanding_arm_and_waist_targets_at_baseline(self):
        self.args.control_mode = "ik"
        self.args.ik_backend = "tiangong2pro_official"
        result = replay.execute_replay(
            self.args, self.client, self.arm, [self.command(0.04, 0.04)],
            np.zeros(14), 0.0, [], {}, {}, self.hand_config,
            waist_status_buffer=self.waist,
        )
        self.assertEqual(result, 0)
        arm_topic = next(topic for topic in FakeTopic.topics if topic.name == replay.ARM_CMD_TOPIC)
        waist_topic = next(topic for topic in FakeTopic.topics if topic.name == replay.WAIST_CMD_TOPIC)
        self.assertEqual(len(arm_topic.messages), 2)
        self.assertEqual(len(waist_topic.messages), 2)
        np.testing.assert_allclose([row["pos"] for row in arm_topic.messages[0]["cmds"]], 0.04)
        np.testing.assert_allclose([row["pos"] for row in arm_topic.messages[-1]["cmds"]], 0.0)
        self.assertAlmostEqual(waist_topic.messages[0]["cmds"][0]["pos"], 0.04)
        self.assertAlmostEqual(waist_topic.messages[-1]["cmds"][0]["pos"], 0.0)

    def test_invalid_hand_output_is_rejected_before_arm_publication(self):
        self.args.control_side = "left"
        config = replace(self.hand_config, publish_enabled=True)
        result = replay.execute_replay(
            self.args, self.client, self.arm, [self.command(0.02)], np.zeros(14), 0.0,
            [{"left": (float("nan"),) * 6}], {"left": self.hand}, {"left": (0.4,) * 6}, config,
        )
        self.assertEqual(result, 14)
        self.assertTrue(all(not topic.messages for topic in FakeTopic.topics))

    def test_hand_only_replay_stops_without_restoring_after_status_loss(self):
        self.args.control_side = "left"
        config = replace(self.hand_config, publish_enabled=True)
        FakeTopic.on_publish = lambda name, message: setattr(self.clock, "now", self.clock.now + 1.0)
        result = replay.execute_hand_only_replay(
            self.args, self.client, [{}, {}], [{"left": (0.5,) * 6}] * 2,
            {"left": self.hand}, {"left": (0.4,) * 6}, config,
        )
        self.assertEqual(result, 14)
        self.assertEqual(sum(len(topic.messages) for topic in FakeTopic.topics), 1)
        self.assertTrue(all(topic.name not in (replay.ARM_CMD_TOPIC, replay.WAIST_CMD_TOPIC)
                            for topic in FakeTopic.topics))

    def test_hand_only_replay_cancels_outstanding_target_at_baseline(self):
        self.args.control_side = "left"
        config = replace(self.hand_config, publish_enabled=True, thumb_rotation_mode="mapped")
        result = replay.execute_hand_only_replay(
            self.args, self.client, [{}], [{"left": (0.44,) * 6}],
            {"left": self.hand}, {"left": (0.4,) * 6}, config,
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(FakeTopic.topics), 1)
        messages = FakeTopic.topics[0].messages
        self.assertEqual(len(messages), 2)
        np.testing.assert_allclose(messages[0]["position"], 0.44)
        np.testing.assert_allclose(messages[-1]["position"], 0.4)

    def test_hand_recovery_limits_every_step_against_measured_feedback(self):
        self.args.control_side = "left"
        config = replace(self.hand_config, publish_enabled=True, thumb_rotation_mode="mapped")
        self.hand_feedback((0.7,) * 6)
        gate = replay.ReplayFeedbackGate(self.args, self.client, None, None, {"left": self.hand}, config)
        topic = FakeTopic(self.client, "hand", "fake")
        previous = np.full(6, 0.7)

        def follow(name, message):
            nonlocal previous
            target = np.asarray(message["position"])
            self.assertLessEqual(np.max(np.abs(target - previous)), config.max_step + 1e-9)
            previous = target
            self.hand_feedback(target)

        FakeTopic.on_publish = follow
        replay.restore_baselines(self.args, gate, {"left": topic}, {"left": (0.4,) * 6}, {"left": None})
        np.testing.assert_allclose(previous, 0.4, atol=1e-3)
        self.assertGreater(len(topic.messages), 1)

    def test_publication_exception_does_not_trigger_a_baseline_jump(self):
        def fail(name, message):
            raise RuntimeError("fake transport failure")

        FakeTopic.on_publish = fail
        result = replay.execute_replay(self.args, self.client, self.arm, [self.command(0.02)],
                                      np.zeros(14), 0.0, [], {}, {}, self.hand_config)
        self.assertEqual(result, 14)
        self.assertEqual(sum(len(topic.messages) for topic in FakeTopic.topics), 1)

    def test_mechanical_limits_and_missing_waist_publisher_are_enforced(self):
        self.arm_feedback(np.full(14, 1.99))
        gate = self.gate(baseline=np.full(14, 1.99))
        safe = self.publish(self.command(3.0), gate)
        np.testing.assert_allclose(safe.positions, 2.0)
        self.arm_feedback(np.zeros(14))
        gate = self.gate(waist=True)
        topic = FakeTopic(self.client, replay.ARM_CMD_TOPIC, "fake")
        with self.assertRaises(RuntimeError):
            self.publish(self.command(0.02, 0.02), gate, topic)
        self.assertEqual(topic.messages, [])

    def prepare_streaming_ik(self, solve):
        self.args.control_mode = "ik"
        self.args.ik_backend = "teleop_dls"
        module = types.ModuleType("teleop_ik_core")
        module.apply_joint_safety = lambda q_next, **kwargs: (np.asarray(q_next), False)
        self.stack.enter_context(stub_modules({"teleop_ik_core": module}))
        self.stack.enter_context(patch.object(replay, "load_workspace_calibration", return_value=None))
        self.stack.enter_context(patch.object(replay, "load_wrist_calibration", return_value=None))
        self.stack.enter_context(patch.object(replay, "replay_anatomical_wrist_pose", return_value=np.eye(4)))
        self.stack.enter_context(patch.object(replay, "replay_mocap_reference_position", return_value=None))
        self.stack.enter_context(patch.object(replay, "replay_wrist_behavior_reasons", return_value={}))
        mapper = types.SimpleNamespace(
            set_runtime_mocap_home=lambda *args, **kwargs: None,
            map_targets=lambda *args, **kwargs: types.SimpleNamespace(left_target=np.eye(4), right_target=np.eye(4)),
        )
        self.stack.enter_context(patch.object(replay, "RelativeMocapTargetMapper", return_value=mapper))

        solver = types.SimpleNamespace(
            wrist_effector_poses=lambda q: (np.eye(4), np.eye(4)),
            wrist_to_tcp_transforms=lambda: (np.eye(4), np.eye(4)),
            solve=solve, joint_limits=[(-2.0, 2.0)] * 14,
        )
        self.stack.enter_context(patch.object(replay, "make_ik_solver", return_value=solver))

    def test_streaming_replay_cancels_outstanding_arm_target_at_baseline(self):
        self.prepare_streaming_ik(lambda request: types.SimpleNamespace(
            command=self.command(0.04), reason="ok", limited=False, error_norm=0.0,
        ))
        result = replay.execute_ik_replay_streaming(
            self.args, self.client, self.arm, None, [{"receive_time": 0.0}],
            np.zeros(14), 0.0, {}, {}, self.hand_config,
        )
        self.assertEqual(result, 0)
        topic = next(topic for topic in FakeTopic.topics if topic.name == replay.ARM_CMD_TOPIC)
        self.assertEqual(len(topic.messages), 2)
        np.testing.assert_allclose([row["pos"] for row in topic.messages[0]["cmds"]], 0.04)
        np.testing.assert_allclose([row["pos"] for row in topic.messages[-1]["cmds"]], 0.0)

    def test_slow_ik_result_is_rejected_even_with_fresh_new_feedback(self):
        def slow_solve(request):
            np.testing.assert_allclose(request.current_q, np.full(14, 0.1))
            self.clock.now += 1.0
            self.arm_feedback(np.full(14, 0.15))
            return types.SimpleNamespace(command=self.command(0.15), reason="ok", limited=False, error_norm=0.0)

        self.prepare_streaming_ik(slow_solve)
        self.arm_feedback(np.full(14, 0.1))
        result = replay.execute_ik_replay_streaming(
            self.args, self.client, self.arm, None, [{"receive_time": 0.0}],
            np.zeros(14), 0.0, {}, {}, self.hand_config,
        )
        self.assertEqual(result, 14)
        self.assertTrue(all(not topic.messages for topic in FakeTopic.topics))

    def test_bad_streaming_calibration_prevents_prepose_and_advertising(self):
        self.args.control_mode = "ik"
        self.args.prepose = "natural_down"
        for bad_kind in ("workspace", "wrist"):
            with self.subTest(kind=bad_kind), ExitStack() as stack:
                self.args.workspace_calibration = Path("invalid-workspace.json")
                self.args.wrist_calibration = Path("invalid-wrist.json")
                loaders = {}
                for kind in ("workspace", "wrist"):
                    loaders[kind] = stack.enter_context(patch.object(
                        replay, f"load_{kind}_calibration", return_value=None,
                        side_effect=ValueError("invalid calibration") if kind == bad_kind else None,
                    ))
                prepose = stack.enter_context(patch.object(replay, "prepose_to_baseline"))
                with self.assertRaisesRegex(ValueError, "invalid calibration"):
                    replay.execute_ik_replay_streaming(
                        self.args, self.client, self.arm, None, [{"receive_time": 0.0}],
                        np.zeros(14), 0.0, {}, {}, self.hand_config,
                    )
                loaders[bad_kind].assert_called_once_with(
                    getattr(self.args, f"{bad_kind}_calibration"), self.args.control_side, optional=False,
                )
                prepose.assert_not_called()
                self.assertEqual(FakeTopic.topics, [])

    def test_main_preflights_teleop_dls_calibration_before_connecting(self):
        self.args.control_mode = "ik"
        self.args.hand_control_json = "{}"
        self.args.workspace_calibration = Path("invalid-workspace.json")
        with patch.object(replay, "parse_args", return_value=self.args), patch.object(
            replay, "load_workspace_calibration", side_effect=ValueError("invalid calibration")
        ) as loader, patch.object(replay, "connect_rosbridge") as connect:
            with self.assertRaisesRegex(ValueError, "invalid calibration"):
                replay.main()
        loader.assert_called_once_with(self.args.workspace_calibration, self.args.control_side, optional=False)
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
