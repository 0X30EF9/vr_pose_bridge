from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from teleop_control_safety import feedback_is_fresh
from teleop_ik_types import IKCommand, TeleopIKResult
from teleop_ik_adapters import build_rosbridge_motor_message
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS, arm_positions_from_status


class Publisher:
    def __init__(self):
        self.messages = []
        self.error = None

    def publish(self, message):
        if self.error is not None:
            raise self.error
        self.messages.append(copy.deepcopy(message))


def load_bridge():
    modules = {name: ModuleType(name) for name in (
        "rclpy", "rclpy.node", "roslibpy", "sensor_msgs", "sensor_msgs.msg")}
    modules["rclpy.node"].Node = type("Node", (), {})
    modules["sensor_msgs.msg"].JointState = type("JointState", (), {})
    path = Path(__file__).resolve().parents[1] / "mocap_ros_2_bridge_node.py"
    spec = importlib.util.spec_from_file_location("bridge_boundaries_under_test", path)
    module = importlib.util.module_from_spec(spec)
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


class BridgeBoundariesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_bridge()
        # These regressions cover the existing explicit IK path.
        self.module.CONTROL_MODE = "ik"
        self.clock = SimpleNamespace(now=100.0, wall=1000.0)
        self.module.time = SimpleNamespace(
            monotonic=lambda: self.clock.now,
            time=lambda: self.clock.wall,
            perf_counter=lambda: self.clock.now,
        )
        self.warnings = []
        node = object.__new__(self.module.MocapBridgeNode)
        node.state_lock = threading.Lock()
        node.data_lock = threading.Lock()
        node.current_lr_arm_motor_q = np.zeros(14)
        node.last_arm_state_time = self.clock.now
        node.current_waist_position = 0.0
        node.last_waist_state_time = self.clock.now
        node.command_joint_limits = [(-1.0, 1.0)] * 14
        node.safe_baseline_q = np.zeros(14)
        node.safe_baseline_waist = 0.0
        node.last_command_q = None
        node.last_waist_command = None
        node.last_command_publish_time = 0.0
        node.command_faulted = False
        node.rosbridge_client = SimpleNamespace(is_connected=True)
        node.left_motors = copy.deepcopy(self.module.ARM_MOTOR_TEMPLATES[:7])
        node.right_motors = copy.deepcopy(self.module.ARM_MOTOR_TEMPLATES[7:])
        node.all_motors = copy.deepcopy(self.module.ARM_MOTOR_TEMPLATES)
        node.waist_motor = copy.deepcopy(self.module.WAIST_MOTOR_TEMPLATE)
        node.arm_cmd_pub = Publisher()
        node.waist_cmd_pub = Publisher()
        node.ik_solver = SimpleNamespace(waist_limit=(-0.5, 0.5), controls_waist=True)
        node.get_logger = lambda: SimpleNamespace(
            warn=self.warnings.append, error=self.warnings.append,
            info=self.warnings.append, debug=self.warnings.append)
        node.should_log = lambda: False
        node._warned_command_disabled = False
        node._warned_missing_arm_state = False
        node._warned_missing_waist_state = False
        node._warned_arm_error = False
        self.node = node

    def command(self, positions=None, **options):
        return IKCommand(
            timestamp=self.clock.wall, control_side="both",
            joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
            positions=np.full(14, 0.02) if positions is None else positions, **options)

    def test_complete_named_and_unnamed_legacy_feedback(self):
        status = [{"name": motor_id, "pos": index / 100.0}
                  for index, motor_id in enumerate(ARM_MOTOR_IDS)]
        self.assertEqual(arm_positions_from_status(list(reversed(status))),
                         tuple(index / 100.0 for index in range(14)))
        self.assertEqual(arm_positions_from_status([{"pos": 0.0}] * 14), (0.0,) * 14)

    def test_bad_feedback_invalidates_previous_good_sample(self):
        valid = [{"name": motor_id, "pos": 0.0} for motor_id in ARM_MOTOR_IDS]
        cases = [
            [{"name": index, "pos": 0.0} for index in range(100, 114)],
            valid[:-1], [valid[0]] + valid[:-1],
            [{"pos": 0.0}] + valid[1:],
            [{"name": 11, "pos": float("nan")}] + valid[1:],
            [{"name": 11, "pos": 0.0, "error": 1}] + valid[1:],
        ]
        for sample in cases:
            with self.subTest(sample=sample):
                self.node.arm_status_callback({"status": valid})
                self.assertIsNotNone(self.node.current_lr_arm_motor_q)
                self.node.arm_status_callback({"status": sample})
                self.assertIsNone(self.node.current_lr_arm_motor_q)
                self.assertIsNone(self.node.last_arm_state_time)

    def test_valid_command_updates_state_after_publication(self):
        self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 1)
        np.testing.assert_allclose(self.node.last_command_q, 0.02)

    def test_stale_frame_or_feedback_does_not_advance_command(self):
        for arm_time, frame_time in ((99.0, 100.0), (100.0, 99.0), (101.0, 100.0)):
            with self.subTest(arm_time=arm_time, frame_time=frame_time):
                self.node.last_arm_state_time = arm_time
                self.node.publish_to_robot(self.command(), mocap_received_at=frame_time)
                self.assertFalse(self.node.arm_cmd_pub.messages)
                self.assertIsNone(self.node.last_command_q)

    def test_wall_clock_changes_do_not_bypass_monotonic_gate(self):
        self.clock.wall = -100000.0
        self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 1)
        self.clock.now += 1.0
        self.node.publish_to_robot(self.command(), mocap_received_at=100.0)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 1)

    def test_hard_limit_and_actual_feedback_step_are_final_boundaries(self):
        self.node.command_joint_limits[0] = (-0.25, 0.25)
        for target in (np.full(14, 0.266), np.full(14, 0.09)):
            self.node.publish_to_robot(self.command(target), mocap_received_at=self.clock.now)
        self.assertFalse(self.node.arm_cmd_pub.messages)
        self.assertIsNone(self.node.last_command_q)

    def test_waist_rejection_does_not_advance_arm_state(self):
        self.node.last_waist_state_time = 99.0
        self.node.publish_to_robot(self.command(waist_position=0.02), mocap_received_at=self.clock.now)
        self.assertFalse(self.node.arm_cmd_pub.messages)
        self.assertIsNone(self.node.last_command_q)
        self.assertIsNone(self.node.last_waist_command)

    def test_transport_failure_latches_session_without_optimistic_state(self):
        self.node.arm_cmd_pub.error = OSError("offline")
        self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertTrue(self.node.command_faulted)
        self.assertIsNone(self.node.last_command_q)
        self.node.arm_cmd_pub.error = None
        self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertFalse(self.node.arm_cmd_pub.messages)

    def test_disconnected_transport_cannot_queue_commands(self):
        self.node.rosbridge_client.is_connected = False
        self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertTrue(self.node.command_faulted)
        self.assertFalse(self.node.arm_cmd_pub.messages)
        self.assertIsNone(self.node.last_command_q)

    def test_frame_expiring_during_message_build_does_not_latch_session(self):
        def slow_build(*args):
            self.clock.now += 1.0
            return build_rosbridge_motor_message(*args)

        with patch.object(self.module, "build_rosbridge_motor_message", side_effect=slow_build):
            self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertFalse(self.node.arm_cmd_pub.messages)
        self.assertFalse(self.node.command_faulted)
        self.node.last_arm_state_time = self.clock.now
        self.node.publish_to_robot(self.command(), mocap_received_at=self.clock.now)
        self.assertEqual(len(self.node.arm_cmd_pub.messages), 1)

    def test_partial_arm_waist_failure_preserves_only_submitted_arm(self):
        self.node.waist_cmd_pub.error = OSError("waist offline")
        self.node.publish_to_robot(self.command(waist_position=0.02), mocap_received_at=self.clock.now)
        self.assertTrue(self.node.command_faulted)
        np.testing.assert_allclose(self.node.last_command_q, 0.02)
        self.assertIsNone(self.node.last_waist_command)

    def test_slow_ik_result_is_rejected_even_with_new_feedback(self):
        node = self.node
        node.latest_mocap_data = {"timestamp": self.clock.wall, "receive_monotonic": self.clock.now}
        node.log_frame_index = 0
        node.process_hand_ik = lambda data: None
        node.mocap_anatomical_wrist_pose = lambda *values: np.eye(4)
        node.mocap_reference_position = lambda data: np.zeros(3)
        node.ensure_natural_down_startup = lambda *values: True
        node.mocap_target_mapper = SimpleNamespace(map_targets=lambda *values, **options: SimpleNamespace(
            left_target=np.eye(4), right_target=np.eye(4),
            meta={"left_control_target": np.eye(4), "right_control_target": np.eye(4)}))
        node.ik_solver.joint_limits = node.command_joint_limits
        node.ik_solver.wrist_effector_poses = lambda q: (np.eye(4), np.eye(4))

        def slow_solve(request):
            self.clock.now += 1.0
            node.last_arm_state_time = self.clock.now
            node.last_waist_state_time = self.clock.now
            return TeleopIKResult(self.command(np.zeros(14)), True, 0.0, 1, False)

        node.ik_solver.solve = slow_solve
        node.control_loop()
        self.assertFalse(node.arm_cmd_pub.messages)
        self.assertIsNone(node.last_command_q)

    def test_freshness_rejects_nonfinite_and_future_times(self):
        for received in (None, float("nan"), float("inf"), 101.0):
            self.assertFalse(feedback_is_fresh(received, 100.0, 0.5))

    def test_right_arm_templates_are_matched_by_id_and_complete(self):
        command = self.command(np.arange(14) / 100.0)
        command = IKCommand(
            timestamp=command.timestamp, control_side="right", joint_names=command.joint_names,
            positions=command.positions, motor_ids=command.motor_ids)
        with self.assertRaises(ValueError):
            build_rosbridge_motor_message(command, self.node.all_motors)
        message = build_rosbridge_motor_message(command, list(reversed(self.node.right_motors)))
        by_id = {motor["name"]: motor["pos"] for motor in message["cmds"]}
        self.assertEqual(by_id, dict(zip(ARM_MOTOR_IDS[7:], np.arange(7, 14) / 100.0)))

    def test_invalid_jointstate_and_waist_samples_clear_cached_feedback(self):
        self.node.ros2_joint_state_callback(SimpleNamespace(
            name=list(ARM_JOINT_NAMES[:-1]) + [ARM_JOINT_NAMES[0]], position=[0.0] * 14))
        self.assertIsNone(self.node.current_lr_arm_motor_q)
        self.assertIsNone(self.node.last_arm_state_time)
        self.node.waist_status_callback({"status": [{"name": 31, "pos": float("nan")}]})
        self.assertIsNone(self.node.current_waist_position)
        self.assertIsNone(self.node.last_waist_state_time)

    def test_hand_command_checks_frame_after_mapping_and_latches_transport_failure(self):
        from dataclasses import replace

        self.module.HAND_CONTROL_CONFIG = replace(self.module.HAND_CONTROL_CONFIG, publish_enabled=True)
        self.node.hand_feedback = {side: (0.0,) * 6 for side in ("left", "right")}
        self.node.hand_feedback_time = {side: self.clock.now for side in ("left", "right")}
        self.node.last_hand_command = {side: None for side in ("left", "right")}
        self.node._warned_hand_state = {side: False for side in ("left", "right")}
        self.node.hand_cmd_pubs = {side: Publisher() for side in ("left", "right")}
        result = SimpleNamespace(left_calibrated=True, right_calibrated=True, joint_positions=[0.0] * 12)
        self.node.publish_hand_result(result, mocap_received_at=99.0)
        self.assertFalse(self.node.hand_cmd_pubs["left"].messages)
        self.node.hand_cmd_pubs["left"].error = OSError("hand offline")
        self.node.publish_hand_result(result, mocap_received_at=self.clock.now)
        self.assertTrue(self.node.command_faulted)
        self.assertIsNone(self.node.last_hand_command["left"])
        self.assertFalse(self.node.hand_cmd_pubs["right"].messages)

    def test_fresh_but_stationary_hand_feedback_prevents_command_accumulation(self):
        from dataclasses import replace

        self.module.HAND_CONTROL_CONFIG = replace(self.module.HAND_CONTROL_CONFIG, publish_enabled=True)
        sides = ("left", "right")
        self.node.hand_feedback = {side: (0.0,) * 6 for side in sides}
        self.node.hand_feedback_time = {side: self.clock.now for side in sides}
        self.node.last_hand_command = {side: None for side in sides}
        self.node._warned_hand_state = {side: False for side in sides}
        self.node.hand_cmd_pubs = {side: Publisher() for side in sides}
        result = SimpleNamespace(left_calibrated=True, right_calibrated=True, joint_positions=[0.0] * 12)
        self.node.publish_hand_result(result, mocap_received_at=self.clock.now)
        self.node.publish_hand_result(result, mocap_received_at=self.clock.now)
        for side in sides:
            self.assertEqual(len(self.node.hand_cmd_pubs[side].messages), 1)
            self.assertLessEqual(max(self.node.last_hand_command[side]), self.module.HAND_CONTROL_CONFIG.max_step)


if __name__ == "__main__":
    unittest.main()
