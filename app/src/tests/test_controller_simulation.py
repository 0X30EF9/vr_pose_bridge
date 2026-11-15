from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from teleop_controller_simulation import (AnchorOracle, ControllerWorld, build_controller_solver, packet_from_robot_poses,
                                          run_controller_scene)
from teleop_vr_controller import parse_controller_packet, pose_in_robot


def poses():
    result = {side: np.eye(4) for side in ("left", "right")}
    for i, pose in enumerate(result.values()):
        pose[:3, 3] = [.2, .3 if i == 0 else -.3, -.4]
        pose[:3, :3] = Rotation.from_euler("xyz", [.2, -.3, .4]).as_matrix()
    return result


class ControllerFixtureTest(unittest.TestCase):
    def packet(self, source):
        return packet_from_robot_poses(poses(), seq=7, timestamp=1., coordinate_system=source,
            enabled={"left": True, "right": True}, tracked={"left": True, "right": True},
            activation_ids={"left": 1, "right": 1})

    def test_independent_unity_and_openxr_inverse_fixtures_match_actual_parser(self):
        for source in ("unity", "openxr"):
            with self.subTest(source=source):
                packet = self.packet(source)
                frame = parse_controller_packet(json.dumps(packet).encode(), received_at=1.)
                self.assertEqual(frame.sequence, 7)
                for side in ("left", "right"):
                    np.testing.assert_allclose(pose_in_robot(getattr(frame, side), source), poses()[side], atol=1e-12)

    def test_wrong_coordinate_and_invalid_values_are_rejected(self):
        for kind in ("axes", "quaternion", "position", "schema"):
            packet = self.packet("unity")
            if kind == "axes":
                packet["coordinate_system"] = "unknown_axes"
            elif kind == "quaternion":
                packet["left"]["rotation_xyzw"] = [0, 0, 0, 0]
            elif kind == "position":
                packet["left"]["position"][0] = float("nan")
            else:
                packet["version"] = 999
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                parse_controller_packet(json.dumps(packet).encode(), received_at=1.)

    def test_each_cartesian_axis_has_the_declared_direction_in_both_raw_conventions(self):
        for source in ("unity", "openxr"):
            for axis in range(3):
                target = {side: np.eye(4) for side in ("left", "right")}
                for pose in target.values():
                    pose[axis, 3] = .02
                packet = packet_from_robot_poses(target, seq=1, timestamp=1., coordinate_system=source,
                    enabled={"left": True, "right": True}, tracked={"left": True, "right": True},
                    activation_ids={"left": 1, "right": 1})
                parsed = parse_controller_packet(json.dumps(packet), received_at=1.)
                with self.subTest(source=source, axis=axis):
                    np.testing.assert_allclose(pose_in_robot(parsed.left, source)[:3, 3], target["left"][:3, 3], atol=1e-12)

    def fixture(self):
        return {"poses": poses(), "enabled": {"left": True, "right": True},
                "tracked": {"left": True, "right": True}, "activation_ids": {"left": 1, "right": 1}, "invalid": False}

    def test_anchor_oracle_repress_does_not_add_disabled_controller_motion(self):
        oracle = AnchorOracle()
        fixture = self.fixture()
        robot = {side: np.eye(4) for side in ("left", "right")}
        first, anchored = oracle.update(fixture, robot)
        self.assertTrue(all(anchored.values()))
        np.testing.assert_allclose(first["left"], robot["left"], atol=1e-12)
        fixture["enabled"] = {"left": False, "right": False}
        fixture["poses"]["left"][0, 3] += .8
        self.assertEqual(oracle.update(fixture, robot)[0], {})
        fixture["enabled"] = {"left": True, "right": True}
        fixture["activation_ids"] = {"left": 2, "right": 2}
        target, _ = oracle.update(fixture, robot)
        np.testing.assert_allclose(target["left"], robot["left"], atol=1e-12)
        fixture["poses"]["left"][0, 3] += .02
        target, _ = oracle.update(fixture, robot)
        self.assertAlmostEqual(target["left"][0, 3], .02)

    def test_fault_oracle_requires_a_tracked_release_before_rearming(self):
        oracle, fixture = AnchorOracle(), self.fixture()
        robot = {side: np.eye(4) for side in ("left", "right")}
        oracle.update(fixture, robot)
        fault = copy.deepcopy(fixture)
        fault["invalid"] = True
        self.assertEqual(oracle.update(fault, robot)[0], {})
        fixture["activation_ids"] = {"left": 2, "right": 2}
        self.assertEqual(oracle.update(fixture, robot)[0], {})
        fixture["enabled"] = {"left": False, "right": False}
        oracle.update(fixture, robot)
        fixture["enabled"] = {"left": True, "right": True}
        self.assertEqual(len(oracle.update(fixture, robot)[0]), 2)


HAS_ENGINE = importlib.util.find_spec("pybullet") is not None and importlib.util.find_spec("pinocchio") is not None


@unittest.skipUnless(HAS_ENGINE, "dedicated Pinocchio/PyBullet simulation environment required")
class ControllerPhysicsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.solver = build_controller_solver(100., .48)
        self.world = ControllerWorld(Path(temporary.name), dt=.01, substeps=4, max_joint_velocity=.48)
        self.addCleanup(self.world.close)

    def test_clutch_anchor_and_inactive_targets_use_actual_production_path(self):
        report, _ = run_controller_scene(self.world, self.solver, "clutch_reanchor", .8, False, "unity")
        self.assertGreaterEqual(report["anchor_side_frames"], 4)
        self.assertLess(report["anchor_target_position_max_m"], 1e-6)
        self.assertEqual(report["anchor_joint_delta_max_rad"], 0.)
        self.assertEqual(report["inactive_motor_target_changes"], 0)
        self.assertEqual(report["committed_history_violations"], 0)
        self.assertEqual(report["unexpected_rejections"], 0)
        self.assertLess(report["mapping_position_max_m"], 1e-6)

    def test_invalid_raw_inputs_stop_and_then_reanchor_through_production_gate(self):
        report, _ = run_controller_scene(self.world, self.solver, "invalid_input", .8, False, "openxr")
        self.assertGreater(report["input_rejections"], 0)
        self.assertEqual(report["invalid_input_submissions"], 0)
        self.assertEqual(report["inactive_motor_target_changes"], 0)
        self.assertEqual(report["unexpected_rejections"], 0)
        self.assertEqual(report["hard_limit_violation_frames"], 0)
        self.assertEqual(report["velocity_violation_frames"], 0)

    def test_zero_pose_preview_anchors_without_a_command_jump(self):
        self.assertEqual(self.solver.config.smoothing_alpha, 1.)
        self.assertAlmostEqual(self.solver.config.max_joint_delta_per_step, .0048)
        report, _ = run_controller_scene(self.world, self.solver, "left_only", .5, False, "unity", initial_q=np.zeros(14))
        self.assertEqual(report["anchor_joint_delta_max_rad"], 0.)
        self.assertEqual(report["unexpected_rejections"], 0)
        self.assertEqual(report["hard_limit_violation_frames"], 0)
        self.assertEqual(report["velocity_violation_frames"], 0)


if __name__ == "__main__":
    unittest.main()
