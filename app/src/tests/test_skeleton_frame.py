from __future__ import annotations

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from teleop_ik_core import TeleopArmIK, apply_joint_safety
from teleop_direct_retarget import (
    DirectRetargetConfig,
    body_frame_from_skeleton,
    calibration_window_is_stable,
    relative_direction_target,
    skeleton_observation,
)


def pose(position: tuple[float, float, float]) -> dict[str, object]:
    return {"pos": list(position), "quat": [0.0, 0.0, 0.0, 1.0], "tracked": True}


def skeleton_frame(*, side_raise: bool = False, folded_right: bool = False) -> dict[str, object]:
    left_elbow = (0.0, 0.50, 1.50) if side_raise else (0.0, 0.20, 1.20)
    left_wrist = (0.0, 0.80, 1.50) if side_raise else (0.0, 0.20, 0.90)
    right_wrist = (0.20, -0.20, 1.30) if folded_right else (0.0, -0.20, 0.90)
    return {
        "receive_time": 1.0,
        "skeleton": {
            "left": {
                "shoulder": pose((0.0, 0.20, 1.50)),
                "elbow": pose(left_elbow),
                "wrist": pose(left_wrist),
                "hand": pose(left_wrist),
            },
            "right": {
                "shoulder": pose((0.0, -0.20, 1.50)),
                "elbow": pose((0.0, -0.20, 1.20)),
                "wrist": pose(right_wrist),
                "hand": pose(right_wrist),
            },
            "torso": {
                "left_shoulder": pose((0.0, 0.20, 1.50)),
                "right_shoulder": pose((0.0, -0.20, 1.50)),
                "left_hip": pose((0.0, 0.15, 1.00)),
                "right_hip": pose((0.0, -0.15, 1.00)),
                "waist": pose((0.0, 0.0, 1.00)),
                "chest": pose((0.0, 0.0, 1.40)),
                "neck": pose((0.0, 0.0, 1.60)),
                "head": pose((0.0, 0.0, 1.80)),
                "spine1": pose((0.0, 0.0, 1.15)),
                "spine2": pose((0.0, 0.0, 1.30)),
            },
        },
    }


class SkeletonFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DirectRetargetConfig(
            joint_names=[str(index) for index in range(14)],
            motor_ids=list(range(14)),
            calibration_window_frames=5,
        )

    def test_body_axes_follow_formotion_x_forward_y_left_z_up(self) -> None:
        body = body_frame_from_skeleton(skeleton_frame())
        self.assertIsNotNone(body)
        assert body is not None
        np.testing.assert_allclose(body.forward, (1.0, 0.0, 0.0), atol=1e-9)
        np.testing.assert_allclose(body.left, (0.0, 1.0, 0.0), atol=1e-9)
        np.testing.assert_allclose(body.up, (0.0, 0.0, 1.0), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(body.world_to_body)), 1.0, places=9)

    def test_side_raise_is_positive_left_not_forward(self) -> None:
        observation, reason = skeleton_observation(skeleton_frame(side_raise=True), self.config)
        self.assertEqual(reason, "ok")
        assert observation is not None
        np.testing.assert_allclose(
            observation.arms["left"].upper_direction_local,
            (0.0, 1.0, 0.0),
            atol=1e-9,
        )

    def test_stable_window_calibrates(self) -> None:
        observation, _ = skeleton_observation(skeleton_frame(), self.config)
        assert observation is not None
        stable, metrics = calibration_window_is_stable([observation] * 5, self.config)
        self.assertTrue(stable)
        self.assertTrue(metrics["variation_stable"])
        self.assertTrue(metrics["stand_pose_valid"])
        self.assertAlmostEqual(metrics["max_direction_rms_rad"], 0.0)
        self.assertAlmostEqual(metrics["max_length_cv"], 0.0)

    def test_stable_but_folded_arm_is_rejected_for_calibration(self) -> None:
        observation, _ = skeleton_observation(skeleton_frame(folded_right=True), self.config)
        assert observation is not None
        stable, metrics = calibration_window_is_stable([observation] * 5, self.config)

        self.assertFalse(stable)
        self.assertTrue(metrics["variation_stable"])
        self.assertFalse(metrics["stand_pose_valid"])
        self.assertGreater(metrics["max_elbow_angle_rad"], self.config.calibration_max_elbow_angle_rad)

    def test_relative_direction_preserves_robot_home_and_applies_human_delta(self) -> None:
        target = relative_direction_target(
            home_human=np.asarray((0.0, 0.0, -1.0)),
            current_human=np.asarray((0.0, 1.0, 0.0)),
            robot_home=np.asarray((0.0, 0.0, -1.0)),
            scale=1.0,
        )
        np.testing.assert_allclose(target, (0.0, 1.0, 0.0), atol=1e-9)


class DirectionSolverSafetyTest(unittest.TestCase):
    def test_frame_safety_is_applied_once_after_internal_iterations(self) -> None:
        solver = object.__new__(TeleopArmIK)
        solver.model = SimpleNamespace(nq=1)
        solver.config = SimpleNamespace(
            max_iterations=3,
            damping=0.1,
            nullspace_gain=0.0,
            max_joint_delta_per_step=0.1,
            smoothing_alpha=0.5,
        )
        solver.home_q = np.zeros(1)
        solver.joint_limits = [(-10.0, 10.0)]

        def build_task(_self, _q, _side, _targets, *, upper_weight, forearm_weight):
            del upper_weight, forearm_weight
            return np.ones((1, 1)), np.ones(1)

        solver._build_segment_direction_task = MethodType(build_task, solver)
        solver._mask_control_side = MethodType(lambda _self, dq, _side: dq, solver)
        solver.segment_directions = MethodType(
            lambda _self, _q, _side: (np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])),
            solver,
        )
        targets = {"left": (np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]))}

        with patch("teleop_ik_core.apply_joint_safety", wraps=apply_joint_safety) as safety:
            command, meta = solver.solve_segment_directions(
                timestamp=1.0,
                control_side="left",
                target_directions=targets,
                current_q=np.zeros(1),
                last_q_cmd=np.zeros(1),
            )

        self.assertEqual(safety.call_count, 1)
        np.testing.assert_allclose(command, [0.05], atol=1e-9)
        self.assertEqual(meta["iterations"], 3)


if __name__ == "__main__":
    unittest.main()