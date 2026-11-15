from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_mocap_protocol import JointPose, MOCAP_JOINT_NAMES, MocapFrame
from teleop_startup_pose import (
    NaturalDownStartupConfig,
    NaturalDownStartupGate,
    check_human_natural_down,
    check_robot_arm_state,
)


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)


def natural_down_positions() -> dict[str, np.ndarray]:
    return {
        "left_shoulder": np.array([0.0, 0.22, 1.45]),
        "right_shoulder": np.array([0.0, -0.22, 1.45]),
        "left_hip": np.array([0.0, 0.14, 0.95]),
        "right_hip": np.array([0.0, -0.14, 0.95]),
        "left_elbow": np.array([0.0, 0.22, 1.12]),
        "right_elbow": np.array([0.0, -0.22, 1.12]),
        "left_wrist": np.array([0.0, 0.22, 0.78]),
        "right_wrist": np.array([0.0, -0.22, 0.78]),
        "left_hand": np.array([0.0, 0.22, 0.72]),
        "right_hand": np.array([0.0, -0.22, 0.72]),
        "chest": np.array([0.0, 0.0, 1.30]),
    }


def make_frame(
    overrides: dict[str, np.ndarray] | None = None,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> MocapFrame:
    positions = natural_down_positions()
    positions.update(overrides or {})
    rotation_safe = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation_safe = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    joints = []
    for name in MOCAP_JOINT_NAMES:
        position = positions.get(name, np.zeros(3))
        transformed = rotation_safe @ position + translation_safe
        joints.append(JointPose(tuple(transformed), IDENTITY_QUATERNION, tracked=True))
    return MocapFrame(
        timestamp_s=1.0,
        timestamp_us=1_000_000,
        receive_time_s=1.0,
        joints=tuple(joints),
        source="startup_pose_test",
    )


class NaturalDownStartupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NaturalDownStartupConfig(hold_seconds=2.0)

    def test_both_arms_natural_down_pass(self) -> None:
        passed, reasons, metrics = check_human_natural_down(make_frame(), self.config)
        self.assertTrue(passed)
        self.assertEqual(reasons, ())
        self.assertGreater(metrics["left_wrist_drop_m"], 0.60)
        self.assertGreater(metrics["right_wrist_drop_m"], 0.60)

    def test_chest_pose_raised_pose_and_single_side_are_rejected(self) -> None:
        poses = (
            {
                "left_elbow": np.array([0.28, 0.22, 1.35]),
                "left_wrist": np.array([0.42, 0.22, 1.30]),
                "right_elbow": np.array([0.28, -0.22, 1.35]),
                "right_wrist": np.array([0.42, -0.22, 1.30]),
            },
            {
                "left_elbow": np.array([0.0, 0.50, 1.45]),
                "left_wrist": np.array([0.0, 0.78, 1.45]),
                "right_elbow": np.array([0.0, -0.50, 1.45]),
                "right_wrist": np.array([0.0, -0.78, 1.45]),
            },
            {
                "right_elbow": np.array([0.25, -0.22, 1.35]),
                "right_wrist": np.array([0.40, -0.22, 1.30]),
            },
        )
        for overrides in poses:
            with self.subTest(overrides=overrides):
                passed, reasons, _ = check_human_natural_down(make_frame(overrides), self.config)
                self.assertFalse(passed)
                self.assertTrue(reasons)

    def test_body_rigid_transform_does_not_change_result(self) -> None:
        rotation = R.from_euler("xyz", [0.45, -0.30, 1.10]).as_matrix()
        translation = np.array([2.5, -1.7, 0.8])
        passed, reasons, _ = check_human_natural_down(
            make_frame(rotation=rotation, translation=translation),
            self.config,
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, ())

    def test_robot_pose_is_unrestricted_but_state_must_be_valid(self) -> None:
        arbitrary_pose = np.linspace(-0.8, 0.8, 14)
        passed, reasons, metrics = check_robot_arm_state(arbitrary_pose)
        self.assertTrue(passed)
        self.assertEqual(reasons, ())
        self.assertEqual(metrics, {})

        for invalid_state in (np.zeros(13), np.full(14, np.nan)):
            with self.subTest(shape=invalid_state.shape):
                passed, reasons, _ = check_robot_arm_state(invalid_state)
                self.assertFalse(passed)
                self.assertEqual(reasons, ("robot_arm_state_invalid",))

    def test_gate_requires_continuous_stable_hold_and_resets_on_motion(self) -> None:
        gate = NaturalDownStartupGate(self.config)
        frame = make_frame()
        left_home = np.array([0.0, 0.22, -0.58])
        right_home = np.array([0.0, -0.22, -0.58])

        robot_q = np.linspace(-0.8, 0.8, 14)
        first = gate.observe(0.0, frame, robot_q, left_home, right_home)
        self.assertFalse(first.ready)
        self.assertEqual(first.reasons, ("holding_natural_down",))

        moved_left_home = left_home + [0.10, 0.0, 0.0]
        unstable = gate.observe(1.0, frame, robot_q, moved_left_home, right_home)
        self.assertFalse(unstable.ready)
        self.assertEqual(unstable.reasons, ("natural_down_pose_unstable",))
        self.assertEqual(unstable.held_seconds, 0.0)

        self.assertFalse(gate.observe(2.0, frame, robot_q, moved_left_home, right_home).ready)
        ready = gate.observe(3.1, frame, robot_q, moved_left_home, right_home)
        self.assertTrue(ready.ready)
        self.assertEqual(ready.reasons, ())


if __name__ == "__main__":
    unittest.main()