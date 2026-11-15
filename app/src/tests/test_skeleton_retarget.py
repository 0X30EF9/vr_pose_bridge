from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from teleop_direct_retarget import DirectRetargetConfig
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
from teleop_skeleton_retarget import SkeletonRetargeter
from test_skeleton_frame import skeleton_frame


def config(**overrides) -> DirectRetargetConfig:
    return DirectRetargetConfig(**{
        "joint_names": list(ARM_JOINT_NAMES), "motor_ids": list(ARM_MOTOR_IDS),
        "calibration_window_frames": 3, **overrides,
    })


def rotate_frame(frame, rotation, translation=(0.0, 0.0, 0.0)):
    result = copy.deepcopy(frame)
    for group in result["skeleton"].values():
        for entry in group.values():
            entry["pos"] = (rotation.apply(entry["pos"]) + translation).tolist()
            entry["quat"] = (rotation * Rotation.from_quat(entry["quat"])).as_quat().tolist()
    return result


def turn_hand(frame, angle, axis="z", side="left"):
    result = copy.deepcopy(frame)
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    for name in ("wrist", "hand"):
        entry = result["skeleton"][side][name]
        entry["quat"] = (rotation * Rotation.from_quat(entry["quat"])).as_quat().tolist()
    return result


class RecordingSolver:
    def __init__(self):
        self.joint_limits = [(-2.0, 2.0)] * 14
        self.config = SimpleNamespace(orientation_weight=0.25)
        self.calls = []
        self.output = None

    def set_home_q(self, values):
        self.home_q = values.copy()

    def end_effector_poses(self, values):
        return np.eye(4), np.eye(4)

    def segment_directions(self, values, side):
        return np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0])

    def solve_segment_directions(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs["current_q"].copy() if self.output is None else self.output.copy()), {"limited": False}


class SkeletonMappingTest(unittest.TestCase):
    def setUp(self):
        self.frame = skeleton_frame()
        self.solver = RecordingSolver()
        self.retarget = SkeletonRetargeter(config(), self.solver)
        self.zero = np.zeros(14)
        self.retarget.align([copy.deepcopy(self.frame) for _ in range(5)], self.zero)

    def test_neutral_uses_measured_robot_baseline(self):
        baseline = np.full(14, 0.15)
        self.retarget.align([self.frame] * 3, baseline)
        command = self.retarget.command_from_frame(self.frame, baseline)
        np.testing.assert_allclose(command.positions, baseline)
        np.testing.assert_allclose(self.solver.home_q, baseline)
        self.assertIsNone(command.waist_position)

    def test_body_rigid_rotation_and_translation_do_not_move_targets(self):
        self.retarget.command_from_frame(self.frame, self.zero)
        expected = self.solver.calls[-1]
        transformed = rotate_frame(self.frame, Rotation.from_euler("zyx", [90, 15, -10], degrees=True), (2.0, -3.0, 0.5))
        self.retarget.command_from_frame(transformed, self.zero)
        actual = self.solver.calls[-1]
        for side in ("left", "right"):
            np.testing.assert_allclose(actual["target_directions"][side], expected["target_directions"][side], atol=1e-12)
            np.testing.assert_allclose(actual["target_orientations"][side], expected["target_orientations"][side], atol=1e-12)

    def test_direction_change_is_present_in_solver_targets(self):
        # A small upward/sideward change stays below the local jump gate.
        moved = copy.deepcopy(self.frame)
        moved["skeleton"]["left"]["elbow"]["pos"][1] += 0.04
        moved["skeleton"]["left"]["wrist"]["pos"][1] += 0.08
        self.retarget.command_from_frame(moved, self.zero)
        upper, forearm = self.solver.calls[-1]["target_directions"]["left"]
        self.assertGreater(upper[1], 0.1)
        self.assertGreater(forearm[1], 0.1)

    def test_same_positions_with_90_degree_hand_twist_change_orientation_only(self):
        self.retarget.command_from_frame(self.frame, self.zero)
        original = self.solver.calls[-1]
        self.retarget.command_from_frame(turn_hand(self.frame, 90), self.zero)
        twisted = self.solver.calls[-1]
        np.testing.assert_allclose(twisted["target_directions"]["left"], original["target_directions"]["left"])
        np.testing.assert_allclose(twisted["target_orientations"]["left"], Rotation.from_euler("z", 90, degrees=True).as_matrix(), atol=1e-12)

    def test_quaternion_sign_and_crossing_180_are_continuous(self):
        for angle in (89, 179, 181):
            self.retarget.command_from_frame(turn_hand(self.frame, angle), self.zero)
        latest = self.solver.calls[-1]["target_orientations"]["left"]
        signed = turn_hand(self.frame, 181)
        for name in ("wrist", "hand"):
            signed["skeleton"]["left"][name]["quat"] = (-np.asarray(signed["skeleton"]["left"][name]["quat"])).tolist()
        self.retarget.command_from_frame(signed, self.zero)
        np.testing.assert_allclose(self.solver.calls[-1]["target_orientations"]["left"], latest, atol=1e-12)
        np.testing.assert_allclose(latest, Rotation.from_euler("z", 181, degrees=True).as_matrix(), atol=1e-12)

    def test_alignment_averages_quaternion_signs_and_uses_full_window(self):
        frames = [turn_hand(self.frame, angle) for angle in (-1, 0, 1)]
        frames[-1]["skeleton"]["left"]["hand"]["quat"] = (-np.asarray(frames[-1]["skeleton"]["left"]["hand"]["quat"])).tolist()
        self.retarget.align(frames, self.zero)
        self.retarget.command_from_frame(self.frame, self.zero)
        np.testing.assert_allclose(self.solver.calls[-1]["target_orientations"]["left"], np.eye(3), atol=1e-12)

    def test_half_orientation_gain_crosses_180_and_360_continuously(self):
        mapper = SkeletonRetargeter(config(), self.solver, orientation_scale=0.5)
        mapper.align([self.frame] * 3, self.zero)
        for angle in (89, 179, 181, 269, 359, 360, 361):
            mapper.command_from_frame(turn_hand(self.frame, angle), self.zero)
            np.testing.assert_allclose(
                self.solver.calls[-1]["target_orientations"]["left"],
                Rotation.from_euler("z", angle / 2.0, degrees=True).as_matrix(), atol=1e-12,
            )

    def test_half_gain_full_turn_then_axis_change_has_no_target_jump(self):
        mapper = SkeletonRetargeter(config(), self.solver, orientation_scale=0.5)
        mapper.align([self.frame] * 3, self.zero)
        for angle in (89, 179, 181, 269, 359):
            mapper.command_from_frame(turn_hand(self.frame, angle), self.zero)
        previous_target = self.solver.calls[-1]["target_orientations"]["left"].copy()
        mapper.command_from_frame(turn_hand(self.frame, 0.1, axis="x"), self.zero)
        current_target = self.solver.calls[-1]["target_orientations"]["left"]
        input_step = (Rotation.from_euler("x", 0.1, degrees=True) * Rotation.from_euler("z", 359, degrees=True).inv()).magnitude()
        target_step = Rotation.from_matrix(current_target @ previous_target.T).magnitude()
        self.assertLess(np.degrees(target_step), 1.0)
        self.assertAlmostEqual(target_step, input_step * 0.5, places=12)

    def test_full_gain_matches_absolute_home_delta_after_mixed_axis_motion(self):
        home = turn_hand(self.frame, 20)
        self.retarget.align([home] * 3, self.zero)
        home_rotation = Rotation.from_euler("z", 20, degrees=True)
        for angle, axis in ((80, "z"), (169, "z"), (179, "z"), (269, "z"), (359, "z"), (0.1, "x"), ([20, 10], "xy")):
            self.retarget.command_from_frame(turn_hand(self.frame, angle, axis=axis), self.zero)
            expected = (Rotation.from_euler(axis, angle, degrees=True) * home_rotation.inv()).as_matrix()
            np.testing.assert_allclose(self.solver.calls[-1]["target_orientations"]["left"], expected, atol=1e-12)

    def test_fractional_gain_repeated_static_frame_does_not_accumulate(self):
        mapper = SkeletonRetargeter(config(), self.solver, orientation_scale=0.5)
        mapper.align([self.frame] * 3, self.zero)
        moved = turn_hand(self.frame, 45)
        mapper.command_from_frame(moved, self.zero)
        expected = self.solver.calls[-1]["target_orientations"]["left"].copy()
        for _ in range(30):
            mapper.command_from_frame(moved, self.zero)
        np.testing.assert_allclose(self.solver.calls[-1]["target_orientations"]["left"], expected, atol=1e-12)

    def test_fractional_gain_starts_at_window_mean_and_realign_resets_integral(self):
        mapper = SkeletonRetargeter(config(), self.solver, orientation_scale=0.5)
        mapper.align([turn_hand(self.frame, angle) for angle in (-1, 0, 1)], self.zero)
        mapper.command_from_frame(turn_hand(self.frame, 1), self.zero)
        np.testing.assert_allclose(
            self.solver.calls[-1]["target_orientations"]["left"],
            Rotation.from_euler("z", 0.5, degrees=True).as_matrix(), atol=1e-12,
        )
        mapper.command_from_frame(turn_hand(self.frame, 60), self.zero)
        new_home = turn_hand(self.frame, 60)
        baseline = np.full(14, 0.2)
        mapper.align([new_home] * 3, baseline)
        command = mapper.command_from_frame(new_home, baseline)
        np.testing.assert_allclose(self.solver.calls[-1]["target_orientations"]["left"], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(command.positions, baseline)

    def test_single_side_preserves_feedback_on_other_side(self):
        mapper = SkeletonRetargeter(config(control_side="left"), self.solver)
        mapper.align([self.frame] * 3, self.zero)
        self.solver.output = np.ones(14)
        feedback = np.zeros(14)
        feedback[7:] = 0.2
        command = mapper.command_from_frame(self.frame, feedback)
        self.assertEqual(set(self.solver.calls[-1]["target_directions"]), {"left"})
        self.assertEqual(set(self.solver.calls[-1]["target_orientations"]), {"left"})
        np.testing.assert_array_equal(command.positions[7:], feedback[7:])
        np.testing.assert_allclose(command.positions[:7], 0.04)

    def test_current_feedback_is_used_and_proposals_are_not_committed(self):
        self.solver.output = np.ones(14)
        feedback = np.full(14, 0.1)
        last = np.full(14, 0.11)
        first = self.retarget.command_from_frame(self.frame, feedback, last)
        second = self.retarget.command_from_frame(self.frame, feedback, last)
        np.testing.assert_array_equal(self.solver.calls[-1]["current_q"], feedback)
        np.testing.assert_array_equal(self.solver.calls[-1]["last_q_cmd"], last)
        np.testing.assert_array_equal(first.positions, second.positions)
        self.assertLessEqual(float(np.max(abs(first.positions-feedback))), 0.04 + 1e-12)
        self.assertLessEqual(float(np.max(abs(first.positions-last))), 0.04 + 1e-12)

    def test_invalid_tracking_and_quaternions_reject_before_solve(self):
        for joint in ("wrist", "hand"):
            for replacement in ({"tracked": False}, {"quat": [0, 0, 0, 0]}, {"quat": [0, float("nan"), 0, 1]}, {"pos": [0, float("inf"), 1]}):
                with self.subTest(joint=joint, replacement=replacement):
                    frame = copy.deepcopy(self.frame)
                    frame["skeleton"]["left"][joint].update(replacement)
                    with self.assertRaises(ValueError):
                        self.retarget.command_from_frame(frame, self.zero)
        self.assertEqual(self.solver.calls, [])

    def test_bone_length_local_jump_and_rotation_jump_reject(self):
        bad_length = copy.deepcopy(self.frame)
        bad_length["skeleton"]["left"]["wrist"]["pos"][2] = 0.3
        local_jump = copy.deepcopy(self.frame)
        for name in ("shoulder", "elbow", "wrist", "hand"):
            local_jump["skeleton"]["left"][name]["pos"][0] += 0.3
        for frame in (bad_length, local_jump, turn_hand(self.frame, 150)):
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    self.retarget.command_from_frame(frame, self.zero)
        self.assertEqual(self.solver.calls, [])

    def test_invalid_alignment_and_nonfinite_parameters_fail_closed(self):
        for frames in ([self.frame] * 2, [skeleton_frame(folded_right=True)] * 3, [self.frame, turn_hand(self.frame, 30), self.frame]):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    self.retarget.align(frames, self.zero)
                self.assertFalse(self.retarget.aligned)
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                SkeletonRetargeter(config(), self.solver, value)

    def test_nonfinite_feedback_last_command_and_solver_output_are_rejected(self):
        bad = self.zero.copy()
        bad[0] = float("nan")
        for current, last in ((bad, None), (self.zero, bad)):
            with self.assertRaises(ValueError):
                self.retarget.command_from_frame(self.frame, current, last)
        self.solver.output = bad
        with self.assertRaises(ValueError):
            self.retarget.command_from_frame(self.frame, self.zero)

    def test_final_bounds_intersect_hard_limits_feedback_last_and_baseline(self):
        self.solver.joint_limits[0] = (-0.2, 0.25)
        baseline = np.full(14, 0.2)
        self.retarget.align([self.frame] * 3, baseline)
        self.solver.output = np.full(14, 10.0)
        command = self.retarget.command_from_frame(self.frame, np.full(14, 0.23), np.full(14, 0.24))
        self.assertEqual(command.positions[0], 0.25)
        with self.assertRaises(ValueError):
            self.retarget.command_from_frame(self.frame, self.zero, np.full(14, 0.2))


class RealUrdfSkeletonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mesh paths are intentionally unavailable in the text-only offline
        # checkout. The real default constructor must need kinematics only.
        with patch.object(pin.RobotWrapper, "BuildFromURDF", side_effect=AssertionError("geometry must not load")):
            cls.mapper = SkeletonRetargeter(config(control_side="left"))
        cls.solver = cls.mapper.solver

    def setUp(self):
        self.frame = skeleton_frame()
        self.zero = np.zeros(14)
        self.mapper.align([self.frame] * 3, self.zero)

    def test_default_real_urdf_neutral_fk_and_solve(self):
        command = self.mapper.command_from_frame(self.frame, self.zero)
        self.assertEqual(command.positions.shape, (14,))
        np.testing.assert_allclose(command.positions, self.zero, atol=1e-10)
        self.assertTrue(np.all(np.isfinite(command.positions)))

    def test_real_orientation_task_follows_wrist_without_legacy_012_envelope(self):
        initial = self.solver.end_effector_poses(self.zero)[0][:3, :3]
        desired_q = self.zero.copy()
        desired_q[5] = 0.3
        desired = self.solver.end_effector_poses(desired_q)[0][:3, :3]
        delta = desired @ initial.T
        frame = copy.deepcopy(self.frame)
        frame["skeleton"]["left"]["hand"]["quat"] = Rotation.from_matrix(delta).as_quat().tolist()
        current = self.zero.copy()
        previous = None
        for _ in range(30):
            command = self.mapper.command_from_frame(frame, current, previous)
            self.assertLessEqual(float(np.max(abs(command.positions-current))), 0.04 + 1e-12)
            previous = command.positions.copy()
            current = previous.copy()  # Explicitly simulate acknowledged feedback.
        actual = self.solver.end_effector_poses(current)[0][:3, :3]
        error = np.linalg.norm(Rotation.from_matrix(desired @ actual.T).as_rotvec())
        self.assertLess(error, 0.06)
        self.assertGreater(current[5], 0.12)
        np.testing.assert_array_equal(current[7:], self.zero[7:])

    def test_invalid_orientation_target_rejected_by_real_solver(self):
        directions = {"left": self.solver.segment_directions(self.zero, "left")}
        for rotation in (np.full((3, 3), np.nan), np.diag([1, 1, -1]), np.ones((3, 3))):
            with self.subTest(rotation=rotation):
                with self.assertRaises(ValueError):
                    self.solver.solve_segment_directions(
                        timestamp=1.0, control_side="left", target_directions=directions,
                        target_orientations={"left": rotation}, current_q=self.zero, last_q_cmd=None,
                    )


if __name__ == "__main__":
    unittest.main()
