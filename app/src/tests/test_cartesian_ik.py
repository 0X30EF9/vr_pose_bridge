from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from teleop_ik_core import TeleopArmIK, compute_cartesian_increment
from teleop_ik_types import TeleopIKConfig, TeleopIKInput
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS


def make_solver(**overrides):
    return TeleopArmIK(TeleopIKConfig(**{
        "joint_names": list(ARM_JOINT_NAMES), "motor_ids": list(ARM_MOTOR_IDS),
        "max_iterations": 6, "max_joint_delta_per_step": .04, "control_dt": .01,
        "enable_local_log": False, "log_to_console": False,
        "show_model_load_progress": False, "log_ik_result": False, **overrides,
    }))


class CartesianIkTest(unittest.TestCase):
    def setUp(self):
        self.solver = make_solver()
        self.zero = np.zeros(14)
        self.targets = self.solver.end_effector_poses(self.zero)

    def request(self, **overrides):
        return TeleopIKInput(**{
            "timestamp": 1., "control_side": "both", "target_pose_l": self.targets[0].copy(),
            "target_pose_r": self.targets[1].copy(), "current_q": self.zero.copy(), **overrides,
        })

    def test_neutral_is_exact_and_never_commands_waist(self):
        result = self.solver.solve_cartesian(self.request(current_waist_position=.2))
        np.testing.assert_allclose(result.command.positions, self.zero, atol=1e-12)
        self.assertTrue(result.converged)
        self.assertIsNone(result.command.waist_position)
        self.assertEqual(result.iteration_count, 1)

    def test_reachable_pose_tracks_and_reports_exact_final_residuals(self):
        desired = self.zero.copy()
        desired[[0, 2, 3, 5, 6, 7, 9, 10, 12, 13]] = [.10, .08, -.14, .16, .1, .10, -.08, -.14, .16, -.1]
        goals = self.solver.end_effector_poses(desired)
        current, previous = self.zero.copy(), None
        for index in range(40):
            result = self.solver.solve_cartesian(self.request(
                target_pose_l=goals[0], target_pose_r=goals[1], current_q=current,
                last_q_cmd=previous, timestamp=index*.01,
            ))
            self.assertLessEqual(float(np.max(abs(result.command.positions-current))), .04+1e-12)
            if previous is not None:
                self.assertLessEqual(float(np.max(abs(result.command.positions-previous))), .04+1e-12)
            previous = result.command.positions.copy()
            current = previous.copy()  # Simulated acknowledged feedback, never inferred by IK.
        actual = self.solver.end_effector_poses(current)
        for index, side in enumerate(("left", "right")):
            residual = result.command.meta["residual_by_side"][side]
            position = np.linalg.norm(goals[index][:3, 3]-actual[index][:3, 3])
            angle = Rotation.from_matrix(goals[index][:3, :3]@actual[index][:3, :3].T).magnitude()
            self.assertAlmostEqual(residual["position_error_m"], position, places=12)
            self.assertAlmostEqual(residual["orientation_error_rad"], angle, places=12)
            self.assertLess(position, .002)
            self.assertLess(angle, .02)

    def test_position_only_ignores_target_rotation_without_resetting_wrist(self):
        self.solver.config = replace(self.solver.config, task_mode="position_only")
        current = self.zero.copy()
        current[[5, 6]] = [.12, .08]
        goals = self.solver.end_effector_poses(current)
        goals[0][:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ goals[0][:3, :3]
        result = self.solver.solve_cartesian(self.request(
            control_side="left", current_q=current, target_pose_l=goals[0], target_pose_r=goals[1],
        ))
        np.testing.assert_allclose(result.command.positions, current, atol=1e-12)
        self.assertTrue(result.converged)
        self.assertGreater(result.command.meta["residual_by_side"]["left"]["orientation_error_rad"], 1.5)
        desired = current.copy()
        desired[5] += .15
        goal = self.solver.end_effector_poses(desired)[0]
        moved = self.solver.solve_cartesian(self.request(control_side="left", current_q=current, target_pose_l=goal))
        self.assertGreater(abs(moved.command.positions[5]-current[5]), 1e-6)

    def test_inactive_last_command_can_differ_and_inactive_feedback_is_preserved(self):
        previous = self.zero.copy()
        previous[12:] = [.15, -.15]
        desired = self.zero.copy()
        desired[5] = .15
        result = self.solver.solve_cartesian(self.request(
            control_side="left", last_q_cmd=previous,
            target_pose_l=self.solver.end_effector_poses(desired)[0],
        ))
        np.testing.assert_array_equal(result.command.positions[7:], self.zero[7:])
        self.assertEqual(set(result.command.meta["residual_by_side"]), {"left"})

    def test_feedback_seeds_first_linearization_and_no_proposal_is_committed(self):
        previous = np.full(14, .02)
        request = self.request(last_q_cmd=previous)
        with patch.object(pin, "computeJointJacobians", wraps=pin.computeJointJacobians) as jacobians:
            first = self.solver.solve_cartesian(request)
        np.testing.assert_array_equal(jacobians.call_args_list[0].args[2], self.zero)
        second = self.solver.solve_cartesian(request)
        np.testing.assert_array_equal(first.command.positions, second.command.positions)
        np.testing.assert_array_equal(self.solver.home_q, self.zero)

    def test_empty_feedback_last_intersection_is_rejected(self):
        previous = self.zero.copy()
        previous[5] = .2
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.solver.solve_cartesian(self.request(last_q_cmd=previous))

    def test_invalid_reference_shape_values_and_hard_bounds_are_rejected(self):
        out_of_bounds = self.zero.copy()
        out_of_bounds[0] = self.solver.joint_limits[0][1]+.01
        for value in (np.zeros((2, 7)), np.zeros(13), np.full(14, np.nan), np.full(14, np.inf), out_of_bounds):
            for field in ("current_q", "last_q_cmd"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.solver.solve_cartesian(self.request(**{field: value}))

    def test_invalid_se3_targets_are_rejected_even_for_inactive_side(self):
        malformed = []
        for change in ("reflection", "scale", "bottom", "nan"):
            target = self.targets[0].copy()
            if change == "reflection":
                target[:3, :3] = np.diag([1, 1, -1])
            elif change == "scale":
                target[:3, :3] *= 2
            elif change == "bottom":
                target[3, 0] = .01
            else:
                target[0, 3] = np.nan
            malformed.append(target)
        malformed.append(np.eye(3))
        for target in malformed:
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    self.solver.solve_cartesian(self.request(control_side="left", target_pose_r=target))

    def test_invalid_side_and_timestamp_are_rejected(self):
        for override in ({"control_side": "none"}, {"timestamp": np.nan}, {"timestamp": np.inf},
                         {"current_waist_position": np.nan}):
            with self.assertRaises(ValueError):
                self.solver.solve_cartesian(self.request(**override))

    def test_explicit_legacy_joint_constraints_are_rejected(self):
        original = self.solver.config
        for overrides in ({"locked_joint_indices": (5,)}, {"stabilized_joint_indices": (5,)},
                          {"stabilized_joint_max_delta": .1}, {"stabilized_joint_max_deltas": (.1,)},
                          {"stabilized_joint_indices": (5,), "stabilized_joint_delta_bounds": ((-.1, .1),)}):
            self.solver.config = replace(original, **overrides)
            with self.assertRaisesRegex(ValueError, "locked/stabilized"):
                self.solver.solve_cartesian(self.request())

    def test_input_arrays_are_unchanged_and_output_is_independent(self):
        current = self.zero.copy()
        last = self.zero.copy()
        left, right = (target.copy() for target in self.targets)
        request = self.request(current_q=current, last_q_cmd=last, target_pose_l=left, target_pose_r=right)
        snapshots = [value.copy() for value in (current, last, left, right)]
        result = self.solver.solve_cartesian(request)
        for value, original in zip((current, last, left, right), snapshots):
            np.testing.assert_array_equal(value, original)
        result.command.positions[:] = .1
        result.command.joint_names[0] = "changed"
        np.testing.assert_array_equal(current, self.zero)
        np.testing.assert_array_equal(last, self.zero)
        self.assertEqual(self.solver.config.joint_names[0], ARM_JOINT_NAMES[0])

    def test_unreachable_target_reports_unclipped_residual_and_obeys_all_bounds(self):
        goal = self.targets[0].copy()
        goal[:3, 3] = [2., 1., 2.]
        current = self.zero.copy()
        current[0] = self.solver.joint_limits[0][1]-.005
        previous = current.copy()
        previous[0] -= .01
        result = self.solver.solve_cartesian(self.request(
            control_side="left", current_q=current, last_q_cmd=previous, target_pose_l=goal,
        ))
        q = result.command.positions
        bounds = np.asarray(self.solver.joint_limits)
        self.assertTrue(np.all(q >= bounds[:, 0]) and np.all(q <= bounds[:, 1]))
        self.assertLessEqual(float(np.max(abs(q[:7]-current[:7]))), .04+1e-12)
        self.assertLessEqual(float(np.max(abs(q[:7]-previous[:7]))), .04+1e-12)
        self.assertGreater(result.command.meta["residual_by_side"]["left"]["position_error_m"], 2.)
        self.assertFalse(result.converged)
        self.assertTrue(result.limited)

    def test_zero_step_freezes_without_nonfinite_fallback(self):
        self.solver.config = replace(self.solver.config, max_joint_delta_per_step=0.)
        goal = self.targets[0].copy()
        goal[0, 3] += .02
        result = self.solver.solve_cartesian(self.request(target_pose_l=goal))
        np.testing.assert_array_equal(result.command.positions, self.zero)
        self.assertTrue(result.limited)
        self.assertFalse(result.converged)

    def test_zero_damping_near_singular_pose_remains_finite_and_bounded(self):
        self.solver.config = replace(self.solver.config, damping=0.)
        goal = self.targets[0].copy()
        goal[0, 3] += .01
        result = self.solver.solve_cartesian(self.request(control_side="left", target_pose_l=goal))
        self.assertTrue(np.all(np.isfinite(result.command.positions)))
        self.assertLessEqual(float(np.max(abs(result.command.positions))), .04+1e-12)
        np.testing.assert_array_equal(result.command.positions[7:], self.zero[7:])

    def test_kinematic_work_is_one_jacobian_update_per_iteration_and_one_final_fk(self):
        goal = self.targets[0].copy()
        goal[0, 3] += .02
        with patch.object(pin, "computeJointJacobians", wraps=pin.computeJointJacobians) as jacobians, \
                patch.object(pin, "forwardKinematics", wraps=pin.forwardKinematics) as fk, \
                patch.object(pin, "computeFrameJacobian", wraps=pin.computeFrameJacobian) as frame_jacobians:
            result = self.solver.solve_cartesian(self.request(target_pose_l=goal))
        self.assertEqual(jacobians.call_count, result.iteration_count)
        self.assertEqual(fk.call_count, 1)
        self.assertEqual(frame_jacobians.call_count, 0)

    def test_projected_posture_preference_matches_dls_nullspace_algebra(self):
        rng = np.random.default_rng(816)
        jacobians = rng.normal(size=(2, 6, 7))
        errors = rng.normal(size=(2, 6))
        bias = rng.normal(size=(2, 7))
        damping = .08
        actual = compute_cartesian_increment(jacobians, errors, damping, bias)
        expected = []
        for jacobian, error, preference in zip(jacobians, errors, bias):
            inverse = jacobian.T @ np.linalg.solve(jacobian@jacobian.T+damping*damping*np.eye(6), np.eye(6))
            expected.append(inverse@error+(np.eye(7)-inverse@jacobian)@preference)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
