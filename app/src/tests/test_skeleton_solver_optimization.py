from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from teleop_ik_core import (
    apply_joint_safety, compute_damped_increment, compute_joint_space_increment,
)
from teleop_direct_retarget import relative_direction_target
from teleop_skeleton_retarget import _relative_unit_direction
from teleop_skeleton_retarget import SkeletonRetargeter
from teleop_direct_retarget import DirectRetargetConfig
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
from test_skeleton_frame import skeleton_frame


def make_mapper():
    return SkeletonRetargeter(DirectRetargetConfig(
        joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS), calibration_window_frames=3,
    ))


def robot_targets(solver, q):
    poses = solver.end_effector_poses(q)
    return ({side: solver.segment_directions(q, side) for side in ("left", "right")},
            {side: pose[:3, :3] for side, pose in zip(("left", "right"), poses)})


class SkeletonSolverOptimizationTest(unittest.TestCase):
    def test_prevalidated_direction_mapping_matches_checked_reference(self):
        rng = np.random.default_rng(887)
        for _ in range(30):
            vectors = rng.normal(size=(3, 3))
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            home, current, robot = vectors
            for direction in (current, home, -home):
                for scale in (0., .25, 1., 2.):
                    np.testing.assert_allclose(
                        _relative_unit_direction(home, direction, robot, scale),
                        relative_direction_target(home, direction, robot, scale), atol=1e-12, rtol=1e-12,
                    )

    def test_joint_space_dls_matches_task_space_with_rank_deficient_tasks(self):
        rng = np.random.default_rng(1742)
        for rows, columns in ((9, 7), (18, 14), (6, 7)):
            jacobian = rng.normal(size=(rows, columns))
            jacobian[-1] = jacobian[0]
            error = rng.normal(size=rows)
            for damping in (.08, .001, .5):
                with self.subTest(shape=jacobian.shape, damping=damping):
                    np.testing.assert_allclose(
                        compute_joint_space_increment(jacobian, error, damping),
                        compute_damped_increment(jacobian, error, damping), atol=2e-8, rtol=2e-8,
                    )

    def test_zero_damping_singular_task_has_finite_least_norm_solution(self):
        jacobian = np.zeros((9, 7))
        jacobian[0, 0] = 1
        result = compute_joint_space_increment(jacobian, np.ones(9), 0.)
        np.testing.assert_array_equal(result, [1, 0, 0, 0, 0, 0, 0])

    def test_batched_two_arm_dls_matches_independent_task_space_solves(self):
        rng = np.random.default_rng(1743)
        jacobians = rng.normal(size=(2, 9, 7))
        jacobians[:, -1] = jacobians[:, 0]
        errors = rng.normal(size=(2, 9))
        actual = compute_joint_space_increment(jacobians, errors, .08)
        expected = np.asarray([compute_damped_increment(jacobian, error, .08) for jacobian, error in zip(jacobians, errors)])
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)

    def test_real_combined_iteration_matches_original_stacked_task(self):
        solver = make_mapper().solver
        current = np.zeros(14)
        current[[0, 3, 5, 7, 10, 12]] = [.1, -.1, .05, .1, -.1, .05]
        desired = current.copy()
        desired[[0, 3, 5, 7, 10, 12]] += [.01, .02, .03, -.01, .02, .03]
        directions, rotations = robot_targets(solver, desired)
        for side in ("left", "right", "both"):
            with self.subTest(side=side):
                jacobian, error = solver._build_segment_direction_task(
                    current, side, directions, upper_weight=1., forearm_weight=.85,
                )
                selected_rotations = {key: value for key, value in rotations.items() if side == "both" or side == key}
                angular_jacobian, angular_error = solver._build_segment_orientation_task(current, selected_rotations)
                jacobian = np.vstack((jacobian, angular_jacobian * .25))
                error = np.concatenate((error, angular_error * .25))
                dq = compute_damped_increment(jacobian, error, .08)
                dq += solver.config.nullspace_gain * (solver.home_q-current)
                dq = solver._mask_control_side(dq, side)
                bounds = np.asarray(solver.joint_limits)
                expected, _ = apply_joint_safety(
                    np.clip(current + dq, bounds[:, 0], bounds[:, 1]), current,
                    solver.joint_limits, .04, .65,
                )
                actual, _ = solver.solve_segment_directions(
                    timestamp=1., control_side=side, target_directions=directions,
                    target_orientations=rotations, current_q=current, last_q_cmd=current,
                    upper_weight=1., forearm_weight=.85, orientation_weight=.25, max_iterations=1,
                )
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)

    def test_mapper_updates_kinematics_once_per_iteration_and_final_output(self):
        mapper = make_mapper()
        home = skeleton_frame()
        frame = skeleton_frame()
        frame["skeleton"]["left"]["elbow"]["pos"][1] += .01
        frame["skeleton"]["left"]["wrist"]["pos"][1] += .02
        mapper.align([home] * 3, np.zeros(14))
        with patch.object(pin, "computeJointJacobians", wraps=pin.computeJointJacobians) as jacobians, \
                patch.object(pin, "forwardKinematics", wraps=pin.forwardKinematics) as fk, \
                patch.object(pin, "computeFrameJacobian", wraps=pin.computeFrameJacobian) as frame_jacobians:
            command = mapper.command_from_frame(frame, np.zeros(14))
        self.assertEqual(jacobians.call_count, command.meta["solver_meta"]["iterations"])
        self.assertEqual(fk.call_count, 1)
        self.assertEqual(frame_jacobians.call_count, 0)


if __name__ == "__main__":
    unittest.main()
