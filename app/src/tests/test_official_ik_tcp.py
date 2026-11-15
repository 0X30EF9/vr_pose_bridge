from __future__ import annotations

import unittest

import numpy as np
import pinocchio as pin

from teleop_ik_types import TeleopIKConfig, TeleopIKInput
from teleop_robot_constants import POSITION_ONLY_LOCKED_JOINT_INDICES
from teleop_ik_ubtech_casadi import (
    ARM_JOINT_NAMES,
    OFFICIAL_SOLVER_SO_PATH,
    OfficialTiangong2ProIK,
)


@unittest.skipUnless(OFFICIAL_SOLVER_SO_PATH.is_file(), "缺少官方 IK 求解器 .so")
class OfficialIkTcpFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = OfficialTiangong2ProIK(
            TeleopIKConfig(
                joint_names=list(ARM_JOINT_NAMES),
                motor_ids=list(range(14)),
                task_mode="pose",
                enable_local_log=False,
                show_model_load_progress=False,
            )
        )

    def test_public_end_effector_pose_uses_hand_tcp(self) -> None:
        q = np.zeros(14)
        left_pose, right_pose = self.solver.end_effector_poses(q)
        full_q = self.solver._full_q(q)
        pin.forwardKinematics(self.solver.model, self.solver.data, full_q)
        pin.updateFramePlacements(self.solver.model, self.solver.data)

        expected_left = self.solver.data.oMf[self.solver.model.getFrameId("left_tcp_link")].homogeneous
        expected_right = self.solver.data.oMf[self.solver.model.getFrameId("right_tcp_link")].homogeneous
        np.testing.assert_allclose(left_pose, expected_left, atol=1e-10)
        np.testing.assert_allclose(right_pose, expected_right, atol=1e-10)

        left_solver_pose = self.solver.data.oMf[self.solver.left_solver_frame_id].translation
        right_solver_pose = self.solver.data.oMf[self.solver.right_solver_frame_id].translation
        self.assertAlmostEqual(float(np.linalg.norm(left_pose[:3, 3] - left_solver_pose)), 0.0848, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(right_pose[:3, 3] - right_solver_pose)), 0.0848, places=6)

    def test_tcp_target_is_converted_for_legacy_solver_frame(self) -> None:
        q = np.zeros(14)
        left_tcp, right_tcp = self.solver.end_effector_poses(q)
        left_solver_target = self.solver._tcp_target_to_solver_target(
            left_tcp,
            self.solver.left_solver_to_tcp,
        )
        right_solver_target = self.solver._tcp_target_to_solver_target(
            right_tcp,
            self.solver.right_solver_to_tcp,
        )

        full_q = self.solver._full_q(q)
        pin.forwardKinematics(self.solver.model, self.solver.data, full_q)
        pin.updateFramePlacements(self.solver.model, self.solver.data)
        np.testing.assert_allclose(
            left_solver_target,
            self.solver.data.oMf[self.solver.left_solver_frame_id].homogeneous,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            right_solver_target,
            self.solver.data.oMf[self.solver.right_solver_frame_id].homogeneous,
            atol=1e-10,
        )

    def test_solve_passes_converted_tcp_targets_to_official_solver(self) -> None:
        q = np.zeros(14)
        left_target, right_target = self.solver.end_effector_poses(q)
        left_target = left_target.copy()
        right_target = right_target.copy()
        left_target[0, 3] += 0.03
        right_target[0, 3] += 0.02
        captured: dict[str, np.ndarray] = {}
        original_solve_ik = self.solver.solver.solve_ik

        def capture_solve_ik(**kwargs):
            captured["left_wrist"] = np.asarray(kwargs["left_wrist"], dtype=float)
            captured["right_wrist"] = np.asarray(kwargs["right_wrist"], dtype=float)
            return np.asarray(kwargs["q_init"], dtype=float)

        self.solver.solver.solve_ik = capture_solve_ik
        try:
            self.solver.solve(
                TeleopIKInput(
                    timestamp=1.0,
                    control_side="both",
                    target_pose_l=left_target,
                    target_pose_r=right_target,
                    current_q=q,
                    current_waist_position=0.0,
                )
            )
        finally:
            self.solver.solver.solve_ik = original_solve_ik

        np.testing.assert_allclose(
            captured["left_wrist"] @ self.solver.left_solver_to_tcp.homogeneous,
            left_target,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            captured["right_wrist"] @ self.solver.right_solver_to_tcp.homogeneous,
            right_target,
            atol=1e-10,
        )

    def test_pose_ik_keeps_tcp_position_while_wrist_rotates_continuously(self) -> None:
        q = np.asarray(
            [0.15, 0.18, 0.05, -0.45, 0.04, 0.08, -0.06, 0.15, -0.18, -0.05, -0.45, -0.04, -0.08, 0.06],
            dtype=float,
        )
        self.solver.last_waist_position = 0.0
        self.solver.set_home_q(q)
        left_home, right_home = self.solver.end_effector_poses(q)
        solved_qs: list[np.ndarray] = []
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        try:
            for frame_index, angle in enumerate(np.linspace(0.0, 0.35, 15)):
                left_target = left_home.copy()
                right_target = right_home.copy()
                world_delta = pin.exp3(np.array([angle, 0.0, 0.0]))
                left_target[:3, :3] = world_delta @ left_home[:3, :3]
                right_target[:3, :3] = world_delta @ right_home[:3, :3]
                result = self.solver.solve(
                    TeleopIKInput(
                        timestamp=frame_index * 0.02,
                        control_side="both",
                        target_pose_l=left_target,
                        target_pose_r=right_target,
                        current_q=q,
                        last_q_cmd=q,
                        current_waist_position=0.0,
                    )
                )
                self.assertEqual(result.reason, "ok")
                q = result.command.positions.copy()
                solved_qs.append(q)
                solved_left, solved_right = self.solver.end_effector_poses(q)
                position_errors.append(
                    max(
                        float(np.linalg.norm(solved_left[:3, 3] - left_target[:3, 3])),
                        float(np.linalg.norm(solved_right[:3, 3] - right_target[:3, 3])),
                    )
                )
                orientation_errors.append(
                    max(
                        float(np.linalg.norm(pin.log3(solved_left[:3, :3] @ left_target[:3, :3].T))),
                        float(np.linalg.norm(pin.log3(solved_right[:3, :3] @ right_target[:3, :3].T))),
                    )
                )
        finally:
            self.solver.last_waist_position = 0.0
            self.solver.set_home_q(np.zeros(14))

        solved_qs_array = np.asarray(solved_qs)
        self.assertLess(max(position_errors), 1e-3)
        self.assertLess(max(orientation_errors), 5e-4)
        self.assertGreater(float(np.max(np.ptp(solved_qs_array[:, [5, 6, 12, 13]], axis=0))), 0.30)
        self.assertLess(float(np.max(np.abs(np.diff(solved_qs_array, axis=0)))), 0.06)

    def test_pose_ik_continuous_lift_keeps_tcp_and_elbow_trajectory_continuous(self) -> None:
        q = np.asarray(
            [0.10, 0.18, 0.03, -0.40, 0.02, 0.05, -0.04, 0.10, -0.18, -0.03, -0.40, -0.02, -0.05, 0.04],
            dtype=float,
        )
        self.solver.last_waist_position = 0.0
        self.solver.set_home_q(q)
        left_home, right_home = self.solver.end_effector_poses(q)
        solved_qs: list[np.ndarray] = []
        elbow_positions: list[np.ndarray] = []
        position_errors: list[float] = []
        try:
            for frame_index, progress in enumerate(np.linspace(0.0, 1.0, 41)):
                left_target = left_home.copy()
                right_target = right_home.copy()
                left_target[2, 3] += 0.16 * progress
                right_target[2, 3] += 0.16 * progress
                world_delta = pin.exp3(np.array([0.0, -0.35 * progress, 0.0]))
                left_target[:3, :3] = world_delta @ left_home[:3, :3]
                right_target[:3, :3] = world_delta @ right_home[:3, :3]
                result = self.solver.solve(
                    TeleopIKInput(
                        timestamp=frame_index * 0.02,
                        control_side="both",
                        target_pose_l=left_target,
                        target_pose_r=right_target,
                        current_q=q,
                        last_q_cmd=q,
                        current_waist_position=0.0,
                    )
                )
                self.assertEqual(result.reason, "ok")
                q = result.command.positions.copy()
                solved_qs.append(q)
                solved_left, solved_right = self.solver.end_effector_poses(q)
                position_errors.append(
                    max(
                        float(np.linalg.norm(solved_left[:3, 3] - left_target[:3, 3])),
                        float(np.linalg.norm(solved_right[:3, 3] - right_target[:3, 3])),
                    )
                )
                full_q = self.solver._full_q(q, result.command.waist_position)
                pin.forwardKinematics(self.solver.model, self.solver.data, full_q)
                elbow_positions.append(
                    np.concatenate(
                        [
                            self.solver.data.oMi[
                                self.solver.model.getJointId("elbow_pitch_l_joint")
                            ].translation,
                            self.solver.data.oMi[
                                self.solver.model.getJointId("elbow_pitch_r_joint")
                            ].translation,
                        ]
                    )
                )
        finally:
            self.solver.last_waist_position = 0.0
            self.solver.set_home_q(np.zeros(14))

        solved_qs_array = np.asarray(solved_qs)
        elbow_array = np.asarray(elbow_positions).reshape(-1, 2, 3)
        self.assertLess(max(position_errors), 1e-3)
        self.assertLess(float(np.max(np.abs(np.diff(solved_qs_array, axis=0)))), 0.10)
        self.assertLess(float(np.max(np.linalg.norm(np.diff(elbow_array, axis=0), axis=2))), 0.02)
        self.assertGreater(float(np.min(elbow_array[:, :, 2] - elbow_array[0, :, 2])), -0.005)

    def test_position_only_ignores_uncalibrated_wrist_orientation(self) -> None:
        original_config = self.solver.config
        original_solve_ik = self.solver.solver.solve_ik
        self.solver.config = TeleopIKConfig(
            joint_names=list(ARM_JOINT_NAMES),
            motor_ids=list(range(14)),
            task_mode="position_only",
            max_iterations=6,
            damping=0.06,
            locked_joint_indices=POSITION_ONLY_LOCKED_JOINT_INDICES,
            enable_local_log=False,
            show_model_load_progress=False,
        )
        q = np.zeros(14)
        self.solver.set_home_q(q)
        left_target, right_target = self.solver.end_effector_poses(q)
        left_target[:3, :3] = pin.exp3(np.array([1.1, -0.7, 0.5])) @ left_target[:3, :3]
        right_target[:3, :3] = pin.exp3(np.array([-0.8, 0.9, -0.6])) @ right_target[:3, :3]

        def reject_pose_solver(**_kwargs):
            raise AssertionError("position_only 不得调用官方完整位姿求解器")

        self.solver.solver.solve_ik = reject_pose_solver
        try:
            result = self.solver.solve(
                TeleopIKInput(
                    timestamp=2.0,
                    control_side="both",
                    target_pose_l=left_target,
                    target_pose_r=right_target,
                    current_q=q,
                    current_waist_position=0.0,
                )
            )
        finally:
            self.solver.config = original_config
            self.solver.solver.solve_ik = original_solve_ik
            self.solver.last_waist_position = 0.0
            self.solver.set_home_q(q)

        np.testing.assert_allclose(result.command.positions, q, atol=1e-10)
        self.assertEqual(result.command.meta["solver"], "official_urdf_position_dls")
        self.assertFalse(result.command.meta["orientation_target_used"])
        self.assertAlmostEqual(result.command.waist_position, 0.0)

    def test_position_only_converges_with_wrist_locked_to_home(self) -> None:
        original_config = self.solver.config
        self.solver.config = TeleopIKConfig(
            joint_names=list(ARM_JOINT_NAMES),
            motor_ids=list(range(14)),
            task_mode="position_only",
            max_iterations=8,
            damping=0.03,
            position_convergence_threshold=1e-3,
            max_joint_delta_per_step=0.06,
            smoothing_alpha=1.0,
            locked_joint_indices=POSITION_ONLY_LOCKED_JOINT_INDICES,
            enable_local_log=False,
            show_model_load_progress=False,
        )
        home_q = np.asarray(
            [0.0, 0.12, 0.08, -0.25, 0.04, 0.18, -0.14, 0.0, -0.12, -0.08, -0.25, -0.04, -0.16, 0.13],
            dtype=float,
        )
        self.solver.last_waist_position = 0.0
        self.solver.set_home_q(home_q)
        left_target, right_target = self.solver.end_effector_poses(home_q)
        left_target[0, 3] += 0.06
        left_target[2, 3] += 0.05
        current_q = home_q.copy()
        result = None
        try:
            for frame_index in range(80):
                result = self.solver.solve(
                    TeleopIKInput(
                        timestamp=3.0 + frame_index * 0.02,
                        control_side="left",
                        target_pose_l=left_target,
                        target_pose_r=right_target,
                        current_q=current_q,
                        last_q_cmd=current_q,
                        current_waist_position=0.0,
                    )
                )
                current_q = result.command.positions.copy()
            solved_left, _ = self.solver.end_effector_poses(current_q)
        finally:
            self.solver.config = original_config
            self.solver.set_home_q(np.zeros(14))

        self.assertIsNotNone(result)
        final_error = float(np.linalg.norm(left_target[:3, 3] - solved_left[:3, 3]))
        self.assertLess(final_error, 1e-3)
        np.testing.assert_allclose(current_q[[5, 6, 12, 13]], home_q[[5, 6, 12, 13]], atol=1e-12)
        np.testing.assert_allclose(current_q[7:], home_q[7:], atol=1e-12)
        self.assertEqual(result.command.meta["locked_arm_joint_indices"], [5, 6, 12, 13])


if __name__ == "__main__":
    unittest.main()