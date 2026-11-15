from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from teleop_direct_retarget import DirectRetargetConfig, DirectRetargeter
from teleop_ik_core import TeleopArmIK, add_operational_frame, apply_joint_safety
from teleop_ik_types import TeleopIKConfig, TeleopIKInput
from teleop_joint_schema import ARM_JOINT_NAMES


SOURCE_DIR = Path(__file__).resolve().parents[1]


def make_model(with_waist: bool = False) -> pin.Model:
    """Small real Pinocchio model, with the original left shoulder's 0.25 rad bound."""
    model = pin.Model()
    root = 0
    if with_waist:
        root = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "body_yaw_joint")
    for side, names, frame_name in (
        ("left", ARM_JOINT_NAMES[:7], "L_ee"),
        ("right", ARM_JOINT_NAMES[7:], "R_ee"),
    ):
        parent = root
        for name in names:
            parent = model.addJoint(
                parent, pin.JointModelRY(), pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])), name
            )
        add_operational_frame(model, frame_name, names[-1], pin.SE3.Identity())
        add_operational_frame(
            model, f"{side}_tcp_link", names[-1],
            pin.SE3(np.eye(3), np.array([0.0848, 0.0, 0.0])),
        )
    model.lowerPositionLimit[:] = -1.0
    model.upperPositionLimit[:] = 1.0
    shoulder_index = model.joints[model.getJointId(ARM_JOINT_NAMES[0])].idx_q
    model.upperPositionLimit[shoulder_index] = 0.25
    if with_waist:
        model.lowerPositionLimit[0] = -0.5
        model.upperPositionLimit[0] = 0.5
    return model


def ik_config(**kwargs) -> TeleopIKConfig:
    defaults = dict(
        joint_names=list(ARM_JOINT_NAMES), motor_ids=list(range(14)),
        enable_local_log=False, log_to_console=False, show_model_load_progress=False,
    )
    return TeleopIKConfig(**{**defaults, **kwargs})


def make_core_solver(**kwargs) -> TeleopArmIK:
    model = make_model()
    with patch.object(pin.RobotWrapper, "BuildFromURDF", return_value=SimpleNamespace(model=model)), \
            patch.object(TeleopArmIK, "_build_reduced_model", return_value=model):
        return TeleopArmIK(ik_config(**kwargs))


def load_official_adapter():
    # Loading this module must not import the native CasADi backend or its committed .so.
    native_stub = ModuleType("tiangong_ik")
    native_stub.TiangongArmIK = object
    spec = importlib.util.spec_from_file_location(
        "joint_safety_official_adapter", SOURCE_DIR / "teleop_ik_ubtech_casadi.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"tiangong_ik": native_stub}):
        spec.loader.exec_module(module)
    return module


class JointLimitSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.official = load_official_adapter()

    def make_official_solver(self, **kwargs):
        model = make_model(with_waist=True)
        backend = SimpleNamespace(
            reduced_robot=SimpleNamespace(model=model),
            L_hand_id=model.getFrameId("L_ee"), R_hand_id=model.getFrameId("R_ee"),
            last_return_status="residual_converged", last_iteration_count=1,
        )
        backend.solve_ik = lambda **inputs: inputs["q_init"].copy()
        with tempfile.TemporaryDirectory() as directory:
            placeholder = Path(directory) / "unused.so"
            placeholder.touch()
            with patch.object(self.official, "OFFICIAL_SOLVER_SO_PATH", placeholder), \
                    patch.object(self.official, "TiangongArmIK", return_value=backend):
                solver = self.official.OfficialTiangong2ProIK(ik_config(**kwargs))
        return solver

    def test_wide_configuration_cannot_expand_either_models_hard_limits(self) -> None:
        core = make_core_solver(joint_limits=[(-6.0, 6.0)] * 14)
        official = self.make_official_solver(joint_limits=[(-6.0, 6.0)] * 15)
        self.assertEqual(core.joint_limits[0], (-1.0, 0.25))
        self.assertEqual(official.joint_limits[0], (-1.0, 0.25))
        self.assertEqual(official.waist_limit, (-0.5, 0.5))

    def test_narrow_configuration_and_single_value_freeze_are_preserved(self) -> None:
        limits = [(-6.0, 6.0)] * 14
        limits[0] = (-0.1, 0.1)
        limits[1] = (0.0, 0.0)
        solver = make_core_solver(joint_limits=limits)
        self.assertEqual(solver.joint_limits[0], (-0.1, 0.1))
        self.assertEqual(solver.joint_limits[1], (0.0, 0.0))

    def test_invalid_and_disjoint_configuration_is_rejected_by_both_constructors(self) -> None:
        for bad_limit in ((np.nan, 1.0), (-1.0, np.inf), (1.0, -1.0), (2.0, 3.0)):
            with self.subTest(limit=bad_limit, backend="core"):
                with self.assertRaises(ValueError):
                    make_core_solver(joint_limits=[bad_limit] + [(-6.0, 6.0)] * 13)
            with self.subTest(limit=bad_limit, backend="official"):
                with self.assertRaises(ValueError):
                    self.make_official_solver(joint_limits=[bad_limit] + [(-6.0, 6.0)] * 14)

    def test_direct_retarget_normal_command_path_cannot_reproduce_0266_radian_violation(self) -> None:
        model = make_model()
        with patch.object(pin.RobotWrapper, "BuildFromURDF", return_value=SimpleNamespace(model=model)), \
                patch.object(TeleopArmIK, "_build_reduced_model", return_value=model):
            retargeter = DirectRetargeter(DirectRetargetConfig(
                joint_names=list(ARM_JOINT_NAMES), motor_ids=list(range(14)), control_side="left",
                direction_iterations=1,
            ))
        baseline = np.zeros(14)
        retargeter.solver.set_home_q(baseline)
        retargeter.state.baseline_q = baseline.copy()
        retargeter.state.last_q_cmd = baseline.copy()
        retargeter.state.last_q_cmd[0] = 0.24
        retargeter.state.calibrated = True
        direction = np.array([0.0, 0.0, 1.0])
        arms = {
            side: SimpleNamespace(upper_direction_local=direction, forearm_direction_local=direction)
            for side in ("left", "right")
        }
        retargeter.state.home_arms = arms
        retargeter.state.robot_home_directions = {
            side: (direction, direction) for side in ("left", "right")
        }
        # An outward target would produce the previous 0.266 rad command with relaxed bounds.
        with patch("teleop_direct_retarget.skeleton_observation", return_value=(SimpleNamespace(arms=arms), "ok")), \
                patch("teleop_direct_retarget.observation_is_safe", return_value=(True, "ok")), \
                patch.object(retargeter.solver, "_build_segment_direction_task", return_value=(np.eye(1, 14), np.ones(1))):
            command = retargeter.command_from_frame({"receive_time": 1.0}, baseline)
        self.assertLessEqual(command.positions[0], 0.25)
        self.assertGreaterEqual(command.positions[0], 0.24)
        np.testing.assert_allclose(command.positions[7:], baseline[7:])
        self.assertTrue(command.meta["limited"])

    def test_position_only_final_command_obeys_the_urdf_intersection(self) -> None:
        solver = self.make_official_solver(
            joint_limits=[(-6.0, 6.0)] * 15, task_mode="position_only",
            max_iterations=1, max_joint_delta_per_step=0.04,
        )
        current = np.zeros(14)
        current[0] = 0.24
        left_target, right_target = solver.end_effector_poses(current)
        jacobian = np.zeros((1, 15))
        jacobian[0, solver.joint_indices[ARM_JOINT_NAMES[0]]] = 1.0
        with patch.object(solver, "_build_position_task", return_value=(jacobian, np.ones(1))):
            result = solver.solve(TeleopIKInput(
                timestamp=1.0, control_side="left", target_pose_l=left_target,
                target_pose_r=right_target, current_q=current, last_q_cmd=current,
                current_waist_position=0.0,
            ))
        self.assertLessEqual(result.command.positions[0], 0.25)
        np.testing.assert_allclose(result.command.positions[7:], current[7:])

    def test_pose_output_and_failure_hold_obey_urdf_limits(self) -> None:
        solver = self.make_official_solver(joint_limits=[(-6.0, 6.0)] * 15)
        left_target, right_target = solver.end_effector_poses(np.zeros(14))
        solver.solver.solve_ik = lambda **_inputs: np.full(15, 5.0)
        ik_input = TeleopIKInput(
            timestamp=1.0, control_side="both", target_pose_l=left_target,
            target_pose_r=right_target, current_q=np.zeros(14), last_q_cmd=np.full(14, 5.0),
            current_waist_position=0.0,
        )
        result = solver.solve(ik_input)
        command = result.command
        self.assertLessEqual(command.positions[0], 0.25)
        self.assertLessEqual(command.waist_position, 0.5)
        self.assertTrue(np.all(command.positions <= np.array(solver.joint_limits)[:, 1]))
        self.assertTrue(result.limited)
        solver.solver.solve_ik = lambda **_inputs: np.full(15, np.nan)
        held = solver.solve(ik_input)
        self.assertTrue(held.reason.startswith("solver_failed"))
        self.assertTrue(np.all(np.isfinite(held.command.positions)))
        self.assertLessEqual(held.command.positions[0], 0.25)

    def test_nonfinite_feedback_and_previous_command_are_rejected(self) -> None:
        solver = make_core_solver()
        for current, previous in ((np.full(14, np.nan), np.zeros(14)), (np.zeros(14), np.full(14, np.inf))):
            with self.subTest(current=current[0], previous=previous[0]):
                with self.assertRaises(ValueError):
                    solver.solve(TeleopIKInput(
                        timestamp=1.0, control_side="both", target_pose_l=np.eye(4),
                        target_pose_r=np.eye(4), current_q=current, last_q_cmd=previous,
                    ))

    def test_core_reports_the_residual_of_the_final_command(self) -> None:
        solver = make_core_solver(
            max_iterations=1, damping=0.0, max_joint_delta_per_step=1.0,
            smoothing_alpha=1.0, nullspace_gain=0.0,
        )
        with patch.object(solver, "_build_task", side_effect=lambda q, _task: (
            np.eye(1, 14), np.asarray([0.1 - q[0]])
        )):
            result = solver.solve(TeleopIKInput(
                timestamp=1.0, control_side="left", target_pose_l=np.eye(4),
                target_pose_r=np.eye(4), current_q=np.zeros(14), last_q_cmd=np.zeros(14),
            ))
        self.assertAlmostEqual(result.command.positions[0], 0.1)
        self.assertAlmostEqual(result.error_norm, 0.0)
        self.assertTrue(result.converged)

    def test_disjoint_stabilized_constraints_are_rejected(self) -> None:
        solver = make_core_solver(
            stabilized_joint_indices=(0,), stabilized_joint_delta_bounds=((-3.0, -2.0),)
        )
        left_target, right_target = solver.end_effector_poses(np.zeros(14))
        with self.assertRaises(ValueError):
            solver.solve(TeleopIKInput(
                timestamp=1.0, control_side="left", target_pose_l=left_target,
                target_pose_r=right_target, current_q=np.zeros(14),
            ))

    def test_direct_retarget_rejects_an_out_of_bounds_baseline_before_holding(self) -> None:
        retargeter = DirectRetargeter(
            DirectRetargetConfig(list(ARM_JOINT_NAMES), list(range(14))), solver=make_core_solver()
        )
        baseline = np.zeros(14)
        baseline[0] = 0.3
        with self.assertRaises(ValueError):
            retargeter.command_from_frame({}, baseline)


class JointSafetyParametersTest(unittest.TestCase):
    def test_finite_reference_is_required_even_when_candidate_also_failed(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    apply_joint_safety(np.array([invalid]), np.array([invalid]), [(-1.0, 1.0)], 0.04, 0.65)
        held, limited = apply_joint_safety(np.array([np.nan]), np.array([0.2]), [(-1.0, 1.0)], 0.04, 0.65)
        np.testing.assert_allclose(held, [0.2])
        self.assertTrue(limited)

    def test_nonfinite_motion_parameters_are_rejected(self) -> None:
        for field in ("damping", "max_joint_delta_per_step", "smoothing_alpha", "position_weight", "nullspace_gain"):
            for invalid in (np.nan, np.inf):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(ValueError):
                        ik_config(**{field: invalid})
        for field in ("position_scale", "max_joint_delta_per_step", "max_command_delta_from_baseline", "smoothing_alpha"):
            with self.subTest(direct_field=field):
                with self.assertRaises(ValueError):
                    DirectRetargetConfig(list(ARM_JOINT_NAMES), list(range(14)), **{field: np.nan})

    def test_shared_safety_rejects_invalid_parameters_and_allows_zero_freeze(self) -> None:
        for max_step, alpha, limits in (
            (np.nan, 0.65, [(-1.0, 1.0)]),
            (0.04, np.inf, [(-1.0, 1.0)]),
            (-0.04, 0.65, [(-1.0, 1.0)]),
            (0.04, 0.65, [(-1.0, np.nan)]),
        ):
            with self.subTest(max_step=max_step, alpha=alpha, limits=limits):
                with self.assertRaises(ValueError):
                    apply_joint_safety(np.array([0.5]), np.array([0.2]), limits, max_step, alpha)
        frozen, _ = apply_joint_safety(np.array([0.5]), np.array([0.2]), [(0.2, 0.2)], 0.0, 0.65)
        np.testing.assert_allclose(frozen, [0.2])


if __name__ == "__main__":
    unittest.main()
