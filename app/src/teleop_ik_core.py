from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin

from teleop_ik_types import IKCommand, TeleopIKConfig, TeleopIKInput, TeleopIKResult

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "walker_tienkung_ei_description"
_URDF_PATH = _MODEL_DIR / "walker_tienkung_ei.urdf"
_IDENTITY_3 = np.eye(3)
_ROTATION_TOLERANCE = 1e-6 + 1e-5 * _IDENTITY_3


class TeleopArmIK:
    """面向遥操作的局部增量 IK。

    这不是一次性全局最优求解器，而是控制环里的“小步修正器”。
    每帧只基于当前关节角、上一帧指令和当前目标位姿做少量 DLS 迭代，
    以换取更稳定的耗时和更连续的下发手感。
    """

    def __init__(self, config: TeleopIKConfig):
        self.config = config
        self.solve_frame_index = 0
        self.logger = self._init_logger()

        total_start = time.perf_counter()
        self._log_event(
            "model_load_start",
            {
                "urdf_path": str(_URDF_PATH),
                "model_dir": str(_MODEL_DIR),
            },
        )

        stage_start = time.perf_counter()
        self.robot = self._run_with_model_load_progress(
            "build_kinematic_model_from_urdf",
            lambda: pin.RobotWrapper(pin.buildModelFromUrdf(str(_URDF_PATH))),
        )
        self._log_event(
            "model_load_stage",
            {
                "stage": "build_kinematic_model_from_urdf",
                "cost_ms": elapsed_ms(stage_start),
                "full_nq": self.robot.model.nq,
                "full_nv": self.robot.model.nv,
            },
        )

        stage_start = time.perf_counter()
        self.model = self._build_reduced_model()
        self._log_event(
            "model_load_stage",
            {
                "stage": "build_reduced_arm_model",
                "cost_ms": elapsed_ms(stage_start),
                "reduced_nq": self.model.nq,
                "reduced_nv": self.model.nv,
            },
        )

        stage_start = time.perf_counter()
        self.data = self.model.createData()
        self.joint_limits = self._resolve_joint_limits()
        self.left_frame_id = self.model.getFrameId(config.left_frame_name)
        self.right_frame_id = self.model.getFrameId(config.right_frame_name)
        self.segment_joint_ids = {
            "left": (
                self.model.getJointId("shoulder_pitch_l_joint"),
                self.model.getJointId("elbow_pitch_l_joint"),
                self.model.getJointId("wrist_pitch_l_joint"),
            ),
            "right": (
                self.model.getJointId("shoulder_pitch_r_joint"),
                self.model.getJointId("elbow_pitch_r_joint"),
                self.model.getJointId("wrist_pitch_r_joint"),
            ),
        }
        if any(joint_id == 0 for ids in self.segment_joint_ids.values() for joint_id in ids):
            raise ValueError("reduced URDF 缺少肩/肘/腕骨段任务关节")
        if config.home_q is None:
            self.home_q = np.zeros(self.model.nq)
        else:
            self.home_q = ensure_vector(config.home_q, self.model.nq, "home_q")
        self.home_q = clip_to_limits(self.home_q, self.joint_limits)
        self._log_event(
            "model_load_stage",
            {
                "stage": "create_runtime_data",
                "cost_ms": elapsed_ms(stage_start),
                "left_frame_id": self.left_frame_id,
                "right_frame_id": self.right_frame_id,
            },
        )

        self._log_event(
            "model_load_done",
            {
                "total_cost_ms": elapsed_ms(total_start),
                "joint_count": self.model.nq,
                "left_frame_name": config.left_frame_name,
                "right_frame_name": config.right_frame_name,
            },
        )

    def set_home_q(self, home_q: np.ndarray) -> None:
        """把当前安全站姿设为 IK 零空间偏好姿态。"""

        self.home_q = clip_to_limits(ensure_vector(home_q, self.model.nq, "home_q"), self.joint_limits)

    def end_effector_poses(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """基于 reduced model 计算当前左右手腕 IK 目标帧。"""

        q_safe = clip_to_limits(ensure_vector(q, self.model.nq, "q"), self.joint_limits)
        pin.forwardKinematics(self.model, self.data, q_safe)
        pin.updateFramePlacements(self.model, self.data)
        return se3_to_matrix(self.data.oMf[self.left_frame_id]), se3_to_matrix(self.data.oMf[self.right_frame_id])

    def segment_directions(self, q: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        """返回机器人基座坐标中的上臂、前臂单位方向。"""
        q_safe = clip_to_limits(ensure_vector(q, self.model.nq, "q"), self.joint_limits)
        pin.forwardKinematics(self.model, self.data, q_safe)
        shoulder_id, elbow_id, wrist_id = self._segment_ids(side)
        shoulder = self.data.oMi[shoulder_id].translation
        elbow = self.data.oMi[elbow_id].translation
        wrist = self.data.oMi[wrist_id].translation
        upper, _ = normalized_direction(shoulder, elbow, f"{side} robot upper arm")
        forearm, _ = normalized_direction(elbow, wrist, f"{side} robot forearm")
        return upper, forearm

    def solve_segment_directions(
        self,
        *,
        timestamp: float,
        control_side: str,
        target_directions: dict[str, tuple[np.ndarray, np.ndarray]],
        current_q: np.ndarray,
        last_q_cmd: np.ndarray | None,
        upper_weight: float = 1.0,
        forearm_weight: float = 1.0,
        max_iterations: int | None = None,
        damping: float | None = None,
        target_orientations: dict[str, np.ndarray] | None = None,
        orientation_weight: float = 0.25,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """以骨段方向和可选手部姿态联合求解，不约束人体的绝对末端位置。"""
        current_raw = ensure_vector(current_q, self.model.nq, "current_q")
        command_raw = current_raw if last_q_cmd is None else ensure_vector(last_q_cmd, self.model.nq, "last_q_cmd")
        current = clip_to_limits(current_raw, self.joint_limits)
        command_ref = clip_to_limits(command_raw, self.joint_limits)
        q = current.copy()
        limited = not np.allclose(current, current_raw, atol=1e-8) or not np.allclose(command_ref, command_raw, atol=1e-8)
        iterations = self.config.max_iterations if max_iterations is None else max_iterations
        if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("max_iterations 必须为正整数")
        dls_damping = float(self.config.damping if damping is None else damping)
        weights = np.asarray([upper_weight, forearm_weight, dls_damping, orientation_weight], dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("骨段任务权重和阻尼必须为有限非负数")
        orientations = None if target_orientations is None else {
            side: ensure_rotation(target_orientations.get(side), f"{side} target orientation")
            for side in selected_control_sides(control_side)
        }
        if orientations is not None and self.model.nq == self.model.nv == 14:
            return self._solve_skeleton_tasks(
                timestamp=timestamp, control_side=control_side,
                directions={side: self._checked_target_directions(target_directions, side) for side in orientations},
                orientations=orientations, current=current, reference=command_ref,
                iterations=iterations, damping=dls_damping,
                upper_weight=float(upper_weight), forearm_weight=float(forearm_weight),
                orientation_weight=float(orientation_weight), initially_limited=limited,
            )
        error_norm = 0.0

        for iteration in range(iterations):
            jacobian, error = self._build_segment_direction_task(
                q,
                control_side,
                target_directions,
                upper_weight=float(upper_weight),
                forearm_weight=float(forearm_weight),
            )
            if orientations is not None and orientation_weight > 0.0:
                angular_jacobian, angular_error = self._build_segment_orientation_task(q, orientations)
                jacobian = np.vstack((jacobian, angular_jacobian * orientation_weight))
                error = np.concatenate((error, angular_error * orientation_weight))
            error_norm = float(np.linalg.norm(error))
            if jacobian.size == 0 or error_norm < 1e-5:
                break
            dq = compute_damped_increment(jacobian, error, dls_damping)
            dq += self.config.nullspace_gain * (self.home_q - q)
            dq = self._mask_control_side(dq, control_side)
            q_next = clip_to_limits(q + dq, self.joint_limits)
            limited = limited or not np.allclose(q_next, q + dq, atol=1e-8)
            q = q_next

        q, command_limited = apply_joint_safety(
            q_next=q,
            q_ref=command_ref,
            limits=self.joint_limits,
            max_step=self.config.max_joint_delta_per_step,
            smoothing_alpha=self.config.smoothing_alpha,
        )
        limited = limited or command_limited
        _, final_error = self._build_segment_direction_task(
            q,
            control_side,
            target_directions,
            upper_weight=float(upper_weight),
            forearm_weight=float(forearm_weight),
        )
        if orientations is not None and orientation_weight > 0.0:
            _, angular_error = self._build_segment_orientation_task(q, orientations)
            final_error = np.concatenate((final_error, angular_error * orientation_weight))
        error_norm = float(np.linalg.norm(final_error))

        residual_by_side: dict[str, dict[str, float]] = {}
        for side in selected_control_sides(control_side):
            target_upper, target_forearm = self._checked_target_directions(target_directions, side)
            actual_upper, actual_forearm = self.segment_directions(q, side)
            residual_by_side[side] = {
                "upper_angle_rad": vector_angle(actual_upper, target_upper),
                "forearm_angle_rad": vector_angle(actual_forearm, target_forearm),
            }
            if orientations is not None:
                frame_id = self.left_frame_id if side == "left" else self.right_frame_id
                pin.updateFramePlacements(self.model, self.data)
                residual_by_side[side]["orientation_angle_rad"] = float(np.linalg.norm(
                    pin.log3(orientations[side] @ self.data.oMf[frame_id].rotation.T)
                ))
        return q, {
            "solver": "urdf_segment_direction_dls",
            "timestamp": float(timestamp),
            "iterations": iteration + 1,
            "error_norm": error_norm,
            "limited": limited,
            "residual_by_side": residual_by_side,
            "upper_weight": float(upper_weight),
            "forearm_weight": float(forearm_weight),
            "damping": dls_damping,
            "orientation_weight": float(orientation_weight),
        }

    def _solve_skeleton_tasks(
        self, *, timestamp: float, control_side: str,
        directions: dict[str, tuple[np.ndarray, np.ndarray]], orientations: dict[str, np.ndarray],
        current: np.ndarray, reference: np.ndarray, iterations: int, damping: float,
        upper_weight: float, forearm_weight: float, orientation_weight: float, initially_limited: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Independent 7-axis DLS blocks, with one kinematic update per iteration.

        The reduced walker has a fixed base/waist and disjoint arm chains.
        Solving its two 9x7 tasks independently is algebraically the same damped
        least-squares problem as the stacked 18x14 task. Inputs are checked once
        by the public entry point; the inner loop only handles model output.
        """
        lower, upper = joint_limit_arrays(self.joint_limits, 14)
        sides = tuple(directions)
        columns = {side: slice(0, 7) if side == "left" else slice(7, 14) for side in sides}
        task_jacobians = np.empty((len(sides), 9, 7))
        task_errors = np.empty((len(sides), 9))
        q = current.copy()
        limited = initially_limited
        for iteration in range(iterations):
            # computeJointJacobians also updates oMi; getFrameJacobian reuses J.
            pin.computeJointJacobians(self.model, self.data, q)
            error_squared = 0.0
            for arm_index, side in enumerate(sides):
                column = columns[side]
                jacobian, error = task_jacobians[arm_index], task_errors[arm_index]
                shoulder_id, elbow_id, wrist_id = self._segment_ids(side)
                ids = (shoulder_id, elbow_id, wrist_id)
                positions = [self.data.oMi[joint_id].translation for joint_id in ids]
                translation_jacobians = [pin.getJointJacobian(
                    self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )[:3, column] for joint_id in ids]
                for row, weight, target, first, second in (
                    (0, upper_weight, directions[side][0], 0, 1),
                    (3, forearm_weight, directions[side][1], 1, 2),
                ):
                    vector = positions[second] - positions[first]
                    length = float(np.linalg.norm(vector))
                    if length <= 1e-9 or not np.isfinite(length):
                        raise ValueError("robot segment has invalid length")
                    direction = vector / length
                    delta_jacobian = translation_jacobians[second] - translation_jacobians[first]
                    jacobian[row:row + 3] = (delta_jacobian - np.outer(direction, direction @ delta_jacobian)) * (weight / length)
                    error[row:row + 3] = (target - direction) * weight
                frame_id = self.left_frame_id if side == "left" else self.right_frame_id
                pin.updateFramePlacement(self.model, self.data, frame_id)
                jacobian[6:] = pin.getFrameJacobian(
                    self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )[3:, column] * orientation_weight
                error[6:] = clip_vector_norm(
                    pin.log3(orientations[side] @ self.data.oMf[frame_id].rotation.T), self.config.max_orientation_error
                ) * orientation_weight
                error_squared += float(error @ error)
            if error_squared < 1e-10:
                break
            dq = np.zeros(14)
            increments = compute_joint_space_increment(task_jacobians, task_errors, damping)
            for arm_index, side in enumerate(sides):
                column = columns[side]
                dq[column] = increments[arm_index]
                dq[column] += self.config.nullspace_gain * (self.home_q[column] - q[column])
            raw_next = q + dq
            limited = limited or bool(np.any(raw_next < lower) or np.any(raw_next > upper))
            q = np.clip(raw_next, lower, upper)
        q, command_limited = apply_joint_safety(
            q, reference, self.joint_limits, self.config.max_joint_delta_per_step, self.config.smoothing_alpha,
        )
        # Only the selected arm may be commanded. The other half stays at the
        # actual feedback even if a caller supplies a different last command.
        if control_side == "left":
            q[7:] = current[7:]
        elif control_side == "right":
            q[:7] = current[:7]
        state = self.segment_state(q, control_side)
        residuals = {}
        final_error_squared = 0.0
        for side in sides:
            actual = state[side]
            angular_error = pin.log3(orientations[side] @ actual["orientation"].T)
            residuals[side] = {
                "upper_angle_rad": float(np.arccos(np.clip(actual["upper"] @ directions[side][0], -1., 1.))),
                "forearm_angle_rad": float(np.arccos(np.clip(actual["forearm"] @ directions[side][1], -1., 1.))),
                "orientation_angle_rad": float(np.linalg.norm(angular_error)),
            }
            for key, target, weight in (("upper", directions[side][0], upper_weight), ("forearm", directions[side][1], forearm_weight)):
                error = (target - actual[key]) * weight
                final_error_squared += float(error @ error)
            weighted_angular = clip_vector_norm(angular_error, self.config.max_orientation_error) * orientation_weight
            final_error_squared += float(weighted_angular @ weighted_angular)
        return q, {
            "solver": "urdf_segment_direction_dls", "timestamp": float(timestamp),
            "iterations": iteration + 1, "error_norm": float(np.sqrt(final_error_squared)),
            "limited": bool(limited or command_limited), "residual_by_side": residuals,
            "upper_weight": upper_weight, "forearm_weight": forearm_weight,
            "damping": damping, "orientation_weight": orientation_weight,
        }

    def segment_state(self, q: np.ndarray, control_side: str = "both") -> dict[str, dict[str, np.ndarray]]:
        """Read directions and hand orientations using a single final FK."""
        q_safe = clip_to_limits(ensure_vector(q, self.model.nq, "q"), self.joint_limits)
        pin.forwardKinematics(self.model, self.data, q_safe)
        result = {}
        for side in selected_control_sides(control_side):
            shoulder_id, elbow_id, wrist_id = self._segment_ids(side)
            upper, _ = normalized_direction(self.data.oMi[shoulder_id].translation, self.data.oMi[elbow_id].translation, "upper")
            forearm, _ = normalized_direction(self.data.oMi[elbow_id].translation, self.data.oMi[wrist_id].translation, "forearm")
            frame_id = self.left_frame_id if side == "left" else self.right_frame_id
            pin.updateFramePlacement(self.model, self.data, frame_id)
            result[side] = {"upper": upper, "forearm": forearm, "orientation": self.data.oMf[frame_id].rotation.copy()}
        return result

    def _build_segment_orientation_task(
        self, q: np.ndarray, targets: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        blocks: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        for side, target in targets.items():
            frame_id = self.left_frame_id if side == "left" else self.right_frame_id
            jacobian = pin.computeFrameJacobian(
                self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            blocks.append(jacobian[3:])
            errors.append(clip_vector_norm(
                pin.log3(target @ self.data.oMf[frame_id].rotation.T), self.config.max_orientation_error
            ))
        return np.vstack(blocks), np.concatenate(errors)

    def _segment_ids(self, side: str) -> tuple[int, int, int]:
        try:
            return self.segment_joint_ids[side]
        except KeyError as exc:
            raise ValueError("side 必须为 left 或 right") from exc

    def _checked_target_directions(
        self,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if side not in targets:
            raise ValueError(f"缺少 {side} 骨段方向目标")
        upper, forearm = targets[side]
        return normalize_direction(upper, f"{side} target upper"), normalize_direction(forearm, f"{side} target forearm")

    def _build_segment_direction_task(
        self,
        q: np.ndarray,
        control_side: str,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        upper_weight: float,
        forearm_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        blocks: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        for side in selected_control_sides(control_side):
            target_upper, target_forearm = self._checked_target_directions(targets, side)
            shoulder_id, elbow_id, wrist_id = self._segment_ids(side)
            upper_jacobian, upper_direction = self._direction_jacobian(q, shoulder_id, elbow_id)
            forearm_jacobian, forearm_direction = self._direction_jacobian(q, elbow_id, wrist_id)
            blocks.extend((upper_jacobian * upper_weight, forearm_jacobian * forearm_weight))
            errors.extend(((target_upper - upper_direction) * upper_weight, (target_forearm - forearm_direction) * forearm_weight))
        if not blocks:
            return np.empty((0, self.model.nv)), np.empty((0,))
        return np.vstack(blocks), np.concatenate(errors)

    def _direction_jacobian(
        self,
        q: np.ndarray,
        start_joint_id: int,
        end_joint_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = self.data.oMi[start_joint_id].translation
        end = self.data.oMi[end_joint_id].translation
        direction, length = normalized_direction(start, end, "robot segment")
        start_jacobian = pin.getJointJacobian(
            self.model, self.data, start_joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3]
        end_jacobian = pin.getJointJacobian(
            self.model, self.data, end_joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3]
        projection = np.eye(3) - np.outer(direction, direction)
        return projection @ (end_jacobian - start_jacobian) / length, direction

    def solve(self, ik_input: TeleopIKInput) -> TeleopIKResult:
        """求解单帧遥操作命令。

        输入：`TeleopIKInput`，包含目标左右手位姿、当前 14 维关节角、上一帧指令。
        输出：`TeleopIKResult.command.positions`，始终是完整 14 维关节命令。
        """

        solve_start = time.perf_counter()
        self.solve_frame_index += 1
        current_q_raw = ensure_vector(ik_input.current_q, self.model.nq, "current_q")
        current_q = clip_to_limits(current_q_raw, self.joint_limits)
        command_ref_raw = current_q if ik_input.last_q_cmd is None else ensure_vector(ik_input.last_q_cmd, self.model.nq, "last_q_cmd")
        command_ref = clip_to_limits(command_ref_raw, self.joint_limits)

        # `current_q` 是真实机器人/仿真的反馈状态，IK 必须从这里线性化；
        # `command_ref` 只约束本帧下发相对上一帧指令的小步连续性，避免反馈滞后时产生命令跳变。
        q = self._lock_configured_joints(current_q.copy(), ik_input.control_side)
        command_ref = self._lock_configured_joints(command_ref, ik_input.control_side)
        limited = (
            not np.allclose(current_q, current_q_raw, atol=1e-8)
            or not np.allclose(command_ref, command_ref_raw, atol=1e-8)
        )
        feedback_command_error = float(np.linalg.norm(current_q - command_ref))
        error_norm = 0.0

        for iteration in range(max(1, self.config.max_iterations)):
            task_jacobian, task_error = self._build_task(q, ik_input)
            error_norm = float(np.linalg.norm(task_error))
            if task_jacobian.size == 0 or error_norm < 1e-5:
                break

            dq = compute_damped_increment(task_jacobian, task_error, self.config.damping)
            dq += self.config.nullspace_gain * (self.home_q - q)
            dq += stabilized_joint_increment(
                q=q,
                home_q=self.home_q,
                indices=self.config.stabilized_joint_indices,
                gain=self.config.stabilized_joint_gain,
            )
            dq = self._mask_control_side(dq, ik_input.control_side)

            q_next = q + dq
            q, step_limited = apply_joint_safety(
                q_next=q_next,
                q_ref=command_ref,
                limits=self.joint_limits,
                max_step=self.config.max_joint_delta_per_step,
                smoothing_alpha=self.config.smoothing_alpha,
            )
            q = self._lock_configured_joints(q, ik_input.control_side)
            limited = limited or step_limited

        q = constrain_stabilized_joints(
            q=q,
            home_q=self.home_q,
            limits=self.joint_limits,
            indices=self.config.stabilized_joint_indices,
            max_delta=self.config.stabilized_joint_max_delta,
            max_deltas=self.config.stabilized_joint_max_deltas,
            delta_bounds=self.config.stabilized_joint_delta_bounds,
        )
        q = clip_to_limits(self._lock_configured_joints(q, ik_input.control_side), self.joint_limits)
        _, final_error = self._build_task(q, ik_input)
        error_norm = float(np.linalg.norm(final_error))

        command = IKCommand(
            timestamp=ik_input.timestamp,
            control_side=ik_input.control_side,
            joint_names=self.config.joint_names,
            positions=q,
            motor_ids=self.config.motor_ids,
            meta={
                "solver": "teleop_dls",
                "task_mode": self.config.task_mode,
                "error_norm": error_norm,
                "limited": limited,
                "iterations": iteration + 1,
                "feedback_command_error": feedback_command_error,
                "joint_limits_source": "urdf" if self.config.joint_limits is None else "urdf_config_intersection",
            },
        )
        result = TeleopIKResult(
            command=command,
            converged=error_norm < self.config.position_convergence_threshold,
            error_norm=error_norm,
            iteration_count=iteration + 1,
            limited=limited,
        )
        self._log_ik_result(ik_input, result, elapsed_ms(solve_start))
        return result

    def _resolve_joint_limits(self) -> list[tuple[float, float]]:
        """从 reduced URDF 模型读取 14 维关节限位。

        q 的顺序由 Pinocchio reduced model 决定，因此限位也必须使用同一个模型顺序，
        避免入口脚本手写数组和真实机器人模型逐渐漂移。
        """

        return resolve_joint_limits(self.model, self.config.joint_limits)

    def _build_reduced_model(self):
        joints_to_lock = [
            "head_roll_joint", "head_pitch_joint", "head_yaw_joint", "waist_yaw_joint",
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
        ]
        reduced_model = pin.buildReducedModel(
            self.robot.model,
            [self.robot.model.getJointId(name) for name in joints_to_lock],
            np.zeros(self.robot.model.nq),
        )
        add_operational_frame(
            reduced_model,
            frame_name="L_ee",
            parent_joint_name="wrist_roll_l_joint",
            placement=pin.SE3(np.eye(3), np.array([0.025, -0.07, -0.21])),
        )
        add_operational_frame(
            reduced_model,
            frame_name="R_ee",
            parent_joint_name="wrist_roll_r_joint",
            placement=pin.SE3(np.eye(3), np.array([0.025, 0.07, -0.21])),
        )
        return reduced_model

    def _build_task(self, q: np.ndarray, ik_input: TeleopIKInput) -> tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        blocks = []
        errors = []
        if ik_input.control_side in {"left", "both"}:
            jacobian, error = self._frame_task(q, self.left_frame_id, ik_input.target_pose_l)
            blocks.append(jacobian)
            errors.append(error)
        if ik_input.control_side in {"right", "both"}:
            jacobian, error = self._frame_task(q, self.right_frame_id, ik_input.target_pose_r)
            blocks.append(jacobian)
            errors.append(error)

        if not blocks:
            return np.empty((0, self.model.nv)), np.empty((0,))
        task_jacobian = np.vstack(blocks)
        for index in self._selected_locked_joint_indices(ik_input.control_side):
            task_jacobian[:, index] = 0.0
        return task_jacobian, np.concatenate(errors)

    def _selected_locked_joint_indices(self, control_side: str) -> tuple[int, ...]:
        selected: list[int] = []
        for index in self.config.locked_joint_indices:
            if index < 0 or index >= self.model.nq:
                raise ValueError(f"locked_joint_indices 包含非法关节索引: {index}")
            if (index < 7 and control_side in {"left", "both"}) or (
                index >= 7 and control_side in {"right", "both"}
            ):
                selected.append(index)
        return tuple(selected)

    def _lock_configured_joints(self, q: np.ndarray, control_side: str) -> np.ndarray:
        locked = np.asarray(q, dtype=float).copy()
        for index in self._selected_locked_joint_indices(control_side):
            locked[index] = self.home_q[index]
        return locked

    def _frame_task(self, q: np.ndarray, frame_id: int, target_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target_pose = ensure_pose(target_pose, "target_pose")
        current_pose = self.data.oMf[frame_id]
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        if self.config.task_mode == "position_only":
            position_error = compute_position_error(current_pose, target_pose, self.config)
            return jacobian[:3] * self.config.position_weight, position_error * self.config.position_weight

        error = compute_pose_error(current_pose, target_pose, self.config)
        weights = np.array(
            [
                self.config.position_weight,
                self.config.position_weight,
                self.config.position_weight,
                self.config.orientation_weight,
                self.config.orientation_weight,
                self.config.orientation_weight,
            ],
            dtype=float,
        )
        return jacobian * weights[:, None], error * weights

    def _init_logger(self) -> logging.Logger | None:
        if not self.config.enable_local_log:
            return None

        log_dir = Path(self.config.log_dir) if self.config.log_dir else Path(__file__).resolve().parents[1] / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / self.config.log_file_name

        logger_name = f"teleop_ik_core.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)

        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(console_handler)

        logger.info(json.dumps({"section": "local_log_ready", "path": str(log_path)}, ensure_ascii=False))
        return logger

    def _run_with_model_load_progress(self, stage: str, action):
        """为阻塞式 URDF 加载提供终端心跳。

        Pinocchio 构建完整 URDF 时可能几十秒没有返回。这里不改变加载逻辑，
        只在旁路线程里输出 elapsed 时间，避免操作者误判程序卡死。
        """

        if not self.config.show_model_load_progress:
            return action()

        interval = max(1.0, float(self.config.model_load_progress_interval_seconds))
        start_time = time.perf_counter()
        stop_event = threading.Event()

        def report_progress():
            while not stop_event.wait(interval):
                cost_ms = elapsed_ms(start_time)
                message = f"[TeleopArmIK] {stage} 正在加载，已耗时 {cost_ms / 1000.0:.1f}s"
                print(message, flush=True)
                self._log_event(
                    "model_load_progress",
                    {
                        "stage": stage,
                        "elapsed_ms": cost_ms,
                    },
                )

        print(f"[TeleopArmIK] {stage} 开始加载模型...", flush=True)
        progress_thread = threading.Thread(target=report_progress, daemon=True)
        progress_thread.start()
        try:
            return action()
        finally:
            stop_event.set()
            progress_thread.join(timeout=0.2)
            cost_ms = elapsed_ms(start_time)
            print(f"[TeleopArmIK] {stage} 加载完成，总耗时 {cost_ms / 1000.0:.1f}s", flush=True)
            self._log_event(
                "model_load_progress_done",
                {
                    "stage": stage,
                    "elapsed_ms": cost_ms,
                },
            )

    def _log_event(self, section: str, payload: dict[str, Any]):
        if self.logger is None:
            return
        self.logger.info(
            json.dumps(
                {
                    "section": section,
                    "time": time.time(),
                    "data": to_jsonable(payload),
                },
                ensure_ascii=False,
            )
        )

    def _log_ik_result(self, ik_input: TeleopIKInput, result: TeleopIKResult, cost_ms: float):
        if not self.config.log_ik_result:
            return
        every_n = max(1, int(self.config.log_every_n_frames))
        if self.solve_frame_index % every_n != 0:
            return

        self._log_event(
            "ik_result",
            {
                "frame": self.solve_frame_index,
                "cost_ms": cost_ms,
                "timestamp": ik_input.timestamp,
                "control_side": ik_input.control_side,
                "current_q": ik_input.current_q,
                "last_q_cmd": ik_input.last_q_cmd,
                "target_pose_l": ik_input.target_pose_l,
                "target_pose_r": ik_input.target_pose_r,
                "positions": result.command.positions,
                "meta": result.command.meta,
                "converged": result.converged,
                "error_norm": result.error_norm,
                "iteration_count": result.iteration_count,
                "limited": result.limited,
                "reason": result.reason,
            },
        )

    @staticmethod
    def _mask_control_side(dq: np.ndarray, control_side: str) -> np.ndarray:
        """单臂遥操作时冻结非控制侧，防止另一只手被局部最小二乘牵连。"""

        masked = dq.copy()
        if control_side == "left":
            masked[7:] = 0.0
        elif control_side == "right":
            masked[:7] = 0.0
        return masked


def selected_control_sides(control_side: str) -> tuple[str, ...]:
    if control_side == "left":
        return ("left",)
    if control_side == "right":
        return ("right",)
    if control_side == "both":
        return ("left", "right")
    raise ValueError(f"Unsupported control_side: {control_side}")


def normalize_direction(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须是 3 维有限向量")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} 长度过小")
    return vector / norm


def normalized_direction(start: np.ndarray, end: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise ValueError(f"{name} 长度过小")
    return vector / length, length


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(normalize_direction(a, "a"), normalize_direction(b, "b")), -1.0, 1.0)))


def se3_to_matrix(value: pin.SE3) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = value.rotation
    transform[:3, 3] = value.translation
    return transform


def compute_position_error(current_pose: pin.SE3, target_pose: np.ndarray, config: TeleopIKConfig) -> np.ndarray:
    target_position = np.asarray(target_pose[:3, 3], dtype=float)
    position_error = target_position - current_pose.translation
    return clip_vector_norm(position_error, config.max_position_error)


def compute_pose_error(current_pose: pin.SE3, target_pose: np.ndarray, config: TeleopIKConfig) -> np.ndarray:
    """计算末端位姿误差。

    输出 6 维误差：[位置误差 xyz, 姿态误差 xyz]。误差会先限幅，避免动捕瞬时跳变直接变成过大的关节步长。
    """

    target_position = np.asarray(target_pose[:3, 3], dtype=float)
    target_rotation = np.asarray(target_pose[:3, :3], dtype=float)
    position_error = target_position - current_pose.translation
    orientation_error = pin.log3(target_rotation @ current_pose.rotation.T)
    return np.concatenate(
        [
            clip_vector_norm(position_error, config.max_position_error),
            clip_vector_norm(orientation_error, config.max_orientation_error),
        ]
    )


def compute_damped_increment(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """Damped Least Squares 增量解。

    DLS 比普通伪逆更适合遥操作：接近奇异位形时不会突然给出很大的关节速度。
    """

    damping_matrix = (damping ** 2) * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_matrix, error)


def compute_joint_space_increment(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """Equivalent DLS in the smaller joint space; zero damping uses least squares."""
    if damping == 0.0:
        if jacobian.ndim == 3:
            return np.asarray([np.linalg.lstsq(block, rhs, rcond=None)[0] for block, rhs in zip(jacobian, error)])
        return np.linalg.lstsq(jacobian, error, rcond=None)[0]
    transpose = np.swapaxes(jacobian, -2, -1)
    hessian = transpose @ jacobian
    diagonal = np.arange(hessian.shape[-1])
    hessian[..., diagonal, diagonal] += damping * damping
    return np.linalg.solve(hessian, transpose @ error[..., None])[..., 0]


def stabilized_joint_increment(
    q: np.ndarray,
    home_q: np.ndarray,
    indices: tuple[int, ...],
    gain: float,
) -> np.ndarray:
    increment = np.zeros_like(q)
    if not indices:
        return increment
    clipped_gain = float(np.clip(gain, 0.0, 1.0))
    for index in indices:
        increment[index] = clipped_gain * (home_q[index] - q[index])
    return increment


def constrain_stabilized_joints(
    q: np.ndarray,
    home_q: np.ndarray,
    limits: list[tuple[float, float]],
    indices: tuple[int, ...],
    max_delta: float | None,
    max_deltas: tuple[float, ...] | None = None,
    delta_bounds: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    if not indices:
        return q
    if delta_bounds is None and max_deltas is None and max_delta is None:
        return q
    if max_deltas is not None and len(max_deltas) != len(indices):
        raise ValueError("stabilized_joint_max_deltas 长度必须与 stabilized_joint_indices 一致")
    if delta_bounds is not None and len(delta_bounds) != len(indices):
        raise ValueError("stabilized_joint_delta_bounds 长度必须与 stabilized_joint_indices 一致")

    constrained = q.copy()
    for order, index in enumerate(indices):
        if delta_bounds is not None:
            lower_delta, upper_delta = delta_bounds[order]
            lower_delta = float(lower_delta)
            upper_delta = float(upper_delta)
            if lower_delta > upper_delta:
                raise ValueError("stabilized_joint_delta_bounds 下界不能大于上界")
        else:
            allowed_delta = abs(float(max_deltas[order] if max_deltas is not None else max_delta))
            lower_delta = -allowed_delta
            upper_delta = allowed_delta

        lower, upper = limits[index]
        constrained_lower = max(lower, home_q[index] + lower_delta)
        constrained_upper = min(upper, home_q[index] + upper_delta)
        if not np.isfinite(lower_delta) or not np.isfinite(upper_delta) or constrained_lower > constrained_upper:
            raise ValueError("稳定关节偏移与关节硬限位没有有效交集")
        constrained[index] = np.clip(constrained[index], constrained_lower, constrained_upper)
    return constrained


def apply_joint_safety(
    q_next: np.ndarray,
    q_ref: np.ndarray,
    limits: list[tuple[float, float]],
    max_step: float,
    smoothing_alpha: float,
) -> tuple[np.ndarray, bool]:
    """安全层：限位、单步速度限制和平滑。

    遥操作不是追求单帧到位，而是追求连续可控；因此这里以 `q_ref` 为中心限制每帧最大变化。
    """

    q_ref = ensure_vector(q_ref, len(limits), "q_ref")
    q_next = np.asarray(q_next, dtype=float).reshape(-1)
    if q_next.shape != q_ref.shape:
        raise ValueError("q_next 和 q_ref 的维度必须一致")
    max_step = float(max_step)
    alpha = float(smoothing_alpha)
    if not np.isfinite(max_step) or max_step < 0.0:
        raise ValueError("max_step 必须为有限非负数")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("smoothing_alpha 必须为 [0, 1] 内的有限数")
    lower, upper = joint_limit_arrays(limits, q_ref.size)
    q_ref_safe = np.clip(q_ref, lower, upper)
    if not np.all(np.isfinite(q_next)):
        return q_ref_safe, True

    q_limited = np.clip(q_next, lower, upper)
    q_step_limited = np.clip(q_limited, q_ref_safe - max_step, q_ref_safe + max_step)
    q_step_limited = np.clip(q_step_limited, lower, upper)
    q_smooth = alpha * q_step_limited + (1.0 - alpha) * q_ref_safe
    q_safe = np.clip(q_smooth, lower, upper)
    limited = not np.allclose(q_safe, q_next, atol=1e-8) or not np.allclose(q_ref_safe, q_ref, atol=1e-8)
    return q_safe, limited


def clip_to_limits(q: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    values = ensure_vector(q, len(limits), "q")
    lower, upper = joint_limit_arrays(limits, values.size)
    return np.clip(values, lower, upper)


def joint_limit_arrays(limits: list[tuple[float, float]], size: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(limits, dtype=float)
    if bounds.shape != (size, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"joint_limits 必须包含 {size} 组有限上下界")
    lower, upper = bounds[:, 0], bounds[:, 1]
    if np.any(lower > upper):
        raise ValueError("joint_limits 下界不能大于上界")
    return lower, upper


def resolve_joint_limits(model: pin.Model, configured_limits: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    lower = ensure_vector(model.lowerPositionLimit, model.nq, "URDF lowerPositionLimit")
    upper = ensure_vector(model.upperPositionLimit, model.nq, "URDF upperPositionLimit")
    if np.any(lower >= upper):
        raise ValueError("URDF 关节限位上下界非法")
    if configured_limits is not None:
        configured_lower, configured_upper = joint_limit_arrays(configured_limits, model.nq)
        lower = np.maximum(lower, configured_lower)
        upper = np.minimum(upper, configured_upper)
        if np.any(lower > upper):
            raise ValueError("配置关节限位与 URDF 硬限位没有交集")
    return [(float(item_lower), float(item_upper)) for item_lower, item_upper in zip(lower, upper)]


def add_operational_frame(model: pin.Model, frame_name: str, parent_joint_name: str, placement: pin.SE3) -> None:
    """兼容不同 Pinocchio 版本的 OP_FRAME 构造签名。"""

    parent_joint_id = model.getJointId(parent_joint_name)
    try:
        frame = pin.Frame(frame_name, parent_joint_id, placement, pin.FrameType.OP_FRAME)
    except Exception:
        parent_frame_id = model.getFrameId(parent_joint_name) if model.existFrame(parent_joint_name) else 0
        frame = pin.Frame(frame_name, parent_joint_id, parent_frame_id, placement, pin.FrameType.OP_FRAME)
    model.addFrame(frame)


def elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000.0, 3)


def to_jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def clip_vector_norm(value: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= max_norm or norm < 1e-12:
        return value
    return value * (max_norm / norm)


def ensure_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape[0] != size:
        raise ValueError(f"{name} 维度错误: got {vector.shape[0]}, expected {size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含 NaN/Inf")
    return vector


def ensure_rotation(value: np.ndarray, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} 必须是 3x3 有限旋转矩阵")
    if np.any(np.abs(rotation @ rotation.T - _IDENTITY_3) > _ROTATION_TOLERANCE) or abs(np.linalg.det(rotation) - 1.0) > 1.1e-5:
        raise ValueError(f"{name} 必须是右手正交旋转矩阵")
    return rotation.copy()


def ensure_pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} 必须是 4x4 有限变换矩阵")
    return pose
