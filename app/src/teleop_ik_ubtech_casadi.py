from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from teleop_ik_types import IKCommand, TeleopIKConfig, TeleopIKInput, TeleopIKResult
from teleop_ik_core import clip_to_limits, ensure_pose, resolve_joint_limits

SOURCE_DIR = Path(__file__).resolve().parent
OFFICIAL_PACKAGE_ROOT = (
    SOURCE_DIR.parent.parent
    / "third_party"
    / "tiangong2pro_ik"
    / "tiangong2pro_ik"
)
OFFICIAL_URDF_PATH = OFFICIAL_PACKAGE_ROOT / "tienkung2_pro" / "urdf" / "tiangong2.0_pro.urdf"
OFFICIAL_MESH_PATH = OFFICIAL_URDF_PATH.parent
OFFICIAL_SOLVER_SO_PATH = OFFICIAL_PACKAGE_ROOT / ".casadi_cache" / "ik_solver.so"

if not OFFICIAL_PACKAGE_ROOT.is_dir():
    raise FileNotFoundError(f"官方天工 2 Pro IK 目录不存在: {OFFICIAL_PACKAGE_ROOT}")
if str(OFFICIAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_PACKAGE_ROOT))

from tiangong_ik import TiangongArmIK  # noqa: E402


WAIST_JOINT_NAME = "body_yaw_joint"
ARM_JOINT_NAMES = (
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)


class OfficialTiangong2ProIK:
    """官方天工 2 Pro IK 的遥操作适配器。

    官方求解器使用 `body_yaw_joint + 双臂 14 轴` 的 15 维模型。本适配器只把
    原有接口需要的 14 维手臂角度放回 `IKCommand.positions`，腰部角度通过新增
    的可选 `IKCommand.waist_position` 交给 ROS 适配层发布，因此旧调用方仍兼容。

    对外末端统一使用 URDF 的 `left_tcp_link/right_tcp_link`。旧官方 `.so` 内部仍约束
    腕滚关节原点 `L_ee/R_ee`，调用前通过固定的 wrist-to-TCP 变换换算目标，避免
    手掌长度缺失后腕部旋转成为近似自由度。
    """

    controls_waist = True

    def __init__(self, config: TeleopIKConfig):
        self.config = config
        self.solve_frame_index = 0
        if not OFFICIAL_SOLVER_SO_PATH.is_file():
            raise FileNotFoundError(
                "缺少官方 IK 预编译求解器: "
                f"{OFFICIAL_SOLVER_SO_PATH}；请先运行 {OFFICIAL_PACKAGE_ROOT / 'gen_so.py'}"
            )
        self.solver = TiangongArmIK(
            urdf_path=str(OFFICIAL_URDF_PATH),
            mesh_path=str(OFFICIAL_MESH_PATH),
            jit=False,
            so_path=str(OFFICIAL_SOLVER_SO_PATH),
        )
        self.solver_mode = "external_so"
        self.model = self.solver.reduced_robot.model
        self.data = self.model.createData()
        self.joint_indices = self._resolve_joint_indices()
        self.left_solver_frame_id = self.solver.L_hand_id
        self.right_solver_frame_id = self.solver.R_hand_id
        self.left_frame_id = self._require_frame("left_tcp_link")
        self.right_frame_id = self._require_frame("right_tcp_link")
        self.left_solver_to_tcp = self._solver_to_tcp_transform(
            self.left_solver_frame_id,
            self.left_frame_id,
            "left",
        )
        self.right_solver_to_tcp = self._solver_to_tcp_transform(
            self.right_solver_frame_id,
            self.right_frame_id,
            "right",
        )
        self.full_joint_limits = self._resolve_limits()
        self.waist_limit = self.full_joint_limits[self.joint_indices[WAIST_JOINT_NAME]]
        self.joint_limits = [self.full_joint_limits[self.joint_indices[name]] for name in ARM_JOINT_NAMES]
        self.home_q = np.zeros(self.model.nq)
        self.reference_q = np.zeros(self.model.nq)
        self.last_waist_position = 0.0
        self.set_home_q(np.zeros(len(ARM_JOINT_NAMES)) if config.home_q is None else config.home_q)

    def _resolve_joint_indices(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        for joint_name in (WAIST_JOINT_NAME, *ARM_JOINT_NAMES):
            joint_id = self.model.getJointId(joint_name)
            if joint_id == 0:
                raise ValueError(f"官方 URDF 缺少关节: {joint_name}")
            indices[joint_name] = self.model.joints[joint_id].idx_q
        if self.model.nq != 15:
            raise ValueError(f"官方天工 2 Pro reduced model 应为 15 维，实际为 {self.model.nq}")
        return indices

    def _require_frame(self, frame_name: str) -> int:
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"官方 URDF 缺少末端 Frame: {frame_name}")
        return frame_id

    def _solver_to_tcp_transform(
        self,
        solver_frame_id: int,
        tcp_frame_id: int,
        side: str,
    ) -> pin.SE3:
        solver_frame = self.model.frames[solver_frame_id]
        tcp_frame = self.model.frames[tcp_frame_id]
        if solver_frame.parentJoint != tcp_frame.parentJoint:
            raise ValueError(f"{side} 官方求解 Frame 与 TCP Frame 不在同一腕关节")
        return solver_frame.placement.inverse() * tcp_frame.placement

    @staticmethod
    def _tcp_target_to_solver_target(target_tcp: np.ndarray, solver_to_tcp: pin.SE3) -> np.ndarray:
        target = np.asarray(target_tcp, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("TCP 目标必须是 4x4 有限变换矩阵")
        return target @ solver_to_tcp.inverse().homogeneous

    def _resolve_limits(self) -> list[tuple[float, float]]:
        return resolve_joint_limits(self.model, self.config.joint_limits)

    @staticmethod
    def _ensure_arm_q(values: np.ndarray, name: str) -> np.ndarray:
        q = np.asarray(values, dtype=float).reshape(-1)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{name} 必须是 14 维有限手臂关节角")
        return q

    def _full_q(self, arm_q: np.ndarray, waist_position: float | None = None) -> np.ndarray:
        q = self.home_q.copy()
        q[self.joint_indices[WAIST_JOINT_NAME]] = (
            self.last_waist_position if waist_position is None else float(waist_position)
        )
        for joint_name, value in zip(ARM_JOINT_NAMES, self._ensure_arm_q(arm_q, "arm_q")):
            q[self.joint_indices[joint_name]] = value
        return np.asarray(q, dtype=float)

    def _arm_q(self, full_q: np.ndarray) -> np.ndarray:
        return np.asarray([full_q[self.joint_indices[name]] for name in ARM_JOINT_NAMES], dtype=float)

    def set_home_q(self, home_q: np.ndarray) -> None:
        arm_q = self._ensure_arm_q(home_q, "home_q")
        self.home_q = self._clip_full_q(self._full_q(arm_q))
        self.reference_q = self.home_q.copy()
        self.solver.init_data = self.home_q.copy()

    def end_effector_poses(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回公共 true TCP 的左右 FK。"""
        return self._frame_poses(q, self.left_frame_id, self.right_frame_id)

    def wrist_effector_poses(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回官方腕滚求解 Frame（wrist-roll 原点）的左右 FK。"""
        return self._frame_poses(q, self.left_solver_frame_id, self.right_solver_frame_id)

    def wrist_to_tcp_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.left_solver_to_tcp.homogeneous.copy(),
            self.right_solver_to_tcp.homogeneous.copy(),
        )

    def _frame_poses(
        self,
        q: np.ndarray,
        left_frame_id: int,
        right_frame_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        full_q = self._full_q(self._ensure_arm_q(q, "q"))
        pin.forwardKinematics(self.model, self.data, full_q)
        pin.updateFramePlacements(self.model, self.data)
        return (
            self.data.oMf[left_frame_id].homogeneous.copy(),
            self.data.oMf[right_frame_id].homogeneous.copy(),
        )

    def solve(self, ik_input: TeleopIKInput) -> TeleopIKResult:
        if self.config.task_mode == "position_only":
            return self._solve_position_only(ik_input)
        return self._solve_pose(ik_input)

    def _solve_position_only(self, ik_input: TeleopIKInput) -> TeleopIKResult:
        """仅跟随 TCP 位置，并把配置的腕关节固定为启动反馈。"""

        from teleop_ik_core import apply_joint_safety

        solve_start = time.perf_counter()
        self.solve_frame_index += 1
        current_arm_q = self._ensure_arm_q(ik_input.current_q, "current_q")
        command_arm_q = (
            current_arm_q
            if ik_input.last_q_cmd is None
            else self._ensure_arm_q(ik_input.last_q_cmd, "last_q_cmd")
        )
        current_waist = float(
            self.last_waist_position
            if ik_input.current_waist_position is None
            else ik_input.current_waist_position
        )
        if not np.isfinite(current_waist):
            raise ValueError("current_waist_position 包含 NaN/Inf")
        current_waist = float(np.clip(current_waist, *self.waist_limit))
        self.last_waist_position = current_waist

        q = self._clip_full_q(self._full_q(current_arm_q, current_waist))
        command_ref = self._clip_full_q(self._full_q(command_arm_q, current_waist))
        q = self._lock_configured_joints(q, ik_input.control_side)
        command_ref = self._lock_configured_joints(command_ref, ik_input.control_side)
        limited = False
        iteration_count = 0
        error_norm = 0.0
        for iteration_count in range(1, max(1, int(self.config.max_iterations)) + 1):
            jacobian, error = self._build_position_task(q, ik_input)
            error_norm = float(np.linalg.norm(error))
            if jacobian.size == 0 or error_norm < 1e-5:
                break

            task_increment = self._hierarchical_position_increment(
                jacobian,
                error,
                q,
                ik_input.control_side,
            )
            predicted_task_motion = jacobian @ task_increment
            directional_progress = float(np.dot(error, predicted_task_motion))
            if (
                iteration_count == 1
                and error_norm > 1e-3
                and directional_progress < 0.20 * error_norm * error_norm
            ):
                dq = self._singularity_escape_increment(ik_input.control_side)
            else:
                dq = task_increment
            dq = self._mask_full_control_side(dq, ik_input.control_side)
            per_iteration_limit = abs(float(self.config.max_joint_delta_per_step))
            max_abs_increment = float(np.max(np.abs(dq)))
            if max_abs_increment > per_iteration_limit:
                dq *= per_iteration_limit / max_abs_increment
                limited = True
            q_next = self._clip_full_q(q + dq)
            q_next = self._lock_configured_joints(q_next, ik_input.control_side)
            limited = limited or not np.allclose(q_next, q + dq, atol=1e-8)
            q = q_next

        q, command_limited = apply_joint_safety(
            q_next=q,
            q_ref=command_ref,
            limits=self.full_joint_limits,
            max_step=self.config.max_joint_delta_per_step,
            smoothing_alpha=self.config.smoothing_alpha,
        )
        limited = limited or command_limited
        locked_q = self._lock_configured_joints(q, ik_input.control_side)
        limited = limited or not np.allclose(locked_q, q, atol=1e-8)
        q = self._clip_full_q(locked_q)
        q[self.joint_indices[WAIST_JOINT_NAME]] = current_waist

        position_error_by_side = self._position_residuals(q, ik_input)
        error_norm = float(np.linalg.norm(list(position_error_by_side.values())))
        arm_positions = self._arm_q(q)
        reference_delta = arm_positions - self._arm_q(self.reference_q)
        wrist_indices = (5, 6, 12, 13)
        command = IKCommand(
            timestamp=ik_input.timestamp,
            control_side=ik_input.control_side,
            joint_names=self.config.joint_names,
            positions=arm_positions,
            motor_ids=self.config.motor_ids,
            waist_position=current_waist,
            meta={
                "solver": "official_urdf_position_dls",
                "task_mode": "position_only",
                "orientation_target_used": False,
                "posture_reference": "wrist_locked_to_startup_feedback",
                "locked_arm_joint_indices": list(self.config.locked_joint_indices),
                "position_error_m": error_norm,
                "position_error_by_side_m": position_error_by_side,
                "position_priority": "strict_primary",
                "max_reference_delta_rad": float(np.max(np.abs(reference_delta))),
                "max_wrist_reference_delta_rad": float(np.max(np.abs(reference_delta[list(wrist_indices)]))),
                "waist_policy": "hold_current_feedback",
                "limited": limited,
                "iterations": iteration_count,
                "cost_ms": (time.perf_counter() - solve_start) * 1000.0,
                "frame_index": self.solve_frame_index,
            },
        )
        return TeleopIKResult(
            command=command,
            converged=error_norm < self.config.position_convergence_threshold,
            error_norm=error_norm,
            iteration_count=iteration_count,
            limited=limited,
            reason="ok",
        )

    def _build_position_task(
        self,
        q: np.ndarray,
        ik_input: TeleopIKInput,
    ) -> tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        blocks: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        side_frames = {
            "left": (self.left_frame_id, ik_input.target_pose_l),
            "right": (self.right_frame_id, ik_input.target_pose_r),
        }
        for side in self._selected_control_sides(ik_input.control_side):
            frame_id, target_pose = side_frames[side]
            target = np.asarray(target_pose, dtype=float)
            if target.shape != (4, 4) or not np.all(np.isfinite(target)):
                raise ValueError(f"{side} TCP 目标必须是 4x4 有限变换矩阵")
            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3]
            position_error = target[:3, 3] - self.data.oMf[frame_id].translation
            error_norm = float(np.linalg.norm(position_error))
            if error_norm > self.config.max_position_error:
                position_error *= float(self.config.max_position_error) / error_norm
            blocks.append(jacobian * float(self.config.position_weight))
            errors.append(position_error * float(self.config.position_weight))
        if not blocks:
            return np.empty((0, self.model.nv)), np.empty((0,))
        return np.vstack(blocks), np.concatenate(errors)

    def _position_residuals(self, q: np.ndarray, ik_input: TeleopIKInput) -> dict[str, float]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        side_frames = {
            "left": (self.left_frame_id, ik_input.target_pose_l),
            "right": (self.right_frame_id, ik_input.target_pose_r),
        }
        residuals: dict[str, float] = {}
        for side in self._selected_control_sides(ik_input.control_side):
            frame_id, target_pose = side_frames[side]
            target_position = np.asarray(target_pose, dtype=float)[:3, 3]
            residuals[side] = float(np.linalg.norm(target_position - self.data.oMf[frame_id].translation))
        return residuals

    def _hierarchical_position_increment(
        self,
        jacobian: np.ndarray,
        error: np.ndarray,
        q: np.ndarray,
        control_side: str,
    ) -> np.ndarray:
        """先求 TCP 位置增量，再把姿态参考严格投影到位置任务零空间。"""

        controlled_indices = self._controlled_full_indices(control_side)
        controlled_jacobian = jacobian[:, controlled_indices]
        joint_cost = self._position_joint_cost(control_side)
        inverse_cost = np.diag(1.0 / joint_cost)
        damping_matrix = (float(self.config.damping) ** 2) * np.eye(controlled_jacobian.shape[0])
        primary_inverse = inverse_cost @ controlled_jacobian.T @ np.linalg.inv(
            controlled_jacobian @ inverse_cost @ controlled_jacobian.T + damping_matrix
        )
        primary_increment = primary_inverse @ error

        exact_inverse = np.linalg.pinv(controlled_jacobian, rcond=1e-5)
        nullspace_projector = np.eye(len(controlled_indices)) - exact_inverse @ controlled_jacobian
        reference_error = self.reference_q[list(controlled_indices)] - q[list(controlled_indices)]
        secondary_increment = nullspace_projector @ (
            float(self.config.nullspace_gain) * reference_error
        )

        full_increment = np.zeros(self.model.nv)
        full_increment[list(controlled_indices)] = primary_increment + secondary_increment
        return full_increment

    def _controlled_full_indices(self, control_side: str) -> tuple[int, ...]:
        locked_arm_indices = set(self.config.locked_joint_indices)
        indices: list[int] = []
        for side in self._selected_control_sides(control_side):
            arm_offset = 0 if side == "left" else 7
            names = ARM_JOINT_NAMES[:7] if side == "left" else ARM_JOINT_NAMES[7:]
            for local_index, name in enumerate(names):
                if arm_offset + local_index not in locked_arm_indices:
                    indices.append(self.joint_indices[name])
        if not indices:
            raise ValueError("position_only 至少需要一个未锁定的受控关节")
        return tuple(indices)

    def _position_joint_cost(self, control_side: str) -> np.ndarray:
        # 代价只选择同一位置解的肩肘分支，不会变成与 TCP 位置竞争的误差项。
        per_joint_cost = {
            "shoulder_pitch_l_joint": 1.0,
            "shoulder_roll_l_joint": 1.4,
            "shoulder_yaw_l_joint": 8.0,
            "elbow_pitch_l_joint": 1.2,
            "elbow_yaw_l_joint": 8.0,
            "shoulder_pitch_r_joint": 1.0,
            "shoulder_roll_r_joint": 1.4,
            "shoulder_yaw_r_joint": 8.0,
            "elbow_pitch_r_joint": 1.2,
            "elbow_yaw_r_joint": 8.0,
        }
        costs: list[float] = []
        locked_arm_indices = set(self.config.locked_joint_indices)
        for side in self._selected_control_sides(control_side):
            arm_offset = 0 if side == "left" else 7
            names = ARM_JOINT_NAMES[:7] if side == "left" else ARM_JOINT_NAMES[7:]
            for local_index, name in enumerate(names):
                if arm_offset + local_index in locked_arm_indices:
                    continue
                costs.append(per_joint_cost.get(name, 1.0))
        return np.asarray(costs, dtype=float)

    def _lock_configured_joints(self, q: np.ndarray, control_side: str) -> np.ndarray:
        locked = np.asarray(q, dtype=float).copy()
        selected_sides = set(self._selected_control_sides(control_side))
        for arm_index in self.config.locked_joint_indices:
            if arm_index < 0 or arm_index >= len(ARM_JOINT_NAMES):
                raise ValueError(f"locked_joint_indices 包含非法手臂索引: {arm_index}")
            side = "left" if arm_index < 7 else "right"
            if side in selected_sides:
                full_index = self.joint_indices[ARM_JOINT_NAMES[arm_index]]
                locked[full_index] = self.reference_q[full_index]
        return locked

    def _singularity_escape_increment(self, control_side: str) -> np.ndarray:
        """在手臂完全伸直、轴向位置误差无一阶梯度时给肘部一个确定的小弯曲。"""

        increment = np.zeros(self.model.nv)
        escape_step = min(abs(float(self.config.max_joint_delta_per_step)), 0.03)
        for side in self._selected_control_sides(control_side):
            joint_name = "elbow_pitch_l_joint" if side == "left" else "elbow_pitch_r_joint"
            increment[self.joint_indices[joint_name]] = -escape_step
        return increment

    def _clip_full_q(self, q: np.ndarray) -> np.ndarray:
        return clip_to_limits(q, self.full_joint_limits)

    def _mask_full_control_side(self, dq: np.ndarray, control_side: str) -> np.ndarray:
        masked = np.asarray(dq, dtype=float).copy()
        masked[self.joint_indices[WAIST_JOINT_NAME]] = 0.0
        if control_side == "left":
            for joint_name in ARM_JOINT_NAMES[7:]:
                masked[self.joint_indices[joint_name]] = 0.0
        elif control_side == "right":
            for joint_name in ARM_JOINT_NAMES[:7]:
                masked[self.joint_indices[joint_name]] = 0.0
        elif control_side != "both":
            raise ValueError(f"Unsupported control_side: {control_side}")
        return masked

    @staticmethod
    def _selected_control_sides(control_side: str) -> tuple[str, ...]:
        if control_side == "left":
            return ("left",)
        if control_side == "right":
            return ("right",)
        if control_side == "both":
            return ("left", "right")
        raise ValueError(f"Unsupported control_side: {control_side}")

    def _solve_pose(self, ik_input: TeleopIKInput) -> TeleopIKResult:
        solve_start = time.perf_counter()
        self.solve_frame_index += 1
        current_arm_q = self._ensure_arm_q(ik_input.current_q, "current_q")
        seed_arm_q = (
            current_arm_q
            if ik_input.last_q_cmd is None
            else self._ensure_arm_q(ik_input.last_q_cmd, "last_q_cmd")
        )
        current_waist = float(
            self.last_waist_position
            if ik_input.current_waist_position is None
            else ik_input.current_waist_position
        )
        if not np.isfinite(current_waist):
            raise ValueError("current_waist_position 包含 NaN/Inf")
        current_waist = float(np.clip(current_waist, *self.waist_limit))
        raw_initial_q = self._full_q(seed_arm_q, current_waist)
        initial_q = self._clip_full_q(raw_initial_q)
        limited = not np.allclose(initial_q, raw_initial_q, atol=1e-8)
        self.last_waist_position = current_waist
        target_pose_l = ensure_pose(ik_input.target_pose_l, "target_pose_l")
        target_pose_r = ensure_pose(ik_input.target_pose_r, "target_pose_r")

        reason = "ok"
        try:
            left_solver_target = self._tcp_target_to_solver_target(
                target_pose_l,
                self.left_solver_to_tcp,
            )
            right_solver_target = self._tcp_target_to_solver_target(
                target_pose_r,
                self.right_solver_to_tcp,
            )
            sol_full_q = self.solver.solve_ik(
                left_wrist=left_solver_target,
                right_wrist=right_solver_target,
                q_init=initial_q,
                max_iters=2,
                pos_threshold=5e-4,
                ori_threshold=2e-3,
                verbose=False,
            )
            sol_full_q = np.asarray(sol_full_q, dtype=float).reshape(-1)
            if sol_full_q.shape != (self.model.nq,) or not np.all(np.isfinite(sol_full_q)):
                raise ValueError(f"官方求解器输出维度或数值非法: {sol_full_q.shape}")
            bounded_q = self._clip_full_q(sol_full_q)
            limited = limited or not np.allclose(bounded_q, sol_full_q, atol=1e-8)
            sol_full_q = bounded_q
            solver_status = getattr(self.solver, "last_return_status", "unknown")
            if solver_status == "exception":
                solver_exception = getattr(self.solver, "last_solver_exception", "unknown error")
                raise RuntimeError(solver_exception)
        except Exception as exc:
            sol_full_q = initial_q
            reason = f"solver_failed: {exc}"

        self.home_q = sol_full_q.copy()
        self.solver.init_data = sol_full_q.copy()
        self.last_waist_position = float(sol_full_q[self.joint_indices[WAIST_JOINT_NAME]])
        arm_positions = self._arm_q(sol_full_q)
        position_error_by_side, orientation_error_by_side = self._pose_residuals(sol_full_q, ik_input)
        max_position_error = max(position_error_by_side.values(), default=0.0)
        max_orientation_error = max(orientation_error_by_side.values(), default=0.0)
        error_norm = self._pose_error_norm(sol_full_q, ik_input)
        command = IKCommand(
            timestamp=ik_input.timestamp,
            control_side=ik_input.control_side,
            joint_names=self.config.joint_names,
            positions=arm_positions,
            motor_ids=self.config.motor_ids,
            waist_position=self.last_waist_position,
            meta={
                "solver": "official_tiangong2pro",
                "official_solver_mode": self.solver_mode,
                "official_solver_so": str(OFFICIAL_SOLVER_SO_PATH),
                "official_urdf": str(OFFICIAL_URDF_PATH),
                "public_end_effector_frames": ["left_tcp_link", "right_tcp_link"],
                "solver_end_effector_frames": ["L_ee", "R_ee"],
                "solver_to_tcp_translation_m": {
                    "left": self.left_solver_to_tcp.translation.tolist(),
                    "right": self.right_solver_to_tcp.translation.tolist(),
                },
                "official_waist_joint": WAIST_JOINT_NAME,
                "full_q_order": [WAIST_JOINT_NAME, *ARM_JOINT_NAMES],
                "solver_return_status": getattr(self.solver, "last_return_status", "unknown"),
                "solver_position_error_m": getattr(self.solver, "last_position_error", float("nan")),
                "solver_orientation_error_rad": getattr(self.solver, "last_orientation_error", float("nan")),
                "tcp_position_error_m": max_position_error,
                "tcp_position_error_by_side_m": position_error_by_side,
                "tcp_orientation_error_rad": max_orientation_error,
                "tcp_orientation_error_by_side_rad": orientation_error_by_side,
                "orientation_target_used": True,
                "task_mode": "pose",
                "warm_start_source": "robot_feedback" if ik_input.last_q_cmd is None else "last_safe_command",
                "solver_exception": getattr(self.solver, "last_solver_exception", None),
                "reason": reason,
                "limited": limited,
                "cost_ms": (time.perf_counter() - solve_start) * 1000.0,
                "frame_index": self.solve_frame_index,
            },
        )
        return TeleopIKResult(
            command=command,
            converged=(
                reason == "ok"
                and max_position_error < 1e-3
                and max_orientation_error < 1e-2
            ),
            error_norm=error_norm,
            iteration_count=self.solver.last_iteration_count,
            limited=limited,
            reason=reason,
        )

    def _pose_residuals(
        self,
        q: np.ndarray,
        ik_input: TeleopIKInput,
    ) -> tuple[dict[str, float], dict[str, float]]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        side_frames = {
            "left": (self.left_frame_id, ik_input.target_pose_l),
            "right": (self.right_frame_id, ik_input.target_pose_r),
        }
        position_errors: dict[str, float] = {}
        orientation_errors: dict[str, float] = {}
        for side in self._selected_control_sides(ik_input.control_side):
            frame_id, target_pose = side_frames[side]
            target = np.asarray(target_pose, dtype=float)
            actual = self.data.oMf[frame_id]
            position_errors[side] = float(np.linalg.norm(actual.translation - target[:3, 3]))
            orientation_errors[side] = float(
                np.linalg.norm(pin.log3(actual.rotation @ target[:3, :3].T))
            )
        return position_errors, orientation_errors

    def _pose_error_norm(self, q: np.ndarray, ik_input: TeleopIKInput) -> float:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        errors: list[np.ndarray] = []
        if ik_input.control_side in {"left", "both"}:
            errors.append(frame_error(self.data.oMf[self.left_frame_id], ik_input.target_pose_l))
        if ik_input.control_side in {"right", "both"}:
            errors.append(frame_error(self.data.oMf[self.right_frame_id], ik_input.target_pose_r))
        return float(np.linalg.norm(np.concatenate(errors))) if errors else 0.0


# 保留旧后端类名，避免外部脚本的导入接口发生变化。
VerifiedCasadiArmIK = OfficialTiangong2ProIK


def frame_error(current: pin.SE3, target_pose: np.ndarray) -> np.ndarray:
    target = np.asarray(target_pose, dtype=float)
    position_error = current.translation - target[:3, 3]
    rotation_error = pin.log3(current.rotation @ target[:3, :3].T)
    return np.concatenate([position_error, rotation_error])
