from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ControlSide = Literal["left", "right", "both"]
IKTaskMode = Literal["pose", "position_only"]


@dataclass(frozen=True)
class TeleopIKConfig:
    """遥操作 IK 的调参入口。

    这一层只描述“求解器如何跟随目标”，不包含 UDP、ROS、HTTP 等通信参数。
    高频遥操作优先保证小步、连续和可预测，因此默认参数比全局优化式 IK 更保守。
    """

    joint_names: list[str]
    motor_ids: list[int]
    joint_limits: list[tuple[float, float]] | None = None
    left_frame_name: str = "L_ee"
    right_frame_name: str = "R_ee"
    task_mode: IKTaskMode = "pose"
    control_dt: float = 0.02
    max_iterations: int = 2
    damping: float = 0.08
    position_weight: float = 1.0
    orientation_weight: float = 0.25
    max_position_error: float = 0.08
    position_convergence_threshold: float = 1e-3
    max_orientation_error: float = 0.35
    max_joint_delta_per_step: float = 0.08
    smoothing_alpha: float = 0.65
    nullspace_gain: float = 0.015
    stabilized_joint_indices: tuple[int, ...] = ()
    stabilized_joint_gain: float = 0.25
    stabilized_joint_max_delta: float | None = None
    stabilized_joint_max_deltas: tuple[float, ...] | None = None
    stabilized_joint_delta_bounds: tuple[tuple[float, float], ...] | None = None
    locked_joint_indices: tuple[int, ...] = ()
    home_q: np.ndarray | None = None
    enable_local_log: bool = True
    log_dir: str | None = None
    log_file_name: str = "teleop_ik_core.log"
    log_to_console: bool = True
    show_model_load_progress: bool = True
    model_load_progress_interval_seconds: float = 5.0
    log_ik_result: bool = True
    log_every_n_frames: int = 1

    def __post_init__(self) -> None:
        if not self.joint_names or len(self.joint_names) != len(self.motor_ids):
            raise ValueError("joint_names 和 motor_ids 必须包含相同数量的关节")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, (int, np.integer)) or self.max_iterations < 1:
            raise ValueError("max_iterations 必须为正整数")
        nonnegative_fields = (
            "damping", "position_weight", "orientation_weight", "max_position_error",
            "max_orientation_error", "max_joint_delta_per_step", "nullspace_gain",
            "stabilized_joint_gain",
        )
        for name in nonnegative_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须为有限非负数")
        for name in ("control_dt", "position_convergence_threshold"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须为有限正数")
        if not np.isfinite(self.smoothing_alpha) or not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha 必须为 [0, 1] 内的有限数")
        if self.stabilized_joint_gain > 1.0:
            raise ValueError("stabilized_joint_gain 必须位于 [0, 1]")
        for indices in (self.stabilized_joint_indices, self.locked_joint_indices):
            if any(not isinstance(index, (int, np.integer)) or index < 0 or index >= len(self.joint_names) for index in indices):
                raise ValueError("关节约束索引超出 joint_names 范围")
        for values in (self.stabilized_joint_max_delta, self.stabilized_joint_max_deltas):
            if values is not None:
                deltas = np.asarray(values, dtype=float)
                if not np.all(np.isfinite(deltas)) or np.any(deltas < 0.0):
                    raise ValueError("稳定关节偏移必须为有限非负数")
        if self.stabilized_joint_delta_bounds is not None:
            bounds = np.asarray(self.stabilized_joint_delta_bounds, dtype=float)
            if bounds.shape != (len(self.stabilized_joint_indices), 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("稳定关节偏移上下界非法")


@dataclass(frozen=True)
class TeleopIKInput:
    """遥操作 IK 的单帧输入。

    输入必须是已经完成坐标系标定后的机器人基坐标系位姿。
    `last_q_cmd` 用于让求解器围绕上一帧指令做小步修正，而不是每帧重新搜索全局最优解。
    """

    timestamp: float
    control_side: ControlSide
    target_pose_l: np.ndarray
    target_pose_r: np.ndarray
    current_q: np.ndarray
    last_q_cmd: np.ndarray | None = None
    current_waist_position: float | None = None


@dataclass(frozen=True)
class IKCommand:
    """下发层唯一关节命令格式。

    核心求解器继续通过 `positions` 输出 14 维手臂关节角；官方 15 轴 IK 的腰部结果放在
    可选 `waist_position` 中。旧调用方无需感知腰部字段，ROS 适配层负责拆分话题。
    """

    timestamp: float
    control_side: ControlSide
    joint_names: list[str]
    positions: np.ndarray
    motor_ids: list[int]
    frame_id: str = "base"
    meta: dict[str, object] = field(default_factory=dict)
    waist_position: float | None = None


@dataclass(frozen=True)
class TeleopIKResult:
    """求解结果和诊断信息。

    `command` 是下发入口；其余字段用于日志、调参和判断是否需要降级。
    """

    command: IKCommand
    converged: bool
    error_norm: float
    iteration_count: int
    limited: bool
    reason: str = "ok"
