from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

MocapTargetMode = Literal["relative", "absolute"]


@dataclass
class MocapTargetMapping:
    left_target: np.ndarray
    right_target: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class WorkspaceCalibration:
    source_path: Path
    base_from_mocap_rotation: np.ndarray
    mocap_home: dict[str, np.ndarray]
    direction_scales: dict[str, dict[str, float]]
    workspace_bounds: dict[str, dict[str, float]]
    control_frame: str
    position_reference: str
    orientation_base_from_mocap_rotations: dict[str, np.ndarray]


def load_workspace_calibration(
    path: str | Path | None,
    control_side: str = "both",
    optional: bool = False,
) -> WorkspaceCalibration | None:
    """Allow an omitted optional calibration, while validating every supplied file."""
    if path in (None, ""):
        if optional:
            print("CALIBRATION_INFO workspace 标定未配置，使用默认相对映射")
            return None
        raise ValueError("未提供工作空间标定文件")
    return _load_workspace_calibration_strict(path, control_side)


def _load_workspace_calibration_strict(
    path: str | Path,
    control_side: str,
) -> WorkspaceCalibration:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到工作空间标定文件: {source_path}")
    with source_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("工作空间标定必须是 JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"不兼容的工作空间标定 schema_version: {payload.get('schema_version')!r}")
    if payload.get("valid") is not True:
        raise ValueError("工作空间标定未通过质量检查，valid 必须为 true")
    if payload.get("control_frame") != "wrist_roll":
        raise ValueError(f"不支持的工作空间 control_frame: {payload.get('control_frame')!r}；请使用腕点控制帧重新标定")

    rotation = np.asarray(payload.get("base_from_mocap_rotation"), dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("base_from_mocap_rotation 必须是 3x3 有限矩阵")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.99:
        raise ValueError("base_from_mocap_rotation 不是有效右手正交旋转矩阵")

    sides = ("left", "right") if control_side == "both" else (control_side,)
    if any(side not in {"left", "right"} for side in sides):
        raise ValueError(f"不支持的 control_side: {control_side}")
    homes: dict[str, np.ndarray] = {}
    scales: dict[str, dict[str, float]] = {}
    bounds: dict[str, dict[str, float]] = {}
    for side in sides:
        home = np.asarray(payload.get("mocap_home", {}).get(side), dtype=float)
        if home.shape != (3,) or not np.all(np.isfinite(home)):
            raise ValueError(f"标定缺少有效 mocap_home.{side}")
        homes[side] = home
        scales[side] = {}
        bounds[side] = {}
        for direction in ("forward", "backward", "outward", "inward", "up", "down"):
            scale = payload.get("direction_scales", {}).get(side, {}).get(direction)
            boundary = payload.get("workspace_bounds", {}).get(side, {}).get(direction, {}).get("safe_boundary_m")
            if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"标定缺少有效 direction_scales.{side}.{direction}")
            if not isinstance(boundary, (int, float)) or not np.isfinite(boundary) or boundary <= 0:
                raise ValueError(f"标定缺少有效 workspace_bounds.{side}.{direction}.safe_boundary_m")
            scales[side][direction] = float(scale)
            bounds[side][direction] = float(boundary)

    for side in ("left", "right"):
        if side not in homes:
            fallback = np.asarray(payload.get("mocap_home", {}).get(side), dtype=float)
            if fallback.shape == (3,) and np.all(np.isfinite(fallback)):
                homes[side] = fallback
        if side not in homes:
            raise ValueError(f"标定缺少有效 mocap_home.{side}")

    position_reference = payload.get("position_reference", "absolute_hand_legacy")
    if position_reference not in {"absolute_hand_legacy", "hand_minus_chest", "wrist_minus_chest"}:
        raise ValueError(f"不支持的工作空间 position_reference: {position_reference!r}")

    orientation_payload = payload.get("orientation_base_from_mocap_rotation", {})
    if not isinstance(orientation_payload, dict):
        raise ValueError("orientation_base_from_mocap_rotation 必须是 object")
    orientation_rotations: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        side_rotation = np.asarray(orientation_payload.get(side, rotation), dtype=float)
        if side_rotation.shape != (3, 3) or not np.all(np.isfinite(side_rotation)):
            raise ValueError(f"orientation_base_from_mocap_rotation.{side} 必须是 3x3 有限矩阵")
        if (
            not np.allclose(side_rotation @ side_rotation.T, np.eye(3), atol=1e-5)
            or np.linalg.det(side_rotation) < 0.99
        ):
            raise ValueError(f"orientation_base_from_mocap_rotation.{side} 不是有效右手正交旋转矩阵")
        orientation_rotations[side] = side_rotation

    return WorkspaceCalibration(
        source_path=source_path,
        base_from_mocap_rotation=rotation,
        mocap_home=homes,
        direction_scales=scales,
        workspace_bounds=bounds,
        control_frame="wrist_roll",
        position_reference=position_reference,
        orientation_base_from_mocap_rotations=orientation_rotations,
    )


def map_workspace_delta(
    delta_mocap: np.ndarray,
    side: str,
    calibration: WorkspaceCalibration,
    gain: float,
) -> tuple[np.ndarray, bool]:
    delta_base = calibration.base_from_mocap_rotation @ np.asarray(delta_mocap, dtype=float)
    directions = (
        ("forward", "backward"),
        ("outward", "inward") if side == "left" else ("inward", "outward"),
        ("up", "down"),
    )
    mapped = np.zeros(3, dtype=float)
    selected: list[str] = []
    for axis, (positive, negative) in enumerate(directions):
        direction = positive if delta_base[axis] >= 0.0 else negative
        selected.append(direction)
        mapped[axis] = delta_base[axis] * calibration.direction_scales[side][direction] * float(gain)

    normalized_radius = np.asarray(
        [abs(mapped[axis]) / calibration.workspace_bounds[side][direction] for axis, direction in enumerate(selected)],
        dtype=float,
    )
    ellipsoid_norm = float(np.linalg.norm(normalized_radius))
    projected = ellipsoid_norm > 1.0
    if projected:
        mapped /= ellipsoid_norm
    return mapped, projected


class RelativeMocapTargetMapper:
    """把动捕世界系手部位姿映射成机器人 IK 目标。

    动捕世界系和机器人 base 都采用前左上坐标轴时，不应该再做轴交换；真正需要处理的是
    人体手腕零位、机器人手腕零位和肢体尺度之间的差异。因此默认用第一帧作为动捕 home，
    后续只把动捕相对位移/相对旋转叠加到机器人 home wrist 上。
    """

    def __init__(
        self,
        left_robot_home: np.ndarray,
        right_robot_home: np.ndarray,
        mode: MocapTargetMode = "relative",
        position_scale: float = 1.0,
        orientation_scale: float = 1.0,
        workspace_calibration: WorkspaceCalibration | None = None,
        control_side: str = "both",
        left_control_to_target: np.ndarray | None = None,
        right_control_to_target: np.ndarray | None = None,
        control_frame: str = "target_tcp",
    ):
        if mode not in {"relative", "absolute"}:
            raise ValueError(f"Unsupported mocap target mode: {mode}")
        if control_side not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported control_side: {control_side}")
        self.mode = mode
        self.position_scale = float(position_scale)
        self.orientation_scale = float(np.clip(orientation_scale, 0.0, 1.0))
        self.workspace_calibration = workspace_calibration
        self.control_side = control_side
        self.left_robot_home = copy_transform(left_robot_home)
        self.right_robot_home = copy_transform(right_robot_home)
        self.left_control_to_target = copy_transform(
            np.eye(4) if left_control_to_target is None else left_control_to_target
        )
        self.right_control_to_target = copy_transform(
            np.eye(4) if right_control_to_target is None else right_control_to_target
        )
        self.control_frame = str(control_frame)
        self.left_mocap_home: np.ndarray | None = None
        self.right_mocap_home: np.ndarray | None = None
        self.mocap_home_source: str | None = None
        self.calibration_frame_index = 0

    def _reference_relative_mocap(
        self,
        left_mocap_wrist: np.ndarray,
        right_mocap_wrist: np.ndarray,
        mocap_reference_position: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_mocap = copy_transform(left_mocap_wrist)
        right_mocap = copy_transform(right_mocap_wrist)
        if (
            self.workspace_calibration is not None
            and self.workspace_calibration.position_reference in {"hand_minus_chest", "wrist_minus_chest"}
        ):
            if mocap_reference_position is None:
                raise ValueError("工作空间标定使用 hand_minus_chest，但当前帧缺少胸口参考点")
            reference = np.asarray(mocap_reference_position, dtype=float)
            if reference.shape != (3,) or not np.all(np.isfinite(reference)):
                raise ValueError("胸口参考点必须是 3 维有限坐标")
            left_mocap[:3, 3] -= reference
            right_mocap[:3, 3] -= reference
        return left_mocap, right_mocap

    def set_runtime_mocap_home(
        self,
        left_mocap_wrist: np.ndarray,
        right_mocap_wrist: np.ndarray,
        mocap_reference_position: np.ndarray | None = None,
        left_home_position: np.ndarray | None = None,
        right_home_position: np.ndarray | None = None,
    ) -> None:
        left_home, right_home = self._reference_relative_mocap(
            left_mocap_wrist,
            right_mocap_wrist,
            mocap_reference_position,
        )
        if left_home_position is not None:
            left_home[:3, 3] = np.asarray(left_home_position, dtype=float)
        if right_home_position is not None:
            right_home[:3, 3] = np.asarray(right_home_position, dtype=float)
        if not np.all(np.isfinite(left_home)) or not np.all(np.isfinite(right_home)):
            raise ValueError("运行时动捕 Home 包含 NaN/Inf")
        self.left_mocap_home = left_home
        self.right_mocap_home = right_home
        self.mocap_home_source = "runtime_natural_down"
        self.calibration_frame_index += 1

    def map_targets(
        self,
        left_mocap_wrist: np.ndarray,
        right_mocap_wrist: np.ndarray,
        mocap_reference_position: np.ndarray | None = None,
    ) -> MocapTargetMapping:
        left_mocap, right_mocap = self._reference_relative_mocap(
            left_mocap_wrist,
            right_mocap_wrist,
            mocap_reference_position,
        )
        calibrated_this_frame = False

        if self.mode == "absolute":
            return MocapTargetMapping(
                left_target=left_mocap,
                right_target=right_mocap,
                meta={
                    "mode": self.mode,
                    "position_scale": self.position_scale,
                    "orientation_scale": self.orientation_scale,
                    "calibrated_this_frame": False,
                },
            )

        if self.left_mocap_home is None or self.right_mocap_home is None:
            self.left_mocap_home = left_mocap.copy()
            self.right_mocap_home = right_mocap.copy()
            if self.workspace_calibration is not None:
                self.left_mocap_home[:3, 3] = self.workspace_calibration.mocap_home["left"]
                self.right_mocap_home[:3, 3] = self.workspace_calibration.mocap_home["right"]
                self.mocap_home_source = "workspace_calibration"
            else:
                self.mocap_home_source = "first_frame"
            self.calibration_frame_index += 1
            calibrated_this_frame = True

        left_target, left_projected, left_control_target = self._map_side(
            "left",
            left_mocap,
            self.left_mocap_home,
            self.left_robot_home,
            self.left_control_to_target,
        )
        right_target, right_projected, right_control_target = self._map_side(
            "right",
            right_mocap,
            self.right_mocap_home,
            self.right_robot_home,
            self.right_control_to_target,
        )

        return MocapTargetMapping(
            left_target=left_target,
            right_target=right_target,
            meta={
                "mode": self.mode,
                "position_scale": self.position_scale,
                "orientation_scale": self.orientation_scale,
                "calibrated_this_frame": calibrated_this_frame,
                "calibration_frame_index": self.calibration_frame_index,
                "mocap_home_source": self.mocap_home_source,
                "left_mocap_home_position": self.left_mocap_home[:3, 3].copy(),
                "right_mocap_home_position": self.right_mocap_home[:3, 3].copy(),
                "workspace_calibration": None
                if self.workspace_calibration is None
                else str(self.workspace_calibration.source_path),
                "position_reference": "absolute_hand_legacy"
                if self.workspace_calibration is None
                else self.workspace_calibration.position_reference,
                "orientation_reference": "world_delta_in_robot_base",
                "control_frame": self.control_frame,
                "left_control_target": left_control_target.copy(),
                "right_control_target": right_control_target.copy(),
                "control_to_target": {
                    "left": self.left_control_to_target.copy(),
                    "right": self.right_control_to_target.copy(),
                },
                "orientation_base_from_mocap_rotation": None
                if self.workspace_calibration is None
                else {
                    side: rotation.copy()
                    for side, rotation in self.workspace_calibration.orientation_base_from_mocap_rotations.items()
                },
                "workspace_projected": {"left": left_projected, "right": right_projected},
            },
        )

    def _map_side(
        self,
        side: str,
        mocap_current: np.ndarray,
        mocap_home: np.ndarray,
        robot_home: np.ndarray,
        control_to_target: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        if self.control_side not in {side, "both"}:
            control_target = copy_transform(robot_home)
            return control_target @ control_to_target, False, control_target
        orientation_base_from_mocap = (
            np.eye(3)
            if self.workspace_calibration is None
            else self.workspace_calibration.orientation_base_from_mocap_rotations[side]
        )
        if self.workspace_calibration is None:
            control_target = build_relative_target(
                mocap_current=mocap_current,
                mocap_home=mocap_home,
                robot_home=robot_home,
                position_scale=self.position_scale,
                orientation_scale=self.orientation_scale,
                base_from_mocap_rotation=orientation_base_from_mocap,
            )
            return control_target @ control_to_target, False, control_target
        control_target = build_relative_target(
            mocap_current=mocap_current,
            mocap_home=mocap_home,
            robot_home=robot_home,
            position_scale=0.0,
            orientation_scale=self.orientation_scale,
            base_from_mocap_rotation=orientation_base_from_mocap,
        )
        mapped_delta, projected = map_workspace_delta(
            mocap_current[:3, 3] - mocap_home[:3, 3],
            side,
            self.workspace_calibration,
            self.position_scale,
        )
        control_target[:3, 3] = robot_home[:3, 3] + mapped_delta
        return control_target @ control_to_target, projected, control_target

    def reset(self):
        self.left_mocap_home = None
        self.right_mocap_home = None


def build_relative_target(
    mocap_current: np.ndarray,
    mocap_home: np.ndarray,
    robot_home: np.ndarray,
    position_scale: float,
    orientation_scale: float = 1.0,
    base_from_mocap_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """用动捕 Home 到当前帧的世界系增量驱动机器人 TCP Home。

    旋转必须先在动捕世界系计算 `R_current @ R_home.T`，再共轭变换到机器人
    base。这样 marker 与 TCP 的固定局部安装差异会在相对旋转中抵消，不会把
    marker 局部轴误当作机器人 TCP 局部轴。
    """

    target = copy_transform(robot_home)
    target[:3, 3] = robot_home[:3, 3] + float(position_scale) * (mocap_current[:3, 3] - mocap_home[:3, 3])
    base_from_mocap = (
        np.eye(3)
        if base_from_mocap_rotation is None
        else np.asarray(base_from_mocap_rotation, dtype=float)
    )
    if base_from_mocap.shape != (3, 3) or not np.all(np.isfinite(base_from_mocap)):
        raise ValueError("base_from_mocap_rotation 必须是 3x3 有限矩阵")
    if (
        not np.allclose(base_from_mocap @ base_from_mocap.T, np.eye(3), atol=1e-5)
        or np.linalg.det(base_from_mocap) < 0.99
    ):
        raise ValueError("base_from_mocap_rotation 不是有效右手正交旋转矩阵")

    delta_rotation_mocap_world = mocap_current[:3, :3] @ mocap_home[:3, :3].T
    delta_rotation_base = (
        base_from_mocap @ delta_rotation_mocap_world @ base_from_mocap.T
    )
    scale = float(np.clip(orientation_scale, 0.0, 1.0))
    if scale <= 1e-9:
        scaled_delta_rotation_base = np.eye(3)
    elif scale >= 1.0 - 1e-9:
        scaled_delta_rotation_base = delta_rotation_base
    else:
        scaled_delta_rotation_base = R.from_rotvec(
            R.from_matrix(delta_rotation_base).as_rotvec() * scale
        ).as_matrix()
    target[:3, :3] = scaled_delta_rotation_base @ robot_home[:3, :3]
    return target


def copy_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"transform shape must be (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("transform contains NaN/Inf")
    return transform.copy()
