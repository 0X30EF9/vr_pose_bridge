from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

WristPositionSource = Literal["wrist_joint", "hand_pose_plus_pivot_offset"]
SIDES = ("left", "right")


@dataclass(frozen=True)
class WristCalibrationThresholds:
    min_samples: int = 60
    min_groups: int = 3
    min_rotation_excitation_rad: float = 0.25
    min_rotation_rank: int = 2
    max_outlier_fraction: float = 0.15
    max_pivot_rms_m: float = 0.010
    max_pivot_p95_m: float = 0.020
    max_wrist_validation_rms_m: float = 0.020
    max_shared_origin_error_m: float = 0.005


@dataclass(frozen=True)
class WristSideCalibration:
    position_source: WristPositionSource
    hand_to_wrist_offset: np.ndarray
    quality: dict[str, Any]


@dataclass(frozen=True)
class WristCalibration:
    source_path: Path
    sides: dict[str, WristSideCalibration]
    thresholds: WristCalibrationThresholds


@dataclass(frozen=True)
class WristPivotFit:
    hand_to_wrist_offset: np.ndarray
    group_centers: dict[str, np.ndarray]
    inlier_mask: np.ndarray
    quality: dict[str, Any]
    valid: bool
    reasons: tuple[str, ...]


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} 必须是 {size} 维有限向量")
    return vector


def _finite_rotation(value: Any, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} 必须是 3x3 有限矩阵")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.99:
        raise ValueError(f"{label} 不是有效右手正交旋转矩阵")
    return rotation


def thresholds_from_mapping(value: dict[str, Any] | None) -> WristCalibrationThresholds:
    source = {} if value is None else value
    defaults = WristCalibrationThresholds()
    return WristCalibrationThresholds(
        min_samples=int(source.get("min_samples", defaults.min_samples)),
        min_groups=int(source.get("min_groups", defaults.min_groups)),
        min_rotation_excitation_rad=float(
            source.get("min_rotation_excitation_rad", defaults.min_rotation_excitation_rad)
        ),
        min_rotation_rank=int(source.get("min_rotation_rank", defaults.min_rotation_rank)),
        max_outlier_fraction=float(source.get("max_outlier_fraction", defaults.max_outlier_fraction)),
        max_pivot_rms_m=float(source.get("max_pivot_rms_m", defaults.max_pivot_rms_m)),
        max_pivot_p95_m=float(source.get("max_pivot_p95_m", defaults.max_pivot_p95_m)),
        max_wrist_validation_rms_m=float(
            source.get("max_wrist_validation_rms_m", defaults.max_wrist_validation_rms_m)
        ),
        max_shared_origin_error_m=float(
            source.get("max_shared_origin_error_m", defaults.max_shared_origin_error_m)
        ),
    )


def thresholds_to_dict(value: WristCalibrationThresholds) -> dict[str, Any]:
    return {
        "min_samples": value.min_samples,
        "min_groups": value.min_groups,
        "min_rotation_excitation_rad": value.min_rotation_excitation_rad,
        "min_rotation_rank": value.min_rotation_rank,
        "max_outlier_fraction": value.max_outlier_fraction,
        "max_pivot_rms_m": value.max_pivot_rms_m,
        "max_pivot_p95_m": value.max_pivot_p95_m,
        "max_wrist_validation_rms_m": value.max_wrist_validation_rms_m,
        "max_shared_origin_error_m": value.max_shared_origin_error_m,
    }


def _quality_number(quality: Mapping[str, Any], key: str, side: str) -> float:
    value = quality.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"手腕标定 sides.{side}.quality.{key} 必须是有限数值")
    return float(value)


def _validate_side_quality(
    side: str,
    side_payload: Mapping[str, Any],
    quality: Mapping[str, Any],
    thresholds: WristCalibrationThresholds,
) -> None:
    if side_payload.get("valid") is not True:
        raise ValueError(f"手腕标定 sides.{side}.valid 必须为 true")
    checks = [
        ("sample_count", _quality_number(quality, "sample_count", side), thresholds.min_samples, "min"),
        ("group_count", _quality_number(quality, "group_count", side), thresholds.min_groups, "min"),
        (
            "rotation_excitation_rad",
            _quality_number(quality, "rotation_excitation_rad", side),
            thresholds.min_rotation_excitation_rad,
            "min",
        ),
        ("rotation_rank", _quality_number(quality, "rotation_rank", side), thresholds.min_rotation_rank, "min"),
        (
            "outlier_fraction",
            _quality_number(quality, "outlier_fraction", side),
            thresholds.max_outlier_fraction,
            "max",
        ),
    ]
    direct_origin_rms = quality.get("hand_wrist_origin_rms_m")
    if side_payload.get("position_source") == "wrist_joint" and direct_origin_rms is not None:
        if quality.get("position_observation") != "explicit_wrist_joint":
            raise ValueError(f"手腕标定 sides.{side}.quality.position_observation 必须为 explicit_wrist_joint")
        if quality.get("hand_wrist_shared_origin") is not True:
            raise ValueError(f"手腕标定 sides.{side}.quality.hand_wrist_shared_origin 必须为 true")
        checks.extend(
            (
                (
                    "hand_wrist_origin_rms_m",
                    _quality_number(quality, "hand_wrist_origin_rms_m", side),
                    thresholds.max_shared_origin_error_m,
                    "max",
                ),
                (
                    "hand_wrist_origin_p95_m",
                    _quality_number(quality, "hand_wrist_origin_p95_m", side),
                    thresholds.max_shared_origin_error_m,
                    "max",
                ),
            )
        )
    else:
        checks.extend(
            (
                ("pivot_rms_m", _quality_number(quality, "pivot_rms_m", side), thresholds.max_pivot_rms_m, "max"),
                ("pivot_p95_m", _quality_number(quality, "pivot_p95_m", side), thresholds.max_pivot_p95_m, "max"),
            )
        )
    for key, value, limit, direction in checks:
        failed = value < limit if direction == "min" else value > limit
        if failed:
            operator = "<" if direction == "min" else ">"
            raise ValueError(f"手腕标定 sides.{side}.quality.{key}={value:g}{operator}{limit:g}")
    validation_rms = quality.get("wrist_validation_rms_m")
    if direct_origin_rms is None and validation_rms is not None:
        validation_value = _quality_number(quality, "wrist_validation_rms_m", side)
        if validation_value > thresholds.max_wrist_validation_rms_m:
            raise ValueError(
                f"手腕标定 sides.{side}.quality.wrist_validation_rms_m="
                f"{validation_value:g}>{thresholds.max_wrist_validation_rms_m:g}"
            )


def load_wrist_calibration(
    path: str | Path | None,
    control_side: str = "both",
    optional: bool = False,
) -> WristCalibration | None:
    if path in (None, ""):
        if optional:
            print("CALIBRATION_INFO wrist 标定未配置，使用 skeleton 腕点")
            return None
        raise ValueError("未提供手腕控制点标定文件")
    if optional:
        try:
            return _load_wrist_calibration_strict(path, control_side)
        except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            print(f"CALIBRATION_WARN 手腕标定不可用，使用 skeleton 腕点: {Path(path).expanduser().resolve()} ({exc})")
            return None
    return _load_wrist_calibration_strict(path, control_side)


def _load_wrist_calibration_strict(
    path: str | Path,
    control_side: str,
) -> WristCalibration:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到手腕控制点标定文件: {source_path}")
    with source_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("手腕控制点标定必须是 JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"不兼容的手腕标定 schema_version: {payload.get('schema_version')!r}")
    if payload.get("valid") is not True:
        raise ValueError("手腕控制点标定未通过质量检查，valid 必须为 true")
    if payload.get("control_point") != "anatomical_wrist":
        raise ValueError(f"不支持的手腕 control_point: {payload.get('control_point')!r}")
    if control_side not in {"left", "right", "both"}:
        raise ValueError(f"不支持的 control_side: {control_side}")

    selected_sides = SIDES if control_side == "both" else (control_side,)
    thresholds = thresholds_from_mapping(payload.get("quality_thresholds"))
    sides: dict[str, WristSideCalibration] = {}
    side_payloads = payload.get("sides")
    if not isinstance(side_payloads, dict):
        raise ValueError("手腕标定缺少 sides object")
    for side in selected_sides:
        side_payload = side_payloads.get(side)
        if not isinstance(side_payload, dict):
            raise ValueError(f"手腕标定缺少 sides.{side}")
        position_source = side_payload.get("position_source")
        if position_source not in {"wrist_joint", "hand_pose_plus_pivot_offset"}:
            raise ValueError(f"不支持的 sides.{side}.position_source: {position_source!r}")
        offset = _finite_vector(
            side_payload.get("hand_to_wrist_offset"),
            3,
            f"sides.{side}.hand_to_wrist_offset",
        )
        quality = side_payload.get("quality")
        if not isinstance(quality, dict):
            raise ValueError(f"手腕标定缺少 sides.{side}.quality")
        _validate_side_quality(side, side_payload, quality, thresholds)
        sides[side] = WristSideCalibration(
            position_source=position_source,
            hand_to_wrist_offset=offset,
            quality=dict(quality),
        )

    return WristCalibration(
        source_path=source_path,
        sides=sides,
        thresholds=thresholds,
    )


def reconstruct_wrist_pose(
    hand_position: Sequence[float],
    hand_quaternion_xyzw: Sequence[float],
    wrist_position: Sequence[float] | None,
    calibration: WristSideCalibration,
) -> np.ndarray:
    hand_pos = _finite_vector(hand_position, 3, "hand_position")
    hand_quat = _finite_vector(hand_quaternion_xyzw, 4, "hand_quaternion_xyzw")
    quat_norm = float(np.linalg.norm(hand_quat))
    if quat_norm <= 1e-8:
        raise ValueError("hand_quaternion_xyzw 长度为 0")
    hand_rotation = R.from_quat(hand_quat / quat_norm).as_matrix()
    if calibration.position_source == "wrist_joint":
        if wrist_position is None:
            raise ValueError("手腕标定要求 wrist_joint，但当前帧缺少 wrist_position")
        position = _finite_vector(wrist_position, 3, "wrist_position")
    else:
        position = hand_pos + hand_rotation @ calibration.hand_to_wrist_offset
    transform = np.eye(4)
    transform[:3, :3] = hand_rotation
    transform[:3, 3] = position
    return transform


def _tracking_state(frame: Mapping[str, Any], side: str, joint_name: str) -> bool | None:
    states: list[bool] = []
    tracking = frame.get("hand_wrist_tracking")
    if isinstance(tracking, Mapping):
        side_tracking = tracking.get(side)
        if isinstance(side_tracking, Mapping) and joint_name in side_tracking:
            states.append(bool(side_tracking[joint_name]))

    skeleton = frame.get("skeleton")
    if isinstance(skeleton, Mapping):
        side_skeleton = skeleton.get(side)
        if isinstance(side_skeleton, Mapping):
            joint = side_skeleton.get(joint_name)
            if isinstance(joint, Mapping) and "tracked" in joint:
                states.append(bool(joint["tracked"]))

    mocap_frame = frame.get("mocap_frame")
    if mocap_frame is not None and hasattr(mocap_frame, "joint"):
        try:
            states.append(bool(mocap_frame.joint(f"{side}_{joint_name}").tracked))
        except (AttributeError, KeyError):
            pass
    if not states:
        return None
    return all(states)


def reconstruct_wrist_pose_from_mapping(
    frame: Mapping[str, Any],
    side: str,
    calibration: WristSideCalibration,
) -> np.ndarray:
    if side not in SIDES:
        raise ValueError(f"不支持的 side: {side}")
    if _tracking_state(frame, side, "hand") is False:
        raise ValueError(f"当前帧 {side} hand 未跟踪")
    if calibration.position_source == "wrist_joint" and _tracking_state(frame, side, "wrist") is False:
        raise ValueError(f"当前帧 {side} wrist 未跟踪")
    hand_position = frame.get(f"{side}_hand_pos", frame.get(f"{side}_pos"))
    hand_quaternion = frame.get(f"{side}_hand_quat", frame.get(f"{side}_quat"))
    wrist_position = frame.get(f"{side}_wrist_pos")
    if wrist_position is None:
        skeleton = frame.get("skeleton")
        if isinstance(skeleton, Mapping):
            side_skeleton = skeleton.get(side)
            if isinstance(side_skeleton, Mapping):
                wrist = side_skeleton.get("wrist")
                if isinstance(wrist, Mapping):
                    wrist_position = wrist.get("pos")
    return reconstruct_wrist_pose(
        hand_position,
        hand_quaternion,
        wrist_position,
        calibration,
    )


def _pivot_system(
    hand_positions: np.ndarray,
    hand_rotations: np.ndarray,
    group_indices: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = hand_positions.shape[0]
    matrix = np.zeros((sample_count * 3, 3 + group_count * 3), dtype=float)
    vector = np.zeros(sample_count * 3, dtype=float)
    for index in range(sample_count):
        row = slice(index * 3, (index + 1) * 3)
        matrix[row, :3] = hand_rotations[index]
        center_column = 3 + int(group_indices[index]) * 3
        matrix[row, center_column:center_column + 3] = -np.eye(3)
        vector[row] = -hand_positions[index]
    return matrix, vector


def _weighted_solution(matrix: np.ndarray, vector: np.ndarray, frame_weights: np.ndarray) -> np.ndarray:
    row_weights = np.repeat(np.sqrt(np.clip(frame_weights, 1e-8, None)), 3)
    weighted_matrix = matrix * row_weights[:, None]
    weighted_vector = vector * row_weights
    solution, _, _, _ = np.linalg.lstsq(weighted_matrix, weighted_vector, rcond=None)
    return solution


def _rotation_excitation(hand_rotations: np.ndarray, group_indices: np.ndarray) -> tuple[float, int, list[float]]:
    rotvecs: list[np.ndarray] = []
    for group_index in np.unique(group_indices):
        indices = np.flatnonzero(group_indices == group_index)
        reference = hand_rotations[indices[0]]
        for index in indices[1:]:
            rotvecs.append(R.from_matrix(hand_rotations[index] @ reference.T).as_rotvec())
    if not rotvecs:
        return 0.0, 0, [0.0, 0.0, 0.0]
    values = np.asarray(rotvecs, dtype=float)
    singular_values = np.linalg.svd(values, compute_uv=False)
    padded = np.pad(singular_values, (0, max(0, 3 - len(singular_values))))[:3]
    max_angle = float(np.max(np.linalg.norm(values, axis=1)))
    rank_threshold = max(0.03, max_angle * 0.10)
    rank = int(np.count_nonzero(padded >= rank_threshold))
    return max_angle, rank, padded.tolist()


def fit_direct_wrist_joint(
    hand_positions: Sequence[Sequence[float]],
    hand_rotations: Sequence[Sequence[Sequence[float]]],
    groups: Sequence[str],
    wrist_positions: Sequence[Sequence[float]],
    thresholds: WristCalibrationThresholds | None = None,
) -> WristPivotFit:
    limits = thresholds or WristCalibrationThresholds()
    positions = np.asarray(hand_positions, dtype=float)
    rotations = np.asarray(hand_rotations, dtype=float)
    wrists = np.asarray(wrist_positions, dtype=float)
    group_names = np.asarray([str(group) for group in groups], dtype=object)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("hand_positions 必须是 Nx3")
    if rotations.shape != (positions.shape[0], 3, 3):
        raise ValueError("hand_rotations 必须是 Nx3x3")
    if wrists.shape != positions.shape or not np.all(np.isfinite(wrists)):
        raise ValueError("wrist_positions 必须是与 hand_positions 对齐的 Nx3 有限矩阵")
    if group_names.shape != (positions.shape[0],):
        raise ValueError("groups 必须与样本数一致")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise ValueError("直接腕点样本包含 NaN/Inf")
    for index, rotation in enumerate(rotations):
        _finite_rotation(rotation, f"hand_rotations[{index}]")

    unique_groups = tuple(dict.fromkeys(group_names.tolist()))
    group_lookup = {name: index for index, name in enumerate(unique_groups)}
    group_indices = np.asarray([group_lookup[name] for name in group_names], dtype=int)
    excitation, rotation_rank, singular_values = _rotation_excitation(rotations, group_indices)
    origin_errors = np.linalg.norm(positions - wrists, axis=1)
    origin_rms = float(np.sqrt(np.mean(np.square(origin_errors))))
    origin_p95 = float(np.percentile(origin_errors, 95))
    reasons: list[str] = []
    if positions.shape[0] < limits.min_samples:
        reasons.append(f"samples={positions.shape[0]}<{limits.min_samples}")
    if len(unique_groups) < limits.min_groups:
        reasons.append(f"groups={len(unique_groups)}<{limits.min_groups}")
    if excitation < limits.min_rotation_excitation_rad:
        reasons.append(f"rotation_excitation={excitation:.4f}<{limits.min_rotation_excitation_rad:.4f}rad")
    if rotation_rank < limits.min_rotation_rank:
        reasons.append(f"rotation_rank={rotation_rank}<{limits.min_rotation_rank}")
    if origin_rms > limits.max_shared_origin_error_m:
        reasons.append(
            f"hand_wrist_origin_rms={origin_rms:.4f}>"
            f"{limits.max_shared_origin_error_m:.4f}m"
        )
    if origin_p95 > limits.max_shared_origin_error_m:
        reasons.append(
            f"hand_wrist_origin_p95={origin_p95:.4f}>"
            f"{limits.max_shared_origin_error_m:.4f}m"
        )

    quality = {
        "sample_count": int(positions.shape[0]),
        "group_count": len(unique_groups),
        "position_observation": "explicit_wrist_joint",
        "hand_wrist_shared_origin": True,
        "hand_wrist_origin_rms_m": origin_rms,
        "hand_wrist_origin_p95_m": origin_p95,
        "outlier_fraction": 0.0,
        "rotation_excitation_rad": excitation,
        "rotation_rank": rotation_rank,
        "rotation_singular_values": singular_values,
    }
    return WristPivotFit(
        hand_to_wrist_offset=np.zeros(3, dtype=float),
        group_centers={name: np.mean(wrists[group_indices == index], axis=0) for name, index in group_lookup.items()},
        inlier_mask=np.ones(positions.shape[0], dtype=bool),
        quality=quality,
        valid=not reasons,
        reasons=tuple(reasons),
    )


def fit_hand_to_wrist_pivot(
    hand_positions: Sequence[Sequence[float]],
    hand_rotations: Sequence[Sequence[Sequence[float]]],
    groups: Sequence[str],
    wrist_positions: Sequence[Sequence[float]] | None = None,
    thresholds: WristCalibrationThresholds | None = None,
    huber_delta_m: float = 0.010,
    max_iterations: int = 20,
) -> WristPivotFit:
    limits = thresholds or WristCalibrationThresholds()
    positions = np.asarray(hand_positions, dtype=float)
    rotations = np.asarray(hand_rotations, dtype=float)
    group_names = np.asarray([str(group) for group in groups], dtype=object)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("hand_positions 必须是 Nx3")
    if rotations.shape != (positions.shape[0], 3, 3):
        raise ValueError("hand_rotations 必须是 Nx3x3")
    if group_names.shape != (positions.shape[0],):
        raise ValueError("groups 必须与样本数一致")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise ValueError("枢轴标定样本包含 NaN/Inf")
    for index, rotation in enumerate(rotations):
        _finite_rotation(rotation, f"hand_rotations[{index}]")

    unique_groups = tuple(dict.fromkeys(group_names.tolist()))
    group_lookup = {name: index for index, name in enumerate(unique_groups)}
    group_indices = np.asarray([group_lookup[name] for name in group_names], dtype=int)
    matrix, vector = _pivot_system(positions, rotations, group_indices, len(unique_groups))
    weights = np.ones(positions.shape[0], dtype=float)
    solution = _weighted_solution(matrix, vector, weights)
    for _ in range(max(1, int(max_iterations))):
        offset = solution[:3]
        centers = solution[3:].reshape(len(unique_groups), 3)
        residual_vectors = positions + np.einsum("nij,j->ni", rotations, offset) - centers[group_indices]
        residuals = np.linalg.norm(residual_vectors, axis=1)
        next_weights = np.ones_like(residuals)
        large = residuals > float(huber_delta_m)
        next_weights[large] = float(huber_delta_m) / residuals[large]
        next_solution = _weighted_solution(matrix, vector, next_weights)
        if np.linalg.norm(next_solution - solution) < 1e-10:
            solution = next_solution
            weights = next_weights
            break
        solution = next_solution
        weights = next_weights

    offset = solution[:3]
    centers_array = solution[3:].reshape(len(unique_groups), 3)
    residual_vectors = positions + np.einsum("nij,j->ni", rotations, offset) - centers_array[group_indices]
    residuals = np.linalg.norm(residual_vectors, axis=1)
    robust_center = float(np.median(residuals))
    robust_sigma = 1.4826 * float(np.median(np.abs(residuals - robust_center)))
    outlier_limit = max(float(huber_delta_m) * 2.0, robust_center + 3.0 * max(robust_sigma, 1e-6))
    inlier_mask = residuals <= outlier_limit
    if np.count_nonzero(inlier_mask) >= 4 and not np.all(inlier_mask):
        inlier_solution = _weighted_solution(matrix, vector, inlier_mask.astype(float))
        offset = inlier_solution[:3]
        centers_array = inlier_solution[3:].reshape(len(unique_groups), 3)
        residual_vectors = positions + np.einsum("nij,j->ni", rotations, offset) - centers_array[group_indices]
        residuals = np.linalg.norm(residual_vectors, axis=1)

    inlier_residuals = residuals[inlier_mask]
    rms = float(np.sqrt(np.mean(np.square(inlier_residuals)))) if inlier_residuals.size else float("inf")
    p95 = float(np.percentile(inlier_residuals, 95)) if inlier_residuals.size else float("inf")
    outlier_fraction = 1.0 - float(np.mean(inlier_mask))
    excitation, rotation_rank, singular_values = _rotation_excitation(rotations, group_indices)

    wrist_validation_rms = None
    direct_wrist_group_rms = None
    if wrist_positions is not None:
        wrists = np.asarray(wrist_positions, dtype=float)
        if wrists.shape != positions.shape or not np.all(np.isfinite(wrists)):
            raise ValueError("wrist_positions 必须是与 hand_positions 对齐的 Nx3 有限矩阵")
        predicted = positions + np.einsum("nij,j->ni", rotations, offset)
        wrist_errors = np.linalg.norm(predicted - wrists, axis=1)
        wrist_validation_rms = float(
            np.sqrt(np.mean(np.square(wrist_errors[inlier_mask])))
        )
        direct_residuals = wrists - np.asarray(
            [np.mean(wrists[group_indices == group], axis=0) for group in group_indices]
        )
        direct_wrist_group_rms = float(
            np.sqrt(np.mean(np.square(np.linalg.norm(direct_residuals, axis=1))))
        )

    reasons: list[str] = []
    if positions.shape[0] < limits.min_samples:
        reasons.append(f"samples={positions.shape[0]}<{limits.min_samples}")
    if len(unique_groups) < limits.min_groups:
        reasons.append(f"groups={len(unique_groups)}<{limits.min_groups}")
    if excitation < limits.min_rotation_excitation_rad:
        reasons.append(f"rotation_excitation={excitation:.4f}<{limits.min_rotation_excitation_rad:.4f}rad")
    if rotation_rank < limits.min_rotation_rank:
        reasons.append(f"rotation_rank={rotation_rank}<{limits.min_rotation_rank}")
    if outlier_fraction > limits.max_outlier_fraction:
        reasons.append(f"outlier_fraction={outlier_fraction:.3f}>{limits.max_outlier_fraction:.3f}")
    if rms > limits.max_pivot_rms_m:
        reasons.append(f"pivot_rms={rms:.4f}>{limits.max_pivot_rms_m:.4f}m")
    if p95 > limits.max_pivot_p95_m:
        reasons.append(f"pivot_p95={p95:.4f}>{limits.max_pivot_p95_m:.4f}m")
    if wrist_validation_rms is not None and wrist_validation_rms > limits.max_wrist_validation_rms_m:
        reasons.append(
            f"wrist_validation_rms={wrist_validation_rms:.4f}>"
            f"{limits.max_wrist_validation_rms_m:.4f}m"
        )

    quality = {
        "sample_count": int(positions.shape[0]),
        "group_count": len(unique_groups),
        "pivot_rms_m": rms,
        "pivot_p95_m": p95,
        "outlier_fraction": outlier_fraction,
        "rotation_excitation_rad": excitation,
        "rotation_rank": rotation_rank,
        "rotation_singular_values": singular_values,
        "wrist_validation_rms_m": wrist_validation_rms,
        "direct_wrist_group_rms_m": direct_wrist_group_rms,
    }
    return WristPivotFit(
        hand_to_wrist_offset=offset,
        group_centers={name: centers_array[index] for index, name in enumerate(unique_groups)},
        inlier_mask=inlier_mask,
        quality=quality,
        valid=not reasons,
        reasons=tuple(reasons),
    )


def select_runtime_position_source(fit: WristPivotFit) -> WristPositionSource:
    if fit.quality.get("position_observation") == "explicit_wrist_joint" and fit.valid:
        return "wrist_joint"
    direct_rms = fit.quality.get("direct_wrist_group_rms_m")
    validation_rms = fit.quality.get("wrist_validation_rms_m")
    if direct_rms is None or validation_rms is None:
        return "hand_pose_plus_pivot_offset"
    pivot_rms = float(fit.quality["pivot_rms_m"])
    if float(direct_rms) <= max(0.010, pivot_rms * 1.25) and float(validation_rms) <= 0.020:
        return "wrist_joint"
    return "hand_pose_plus_pivot_offset"