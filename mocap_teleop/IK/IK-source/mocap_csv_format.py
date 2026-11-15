from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from teleop_mocap_protocol import (
    MOCAP_JOINT_NAMES,
    MocapFrame,
    MocapPacketError,
    encode_mocap_g2_frame,
    frame_from_joint_rows,
    parse_mocap_packet,
)

CSV_SKELETON_NAMES = list(MOCAP_JOINT_NAMES)
CSV_EXTRA_FIELDS = [
    "leftButtonState", "leftButton_mode", "leftButton_joystick", "leftButton_x",
    "leftButton_y", "leftButton_trigger", "leftButton_grip", "leftButton_home",
    "leftJoyStickX", "leftJoyStickY", "rightButtonState", "rightButton_mode",
    "rightButton_joystick", "rightButton_a", "rightButton_b", "rightButton_trigger",
    "rightButton_grip", "rightButton_home", "rightJoyStickX", "rightJoyStickY",
    "reserve0", "reserve1", "reserve2", "reserve3",
]
CSV_FIELDNAMES = ["systemTimestampUs"]
for skeleton_name in CSV_SKELETON_NAMES:
    CSV_FIELDNAMES.extend(
        (
            f"pos_{skeleton_name}_X", f"pos_{skeleton_name}_Y", f"pos_{skeleton_name}_Z",
            f"quat_{skeleton_name}_W", f"quat_{skeleton_name}_X",
            f"quat_{skeleton_name}_Y", f"quat_{skeleton_name}_Z",
        )
    )
CSV_FIELDNAMES.extend(CSV_EXTRA_FIELDS)

UPPER_BODY_JOINTS = ("shoulder", "elbow", "wrist", "hand")
TORSO_REFERENCE_JOINTS = (
    "waist", "spine1", "spine2", "chest", "neck", "head",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
)


def make_vr_csv_name(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else timestamp
    return "vr_" + time.strftime("%Y%m%d_%H%M%S", time.localtime(value)) + ".csv"


def normalize_quat_xyzw(values: Iterable[float]) -> list[float]:
    quat = [float(value) for value in values]
    if len(quat) != 4 or not all(math.isfinite(value) for value in quat):
        raise ValueError("CSV 四元数必须包含 4 个有限值")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 1e-12:
        raise ValueError("CSV 四元数长度为 0")
    return [value / norm for value in quat]


def quat_wxyz_to_xyzw(row: Mapping[str, Any], name: str) -> list[float]:
    return normalize_quat_xyzw(
        (
            float(row[f"quat_{name}_X"]), float(row[f"quat_{name}_Y"]),
            float(row[f"quat_{name}_Z"]), float(row[f"quat_{name}_W"]),
        )
    )


def pos_xyz(row: Mapping[str, Any], name: str) -> list[float]:
    values = [float(row[f"pos_{name}_{axis}"]) for axis in "XYZ"]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"CSV 节点 {name} 位置包含 NaN/Inf")
    return values


def timestamp_us_from_csv(value: Any) -> tuple[int, str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV 时间戳无效: {value!r}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"CSV 时间戳无效: {value!r}")
    if numeric >= 1e14:
        return int(round(numeric)), "microseconds"
    if numeric >= 1e11:
        return int(round(numeric * 1000.0)), "milliseconds_in_systemTimestampUs"
    # 历史 FSS 的 uint32 时间戳以及本项目录制文件均按毫秒递增。
    return int(round(numeric * 1000.0)), "milliseconds_legacy"


def _require_pose_columns(row: Mapping[str, Any], line_number: int) -> None:
    missing: list[str] = []
    for name in CSV_SKELETON_NAMES:
        for column in (
            f"pos_{name}_X", f"pos_{name}_Y", f"pos_{name}_Z",
            f"quat_{name}_W", f"quat_{name}_X", f"quat_{name}_Y", f"quat_{name}_Z",
        ):
            if row.get(column) in (None, ""):
                missing.append(column)
                if len(missing) >= 8:
                    break
        if len(missing) >= 8:
            break
    if missing:
        raise ValueError(f"CSV 第 {line_number} 行缺少 64 节点字段: {missing}")


def csv_row_to_mocap_frame(row: Mapping[str, Any], line_number: int) -> MocapFrame:
    if row.get("systemTimestampUs") in (None, ""):
        raise ValueError(f"CSV 第 {line_number} 行缺少字段: systemTimestampUs")
    _require_pose_columns(row, line_number)
    timestamp_us, timestamp_unit = timestamp_us_from_csv(row["systemTimestampUs"])
    rows_xyzw: list[tuple[float, ...]] = []
    for name in CSV_SKELETON_NAMES:
        position = pos_xyz(row, name)
        quat = quat_wxyz_to_xyzw(row, name)
        rows_xyzw.append(tuple(position + quat))
    return frame_from_joint_rows(
        rows_xyzw,
        timestamp_s=timestamp_us / 1_000_000.0,
        timestamp_us=timestamp_us,
        source="formotion_g2_csv",
        wire={
            "layout": "formotion_g2_csv_v1",
            "line_number": line_number,
            "timestamp_unit_detected": timestamp_unit,
        },
    )


def csv_row_to_replay_frame(row: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    return csv_row_to_mocap_frame(row, line_number).to_replay_dict(frame_index=line_number - 2)


def upper_body_skeleton_from_csv_row(row: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return csv_row_to_mocap_frame(row, 0).to_replay_dict()["skeleton"]


def load_mocap_frames_from_csv(path: Path, max_frames: int | None, stride: int) -> list[MocapFrame]:
    if stride <= 0:
        raise ValueError("--stride 必须 >= 1")
    frames: list[MocapFrame] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头: {path}")
        for row_index, row in enumerate(reader):
            if row_index % stride != 0:
                continue
            frames.append(csv_row_to_mocap_frame(row, row_index + 2))
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"没有可回放帧: {path}")
    return frames


def load_replay_frames_from_csv(path: Path, max_frames: int | None, stride: int) -> list[dict[str, Any]]:
    return [frame.to_replay_dict(frame_index=index) for index, frame in enumerate(load_mocap_frames_from_csv(path, max_frames, stride))]


def resolve_csv_input(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("vr_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        path = candidates[0] if candidates else path / "frames.csv"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV: {path}")
    return path


def blank_csv_row(timestamp_us: int | float) -> dict[str, Any]:
    row: dict[str, Any] = {"systemTimestampUs": int(timestamp_us)}
    for name in CSV_SKELETON_NAMES:
        row[f"pos_{name}_X"] = 0.0
        row[f"pos_{name}_Y"] = 0.0
        row[f"pos_{name}_Z"] = 0.0
        row[f"quat_{name}_W"] = 1.0
        row[f"quat_{name}_X"] = 0.0
        row[f"quat_{name}_Y"] = 0.0
        row[f"quat_{name}_Z"] = 0.0
    row.update({field: 0.0 for field in CSV_EXTRA_FIELDS})
    return row


def mocap_frame_to_csv_row(frame: MocapFrame) -> dict[str, Any]:
    row = blank_csv_row(frame.timestamp_us)
    for name, pose in zip(CSV_SKELETON_NAMES, frame.joints):
        row[f"pos_{name}_X"], row[f"pos_{name}_Y"], row[f"pos_{name}_Z"] = pose.position
        qx, qy, qz, qw = pose.quaternion_xyzw
        row[f"quat_{name}_W"] = qw
        row[f"quat_{name}_X"] = qx
        row[f"quat_{name}_Y"] = qy
        row[f"quat_{name}_Z"] = qz
    return row


def udp_packet_to_csv_row(
    data: bytes,
    receive_time: float,
    *,
    device_mode: str = "glove",
) -> dict[str, Any]:
    try:
        return mocap_frame_to_csv_row(
            parse_mocap_packet(data, receive_time, device_mode=device_mode)
        )
    except MocapPacketError:
        raise
    except (ValueError, OverflowError) as exc:
        raise MocapPacketError(f"UDP 包规范化失败: {exc}") from exc


def csv_row_to_g2_packet(row: Mapping[str, Any], line_number: int) -> bytes:
    return encode_mocap_g2_frame(csv_row_to_mocap_frame(row, line_number))


def write_csv_header(file) -> csv.DictWriter:
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    return writer