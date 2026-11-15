"""Per-frame raw-input, mapped-input, and final-radian CSV diagnostics."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .control_output import FINGER_DATA_SET_ORDER

RAW_POINT_NAMES: tuple[str, ...] = (
    "hand",
    "thumb_near", "thumb_middle", "thumb_far",
    "index_near", "index_middle", "index_far",
    "middle_near", "middle_middle", "middle_far",
    "ring_near", "ring_middle", "ring_far",
    "little_near", "little_middle", "little_far",
)

MAPPED_POINT_NAMES: tuple[str, ...] = RAW_POINT_NAMES[1:]
THUMB_DEBUG_POINT_NAMES: tuple[str, ...] = (
    "thumb_target_middle",
    "thumb_target_far",
    "thumb_robot_middle",
    "thumb_robot_far",
)
LONG_FINGER_NAMES: tuple[str, ...] = ("index", "middle", "ring", "little")
LONG_FINGER_POINT_NAMES: tuple[str, ...] = (
    "target_middle", "target_far", "robot_middle", "robot_far",
)


@dataclass(frozen=True)
class DiagnosticRecordPath:
    path: Path


def _point_fields(prefix: str) -> tuple[str, ...]:
    return (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z", f"{prefix}_valid")


def _quat_fields(prefix: str) -> tuple[str, ...]:
    return (f"{prefix}_qw", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_quat_valid")


def _point_row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "x": value.get("x", ""),
            "y": value.get("y", ""),
            "z": value.get("z", ""),
            "valid": int(bool(value.get("valid", True))),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return {"x": value[0], "y": value[1], "z": value[2], "valid": 1}
    return {"x": "", "y": "", "z": "", "valid": 0}


def _quat_row(value: Any) -> dict[str, Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return {"qw": value[0], "qx": value[1], "qy": value[2], "qz": value[3], "quat_valid": 1}
    return {"qw": "", "qx": "", "qy": "", "qz": "", "quat_valid": 0}


class IkDiagnosticRecorder:
    """Write a self-contained CSV that can correlate SDK input and final radian output."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._stream.flush()

    @staticmethod
    def _fieldnames() -> tuple[str, ...]:
        fields: list[str] = [
            "sequence", "timestamp_ms", "left_solve_time_ms", "right_solve_time_ms",
            *FINGER_DATA_SET_ORDER,
        ]
        for side in ("left", "right"):
            for name in RAW_POINT_NAMES:
                fields.extend(_point_fields(f"{side}_raw_{name}"))
                fields.extend(_quat_fields(f"{side}_raw_{name}"))
            fields.append(f"{side}_debug_calibrated")
            fields.append(f"{side}_debug_frame_id")
            for name in MAPPED_POINT_NAMES:
                fields.extend(_point_fields(f"{side}_mapped_{name}"))
            for name in THUMB_DEBUG_POINT_NAMES:
                fields.extend(_point_fields(f"{side}_{name}"))
            for finger in LONG_FINGER_NAMES:
                for name in LONG_FINGER_POINT_NAMES:
                    fields.extend(_point_fields(f"{side}_{finger}_{name}"))
                fields.append(f"{side}_{finger}_task_valid")
        return tuple(fields)

    @staticmethod
    def _write_point(row: dict[str, Any], prefix: str, value: Any) -> None:
        parsed = _point_row(value)
        for key, cell in parsed.items():
            row[f"{prefix}_{key}"] = cell

    @staticmethod
    def _write_quat(row: dict[str, Any], prefix: str, value: Any) -> None:
        parsed = _quat_row(value)
        for key, cell in parsed.items():
            row[f"{prefix}_{key}"] = cell

    def _write_hand(
        self,
        row: dict[str, Any],
        side: str,
        points: Mapping[str, Any],
        quats: Mapping[str, Any],
        debug: Mapping[str, Any],
    ) -> None:
        for name in RAW_POINT_NAMES:
            self._write_point(row, f"{side}_raw_{name}", points.get(name))
            self._write_quat(row, f"{side}_raw_{name}", quats.get(name))
        row[f"{side}_debug_calibrated"] = int(bool(debug.get("calibrated", False)))
        row[f"{side}_debug_frame_id"] = str(debug.get("frame_id", ""))
        mapped = debug.get("points", {})
        if not isinstance(mapped, Mapping):
            mapped = {}
        for name in MAPPED_POINT_NAMES:
            self._write_point(row, f"{side}_mapped_{name}", mapped.get(name))
        for name in THUMB_DEBUG_POINT_NAMES:
            self._write_point(row, f"{side}_{name}", debug.get(name))
        raw_long_tasks = debug.get("long_finger_tasks", [])
        tasks_by_name = {
            str(task.get("name", "")): task
            for task in raw_long_tasks
            if isinstance(task, Mapping)
        } if isinstance(raw_long_tasks, list) else {}
        for finger in LONG_FINGER_NAMES:
            task = tasks_by_name.get(finger, {})
            for name in LONG_FINGER_POINT_NAMES:
                self._write_point(row, f"{side}_{finger}_{name}", task.get(name) if isinstance(task, Mapping) else None)
            row[f"{side}_{finger}_task_valid"] = int(bool(task.get("valid", False))) if isinstance(task, Mapping) else 0

    def write(
        self,
        *,
        sequence: int,
        timestamp_ms: int,
        values: Sequence[float],
        left_points: Mapping[str, Any],
        left_quats: Mapping[str, Any],
        right_points: Mapping[str, Any],
        right_quats: Mapping[str, Any],
        left_result: Mapping[str, Any],
        right_result: Mapping[str, Any],
    ) -> None:
        if len(values) != len(FINGER_DATA_SET_ORDER):
            raise ValueError("diagnostic recording requires exactly 12 radian joint outputs")
        row: dict[str, Any] = {
            "sequence": int(sequence),
            "timestamp_ms": int(timestamp_ms),
            "left_solve_time_ms": float(left_result.get("solve_time_ms", 0.0)),
            "right_solve_time_ms": float(right_result.get("solve_time_ms", 0.0)),
        }
        row.update({name: float(value) for name, value in zip(FINGER_DATA_SET_ORDER, values)})
        left_debug = left_result.get("debug", {})
        right_debug = right_result.get("debug", {})
        self._write_hand(row, "left", left_points, left_quats,
                         left_debug if isinstance(left_debug, Mapping) else {})
        self._write_hand(row, "right", right_points, right_quats,
                         right_debug if isinstance(right_debug, Mapping) else {})
        self._writer.writerow(row)
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()
