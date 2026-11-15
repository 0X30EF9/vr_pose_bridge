"""Durable per-frame recording for the direct-call Python SDK."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .control_output import FINGER_DATA_SET_ORDER, finger_data_set_named


@dataclass(frozen=True)
class ControlRecordPaths:
    directory: Path
    csv_path: Path
    jsonl_path: Path


class DirectControlRecorder:
    """Write exactly the radian values emitted by the Python SDK API.

    Files are truncated once at SDK startup.  This makes one validation run
    correspond to one coherent capture and prevents stale values from previous
    tests being mistaken for current output.
    """

    def __init__(self, paths: ControlRecordPaths) -> None:
        paths.directory.mkdir(parents=True, exist_ok=True)
        self.paths = paths
        self._csv_stream = paths.csv_path.open("w", encoding="utf-8", newline="")
        self._jsonl_stream = paths.jsonl_path.open("w", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_stream,
            fieldnames=(
                "sequence",
                "timestamp_ms",
                "left_solve_time_ms",
                "right_solve_time_ms",
                *FINGER_DATA_SET_ORDER,
            ),
        )
        self._csv_writer.writeheader()
        self._csv_stream.flush()

    def write(
        self,
        *,
        sequence: int,
        timestamp_ms: int,
        values: Sequence[float],
        left_solve_time_ms: float,
        right_solve_time_ms: float,
    ) -> None:
        if len(values) != len(FINGER_DATA_SET_ORDER):
            raise ValueError("joint position recording requires exactly 12 values")
        named = finger_data_set_named(list(values))
        row: dict[str, Any] = {
            "sequence": int(sequence),
            "timestamp_ms": int(timestamp_ms),
            "left_solve_time_ms": float(left_solve_time_ms),
            "right_solve_time_ms": float(right_solve_time_ms),
            **named,
        }
        self._csv_writer.writerow(row)
        self._jsonl_stream.write(json.dumps({
            "protocol": "revo2_dual_hand_ik_sdk_v2",
            "sequence": int(sequence),
            "timestamp_ms": int(timestamp_ms),
            "joint_positions": [float(value) for value in values],
            "joint_order": list(FINGER_DATA_SET_ORDER),
            "joint_unit": "rad",
            "joint_positions_named": named,
            "left_solve_time_ms": float(left_solve_time_ms),
            "right_solve_time_ms": float(right_solve_time_ms),
        }, ensure_ascii=False, separators=(",", ":")) + "\n")
        # Validation logs need to survive Ctrl-C or a downstream process crash.
        self._csv_stream.flush()
        self._jsonl_stream.flush()

    def close(self) -> None:
        for stream in (getattr(self, "_csv_stream", None), getattr(self, "_jsonl_stream", None)):
            if stream is not None and not stream.closed:
                stream.close()
