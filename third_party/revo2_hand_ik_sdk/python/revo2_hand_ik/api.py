"""Interface-call Python SDK for Revo2 dual-hand IK.

No UDP socket is created in this module.  The host program owns its data source
and calls :class:`Revo2DualHandIkSdk.solve` once per frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .control_output import (
    DIRECT_JOINT_OUTPUT_ORDER,
    joint_position_rad_from_sdk,
    joint_position_rad_named,
    make_joint_position_rad_payload,
)
from .control_recorder import ControlRecordPaths, DirectControlRecorder
from .diagnostic_recorder import DiagnosticRecordPath, IkDiagnosticRecorder
from .native import Revo2HandIkSdk


@dataclass(frozen=True)
class DirectOutputRecordConfig:
    enabled: bool
    paths: ControlRecordPaths | None


@dataclass(frozen=True)
class DiagnosticRecordConfig:
    enabled: bool
    path: DiagnosticRecordPath | None


@dataclass(frozen=True)
class DualSdkConfig:
    config_path: Path
    left_sdk_config: Path
    right_sdk_config: Path
    output_record: DirectOutputRecordConfig
    diagnostic_record: DiagnosticRecordConfig


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    root = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("JSON root must be an object")
    return resolved, root


def _require_object(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config requires object: {key}")
    return value


def _resolve_sdk_config(parent_config: Path, hand: dict[str, Any], side: str) -> Path:
    value = hand.get("sdk_config")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{side}_hand.sdk_config must be a non-empty string")
    path = (parent_config.parent / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{side}_hand.sdk_config does not exist: {path}")
    return path


def _load_record_config(config_path: Path, root: dict[str, Any]) -> DirectOutputRecordConfig:
    raw = root.get("output_record", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("output_record must be an object")
    if not bool(raw.get("enabled", False)):
        return DirectOutputRecordConfig(enabled=False, paths=None)

    directory = (config_path.parent / str(raw.get("directory", "../../log/python_sdk_output"))).resolve()
    csv_name = str(raw.get("csv_filename", "joint_positions_rad.csv"))
    jsonl_name = str(raw.get("jsonl_filename", "joint_positions_rad.jsonl"))
    if Path(csv_name).name != csv_name or Path(jsonl_name).name != jsonl_name:
        raise ValueError("output_record file names must not contain path components")
    return DirectOutputRecordConfig(
        enabled=True,
        paths=ControlRecordPaths(directory=directory, csv_path=directory / csv_name, jsonl_path=directory / jsonl_name),
    )


def _load_diagnostic_record_config(config_path: Path, root: dict[str, Any]) -> DiagnosticRecordConfig:
    raw = root.get("diagnostic_record", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("diagnostic_record must be an object")
    if not bool(raw.get("enabled", False)):
        return DiagnosticRecordConfig(enabled=False, path=None)
    directory = (config_path.parent / str(raw.get("directory", "../../log/python_sdk_output"))).resolve()
    csv_name = str(raw.get("csv_filename", "ik_diagnostic_rad.csv"))
    if Path(csv_name).name != csv_name:
        raise ValueError("diagnostic_record.csv_filename must not contain path components")
    return DiagnosticRecordConfig(enabled=True, path=DiagnosticRecordPath(path=directory / csv_name))


def load_dual_sdk_config(config_path: str | Path) -> DualSdkConfig:
    """Load the direct-call Python SDK configuration.

    The config intentionally has no ``input_udp`` or ``output_udp`` sections.
    """
    path, root = _read_json(config_path)
    mode = root.get("sdk_mode", root.get("bridge_mode"))
    if mode != "dual_hand_direct_api":
        raise ValueError("sdk_mode must be dual_hand_direct_api")
    return DualSdkConfig(
        config_path=path,
        left_sdk_config=_resolve_sdk_config(path, _require_object(root, "left_hand"), "left"),
        right_sdk_config=_resolve_sdk_config(path, _require_object(root, "right_hand"), "right"),
        output_record=_load_record_config(path, root),
        diagnostic_record=_load_diagnostic_record_config(path, root),
    )


def _public_hand_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Filter native output to the public Python result contract.

    Native still returns normalized values for C ABI compatibility; this public
    Python layer hides them and exposes radians only.
    """
    joints = []
    for item in result.get("joints", []):
        if not isinstance(item, Mapping):
            continue
        joints.append({"name": str(item.get("name", "")), "rad": float(item.get("rad", 0.0))})
    return {
        "joint_count": int(result.get("joint_count", len(joints))),
        "joints": joints,
        "joint_unit": "rad",
        "solve_time_ms": float(result.get("solve_time_ms", 0.0)),
        "python_call_time_ms": float(result.get("python_call_time_ms", 0.0)),
        "debug": result.get("debug", {}),
    }


class Revo2DualHandIkSdk:
    """Stateful dual-hand SDK called directly from Python code.

    A typical integration creates this object once, calls ``solve`` every frame,
    and calls ``close`` when the process exits.  Startup calibration ignores a
    short warm-up, then accepts only consecutive stable open-hand frames and
    performs a window quality check. Callers should keep both hands naturally
    open and still until both debug frames report ``calibrated=True``.
    """

    def __init__(self, config_path: str | Path, *, library_path: str | Path | None = None):
        self.cfg = load_dual_sdk_config(config_path)
        self.left_sdk = Revo2HandIkSdk(self.cfg.left_sdk_config, library_path)
        try:
            self.right_sdk = Revo2HandIkSdk(self.cfg.right_sdk_config, library_path)
        except Exception:
            self.left_sdk.close()
            raise
        self.recorder = DirectControlRecorder(self.cfg.output_record.paths) if self.cfg.output_record.enabled and self.cfg.output_record.paths else None
        self.diagnostic_recorder = IkDiagnosticRecorder(self.cfg.diagnostic_record.path.path) if self.cfg.diagnostic_record.enabled and self.cfg.diagnostic_record.path else None
        self.solved_frames = 0

    def __enter__(self) -> "Revo2DualHandIkSdk":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "recorder", None) is not None:
            self.recorder.close()
        if getattr(self, "diagnostic_recorder", None) is not None:
            self.diagnostic_recorder.close()
        if getattr(self, "left_sdk", None) is not None:
            self.left_sdk.close()
        if getattr(self, "right_sdk", None) is not None:
            self.right_sdk.close()

    def reset(self) -> None:
        """Reset open-hand calibration and internal solver seed for both hands."""
        self.left_sdk.reset()
        self.right_sdk.reset()
        self.solved_frames = 0

    def solve(
        self,
        *,
        left_points: Mapping[str, Any],
        right_points: Mapping[str, Any],
        left_quaternions_wxyz: Mapping[str, Any] | None = None,
        right_quaternions_wxyz: Mapping[str, Any] | None = None,
        sequence: int | None = None,
        timestamp_ms: int = 0,
    ) -> dict[str, Any]:
        """Solve one dual-hand frame and return joint positions in radians.

        ``left_points`` and ``right_points`` use the canonical point names:
        ``hand``, ``thumb_near/middle/far``, ``index_near/middle/far``,
        ``middle_near/middle/far``, ``ring_near/middle/far``, and
        ``little_near/middle/far``.  Point values are ``(x, y, z)`` in metres.
        Quaternions are optional and must be in ``(w, x, y, z)`` order.
        """
        if sequence is None:
            sequence = self.solved_frames
        left_result_raw = self.left_sdk.solve(
            left_points,
            left_quaternions_wxyz,
            sequence=int(sequence),
            timestamp_ms=int(timestamp_ms),
        )
        right_result_raw = self.right_sdk.solve(
            right_points,
            right_quaternions_wxyz,
            sequence=int(sequence),
            timestamp_ms=int(timestamp_ms),
        )
        values = joint_position_rad_from_sdk(left_result_raw, right_result_raw)
        payload = make_joint_position_rad_payload(
            left_result_raw,
            right_result_raw,
            sequence=int(sequence),
            timestamp_ms=int(timestamp_ms),
        )
        payload["left"] = _public_hand_result(left_result_raw)
        payload["right"] = _public_hand_result(right_result_raw)

        if self.recorder is not None:
            self.recorder.write(
                sequence=int(sequence),
                timestamp_ms=int(timestamp_ms),
                values=values,
                left_solve_time_ms=float(left_result_raw.get("solve_time_ms", 0.0)),
                right_solve_time_ms=float(right_result_raw.get("solve_time_ms", 0.0)),
            )
        if self.diagnostic_recorder is not None:
            self.diagnostic_recorder.write(
                sequence=int(sequence),
                timestamp_ms=int(timestamp_ms),
                values=values,
                left_points=left_points,
                left_quats=left_quaternions_wxyz or {},
                right_points=right_points,
                right_quats=right_quaternions_wxyz or {},
                left_result=left_result_raw,
                right_result=right_result_raw,
            )

        self.solved_frames += 1
        return payload


__all__ = [
    "DIRECT_JOINT_OUTPUT_ORDER",
    "DualSdkConfig",
    "Revo2DualHandIkSdk",
    "load_dual_sdk_config",
    "joint_position_rad_named",
]
