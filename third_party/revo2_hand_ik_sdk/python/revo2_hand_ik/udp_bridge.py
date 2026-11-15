"""Production FSS UDP input to Revo2 direct-control UDP output."""
from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .control_output import finger_data_set_from_sdk, make_finger_data_set_json_payload
from .control_recorder import ControlRecordPaths, DirectControlRecorder
from .diagnostic_recorder import DiagnosticRecordPath, IkDiagnosticRecorder
from .fss_udp import FssPacketError, extract_hand_mocap, parse_fss_packet
from .native import Revo2HandIkSdk

DIRECT_CONTROL_FORMAT = "finger_data_set_json_v1"


@dataclass(frozen=True)
class UdpSocketConfig:
    listen_ip: str
    listen_port: int
    recv_buffer_size: int
    recv_timeout_ms: int
    check_crc: bool
    target_ip: str
    target_port: int
    role_name: str
    print_interval_s: float


@dataclass(frozen=True)
class DirectOutputRecordConfig:
    enabled: bool
    paths: ControlRecordPaths | None


@dataclass(frozen=True)
class DebugMirrorUdpConfig:
    enabled: bool
    target_ip: str
    target_port: int


@dataclass(frozen=True)
class DebugOverlayUdpConfig:
    enabled: bool
    target_ip: str
    target_port: int


@dataclass(frozen=True)
class DiagnosticRecordConfig:
    enabled: bool
    path: DiagnosticRecordPath | None


@dataclass(frozen=True)
class DualBridgeConfig:
    config_path: Path
    left_sdk_config: Path
    right_sdk_config: Path
    udp: UdpSocketConfig
    output_record: DirectOutputRecordConfig
    debug_mirror_udp: DebugMirrorUdpConfig
    debug_overlay_udp: DebugOverlayUdpConfig
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

    directory = (config_path.parent / str(raw.get("directory", "../../log/python_control_output"))).resolve()
    csv_name = str(raw.get("csv_filename", "finger_data_set.csv"))
    jsonl_name = str(raw.get("jsonl_filename", "finger_data_set.jsonl"))
    if Path(csv_name).name != csv_name or Path(jsonl_name).name != jsonl_name:
        raise ValueError("output_record file names must not contain path components")
    return DirectOutputRecordConfig(
        enabled=True,
        paths=ControlRecordPaths(directory=directory, csv_path=directory / csv_name, jsonl_path=directory / jsonl_name),
    )


def _load_mirror_config(root: dict[str, Any]) -> DebugMirrorUdpConfig:
    raw = root.get("debug_mirror_udp", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("debug_mirror_udp must be an object")
    return DebugMirrorUdpConfig(
        enabled=bool(raw.get("enabled", False)),
        target_ip=str(raw.get("target_ip", "127.0.0.1")),
        target_port=int(raw.get("target_port", 16124)),
    )


def _load_overlay_config(root: dict[str, Any]) -> DebugOverlayUdpConfig:
    raw = root.get("debug_overlay_udp", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("debug_overlay_udp must be an object")
    return DebugOverlayUdpConfig(
        enabled=bool(raw.get("enabled", False)),
        target_ip=str(raw.get("target_ip", "127.0.0.1")),
        target_port=int(raw.get("target_port", 16125)),
    )


def _load_diagnostic_record_config(config_path: Path, root: dict[str, Any]) -> DiagnosticRecordConfig:
    raw = root.get("diagnostic_record", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("diagnostic_record must be an object")
    if not bool(raw.get("enabled", False)):
        return DiagnosticRecordConfig(enabled=False, path=None)
    directory = (config_path.parent / str(raw.get("directory", "../../log/python_control_output"))).resolve()
    csv_name = str(raw.get("csv_filename", "ik_diagnostic.csv"))
    if Path(csv_name).name != csv_name:
        raise ValueError("diagnostic_record.csv_filename must not contain path components")
    return DiagnosticRecordConfig(enabled=True, path=DiagnosticRecordPath(path=directory / csv_name))


def make_debug_overlay_payload(
    left_debug: Mapping[str, Any],
    right_debug: Mapping[str, Any],
    *,
    sequence: int,
    timestamp_ms: int,
    left_raw_points: Mapping[str, Any] | None = None,
    right_raw_points: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize raw FSS plus mapped/task/FK diagnostics for RViz."""
    def clean_point(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not bool(value.get("valid", False)):
            return {"x": None, "y": None, "z": None, "valid": False}
        return {
            "x": float(value.get("x", 0.0)),
            "y": float(value.get("y", 0.0)),
            "z": float(value.get("z", 0.0)),
            "valid": True,
        }

    def raw_input_point(value: Any) -> dict[str, Any]:
        if isinstance(value, (tuple, list)) and len(value) == 3:
            try:
                return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2]), "valid": True}
            except (TypeError, ValueError):
                pass
        return {"x": None, "y": None, "z": None, "valid": False}

    def hand(side: str, debug: Mapping[str, Any], input_points: Mapping[str, Any] | None) -> dict[str, Any]:
        mapped_points = debug.get("points", {})
        if not isinstance(mapped_points, Mapping):
            mapped_points = {}
        if not isinstance(input_points, Mapping):
            input_points = {}
        order = (
            "thumb_near", "thumb_middle", "thumb_far",
            "index_near", "index_middle", "index_far",
            "middle_near", "middle_middle", "middle_far",
            "ring_near", "ring_middle", "ring_far",
            "little_near", "little_middle", "little_far",
        )
        raw_long_tasks = debug.get("long_finger_tasks", [])
        long_tasks = []
        if isinstance(raw_long_tasks, list):
            for task in raw_long_tasks:
                if not isinstance(task, Mapping):
                    continue
                long_tasks.append({
                    "name": str(task.get("name", "")),
                    "valid": bool(task.get("valid", False)),
                    "target_middle": clean_point(task.get("target_middle")),
                    "target_far": clean_point(task.get("target_far")),
                    "robot_middle": clean_point(task.get("robot_middle")),
                    "robot_far": clean_point(task.get("robot_far")),
                })
        return {
            "side": side,
            # Raw landmarks are emitted in the original FSS world coordinates
            # as received by the adapter. No palm-frame transform or scaling.
            "raw_frame_id": "world",
            "raw_points": [{"name": name, "point": raw_input_point(input_points.get(name))}
                           for name in ("hand", *order)],
            "frame_id": str(debug.get("frame_id", "")),
            "calibrated": bool(debug.get("calibrated", False)),
            "points": [{"name": name, "point": clean_point(mapped_points.get(name))} for name in order],
            "thumb_task": {
                "target_middle": clean_point(debug.get("thumb_target_middle")),
                "target_far": clean_point(debug.get("thumb_target_far")),
                "robot_middle": clean_point(debug.get("thumb_robot_middle")),
                "robot_far": clean_point(debug.get("thumb_robot_far")),
            },
            "long_finger_tasks": long_tasks,
        }

    payload = {
        "protocol": "revo2_ik_debug_overlay_v1",
        "sequence": int(sequence),
        "timestamp_ms": int(timestamp_ms),
        "hands": [hand("left", left_debug, left_raw_points), hand("right", right_debug, right_raw_points)],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_dual_bridge_config(config_path: str | Path) -> DualBridgeConfig:
    """Load the only supported Python runtime configuration."""
    path, root = _read_json(config_path)
    if root.get("bridge_mode") != "dual_hand":
        raise ValueError("bridge_mode must be dual_hand")

    incoming = _require_object(root, "input_udp")
    outgoing = _require_object(root, "output_udp")
    if outgoing.get("packet_format") != DIRECT_CONTROL_FORMAT:
        raise ValueError(f"output_udp.packet_format must be {DIRECT_CONTROL_FORMAT}")

    udp = UdpSocketConfig(
        listen_ip=str(incoming.get("listen_ip", "0.0.0.0")),
        listen_port=int(incoming.get("listen_port", 15000)),
        recv_buffer_size=max(2048, int(incoming.get("recv_buffer_size", 4096))),
        recv_timeout_ms=max(1, int(incoming.get("recv_timeout_ms", 20))),
        check_crc=bool(incoming.get("enable_crc_check", True)),
        target_ip=str(outgoing.get("target_ip", "127.0.0.1")),
        target_port=int(outgoing.get("target_port", 5555)),
        role_name=str(outgoing.get("role_name", "Revo2HandIK")),
        print_interval_s=max(0.1, float(root.get("status_print_interval_s", 1.0))),
    )
    return DualBridgeConfig(
        config_path=path,
        left_sdk_config=_resolve_sdk_config(path, _require_object(root, "left_hand"), "left"),
        right_sdk_config=_resolve_sdk_config(path, _require_object(root, "right_hand"), "right"),
        udp=udp,
        output_record=_load_record_config(path, root),
        debug_mirror_udp=_load_mirror_config(root),
        debug_overlay_udp=_load_overlay_config(root),
        diagnostic_record=_load_diagnostic_record_config(path, root),
    )


class FssUdpDualHandIkBridge:
    """One FSS input packet, two SDK solves, one direct-control UDP output."""

    def __init__(self, config_path: str | Path, *, library_path: str | Path | None = None):
        self.cfg = load_dual_bridge_config(config_path)
        self.left_sdk = Revo2HandIkSdk(self.cfg.left_sdk_config, library_path)
        try:
            self.right_sdk = Revo2HandIkSdk(self.cfg.right_sdk_config, library_path)
        except Exception:
            self.left_sdk.close()
            raise

        self.udp = self.cfg.udp
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind((self.udp.listen_ip, self.udp.listen_port))
        self.rx.settimeout(self.udp.recv_timeout_ms / 1000.0)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mirror_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.cfg.debug_mirror_udp.enabled else None
        self.overlay_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.cfg.debug_overlay_udp.enabled else None
        self.recorder = DirectControlRecorder(self.cfg.output_record.paths) if self.cfg.output_record.enabled and self.cfg.output_record.paths else None
        self.diagnostic_recorder = IkDiagnosticRecorder(self.cfg.diagnostic_record.path.path) if self.cfg.diagnostic_record.enabled and self.cfg.diagnostic_record.path else None
        self.running = False
        self.received = 0
        self.forwarded = 0
        self.dropped = 0
        self._last_status = time.monotonic()
        self._last_overlay: bytes | None = None

    def __enter__(self) -> "FssUdpDualHandIkBridge":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.running = False
        for sock in (getattr(self, "rx", None), getattr(self, "tx", None), getattr(self, "mirror_tx", None), getattr(self, "overlay_tx", None)):
            if sock is not None:
                sock.close()
        if getattr(self, "recorder", None) is not None:
            self.recorder.close()
        if getattr(self, "diagnostic_recorder", None) is not None:
            self.diagnostic_recorder.close()
        self.left_sdk.close()
        self.right_sdk.close()

    def _status(self) -> None:
        now = time.monotonic()
        if now - self._last_status >= self.udp.print_interval_s:
            print(f"[revo2-python-udp] received={self.received} forwarded={self.forwarded} dropped={self.dropped}")
            self._last_status = now

    def process_datagram(self, datagram: bytes) -> bytes:
        packet = parse_fss_packet(datagram, check_crc=self.udp.check_crc)
        left_points, left_quats = extract_hand_mocap(packet, "left")
        right_points, right_quats = extract_hand_mocap(packet, "right")
        sequence = self.received
        left_result = self.left_sdk.solve(left_points, left_quats, sequence=sequence, timestamp_ms=packet.timestamp_ms)
        right_result = self.right_sdk.solve(right_points, right_quats, sequence=sequence, timestamp_ms=packet.timestamp_ms)
        values = finger_data_set_from_sdk(left_result, right_result)
        if self.recorder is not None:
            self.recorder.write(
                sequence=sequence,
                timestamp_ms=packet.timestamp_ms,
                values=values,
                left_solve_time_ms=float(left_result.get("solve_time_ms", 0.0)),
                right_solve_time_ms=float(right_result.get("solve_time_ms", 0.0)),
            )
        if getattr(self, "diagnostic_recorder", None) is not None:
            self.diagnostic_recorder.write(
                sequence=sequence,
                timestamp_ms=packet.timestamp_ms,
                values=values,
                left_points=left_points,
                left_quats=left_quats,
                right_points=right_points,
                right_quats=right_quats,
                left_result=left_result,
                right_result=right_result,
            )
        self._last_overlay = make_debug_overlay_payload(
            left_result.get("debug", {}) if isinstance(left_result.get("debug", {}), dict) else {},
            right_result.get("debug", {}) if isinstance(right_result.get("debug", {}), dict) else {},
            sequence=sequence,
            timestamp_ms=packet.timestamp_ms,
            left_raw_points=left_points,
            right_raw_points=right_points,
        )
        return make_finger_data_set_json_payload(
            left_result,
            right_result,
            role_name=self.udp.role_name,
            sequence=sequence,
            timestamp_ms=packet.timestamp_ms,
        )

    def run_forever(self) -> None:
        self.running = True
        print(
            "[revo2-python-udp] mode=dual_hand "
            f"input={self.udp.listen_ip}:{self.udp.listen_port} "
            f"output={self.udp.target_ip}:{self.udp.target_port} "
            f"record={'on' if self.recorder else 'off'} "
            f"diagnostic={'on' if self.diagnostic_recorder else 'off'} "
            f"mirror={'on' if self.mirror_tx else 'off'} "
            f"overlay={'on' if self.overlay_tx else 'off'}"
        )
        try:
            while self.running:
                try:
                    datagram, _ = self.rx.recvfrom(self.udp.recv_buffer_size)
                except TimeoutError:
                    self._status()
                    continue
                self.received += 1
                try:
                    output = self.process_datagram(datagram)
                    self.tx.sendto(output, (self.udp.target_ip, self.udp.target_port))
                    if self.mirror_tx is not None:
                        self.mirror_tx.sendto(output, (self.cfg.debug_mirror_udp.target_ip, self.cfg.debug_mirror_udp.target_port))
                    if self.overlay_tx is not None and self._last_overlay is not None:
                        self.overlay_tx.sendto(
                            self._last_overlay,
                            (self.cfg.debug_overlay_udp.target_ip, self.cfg.debug_overlay_udp.target_port),
                        )
                    self.forwarded += 1
                except (FssPacketError, ValueError, RuntimeError) as exc:
                    self.dropped += 1
                    if self.dropped <= 5 or self.dropped % 100 == 0:
                        print(f"[revo2-python-udp][WARN] drop packet: {exc}")
                self._status()
        except KeyboardInterrupt:
            pass


def make_bridge(config_path: str | Path, *, library_path: str | Path | None = None) -> FssUdpDualHandIkBridge:
    return FssUdpDualHandIkBridge(config_path, library_path=library_path)
