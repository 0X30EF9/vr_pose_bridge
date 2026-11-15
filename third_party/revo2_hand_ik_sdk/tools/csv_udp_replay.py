#!/usr/bin/env python3
"""Replay native Formotion G2 or legacy FSS CSV recordings as G2 UDP.

Supported CSV inputs:

1. Native Formotion G2 export (preferred)
   - ``systemTimestampUs`` (optional but normally present)
   - 64 complete pose groups:
     ``pos_<joint>_X/Y/Z`` + ``quat_<joint>_W/X/Y/Z``

2. Legacy project FSS export
   - legacy 47-joint names using ``quat_*_X/Y/Z/W`` and ``pos_*_X/Y/Z``

Both formats are encoded as the current UDP protocol:

    AA 55 | id=0x0000 | length=1850 | timestamp_us | float[64][7]

The complete UDP datagram remains 1860 bytes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "python"))

from revo2_hand_ik.fss_udp import (  # noqa: E402
    JOINT_NAMES,
    MOCAP_G2_LIVE_LENGTH_FIELD,
    MOCAP_G2_PACKET_TOTAL_BYTES,
    build_mocap_g2_frame_packet,
    legacy_csv_rows_to_g2_joint_poses,
    mocap_g2_csv_required_columns,
    mocap_g2_csv_row_to_joint_poses,
    parse_fss_packet,
)

DEFAULT_TARGET_IP = "127.0.0.1"
DEFAULT_TARGET_PORT = 15000
DEFAULT_REPLAY_HZ = 100.0

CSV_FORMAT_G2 = "formotion_g2_csv_v1"
CSV_FORMAT_LEGACY = "legacy_fss_csv_v1"
TIMESTAMP_COLUMN_CANDIDATES = (
    "systemTimestampUs",
    "system_timestamp_us",
    "timestamp_us",
    "timestampUs",
)


class CsvReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class Recording:
    path: Path
    csv_format: str
    rows: list[dict[str, str]]
    timestamp_column: str | None


def _float(row: Mapping[str, object], column: str) -> float:
    value = row.get(column)
    if value is None or value == "":
        raise CsvReplayError(f"missing required CSV value: {column}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CsvReplayError(f"invalid float in {column}: {value!r}") from exc


def _resolve_csv_paths(arguments: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for argument in arguments:
        expanded = [Path(item) for item in glob.glob(argument, recursive=True)]
        if not expanded:
            candidate = Path(argument).expanduser()
            if candidate.exists():
                expanded = [candidate]
            else:
                raise CsvReplayError(f"CSV path or pattern matched no files: {argument}")
        for candidate in expanded:
            resolved = candidate.expanduser().resolve()
            if not resolved.is_file():
                raise CsvReplayError(f"CSV path is not a file: {candidate}")
            if resolved.suffix.lower() not in {".csv", ".cvs"}:
                raise CsvReplayError(f"unsupported CSV suffix: {resolved}")
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    if not paths:
        raise CsvReplayError("no CSV files were selected")
    return paths


def _legacy_required_columns() -> set[str]:
    required: set[str] = set()
    for joint in JOINT_NAMES:
        required.update(
            {
                f"quat_{joint}_X",
                f"quat_{joint}_Y",
                f"quat_{joint}_Z",
                f"quat_{joint}_W",
                f"pos_{joint}_X",
                f"pos_{joint}_Y",
                f"pos_{joint}_Z",
            }
        )
    return required


def _find_timestamp_column(available: set[str]) -> str | None:
    for candidate in TIMESTAMP_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def _detect_csv_format(path: Path, fieldnames: list[str]) -> tuple[str, str | None]:
    available = set(fieldnames)
    g2_required = mocap_g2_csv_required_columns()
    legacy_required = _legacy_required_columns()

    if g2_required.issubset(available):
        return CSV_FORMAT_G2, _find_timestamp_column(available)
    if legacy_required.issubset(available):
        return CSV_FORMAT_LEGACY, _find_timestamp_column(available)

    missing_g2 = sorted(g2_required - available)
    missing_legacy = sorted(legacy_required - available)
    g2_preview = ", ".join(missing_g2[:6]) + (" ..." if len(missing_g2) > 6 else "")
    legacy_preview = ", ".join(missing_legacy[:6]) + (" ..." if len(missing_legacy) > 6 else "")
    raise CsvReplayError(
        f"unsupported CSV schema: {path}\n"
        f"  native G2 missing {len(missing_g2)} columns: {g2_preview}\n"
        f"  legacy FSS missing {len(missing_legacy)} columns: {legacy_preview}"
    )


def _load_recording(path: Path) -> Recording:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise CsvReplayError(f"CSV has no header: {path}")
        csv_format, timestamp_column = _detect_csv_format(path, reader.fieldnames)
        rows = [row for row in reader if any((value or "").strip() for value in row.values())]
    if not rows:
        raise CsvReplayError(f"CSV contains no motion frames: {path}")
    return Recording(path=path, csv_format=csv_format, rows=rows, timestamp_column=timestamp_column)


def _legacy_joint_poses(row: Mapping[str, object]) -> list[tuple[float, float, float, float, float, float, float]]:
    positions_xyz: dict[str, tuple[float, float, float]] = {}
    rotations_xyzw: dict[str, tuple[float, float, float, float]] = {}
    for joint in JOINT_NAMES:
        rotations_xyzw[joint] = (
            _float(row, f"quat_{joint}_X"),
            _float(row, f"quat_{joint}_Y"),
            _float(row, f"quat_{joint}_Z"),
            _float(row, f"quat_{joint}_W"),
        )
        positions_xyz[joint] = (
            _float(row, f"pos_{joint}_X"),
            _float(row, f"pos_{joint}_Y"),
            _float(row, f"pos_{joint}_Z"),
        )
    return legacy_csv_rows_to_g2_joint_poses(
        positions_xyz=positions_xyz,
        rotations_xyzw=rotations_xyzw,
    )


def _joint_poses(recording: Recording, row: Mapping[str, object]):
    if recording.csv_format == CSV_FORMAT_G2:
        try:
            return mocap_g2_csv_row_to_joint_poses(row)
        except ValueError as exc:
            raise CsvReplayError(str(exc)) from exc
    return _legacy_joint_poses(row)


def _normalize_source_timestamp_us(raw_value: float) -> int:
    """Normalize recorder timestamp to protocol microseconds.

    The supplied new recorder names the column ``systemTimestampUs`` but the
    observed values are millisecond-scale (around 1e12 with ~10-unit frame
    deltas). Epoch microseconds are around 1e15. Values below 1e14 are therefore
    interpreted as milliseconds and multiplied by 1000.
    """
    if raw_value < 0:
        raise CsvReplayError(f"negative source timestamp: {raw_value}")
    if raw_value < 100_000_000_000_000:
        raw_value *= 1000.0
    return int(round(raw_value)) & 0xFFFFFFFFFFFFFFFF


def _timestamp_us(recording: Recording, row: Mapping[str, object], generated_timestamp_us: int) -> int:
    if recording.timestamp_column is None:
        return generated_timestamp_us
    return _normalize_source_timestamp_us(_float(row, recording.timestamp_column))


def _frame_from_row(recording: Recording, row: Mapping[str, object], generated_timestamp_us: int) -> bytes:
    return build_mocap_g2_frame_packet(
        timestamp_us=_timestamp_us(recording, row, generated_timestamp_us),
        joint_poses=_joint_poses(recording, row),
    )


def _validate_recording(recording: Recording) -> None:
    probe = _frame_from_row(recording, recording.rows[0], 0)
    if len(probe) != MOCAP_G2_PACKET_TOTAL_BYTES:
        raise CsvReplayError(f"internal G2 packet length is not {MOCAP_G2_PACKET_TOTAL_BYTES} bytes")
    parsed = parse_fss_packet(probe, check_crc=False)
    if parsed.layout != "formotion_g2_mocap_v1":
        raise CsvReplayError(f"internal packet is not Formotion G2 layout: {parsed.layout}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV回放为新版 Formotion G2 Mocap Frame UDP")
    parser.add_argument("csv_paths", nargs="+", help="新版G2或旧版FSS CSV路径/glob")
    parser.add_argument("--target-ip", default=DEFAULT_TARGET_IP, help=f"UDP目标IP，默认 {DEFAULT_TARGET_IP}")
    parser.add_argument("--target-port", type=int, default=DEFAULT_TARGET_PORT, help=f"UDP目标端口，默认 {DEFAULT_TARGET_PORT}")
    parser.add_argument("--hz", type=float, default=DEFAULT_REPLAY_HZ, help=f"回放频率，默认 {DEFAULT_REPLAY_HZ}")
    parser.add_argument("--once", action="store_true", help="只回放一轮，默认循环回放")
    parser.add_argument("--dry-run", action="store_true", help="只检查CSV格式和UDP编码，不发送数据")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.hz <= 0.0:
        print("[csv-udp-replay][ERROR] --hz must be positive", file=sys.stderr)
        return 2

    try:
        selected_paths = _resolve_csv_paths(args.csv_paths)
        recordings = [_load_recording(path) for path in selected_paths]
        for recording in recordings:
            _validate_recording(recording)
    except CsvReplayError as exc:
        print(f"[csv-udp-replay][ERROR] {exc}", file=sys.stderr)
        return 1

    for recording in recordings:
        timestamp_info = recording.timestamp_column or "generated"
        print(
            "[csv-udp-replay][INFO] "
            f"file={recording.path.name} format={recording.csv_format} "
            f"frames={len(recording.rows)} timestamp={timestamp_info}"
        )

    if args.dry_run:
        print(
            "[csv-udp-replay][INFO] dry-run passed: "
            f"files={len(recordings)} protocol=Formotion-G2-MocapFrame "
            f"id=0x0000 length_field={MOCAP_G2_LIVE_LENGTH_FIELD} datagram={MOCAP_G2_PACKET_TOTAL_BYTES}"
        )
        return 0

    period_s = 1.0 / args.hz
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = 0
    sent = 0
    playlist = ", ".join(recording.path.name for recording in recordings)
    print(
        "[csv-udp-replay][INFO] "
        f"files={len(recordings)} [{playlist}] loop={not args.once} "
        f"target={args.target_ip}:{args.target_port} rate={args.hz:.3f}Hz "
        f"protocol=Formotion-G2-MocapFrame id=0x0000 length_field={MOCAP_G2_LIVE_LENGTH_FIELD} datagram={MOCAP_G2_PACKET_TOTAL_BYTES}"
    )
    try:
        while True:
            for recording in recordings:
                print(f"[csv-udp-replay][INFO] replay={recording.path}")
                for frame_index, row in enumerate(recording.rows):
                    started = time.monotonic()
                    generated_timestamp_us = int(round(sequence * 1_000_000.0 / args.hz)) & 0xFFFFFFFFFFFFFFFF
                    datagram = _frame_from_row(recording, row, generated_timestamp_us)
                    sock.sendto(datagram, (args.target_ip, args.target_port))
                    sequence += 1
                    sent += 1
                    if sent % max(1, int(args.hz)) == 0:
                        print(
                            f"[csv-udp-replay][INFO] sent={sent} "
                            f"frame={frame_index + 1}/{len(recording.rows)} file={recording.path.name}"
                        )
                    remaining = period_s - (time.monotonic() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
            if args.once:
                break
    except KeyboardInterrupt:
        print(f"\n[csv-udp-replay][INFO] stopped after {sent} frames")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
