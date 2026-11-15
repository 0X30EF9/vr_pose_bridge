#!/usr/bin/env python3
"""Bridge Formotion G2 full-body mocap UDP with Revo2 hand IK output.

Input:
    UDP 15000, Formotion G2 Mocap Frame parsed by revo2_hand_ik.fss_udp.

Output:
    JSON UDP packet containing all 64 body joint poses plus the 12 Revo2 hand
    IK joint angles in radians.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SDK_CONFIG = "bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json"

BODY_JOINT_NAMES: tuple[str, ...] = (
    "head",
    "neck",
    "chest",
    "spine1",
    "spine2",
    "waist",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
)


def _default_sdk_root() -> Path:
    """Return the most likely SDK root for both embedded and standalone use."""
    here = Path(__file__).resolve()

    # Embedded package layout:
    #   revo2_hand_ik_sdk/python/examples/fullbody_hand_ik_bridge.py
    embedded_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    if (embedded_root / "python" / "revo2_hand_ik").is_dir():
        return embedded_root

    # Workspace helper layout used while developing this bridge:
    #   <workspace>/revo2_fullbody_hand_bridge/fullbody_hand_ik_bridge.py
    workspace_runtime = here.parents[1] / "_revo2_ik_runtime" / "revo2_hand_ik_runtime"
    return workspace_runtime


def _insert_sdk_python_path(sdk_root: Path | None, sdk_python_root: Path | None) -> Path:
    if sdk_python_root is None:
        if sdk_root is None:
            sdk_root = _default_sdk_root()
        sdk_python_root = sdk_root / "python"
    sdk_python_root = sdk_python_root.expanduser().resolve()
    if not (sdk_python_root / "revo2_hand_ik").is_dir():
        raise FileNotFoundError(
            "SDK Python package not found. Pass --sdk-root or --sdk-python-root. "
            f"Tried: {sdk_python_root}"
        )
    sys.path.insert(0, str(sdk_python_root))
    return sdk_python_root


def _resolve_sdk_config(sdk_root: Path | None, sdk_config: str) -> Path:
    path = Path(sdk_config).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = sdk_root.expanduser().resolve() if sdk_root is not None else _default_sdk_root()
    return (root / path).resolve()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _extract_body_pose(packet: Any, fss_udp: Any) -> dict[str, dict[str, list[float]]]:
    if packet.layout != "formotion_g2_mocap_v1":
        raise ValueError(f"body output requires formotion_g2_mocap_v1, got {packet.layout}")
    if packet.joint_pose_offset is None:
        raise ValueError("G2 packet has no joint_pose_offset")

    prefix = "<" if packet.pose_byte_order == "little" else ">"
    body: dict[str, dict[str, list[float]]] = {}
    for name in BODY_JOINT_NAMES:
        index = fss_udp.MOCAP_G2_JOINT_INDEX[name]
        offset = packet.joint_pose_offset + index * fss_udp.MOCAP_G2_FLOATS_PER_JOINT * 4
        x, y, z, qw, qx, qy, qz = struct.unpack_from(prefix + "7f", packet.raw, offset)
        body[name] = {
            "pos": [float(x), float(y), float(z)],
            "quat_wxyz": [float(qw), float(qx), float(qy), float(qz)],
        }
    return body


def _hand_debug_flag(result: Mapping[str, Any], side: str, key: str) -> bool:
    hand = result.get(side, {})
    if not isinstance(hand, Mapping):
        return False
    debug = hand.get("debug", {})
    return bool(debug.get(key, False)) if isinstance(debug, Mapping) else False


def build_combined_payload(
    *,
    packet: Any,
    sdk_result: Mapping[str, Any],
    body: Mapping[str, Any],
    joint_names: tuple[str, ...],
    sequence: int,
) -> dict[str, Any]:
    return {
        "protocol": "revo2_body_pose_plus_hand_ik_v1",
        "sequence": int(sequence),
        "timestamp_ms": int(packet.timestamp_ms),
        "source": {
            "protocol": packet.layout,
            "packet_id": int(packet.packet_id),
            "packet_size": len(packet.raw),
            "length_field": int(packet.payload_length),
            "timestamp_us": packet.timestamp_us,
            "header_byte_order": packet.header_byte_order,
            "pose_byte_order": packet.pose_byte_order,
            "sync": "55AA" if packet.sync_reversed else "AA55",
        },
        "body": {
            "joint_count": len(joint_names),
            "joint_order": list(joint_names),
            "position_unit": "same_as_input",
            "quaternion_order": "wxyz",
            "note": "Body joints only. Finger joints are removed from this output; hand IK angles are in hand_ik.",
            "joints": body,
        },
        "hand_ik": {
            "joint_count": 12,
            "joint_unit": "rad",
            "joint_order": list(sdk_result.get("joint_order", [])),
            "joint_positions": list(sdk_result.get("joint_positions", [])),
            "joint_positions_named": sdk_result.get("joint_positions_named", {}),
            "left_calibrated": _hand_debug_flag(sdk_result, "left", "calibrated"),
            "right_calibrated": _hand_debug_flag(sdk_result, "right", "calibrated"),
            "left_solve_time_ms": float(sdk_result.get("left_solve_time_ms", 0.0)),
            "right_solve_time_ms": float(sdk_result.get("right_solve_time_ms", 0.0)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive Formotion G2 UDP, run Revo2 hand IK, output body pose plus hand angles."
    )
    parser.add_argument("--sdk-root", default=str(_default_sdk_root()), help="Revo2 SDK runtime root")
    parser.add_argument("--sdk-python-root", default="", help="optional SDK python directory override")
    parser.add_argument("--sdk-config", default=DEFAULT_SDK_CONFIG, help="dual-hand SDK config path")
    parser.add_argument("--library", default=None, help="optional librevo2_ik_sdk.so path")
    parser.add_argument("--listen-ip", default="0.0.0.0", help="input UDP bind address")
    parser.add_argument("--listen-port", type=int, default=15000, help="input UDP port")
    parser.add_argument("--source-ip", default="", help="accept only this sender IP; empty accepts any")
    parser.add_argument("--recv-buffer-size", type=int, default=65535, help="recvfrom buffer size")
    parser.add_argument("--recv-timeout-ms", type=int, default=20, help="receive timeout in ms")
    parser.add_argument("--no-crc", action="store_true", help="disable legacy id=0x0001 CRC check")
    parser.add_argument("--output-ip", default="127.0.0.1", help="output UDP target address")
    parser.add_argument("--output-port", type=int, default=51000, help="output UDP target port")
    parser.add_argument("--print-only", action="store_true", help="print one-line status instead of sending output UDP")
    parser.add_argument("--print-json", action="store_true", help="print each output JSON packet")
    parser.add_argument("--status-interval-s", type=float, default=1.0, help="status print interval")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N solved frames; 0 runs forever")
    args = parser.parse_args()

    if not 1 <= args.listen_port <= 65535:
        parser.error("--listen-port must be in 1..65535")
    if not 1 <= args.output_port <= 65535:
        parser.error("--output-port must be in 1..65535")
    args.recv_buffer_size = max(2048, min(65535, int(args.recv_buffer_size)))
    args.recv_timeout_ms = max(1, int(args.recv_timeout_ms))
    args.status_interval_s = max(0.1, float(args.status_interval_s))
    args.max_frames = max(0, int(args.max_frames))
    return args


def main() -> int:
    args = parse_args()
    sdk_root = Path(args.sdk_root).expanduser().resolve() if args.sdk_root else None
    sdk_python_root = Path(args.sdk_python_root).expanduser().resolve() if args.sdk_python_root else None
    _insert_sdk_python_path(sdk_root, sdk_python_root)

    from revo2_hand_ik import fss_udp
    from revo2_hand_ik.api import Revo2DualHandIkSdk
    from revo2_hand_ik.fss_udp import FssPacketError, extract_hand_mocap, parse_fss_packet

    sdk_config = _resolve_sdk_config(sdk_root, args.sdk_config)
    if not sdk_config.is_file():
        raise FileNotFoundError(f"SDK config not found: {sdk_config}")

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((args.listen_ip, args.listen_port))
    rx.settimeout(args.recv_timeout_ms / 1000.0)

    received = 0
    solved = 0
    dropped = 0
    filtered = 0
    last_status = time.monotonic()

    print(
        "[fullbody-hand-bridge] started "
        f"input={args.listen_ip}:{args.listen_port} "
        f"output={'print-only' if args.print_only else f'{args.output_ip}:{args.output_port}'} "
        f"sdk_config={sdk_config}",
        flush=True,
    )

    try:
        with Revo2DualHandIkSdk(sdk_config, library_path=args.library) as sdk:
            while True:
                try:
                    datagram, source = rx.recvfrom(args.recv_buffer_size)
                except (TimeoutError, socket.timeout):
                    now = time.monotonic()
                    if now - last_status >= args.status_interval_s:
                        print(
                            f"[fullbody-hand-bridge] received={received} solved={solved} "
                            f"dropped={dropped} filtered={filtered} waiting_for_udp=1",
                            flush=True,
                        )
                        last_status = now
                    continue

                source_ip = source[0]
                if args.source_ip and source_ip != args.source_ip:
                    filtered += 1
                    continue

                received += 1
                try:
                    packet = parse_fss_packet(datagram, check_crc=not args.no_crc)
                    body = _extract_body_pose(packet, fss_udp)
                    left_points, left_quats = extract_hand_mocap(packet, "left")
                    right_points, right_quats = extract_hand_mocap(packet, "right")
                    sdk_result = sdk.solve(
                        left_points=left_points,
                        right_points=right_points,
                        left_quaternions_wxyz=left_quats,
                        right_quaternions_wxyz=right_quats,
                        sequence=received,
                        timestamp_ms=packet.timestamp_ms,
                    )
                    output = build_combined_payload(
                        packet=packet,
                        sdk_result=sdk_result,
                        body=body,
                        joint_names=BODY_JOINT_NAMES,
                        sequence=received,
                    )
                    output_bytes = _json_bytes(output)
                    if args.print_json:
                        print(output_bytes.decode("utf-8"), flush=True)
                    elif args.print_only:
                        hand_positions = output["hand_ik"]["joint_positions"]
                        print(
                            f"[fullbody-hand-bridge][FRAME] seq={received} "
                            f"timestamp_ms={packet.timestamp_ms} body_joints={len(body)} "
                            f"hand_angles={len(hand_positions)}",
                            flush=True,
                        )
                    else:
                        tx.sendto(output_bytes, (args.output_ip, args.output_port))
                    solved += 1
                except (FssPacketError, ValueError, RuntimeError) as exc:
                    dropped += 1
                    if dropped <= 10 or dropped % 100 == 0:
                        print(
                            f"[fullbody-hand-bridge][WARN] drop packet #{dropped}: "
                            f"source={source[0]}:{source[1]} size={len(datagram)} reason={exc}",
                            flush=True,
                        )

                now = time.monotonic()
                if now - last_status >= args.status_interval_s:
                    print(
                        f"[fullbody-hand-bridge] received={received} solved={solved} "
                        f"dropped={dropped} filtered={filtered}",
                        flush=True,
                    )
                    last_status = now

                if args.max_frames and solved >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        tx.close()

    print(
        f"[fullbody-hand-bridge] stopped received={received} solved={solved} "
        f"dropped={dropped} filtered={filtered}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
