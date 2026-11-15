#!/usr/bin/env python3
"""Formotion G2 UDP -> Python SDK API -> RViz test adapter.

UDP is only the example/test input source.  The SDK boundary remains the direct
``Revo2DualHandIkSdk.solve(...)`` Python call.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revo2_hand_ik.api import Revo2DualHandIkSdk  # noqa: E402
from revo2_hand_ik.fss_udp import FssPacket, FssPacketError, extract_hand_mocap, parse_fss_packet  # noqa: E402
from revo2_hand_ik.udp_bridge import make_debug_overlay_payload  # noqa: E402


DEFAULT_SDK_CONFIG = Path("bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json")


def _log(message: str) -> None:
    print(message, flush=True)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _result_debug(result: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    hand = result.get(side, {})
    if isinstance(hand, Mapping):
        debug = hand.get("debug", {})
        if isinstance(debug, Mapping):
            return debug
    return {}


def _calibration_text(debug: Mapping[str, Any]) -> str:
    status = str(debug.get("calibration_status", "unknown"))
    samples = int(debug.get("calibration_samples", 0) or 0)
    required = int(debug.get("calibration_required", 0) or 0)
    warmup = int(debug.get("calibration_warmup_remaining", 0) or 0)
    rejected = int(debug.get("calibration_rejected", 0) or 0)
    quality_failures = int(debug.get("calibration_quality_failures", 0) or 0)
    return (
        f"{status}:{samples}/{required},warmup={warmup},"
        f"rejected={rejected},quality_fail={quality_failures}"
    )


def _make_socket(
    listen_ip: str,
    listen_port: int,
    recv_timeout_ms: int,
    kernel_recv_buffer_bytes: int,
    allow_port_reuse: bool,
) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if allow_port_reuse:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, kernel_recv_buffer_bytes)
        sock.bind((listen_ip, listen_port))
        sock.settimeout(max(1, recv_timeout_ms) / 1000.0)
        return sock
    except Exception:
        sock.close()
        raise


def _valid_port(parser: argparse.ArgumentParser, name: str, value: int) -> int:
    if not 1 <= int(value) <= 65535:
        parser.error(f"{name} must be in 1..65535")
    return int(value)


def _wire_description(packet: FssPacket) -> str:
    if packet.layout != "formotion_g2_mocap_v1":
        return packet.layout
    return (
        f"{packet.layout},header={packet.header_byte_order},pose={packet.pose_byte_order},"
        f"timestamp_offset={packet.timestamp_offset},pose_offset={packet.joint_pose_offset},"
        f"length_field={packet.payload_length},packet_size={len(packet.raw)},"
        f"sync={'55AA' if packet.sync_reversed else 'AA55'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive Formotion G2 Mocap Frame UDP, convert it to "
            "Revo2DualHandIkSdk.solve(...) parameters, and mirror radian output to RViz."
        )
    )
    parser.add_argument(
        "--sdk-config",
        "--config",
        dest="sdk_config",
        default=str(DEFAULT_SDK_CONFIG),
        help="dual-hand direct SDK config JSON, default: %(default)s",
    )
    parser.add_argument("--library", default=None, help="optional librevo2_ik_sdk.so path")
    parser.add_argument("--listen-ip", default="0.0.0.0", help="UDP bind IPv4 address, default: %(default)s")
    parser.add_argument(
        "--listen-port", "--port", dest="listen_port", type=int, default=15000,
        help="UDP destination port to bind, default: %(default)s",
    )
    parser.add_argument(
        "--source-ip", default="",
        help="accept only this sender IPv4 address; empty accepts any sender",
    )
    parser.add_argument(
        "--recv-buffer-size", type=int, default=65535,
        help="maximum UDP datagram bytes read by one recvfrom, default: %(default)s",
    )
    parser.add_argument(
        "--kernel-recv-buffer-bytes", type=int, default=4 * 1024 * 1024,
        help="requested SO_RCVBUF size, default: %(default)s",
    )
    parser.add_argument("--recv-timeout-ms", type=int, default=20, help="UDP receive timeout in ms, default: %(default)s")
    parser.add_argument(
        "--allow-port-reuse", action="store_true",
        help="allow another process to bind the same port; disabled by default to prevent packets being stolen",
    )
    parser.add_argument(
        "--no-crc", action="store_true",
        help="disable legacy id=0x0001 CRC validation; G2 id=0x0000 has no documented outer CRC",
    )
    parser.add_argument("--control-target-ip", default="127.0.0.1", help="RViz control mirror target IP")
    parser.add_argument("--control-target-port", type=int, default=16124, help="RViz control mirror target port")
    parser.add_argument("--overlay-target-ip", default="127.0.0.1", help="RViz debug overlay target IP")
    parser.add_argument("--overlay-target-port", type=int, default=16125, help="RViz debug overlay target port")
    parser.add_argument("--disable-control-udp", action="store_true", help="do not send joint_positions(rad) mirror")
    parser.add_argument("--disable-overlay-udp", action="store_true", help="do not send raw/mapped/debug overlay")
    parser.add_argument("--print-interval-s", type=float, default=1.0, help="status print interval")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N solved frames; 0 runs forever")
    args = parser.parse_args()

    args.listen_port = _valid_port(parser, "--listen-port", args.listen_port)
    args.control_target_port = _valid_port(parser, "--control-target-port", args.control_target_port)
    args.overlay_target_port = _valid_port(parser, "--overlay-target-port", args.overlay_target_port)
    args.recv_buffer_size = max(2048, min(65535, int(args.recv_buffer_size)))
    args.kernel_recv_buffer_bytes = max(256 * 1024, int(args.kernel_recv_buffer_bytes))
    args.recv_timeout_ms = max(1, int(args.recv_timeout_ms))
    args.print_interval_s = max(0.1, float(args.print_interval_s))
    args.max_frames = max(0, int(args.max_frames))
    return args


def main() -> int:
    args = parse_args()
    sdk_config = Path(args.sdk_config).expanduser()
    if not sdk_config.is_absolute():
        sdk_config = Path.cwd() / sdk_config
    if not sdk_config.is_file():
        raise FileNotFoundError(f"SDK config not found: {sdk_config}")

    try:
        rx = _make_socket(
            args.listen_ip,
            args.listen_port,
            args.recv_timeout_ms,
            args.kernel_recv_buffer_bytes,
            args.allow_port_reuse,
        )
    except OSError as exc:
        raise RuntimeError(
            f"cannot bind UDP {args.listen_ip}:{args.listen_port}: {exc}. "
            f"Check competing listeners with: ss -H -lunp 'sport = :{args.listen_port}'"
        ) from exc
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    actual_rcvbuf = rx.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    received = 0
    solved = 0
    dropped = 0
    filtered = 0
    moving_frames = 0
    previous_joint_positions = [0.0] * 12
    last_joint_delta = 0.0
    last_joint_abs_max = 0.0
    left_calibrated = False
    right_calibrated = False
    left_calibration_text = "unknown:0/0,warmup=0,rejected=0,quality_fail=0"
    right_calibration_text = "unknown:0/0,warmup=0,rejected=0,quality_fail=0"
    last_status = time.monotonic()
    last_source = "none"
    last_size = 0
    last_wire = "unknown"
    first_datagram_logged = False
    first_valid_logged = False

    _log(
        "[udp-to-sdk] Formotion G2 UDP -> Python SDK API test started: "
        f"bind={args.listen_ip}:{args.listen_port} source_filter={args.source_ip or 'any'} "
        f"recv_datagram_max={args.recv_buffer_size} kernel_rcvbuf={actual_rcvbuf} "
        f"port_reuse={int(args.allow_port_reuse)} sdk_config={sdk_config}"
    )

    try:
        with Revo2DualHandIkSdk(sdk_config, library_path=args.library) as sdk:
            while True:
                try:
                    datagram, source = rx.recvfrom(args.recv_buffer_size)
                except (TimeoutError, socket.timeout):
                    now = time.monotonic()
                    if now - last_status >= args.print_interval_s:
                        _log(
                            f"[udp-to-sdk] received={received} solved={solved} dropped={dropped} filtered={filtered} "
                            f"moving_frames={moving_frames} joint_delta={last_joint_delta:.6f}rad "
                            f"joint_abs_max={last_joint_abs_max:.6f}rad "
                            f"calibrated=L{int(left_calibrated)}/R{int(right_calibrated)} "
                            f"Lcal=[{left_calibration_text}] Rcal=[{right_calibration_text}] "
                            f"last_source={last_source} last_size={last_size} wire={last_wire} waiting_for_udp=1"
                        )
                        last_status = now
                    continue

                source_ip, source_port = source[0], int(source[1])
                last_source = f"{source_ip}:{source_port}"
                last_size = len(datagram)
                if args.source_ip and source_ip != args.source_ip:
                    filtered += 1
                    continue

                received += 1
                head_hex = datagram[:32].hex(" ")
                if not first_datagram_logged:
                    _log(
                        f"[udp-to-sdk][RX] first_datagram source={last_source} size={len(datagram)} "
                        f"head32={head_hex}"
                    )
                    first_datagram_logged = True

                try:
                    packet = parse_fss_packet(datagram, check_crc=not args.no_crc)
                    last_wire = _wire_description(packet)
                    if not first_valid_logged:
                        _log(
                            f"[udp-to-sdk][OK] first_valid_packet source={last_source} size={len(datagram)} "
                            f"wire={last_wire} timestamp_us={packet.timestamp_us}"
                        )
                        first_valid_logged = True

                    left_points, left_quats = extract_hand_mocap(packet, "left")
                    right_points, right_quats = extract_hand_mocap(packet, "right")
                    result = sdk.solve(
                        left_points=left_points,
                        right_points=right_points,
                        left_quaternions_wxyz=left_quats,
                        right_quaternions_wxyz=right_quats,
                        sequence=received,
                        timestamp_ms=packet.timestamp_ms,
                    )

                    joint_positions = result.get("joint_positions", [])
                    if not isinstance(joint_positions, list) or len(joint_positions) != 12:
                        raise ValueError("SDK result joint_positions must contain exactly 12 values")
                    joint_positions = [float(value) for value in joint_positions]
                    last_joint_delta = max(
                        abs(current - previous)
                        for current, previous in zip(joint_positions, previous_joint_positions)
                    )
                    last_joint_abs_max = max(abs(value) for value in joint_positions)
                    if last_joint_delta > 1e-6:
                        moving_frames += 1
                    previous_joint_positions = list(joint_positions)
                    left_debug = _result_debug(result, "left")
                    right_debug = _result_debug(result, "right")
                    left_calibrated = bool(left_debug.get("calibrated", False))
                    right_calibrated = bool(right_debug.get("calibrated", False))
                    left_calibration_text = _calibration_text(left_debug)
                    right_calibration_text = _calibration_text(right_debug)

                    if not args.disable_control_udp:
                        tx.sendto(_json_bytes(result), (args.control_target_ip, args.control_target_port))

                    if not args.disable_overlay_udp:
                        overlay = make_debug_overlay_payload(
                            _result_debug(result, "left"),
                            _result_debug(result, "right"),
                            sequence=received,
                            timestamp_ms=packet.timestamp_ms,
                            left_raw_points=left_points,
                            right_raw_points=right_points,
                        )
                        tx.sendto(overlay, (args.overlay_target_ip, args.overlay_target_port))

                    solved += 1
                except (FssPacketError, ValueError, RuntimeError) as exc:
                    dropped += 1
                    if dropped <= 10 or dropped % 100 == 0:
                        _log(
                            f"[udp-to-sdk][WARN] drop packet #{dropped}: source={last_source} "
                            f"size={len(datagram)} head32={head_hex} reason={exc}"
                        )

                now = time.monotonic()
                if now - last_status >= args.print_interval_s:
                    _log(
                        f"[udp-to-sdk] received={received} solved={solved} dropped={dropped} filtered={filtered} "
                        f"moving_frames={moving_frames} joint_delta={last_joint_delta:.6f}rad "
                        f"joint_abs_max={last_joint_abs_max:.6f}rad "
                        f"calibrated=L{int(left_calibrated)}/R{int(right_calibrated)} "
                        f"Lcal=[{left_calibration_text}] Rcal=[{right_calibration_text}] "
                        f"last_source={last_source} last_size={last_size} wire={last_wire}"
                    )
                    last_status = now

                if args.max_frames and solved >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        tx.close()

    _log(
        f"[udp-to-sdk] stopped: received={received} solved={solved} dropped={dropped} filtered={filtered} "
        f"moving_frames={moving_frames} joint_abs_max={last_joint_abs_max:.6f}rad"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
