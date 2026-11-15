from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from pathlib import Path
from typing import BinaryIO

from mocap_csv_format import make_vr_csv_name, udp_packet_to_csv_row, write_csv_header
from teleop_mocap_protocol import (
    DEVICE_MODES,
    EXPECTED_UDP_PACKET_SIZE,
    LEGACY_PACKET_SIZES,
    MocapPacketError,
    to_jsonable,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "vr_udp_capture"
RAW_FRAME_HEADER = "<QI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VR UDP 数采：只监听动捕 UDP 并保存原始包和解析帧")
    parser.add_argument("--ip", default="0.0.0.0", help="UDP 监听地址")
    parser.add_argument("--port", type=int, default=15000, help="UDP 监听端口")
    parser.add_argument(
        "--device-mode",
        choices=DEVICE_MODES,
        default="glove",
        help="设备协议模式：glove=手套，controller=手柄，auto=按已验证帧头分派",
    )
    parser.add_argument("--duration", type=float, default=None, help="采集秒数；不传则持续采集到 Ctrl+C")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="CSV 输出目录；文件名自动为 vr_时间戳.csv")
    parser.add_argument("--flush-every", type=int, default=20, help="每多少个有效包 flush 一次")
    parser.add_argument("--diagnostic-samples", type=int, default=5, help="最多保存多少个无法解码的数据报；0=不保存")
    parser.add_argument("--diagnostic-bytes", type=int, default=256, help="无效包日志最多显示的头部字节数")
    parser.add_argument("--no-raw", action="store_true", help="不保存 vr_时间戳.raw_packets.bin")
    parser.add_argument("--no-jsonl", action="store_true", help="兼容旧参数；CSV 模式下忽略")
    return parser.parse_args()


def make_capture_paths(output_root: Path) -> tuple[Path, Path, Path, Path]:
    output_dir = output_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / make_vr_csv_name()
    return (
        csv_path,
        csv_path.with_suffix(".meta.json"),
        csv_path.with_suffix(".raw_packets.bin"),
        csv_path.with_suffix(".unsupported_packets.bin"),
    )


def write_raw_packet(file: BinaryIO, receive_time: float, data: bytes) -> None:
    receive_time_ns = int(receive_time * 1_000_000_000)
    file.write(struct.pack(RAW_FRAME_HEADER, receive_time_ns, len(data)))
    file.write(data)


def write_meta(meta_path: Path, meta: dict) -> None:
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(meta), file, ensure_ascii=False, indent=2)
        file.write("\n")


def capture_udp(args: argparse.Namespace) -> int:
    csv_path, meta_path, raw_path, diagnostic_path = make_capture_paths(args.output)
    raw_file = None if args.no_raw else raw_path.open("wb")
    diagnostic_file: BinaryIO | None = None
    diagnostic_limit = max(0, int(getattr(args, "diagnostic_samples", 5)))
    diagnostic_bytes = max(0, int(getattr(args, "diagnostic_bytes", 256)))
    device_mode = str(getattr(args, "device_mode", "glove"))
    csv_file = csv_path.open("w", encoding="utf-8", newline="")
    csv_writer = write_csv_header(csv_file)

    start_time = time.time()
    meta = {
        "format": "csv",
        "csv_path": str(csv_path),
        "device_mode": device_mode,
        "decoder_status": (
            "verified_glove"
            if device_mode == "glove"
            else "probe_only_controller" if device_mode == "controller" else "verified_glove_probe_controller"
        ),
        "udp_ip": args.ip,
        "udp_port": args.port,
        "expected_packet_size": EXPECTED_UDP_PACKET_SIZE if device_mode != "controller" else None,
        "accepted_packet_sizes": (
            [EXPECTED_UDP_PACKET_SIZE, *LEGACY_PACKET_SIZES]
            if device_mode != "controller"
            else []
        ),
        "start_time": start_time,
        "end_time": None,
        "duration_limit": args.duration,
        "received_packets": 0,
        "valid_packets": 0,
        "invalid_packets": 0,
        "raw_saved": raw_file is not None,
        "unsupported_samples_path": str(diagnostic_path),
        "unsupported_samples_saved": 0,
        "invalid_packet_signatures": {},
    }
    write_meta(meta_path, meta)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    sock.settimeout(0.2)

    print(f"RECORDER_CSV {csv_path}")
    print(
        f"RECORDER_LISTEN udp://{args.ip}:{args.port} duration={args.duration} "
        f"device_mode={device_mode}"
    )
    if device_mode == "controller":
        print(
            "RECORDER_CONTROLLER_PROBE 手柄帧布局尚未验证；"
            "仅保存限量原始样本，不生成伪造骨架 CSV"
        )

    try:
        while True:
            now = time.time()
            if args.duration is not None and now - start_time >= args.duration:
                break

            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue

            receive_time = time.time()
            meta["received_packets"] += 1

            if raw_file is not None:
                write_raw_packet(raw_file, receive_time, data)

            try:
                csv_writer.writerow(
                    udp_packet_to_csv_row(
                        data,
                        receive_time,
                        device_mode=device_mode,
                    )
                )
            except MocapPacketError as exc:
                meta["invalid_packets"] += 1
                signature = f"size={len(data)} sync={data[:2].hex(' ').upper()}"
                count = int(meta["invalid_packet_signatures"].get(signature, 0)) + 1
                meta["invalid_packet_signatures"][signature] = count
                if count == 1 or count % 500 == 0:
                    print(
                        "RECORDER_INVALID_PACKET "
                        f"mode={device_mode} count={count} len={len(data)} "
                        f"head={data[:diagnostic_bytes].hex(' ')} error={exc}"
                    )
                if meta["unsupported_samples_saved"] < diagnostic_limit:
                    if diagnostic_file is None:
                        diagnostic_file = diagnostic_path.open("wb")
                    write_raw_packet(diagnostic_file, receive_time, data)
                    meta["unsupported_samples_saved"] += 1
                continue

            meta["valid_packets"] += 1

            if args.flush_every > 0 and meta["valid_packets"] % args.flush_every == 0:
                if raw_file is not None:
                    raw_file.flush()
                if diagnostic_file is not None:
                    diagnostic_file.flush()
                csv_file.flush()
                write_meta(meta_path, {**meta, "end_time": time.time()})
    except KeyboardInterrupt:
        print("RECORDER_STOP KeyboardInterrupt")
    finally:
        meta["end_time"] = time.time()
        sock.close()
        if raw_file is not None:
            raw_file.flush()
            raw_file.close()
        if diagnostic_file is not None:
            diagnostic_file.flush()
            diagnostic_file.close()
        csv_file.flush()
        csv_file.close()
        write_meta(meta_path, meta)

    print(
        "RECORDER_RESULT "
        f"mode={device_mode} received={meta['received_packets']} "
        f"valid={meta['valid_packets']} invalid={meta['invalid_packets']} "
        f"unsupported_samples={meta['unsupported_samples_saved']} csv={csv_path}"
    )
    return 0


def main() -> int:
    return capture_udp(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())