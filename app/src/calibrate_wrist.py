from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_mocap_protocol import MocapPacketError, parse_mocap_packet
from teleop_wrist_calibration import (
    SIDES,
    WristCalibrationThresholds,
    fit_direct_wrist_joint,
    fit_hand_to_wrist_pivot,
    load_wrist_calibration,
    select_runtime_position_source,
    thresholds_to_dict,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "wrist_calibration"
GROUP_GUIDES = {
    "front": "目标腕点放在胸前；保持腕点不移动，依次缓慢做屈伸、左右偏和旋前旋后。",
    "side": "目标腕点移到身体同侧；保持腕点不移动，重复完整转腕动作。",
    "raised": "目标腕点轻微抬高；保持腕点不移动，重复完整转腕动作。",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"标定输入必须是 JSON object: {path}")
    return payload


def prompt(message: str) -> None:
    try:
        input(f"\n{message}\n准备好后按回车，Ctrl+C 可停止: ")
    except EOFError as exc:
        raise RuntimeError("capture 阶段需要交互式终端") from exc


def create_socket(ip: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    sock.settimeout(0.2)
    print(f"WRIST_CALIBRATION_UDP_LISTEN udp://{ip}:{port}")
    return sock


def capture_group(
    sock: socket.socket,
    side: str,
    group: str,
    seconds: float,
    device_mode: str,
) -> dict[str, Any]:
    prompt(f"{side} / {group}: {GROUP_GUIDES[group]} 非目标手自然放松，不要移动躯干。")
    started = time.monotonic()
    frames: list[dict[str, Any]] = []
    invalid_packets = 0
    next_report = 0.0
    while time.monotonic() - started < seconds:
        elapsed = time.monotonic() - started
        if elapsed >= next_report:
            print(f"CAPTURING side={side} group={group} remaining={max(0.0, seconds - elapsed):.1f}s")
            next_report += 1.0
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        receive_time = time.time()
        try:
            frame = parse_mocap_packet(data, receive_time, device_mode=device_mode)
        except (MocapPacketError, ValueError):
            invalid_packets += 1
            continue
        hand = frame.joint(f"{side}_hand")
        wrist = frame.joint(f"{side}_wrist")
        chest = frame.joint("chest")
        if not hand.tracked or not wrist.tracked or not chest.tracked:
            continue
        frames.append(
            {
                "timestamp_s": frame.timestamp_s,
                "hand_pos": list(hand.position),
                "hand_quat": list(hand.quaternion_xyzw),
                "wrist_pos": list(wrist.position),
                "wrist_quat": list(wrist.quaternion_xyzw),
                "chest_pos": list(chest.position),
            }
        )
    print(
        f"CAPTURED side={side} group={group} frames={len(frames)} "
        f"invalid_packets={invalid_packets}"
    )
    if len(frames) < 3:
        raise RuntimeError(f"有效样本不足: side={side} group={group} frames={len(frames)}")
    return {
        "side": side,
        "group": group,
        "duration_seconds": seconds,
        "invalid_packets": invalid_packets,
        "frames": frames,
    }


def capture(args: argparse.Namespace) -> Path:
    selected_sides = SIDES if args.side == "both" else (args.side,)
    session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir = args.output.expanduser().resolve()
    raw_path = output_dir / session_name / "capture.json"
    windows: dict[str, Any] = {}
    sock = create_socket(args.ip, args.port)
    try:
        for side in selected_sides:
            for group in GROUP_GUIDES:
                windows[f"{side}_{group}"] = capture_group(
                    sock,
                    side,
                    group,
                    float(args.group_seconds),
                    args.device_mode,
                )
    finally:
        sock.close()
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "device_mode": args.device_mode,
        "udp": {"ip": args.ip, "port": args.port},
        "sides": list(selected_sides),
        "groups": list(GROUP_GUIDES),
        "windows": windows,
    }
    save_json(raw_path, payload)
    print(f"WRIST_CAPTURE_SAVED {raw_path}")
    return raw_path


def side_samples(payload: dict[str, Any], side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    groups: list[str] = []
    windows = payload.get("windows", {})
    if not isinstance(windows, dict):
        raise ValueError("capture 缺少 windows object")
    for group in GROUP_GUIDES:
        window = windows.get(f"{side}_{group}")
        if not isinstance(window, dict):
            raise ValueError(f"capture 缺少 {side}_{group}")
        frames = window.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"capture.{side}_{group}.frames 必须是 array")
        for frame in frames:
            hand_pos = np.asarray(frame["hand_pos"], dtype=float)
            hand_quat = np.asarray(frame["hand_quat"], dtype=float)
            wrist_pos = np.asarray(frame["wrist_pos"], dtype=float)
            chest_pos = np.asarray(frame["chest_pos"], dtype=float)
            if hand_pos.shape != (3,) or wrist_pos.shape != (3,) or chest_pos.shape != (3,):
                continue
            if hand_quat.shape != (4,) or not np.all(np.isfinite(hand_quat)):
                continue
            quat_norm = float(np.linalg.norm(hand_quat))
            if quat_norm <= 1e-8:
                continue
            positions.append(hand_pos - chest_pos)
            wrists.append(wrist_pos - chest_pos)
            rotations.append(R.from_quat(hand_quat / quat_norm).as_matrix())
            groups.append(group)
    if not positions:
        raise ValueError(f"{side} 没有可分析样本")
    return np.asarray(positions), np.asarray(rotations), np.asarray(wrists), groups


def thresholds_from_args(args: argparse.Namespace) -> WristCalibrationThresholds:
    return WristCalibrationThresholds(
        min_samples=args.min_samples,
        min_groups=3,
        min_rotation_excitation_rad=args.min_rotation_excitation,
        min_rotation_rank=args.min_rotation_rank,
        max_outlier_fraction=args.max_outlier_fraction,
        max_pivot_rms_m=args.max_pivot_rms,
        max_pivot_p95_m=args.max_pivot_p95,
        max_wrist_validation_rms_m=args.max_wrist_validation_rms,
    )


def analyze(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()
    capture_payload = load_json(input_path)
    selected_sides = tuple(capture_payload.get("sides", SIDES))
    if any(side not in SIDES for side in selected_sides):
        raise ValueError(f"capture.sides 非法: {selected_sides}")
    thresholds = thresholds_from_args(args)
    side_results: dict[str, Any] = {}
    valid = True
    for side in selected_sides:
        positions, rotations, wrists, groups = side_samples(capture_payload, side)
        direct_fit = fit_direct_wrist_joint(
            positions,
            rotations,
            groups,
            wrist_positions=wrists,
            thresholds=thresholds,
        )
        fit = direct_fit if direct_fit.valid else fit_hand_to_wrist_pivot(
            positions,
            rotations,
            groups,
            wrist_positions=wrists,
            thresholds=thresholds,
            huber_delta_m=args.huber_delta,
        )
        position_source = select_runtime_position_source(fit)
        side_results[side] = {
            "position_source": position_source,
            "orientation_source": "hand_joint",
            "hand_to_wrist_offset": fit.hand_to_wrist_offset.tolist(),
            "group_centers": {name: center.tolist() for name, center in fit.group_centers.items()},
            "quality": fit.quality,
            "valid": fit.valid,
            "reasons": list(fit.reasons),
        }
        valid = valid and fit.valid
        print(
            f"WRIST_ANALYSIS side={side} valid={fit.valid} source={position_source} "
            f"offset={fit.hand_to_wrist_offset.tolist()} quality={fit.quality} reasons={list(fit.reasons)}"
        )
    if set(selected_sides) != set(SIDES):
        valid = False
    output_path = args.calibration_output.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_capture": str(input_path),
        "valid": valid,
        "control_point": "anatomical_wrist",
        "sides": side_results,
        "quality_thresholds": thresholds_to_dict(thresholds),
    }
    save_json(output_path, payload)
    print(f"WRIST_CALIBRATION_SAVED valid={valid} {output_path}")
    return output_path


def validate(args: argparse.Namespace) -> None:
    calibration = load_wrist_calibration(args.input, control_side=args.side)
    print(
        f"WRIST_CALIBRATION_VALID path={calibration.source_path} "
        f"sides={list(calibration.sides)} thresholds={thresholds_to_dict(calibration.thresholds)}"
    )


def run_all(args: argparse.Namespace) -> Path:
    if args.side != "both":
        raise ValueError("一步式标定必须同时采集左右手，请使用 --side both")
    print(
        "\n一步式腕点标定开始。按提示完成胸前、身体侧方、轻微抬高三个位置；"
        "每个位置尽量固定腕点，缓慢做屈伸、左右偏和掌心翻转。"
    )
    raw_capture_path = capture(args)
    final_path = Path(args.calibration_output).expanduser().resolve()
    candidate_path = final_path.with_name(
        f".{final_path.stem}.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.candidate{final_path.suffix}"
    )

    args.input = raw_capture_path
    args.calibration_output = candidate_path
    try:
        analyze(args)
        args.input = candidate_path
        validate(args)
        candidate_path.replace(final_path)
    except BaseException:
        candidate_path.unlink(missing_ok=True)
        raise
    finally:
        args.calibration_output = final_path
        args.input = final_path if final_path.is_file() else raw_capture_path

    print(f"\n一步式腕点标定完成: {final_path}")
    print("原始采样已保留，可用于复核和重新分析。")
    return final_path


def add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--device-mode", choices=("glove", "auto"), default="glove")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--group-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "calibration.json",
    )
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-rotation-excitation", type=float, default=0.25)
    parser.add_argument("--min-rotation-rank", type=int, default=2)
    parser.add_argument("--max-outlier-fraction", type=float, default=0.15)
    parser.add_argument("--max-pivot-rms", type=float, default=0.010)
    parser.add_argument("--max-pivot-p95", type=float, default=0.020)
    parser.add_argument("--max-wrist-validation-rms", type=float, default=0.020)
    parser.add_argument("--huber-delta", type=float, default=0.010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人体 hand→anatomical wrist 多姿态枢轴标定")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    all_parser = subparsers.add_parser("all", help="一键完成采集、分析和质量验证")
    add_capture_arguments(all_parser)
    add_analysis_arguments(all_parser)

    capture_parser = subparsers.add_parser("capture", help="仅监听动捕 UDP 并保存原始样本")
    add_capture_arguments(capture_parser)

    analyze_parser = subparsers.add_parser("analyze", help="离线稳健拟合 hand→wrist 杆臂")
    analyze_parser.add_argument("input", type=Path)
    add_analysis_arguments(analyze_parser)

    validate_parser = subparsers.add_parser("validate", help="验证持久化标定 schema 与质量门禁")
    validate_parser.add_argument("input", type=Path)
    validate_parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage == "all":
            run_all(args)
        elif args.stage == "capture":
            capture(args)
        elif args.stage == "analyze":
            analyze(args)
        else:
            validate(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"WRIST_CALIBRATION_FAILED {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())