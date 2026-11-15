from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from teleop_mocap_protocol import MocapPacketError, parse_mocap_packet
from teleop_wrist_calibration import load_wrist_calibration, reconstruct_wrist_pose_from_mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT_DIR / "data" / "workspace_calibration"
DIRECTIONS = ("forward", "backward", "outward", "inward", "up", "down")
SIDES = ("left", "right")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=json_default)
        file.write("\n")


def artifact_path(requested: Path, stem: str, overwrite: bool = False) -> Path:
    requested = requested.expanduser()
    if requested.suffix.lower() == ".json":
        target = requested if stem == "calibration" or requested.name.startswith(stem) else requested.with_name(f"{stem}.json")
    else:
        target = requested / f"{stem}.json"
    if overwrite or not target.exists():
        return target
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = target.with_name(f"{target.stem}_{timestamp}{target.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = target.with_name(f"{target.stem}_{timestamp}_{suffix}{target.suffix}")
        suffix += 1
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"标定输入必须是 JSON object: {path}")
    return value


def print_guide() -> None:
    guide = ROOT_DIR / "WORKSPACE_CALIBRATION.md"
    if not guide.is_file():
        raise FileNotFoundError(f"找不到标定说明: {guide}")
    print(guide.read_text(encoding="utf-8"))


def prompt_continue(message: str) -> None:
    try:
        input(f"\n{message}\n准备好后按回车，Ctrl+C 可停止: ")
    except EOFError as exc:
        raise RuntimeError("标定向导需要交互式终端；请使用 --stage analyze 或 --stage scan 处理已有数据") from exc


def create_socket(ip: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    sock.settimeout(0.2)
    print(f"CALIBRATION_UDP_LISTEN udp://{ip}:{port}")
    return sock


def receive_window(
    sock: socket.socket,
    duration: float,
    device_mode: str,
    capture_rate: float,
    wrist_calibration,
    move_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    deadline = started + duration
    frames: list[dict[str, Any]] = []
    invalid = 0
    next_hold_report = 0.0
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        if move_seconds is not None and elapsed >= move_seconds:
            hold_elapsed = elapsed - move_seconds
            if hold_elapsed >= next_hold_report:
                remaining = max(0.0, duration - elapsed)
                print(f"HOLD_CAPTURE remaining_seconds={remaining:.1f}")
                next_hold_report += 1.0
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        receive_time = time.time()
        try:
            frame = parse_mocap_packet(data, receive_time, device_mode=device_mode)
        except (MocapPacketError, ValueError):
            invalid += 1
            continue
        replay = frame.to_replay_dict(frame_index=len(frames))
        try:
            left_wrist = reconstruct_wrist_pose_from_mapping(
                replay,
                "left",
                wrist_calibration.sides["left"],
            )
            right_wrist = reconstruct_wrist_pose_from_mapping(
                replay,
                "right",
                wrist_calibration.sides["right"],
            )
        except (KeyError, ValueError):
            invalid += 1
            continue
        frames.append(
            {
                "receive_time": float(replay["receive_time"]),
                "left_pos": left_wrist[:3, 3].tolist(),
                "left_quat": replay["left_hand_quat"],
                "right_pos": right_wrist[:3, 3].tolist(),
                "right_quat": replay["right_hand_quat"],
                "body_pos": replay["skeleton"]["torso"]["chest"]["pos"],
            }
        )
    expected = max(1, int(round(duration * capture_rate)))
    valid_rate = min(1.0, len(frames) / expected)
    stats = {
        "duration_seconds": float(duration),
        "expected_frames": expected,
        "valid_frames": len(frames),
        "invalid_packets": invalid,
        "valid_rate": valid_rate,
    }
    if invalid:
        print(f"CALIBRATION_INVALID_PACKETS count={invalid}")
    return frames, stats


def capture_pose(
    sock: socket.socket,
    pose: str,
    side: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pose == "home":
        print("HOME：双手都保持胸前、肘部弯曲、手腕中立，不要移动身体。")
        prompt_continue("请摆好 Home 姿态")
        duration = float(args.home_seconds)
        move_seconds = None
        print(f"CAPTURE_HOME_READY hold_seconds={duration:.1f}")
    else:
        descriptions = {
            "forward": "目标手向机器人正前方移动 15～25cm，保持肘部弯曲。",
            "backward": "目标手向身体方向收回 10～15cm，不要后仰。",
            "outward": "左手向左、右手向右移动 15～20cm。",
            "inward": "目标手向身体中线移动 8～15cm，不要越过中线。",
            "up": "目标手垂直向上移动 15～20cm，不要耸肩。",
            "down": "目标手垂直向下移动 10～15cm，不要完全伸直。",
        }
        print(f"{side} {pose}：{descriptions[pose]}")
        prompt_continue("请先回到 Home，再按提示移动目标手")
        print(f"MOVE_NOW 现在开始移动，抵达后保持静止。move_seconds={args.move_seconds:.1f} hold_seconds={args.hold_seconds:.1f}")
        duration = float(args.move_seconds + args.hold_seconds)
        move_seconds = float(args.move_seconds)
    frames, stats = receive_window(
        sock,
        duration,
        args.device_mode,
        float(args.capture_rate),
        args._wrist_calibration,
        move_seconds=move_seconds,
    )
    print(
        f"CAPTURE_WINDOW pose={pose} side={side} frames={len(frames)} "
        f"valid_rate={stats['valid_rate']:.3f}"
    )
    if not frames:
        raise RuntimeError(f"采集窗口没有有效动捕帧: pose={pose} side={side}")
    return frames, stats


def capture_quality_reasons(
    frames: list[dict[str, Any]],
    window_stats: dict[str, Any],
    pose: str,
    side: str,
    args: argparse.Namespace,
    home_centers: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    centers: dict[str, np.ndarray] = {}
    reasons: list[str] = []
    if window_stats["valid_rate"] < float(args.min_valid_rate):
        reasons.append(
            f"valid_rate={window_stats['valid_rate']:.3f}<"
            f"{float(args.min_valid_rate):.3f}"
        )
    for current_side in SIDES:
        center, jitter, count = stable_pose(frames, current_side)
        centers[current_side] = center
        if count < 3:
            reasons.append(f"{current_side}_samples={count}<3")
        if jitter > float(args.max_direction_rms):
            reasons.append(
                f"{current_side}_jitter={jitter:.4f}>{float(args.max_direction_rms):.4f}m"
            )
    if pose != "home" and home_centers is not None:
        displacement = float(np.linalg.norm(centers[side] - home_centers[side]))
        if displacement < float(args.min_direction_displacement):
            reasons.append(
                f"{side}_displacement={displacement:.4f}<"
                f"{float(args.min_direction_displacement):.4f}m"
            )
        other_side = "right" if side == "left" else "left"
        other_motion = float(np.linalg.norm(centers[other_side] - home_centers[other_side]))
        other_limit = max(0.03, float(args.max_direction_rms) * 2.0)
        if other_motion > other_limit:
            reasons.append(
                f"{other_side}_body_relative_motion={other_motion:.4f}>{other_limit:.4f}m"
            )
    return centers, reasons


def capture_until_valid(
    sock: socket.socket,
    pose: str,
    side: str,
    args: argparse.Namespace,
    home_centers: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    retries: list[dict[str, Any]] = []
    while True:
        frames, window_stats = capture_pose(sock, pose, side, args)
        centers, reasons = capture_quality_reasons(
            frames,
            window_stats,
            pose,
            side,
            args,
            home_centers,
        )
        if pose == "home":
            left_jitter = stable_pose(frames, "left")[1]
            right_jitter = stable_pose(frames, "right")[1]
            print(
                "CAPTURE_QUALITY "
                f"pose=home frames={len(frames)} valid_rate={window_stats['valid_rate']:.3f} "
                f"left_rms_m={left_jitter:.4f} right_rms_m={right_jitter:.4f}"
            )
        else:
            if home_centers is None:
                raise ValueError("方向动作质量检查缺少 Home 中心")
            other_side = "right" if side == "left" else "left"
            displacement = float(np.linalg.norm(centers[side] - home_centers[side]))
            other_motion = float(np.linalg.norm(centers[other_side] - home_centers[other_side]))
            target_jitter = stable_pose(frames, side)[1]
            print(
                "CAPTURE_QUALITY "
                f"pose={pose} side={side} frames={len(frames)} "
                f"valid_rate={window_stats['valid_rate']:.3f} "
                f"body_relative_displacement_m={displacement:.4f} "
                f"body_relative_hold_rms_m={target_jitter:.4f} "
                f"other_hand_body_relative_motion_m={other_motion:.4f}"
            )
        if not reasons:
            print(f"CAPTURE_ACCEPTED pose={pose} side={side} frames={len(frames)}")
            return frames, centers, window_stats, retries
        retries.append({"stats": window_stats, "reasons": reasons, "frames": frames})
        print(f"CAPTURE_RETRY pose={pose} side={side} reason={';'.join(reasons)}")


def run_capture(args: argparse.Namespace) -> Path:
    args._wrist_calibration = load_wrist_calibration(args.wrist_calibration, "both")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = args.capture_output / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=False)
    windows: dict[str, Any] = {}
    print(
        "CALIBRATION_PREFLIGHT "
        f"publish=false side={args.side} home_seconds={args.home_seconds} "
        f"move_seconds={args.move_seconds} hold_seconds={args.hold_seconds}"
    )
    sock = create_socket(args.ip, args.port)
    try:
        home_frames, home_centers, home_stats, home_retries = capture_until_valid(sock, "home", "both", args)
        windows["home"] = {
            "side": "both",
            "frames": home_frames,
            "stats": home_stats,
            "retries": home_retries,
        }
        for side_index, side in enumerate(("left", "right") if args.side == "both" else (args.side,), start=1):
            for direction_index, direction in enumerate(DIRECTIONS, start=1):
                key = f"{side}_{direction}"
                print(
                    f"DIRECTION_PROMPT side={side} direction={direction} "
                    f"step={side_index}/{2 if args.side == 'both' else 1}:{direction_index}/{len(DIRECTIONS)}"
                )
                frames, _, stats, retries = capture_until_valid(
                    sock,
                    direction,
                    side,
                    args,
                    home_centers=home_centers,
                )
                windows[key] = {
                    "side": side,
                    "direction": direction,
                    "frames": frames,
                    "stats": stats,
                    "retries": retries,
                }
    finally:
        sock.close()
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "position_reference": "wrist_minus_chest",
        "control_point": "anatomical_wrist",
        "wrist_calibration": str(args._wrist_calibration.source_path),
        "device_mode": args.device_mode,
        "udp": {"ip": args.ip, "port": args.port},
        "windows": windows,
    }
    path = session_dir / "capture.json"
    save_json(path, payload)
    print(f"CAPTURE_OUTPUT {path}")
    return path


def resolve_capture_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "capture.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到标定采集文件: {path}")
    return path


def stable_pose(
    frames: list[dict[str, Any]],
    side: str,
    tail_ratio: float = 0.5,
    *,
    body_relative: bool = True,
) -> tuple[np.ndarray, float, int]:
    if not frames:
        raise ValueError(f"窗口没有帧: {side}")
    tail = frames[max(0, int(len(frames) * (1.0 - tail_ratio))):]
    positions = np.asarray([item[f"{side}_pos"] for item in tail], dtype=float)
    if body_relative and all("body_pos" in item for item in tail):
        body_positions = np.asarray([item["body_pos"] for item in tail], dtype=float)
        positions = positions - body_positions
    center = np.median(positions, axis=0)
    jitter = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))
    return center, jitter, len(tail)


def analyze_capture(args: argparse.Namespace) -> Path:
    input_path = resolve_capture_input(args.input)
    payload = load_json(input_path)
    if payload.get("control_point") != "anatomical_wrist":
        raise ValueError("工作空间采集不是 anatomical_wrist 控制点，请先完成手腕标定并重新采集")
    windows = payload.get("windows")
    if not isinstance(windows, dict) or "home" not in windows:
        raise ValueError("标定采集缺少 windows.home")
    home_frames = windows["home"].get("frames", [])
    homes: dict[str, np.ndarray] = {}
    quality: dict[str, Any] = {}
    valid = True
    for side in SIDES:
        center, jitter, count = stable_pose(home_frames, side)
        homes[side] = center
        home_stats = windows["home"].get("stats", {})
        quality[f"{side}_home_jitter"] = jitter
        quality[f"{side}_home_samples"] = count
        quality[f"{side}_home_valid_rate"] = float(home_stats.get("valid_rate", 1.0))
        valid = valid and jitter <= float(args.max_direction_rms)
        valid = valid and quality[f"{side}_home_valid_rate"] >= float(args.min_valid_rate)

    displacements: dict[str, dict[str, list[float]]] = {side: {} for side in SIDES}
    direction_vectors: dict[str, list[np.ndarray]] = {direction: [] for direction in DIRECTIONS}
    for side in SIDES:
        for direction in DIRECTIONS:
            key = f"{side}_{direction}"
            if key not in windows:
                if args.side == "both":
                    valid = False
                continue
            center, jitter, count = stable_pose(windows[key].get("frames", []), side)
            delta = center - homes[side]
            displacements[side][direction] = delta
            quality[f"{side}_{direction}_jitter"] = jitter
            quality[f"{side}_{direction}_samples"] = count
            quality[f"{side}_{direction}_valid_rate"] = float(
                windows[key].get("stats", {}).get("valid_rate", 1.0)
            )
            quality[f"{side}_{direction}_displacement"] = float(np.linalg.norm(delta))
            valid = valid and jitter <= float(args.max_direction_rms)
            valid = valid and quality[f"{side}_{direction}_valid_rate"] >= float(args.min_valid_rate)
            valid = valid and float(np.linalg.norm(delta)) >= float(args.min_direction_displacement)
            if direction == "forward":
                direction_vectors["forward"].append(delta)
            elif direction == "outward":
                direction_vectors["outward"].append(delta if side == "left" else -delta)
            elif direction == "up":
                direction_vectors["up"].append(delta)

    try:
        axes = []
        for direction in ("forward", "outward", "up"):
            vectors = direction_vectors[direction]
            if not vectors:
                raise ValueError(f"缺少坐标轴动作: {direction}")
            vector = np.mean([item / np.linalg.norm(item) for item in vectors], axis=0)
            axes.append(vector / np.linalg.norm(vector))
        x_axis, y_axis = axes[0], axes[1]
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        basis_mocap = np.column_stack((x_axis, y_axis, z_axis))
        base_from_mocap = basis_mocap.T
        determinant = float(np.linalg.det(base_from_mocap))
        up_alignment = float(np.mean([
            np.dot(item / np.linalg.norm(item), z_axis)
            for item in direction_vectors["up"]
        ]))
        quality["up_alignment"] = up_alignment
        valid = valid and determinant > 0.95 and up_alignment > 0.5
    except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
        base_from_mocap = np.eye(3)
        determinant = 0.0
        quality["axis_error"] = str(exc)
        valid = False

    for side in SIDES:
        side_displacements = displacements[side]
        expected_axes = {
            "forward": np.array([1.0, 0.0, 0.0]),
            "backward": np.array([-1.0, 0.0, 0.0]),
            "outward": np.array([0.0, 1.0 if side == "left" else -1.0, 0.0]),
            "inward": np.array([0.0, -1.0 if side == "left" else 1.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0]),
            "down": np.array([0.0, 0.0, -1.0]),
        }
        for direction, delta in side_displacements.items():
            transformed = base_from_mocap @ np.asarray(delta, dtype=float)
            norm = float(np.linalg.norm(transformed))
            cosine = float(np.dot(transformed / norm, expected_axes[direction])) if norm > 1e-9 else -1.0
            quality[f"{side}_{direction}_direction_cosine"] = cosine
            valid = valid and cosine >= float(args.min_direction_cosine)

    for side in SIDES:
        side_displacements = displacements[side]
        for positive, negative in (("forward", "backward"), ("outward", "inward"), ("up", "down")):
            if positive not in side_displacements or negative not in side_displacements:
                continue
            positive_vector = np.asarray(side_displacements[positive], dtype=float)
            negative_vector = np.asarray(side_displacements[negative], dtype=float)
            cosine = float(
                np.dot(positive_vector, negative_vector)
                / (np.linalg.norm(positive_vector) * np.linalg.norm(negative_vector))
            )
            quality[f"{side}_{positive}_{negative}_cosine"] = cosine
            valid = valid and cosine <= float(args.max_opposite_direction_cosine)

    retry_summary = {
        key: {
            "retry_count": len(window.get("retries", [])),
            "failed_frames": sum(len(item.get("frames", [])) for item in window.get("retries", [])),
            "reasons": [reason for item in window.get("retries", []) for reason in item.get("reasons", [])],
        }
        for key, window in windows.items()
        if window.get("retries")
    }
    analysis = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_capture": str(input_path),
        "position_reference": payload.get("position_reference", "absolute_hand_legacy"),
        "control_point": payload.get("control_point", "legacy_hand_or_tcp"),
        "wrist_calibration": payload.get("wrist_calibration"),
        "valid": bool(valid),
        "base_from_mocap_rotation": base_from_mocap,
        "mocap_home": {side: homes[side] for side in SIDES},
        "displacements": displacements,
        "retry_summary": retry_summary,
        "quality": {**quality, "rotation_determinant": determinant},
        "quality_thresholds": {
            "min_direction_displacement_m": float(args.min_direction_displacement),
            "max_direction_rms_m": float(args.max_direction_rms),
            "min_valid_rate": float(args.min_valid_rate),
            "min_direction_cosine": float(args.min_direction_cosine),
            "max_opposite_direction_cosine": float(args.max_opposite_direction_cosine),
        },
    }
    output = artifact_path(args.calibration_output, "analysis", overwrite=bool(args.overwrite))
    save_json(output, analysis)
    print(f"ANALYZE_RESULT {'OK' if valid else 'FAIL'}")
    print(f"ANALYSIS_OUTPUT {output}")
    if not valid:
        print("ANALYZE_RETRY 请按 WORKSPACE_CALIBRATION.md 重做失败方向")
        return output
    return output


def analysis_qualifies_for_scan(analysis: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    quality = analysis.get("quality", {})
    reasons: list[str] = []
    for side in SIDES:
        home_jitter = float(quality.get(f"{side}_home_jitter", float("inf")))
        home_rate = float(quality.get(f"{side}_home_valid_rate", 0.0))
        if home_jitter > float(args.max_direction_rms):
            reasons.append(f"{side}_home_jitter={home_jitter:.4f}>{float(args.max_direction_rms):.4f}m")
        if home_rate < float(args.min_valid_rate):
            reasons.append(f"{side}_home_valid_rate={home_rate:.3f}<{float(args.min_valid_rate):.3f}")
        for direction in DIRECTIONS:
            jitter_key = f"{side}_{direction}_jitter"
            rate_key = f"{side}_{direction}_valid_rate"
            disp_key = f"{side}_{direction}_displacement"
            cosine_key = f"{side}_{direction}_direction_cosine"
            if jitter_key in quality and float(quality[jitter_key]) > float(args.max_direction_rms):
                reasons.append(f"{jitter_key}={float(quality[jitter_key]):.4f}>{float(args.max_direction_rms):.4f}m")
            if rate_key in quality and float(quality[rate_key]) < float(args.min_valid_rate):
                reasons.append(f"{rate_key}={float(quality[rate_key]):.3f}<{float(args.min_valid_rate):.3f}")
            if disp_key in quality and float(quality[disp_key]) < float(args.min_direction_displacement):
                reasons.append(f"{disp_key}={float(quality[disp_key]):.4f}<{float(args.min_direction_displacement):.4f}m")
            if cosine_key in quality and float(quality[cosine_key]) < float(args.min_direction_cosine):
                reasons.append(f"{cosine_key}={float(quality[cosine_key]):.4f}<{float(args.min_direction_cosine):.4f}")
        for positive, negative in (("forward", "backward"), ("outward", "inward"), ("up", "down")):
            key = f"{side}_{positive}_{negative}_cosine"
            if key in quality and float(quality[key]) > float(args.max_opposite_direction_cosine):
                reasons.append(f"{key}={float(quality[key]):.4f}>{float(args.max_opposite_direction_cosine):.4f}")
    if float(quality.get("rotation_determinant", 0.0)) <= 0.95:
        reasons.append(f"rotation_determinant={float(quality.get('rotation_determinant', 0.0)):.4f}<=0.9500")
    if float(quality.get("up_alignment", 0.0)) <= 0.5:
        reasons.append(f"up_alignment={float(quality.get('up_alignment', 0.0)):.4f}<=0.5000")
    return len(reasons) == 0, reasons


def make_solver(home_q: np.ndarray):
    sys.path.insert(0, str(SOURCE_DIR))
    from teleop_ik_types import TeleopIKConfig
    from teleop_ik_ubtech_casadi import ARM_JOINT_NAMES, OfficialTiangong2ProIK
    from teleop_robot_constants import build_relaxed_joint_limits

    config = TeleopIKConfig(
        joint_names=list(ARM_JOINT_NAMES),
        motor_ids=list(range(14)),
        joint_limits=build_relaxed_joint_limits(15),
        task_mode="pose",
        home_q=home_q,
    )
    return OfficialTiangong2ProIK(config)


def pose_metrics(solver: Any, q: np.ndarray, waist: float, target_l: np.ndarray, target_r: np.ndarray) -> tuple[float, float]:
    import pinocchio as pin

    full_q = solver._full_q(q, waist)
    pin.forwardKinematics(solver.model, solver.data, full_q)
    pin.updateFramePlacements(solver.model, solver.data)
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    for frame_id, target in ((solver.left_frame_id, target_l), (solver.right_frame_id, target_r)):
        current = solver.data.oMf[frame_id]
        position_errors.append(float(np.linalg.norm(current.translation - target[:3, 3])))
        orientation_errors.append(float(np.linalg.norm(pin.log3(current.rotation @ target[:3, :3].T))))
    return max(position_errors), max(orientation_errors)


def scan_workspace(args: argparse.Namespace) -> Path:
    analysis_path = resolve_capture_input(args.input)
    analysis = load_json(analysis_path)
    if not analysis.get("valid", False):
        rechecked_valid, _ = analysis_qualifies_for_scan(analysis, args)
        if not rechecked_valid:
            raise ValueError("analysis.json 未通过质量检查，禁止扫描")
        analysis = {**analysis, "valid": True}
    home_q = np.asarray(args.robot_home_q, dtype=float)
    if home_q.shape != (14,):
        raise ValueError("robot_home_q 必须是 14 维")
    solver = make_solver(home_q)
    from teleop_ik_types import TeleopIKInput

    left_wrist_home, right_wrist_home = solver.wrist_effector_poses(home_q)
    left_wrist_to_tcp, right_wrist_to_tcp = solver.wrist_to_tcp_transforms()
    left_home = left_wrist_home @ left_wrist_to_tcp
    right_home = right_wrist_home @ right_wrist_to_tcp
    directions = {
        "forward": np.array([1.0, 0.0, 0.0]),
        "backward": np.array([-1.0, 0.0, 0.0]),
        "outward_left": np.array([0.0, 1.0, 0.0]),
        "inward_left": np.array([0.0, -1.0, 0.0]),
        "outward_right": np.array([0.0, -1.0, 0.0]),
        "inward_right": np.array([0.0, 1.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "down": np.array([0.0, 0.0, -1.0]),
    }
    bounds: dict[str, Any] = {}
    scan_sides = SIDES if args.side == "both" else (args.side,)
    for side, wrist_home, wrist_to_tcp in (
        ("left", left_wrist_home, left_wrist_to_tcp),
        ("right", right_wrist_home, right_wrist_to_tcp),
    ):
        if side not in scan_sides:
            continue
        bounds[side] = {}
        for direction in DIRECTIONS:
            axis_name = direction if direction in {"forward", "backward", "up", "down"} else f"{direction}_{side}"
            axis = directions[axis_name]
            best: dict[str, Any] | None = None
            failure: dict[str, Any] | None = None
            for distance in np.arange(float(args.scan_step), float(args.scan_max) + 1e-9, float(args.scan_step)):
                solver.set_home_q(home_q)
                target_l = left_home.copy()
                target_r = right_home.copy()
                target_wrist = wrist_home.copy()
                target_wrist[:3, 3] = wrist_home[:3, 3] + axis * distance
                target = target_wrist @ wrist_to_tcp
                if side == "left":
                    target_l = target
                else:
                    target_r = target
                result = solver.solve(
                    TeleopIKInput(
                        timestamp=time.time(),
                        control_side="both",
                        target_pose_l=target_l,
                        target_pose_r=target_r,
                        current_q=home_q,
                        current_waist_position=0.0,
                    )
                )
                q = np.asarray(result.command.positions, dtype=float)
                position_error, orientation_error = pose_metrics(
                    solver,
                    q,
                    float(result.command.waist_position or 0.0),
                    target_l,
                    target_r,
                )
                full_q = solver._full_q(q, result.command.waist_position)
                active_limits = np.asarray(solver.full_joint_limits, dtype=float)
                lower = active_limits[:, 0]
                upper = active_limits[:, 1]
                margin = float(args.joint_margin_rad)
                margin_by_joint = np.minimum(full_q - lower, upper - full_q)
                minimum_joint_margin = float(np.min(margin_by_joint))
                waist_index = solver.joint_indices["body_yaw_joint"]
                waist_margin = float(margin_by_joint[waist_index])
                joint_margin_ok = bool(minimum_joint_margin >= margin)
                solver_ok = result.reason == "ok"
                feasible = bool(
                    solver_ok
                    and np.isfinite(position_error)
                    and np.isfinite(orientation_error)
                    and position_error <= float(args.position_error_threshold)
                    and orientation_error <= float(args.orientation_error_threshold)
                    and joint_margin_ok
                )
                failed_checks = [
                    name
                    for name, passed in (
                        ("solver", solver_ok),
                        ("position_error", position_error <= float(args.position_error_threshold)),
                        ("orientation_error", orientation_error <= float(args.orientation_error_threshold)),
                        ("joint_margin", joint_margin_ok),
                    )
                    if not passed
                ]
                metrics = {
                    "distance_m": float(distance),
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "minimum_joint_margin_rad": minimum_joint_margin,
                    "waist_margin_rad": waist_margin,
                    "joint_margin_ok": joint_margin_ok,
                    "solver_reason": result.reason,
                    "solver_return_status": result.command.meta.get("solver_return_status", "unknown"),
                    "solver_position_error_m": result.command.meta.get("solver_position_error_m"),
                    "solver_orientation_error_rad": result.command.meta.get("solver_orientation_error_rad"),
                    "solver_exception": result.command.meta.get("solver_exception"),
                    "iteration_count": int(result.iteration_count),
                }
                if feasible:
                    best = metrics
                else:
                    failure = {**metrics, "failed_checks": failed_checks}
                    break
            bounds[side][direction] = {
                "math_boundary_m": None if best is None else best["distance_m"],
                "safe_boundary_m": None if best is None else best["distance_m"] * float(args.safe_boundary_ratio),
                "position_error_m": None if best is None else best["position_error_m"],
                "orientation_error_rad": None if best is None else best["orientation_error_rad"],
                "minimum_joint_margin_rad": None if best is None else best["minimum_joint_margin_rad"],
                "waist_margin_rad": None if best is None else best["waist_margin_rad"],
                "joint_margin_ok": False if best is None else best["joint_margin_ok"],
                "solver_return_status": None if best is None else best["solver_return_status"],
                "failure": failure,
            }
            print(f"SCAN_BOUND side={side} direction={direction} boundary={bounds[side][direction]['math_boundary_m']}")

    direction_scales: dict[str, dict[str, float | None]] = {}
    for side, side_bounds in bounds.items():
        direction_scales[side] = {}
        for direction, bound in side_bounds.items():
            safe_boundary = bound["safe_boundary_m"]
            human_delta = np.asarray(analysis["displacements"][side][direction], dtype=float)
            human_distance = float(np.linalg.norm(human_delta))
            direction_scales[side][direction] = (
                None if safe_boundary is None or human_distance <= 1e-9 else float(safe_boundary / human_distance)
            )

    scan_valid = bool(bounds) and all(
        bound["safe_boundary_m"] is not None
        for side_bounds in bounds.values()
        for bound in side_bounds.values()
    )
    scan_has_boundary = any(
        bound["safe_boundary_m"] is not None
        for side_bounds in bounds.values()
        for bound in side_bounds.values()
    )
    output = artifact_path(args.calibration_output, "calibration", overwrite=bool(args.overwrite))
    from teleop_ik_ubtech_casadi import OFFICIAL_SOLVER_SO_PATH, OFFICIAL_URDF_PATH

    result = {
        **analysis,
        "schema_version": 1,
        "valid": scan_valid,
        "created_at": utc_now(),
        "source_analysis": str(analysis_path),
        "control_frame": args.control_frame,
        "urdf_path": str(OFFICIAL_URDF_PATH),
        "solver_path": str(OFFICIAL_SOLVER_SO_PATH),
        "solver_mode": getattr(solver, "solver_mode", "unknown"),
        "robot_home_q": home_q,
        "workspace_bounds": bounds,
        "direction_scales": direction_scales,
        "safe_boundary_ratio": float(args.safe_boundary_ratio),
        "scan": {
            "step_m": float(args.scan_step),
            "max_m": float(args.scan_max),
            "position_error_threshold": float(args.position_error_threshold),
            "orientation_error_threshold": float(args.orientation_error_threshold),
            "joint_limit_policy": "config_override_relaxed_bounds",
            "joint_margin_rad": float(args.joint_margin_rad),
            "additional_joint_margin_rad": float(args.joint_margin_rad),
        },
    }
    save_json(output, result)
    print(f"SCAN_RESULT {'OK' if scan_valid else 'FAIL'}")
    print(f"CALIBRATION_OUTPUT {output}")
    if not scan_has_boundary:
        raise RuntimeError("官方 IK 在所有扫描方向均未找到有效边界")
    return output


def report(path: Path) -> int:
    value = load_json(resolve_capture_input(path))
    print(f"CALIBRATION_REPORT valid={value.get('valid', False)} schema_version={value.get('schema_version')}")
    print(f"source_capture={value.get('source_capture', '')}")
    print(f"control_frame={value.get('control_frame', 'unknown')}")
    print(f"solver_mode={value.get('solver_mode', 'unknown')} solver_path={value.get('solver_path', '')}")
    quality = value.get("quality", {})
    print(
        "COORDINATE_QUALITY "
        f"rotation_determinant={quality.get('rotation_determinant')} "
        f"up_alignment={quality.get('up_alignment')}"
    )
    for side, values in value.get("workspace_bounds", {}).items():
        for direction, bound in values.items():
            failure = bound.get("failure") or {}
            print(
                f"BOUND side={side} direction={direction} "
                f"math_m={bound.get('math_boundary_m')} safe_m={bound.get('safe_boundary_m')} "
                f"position_error_m={bound.get('position_error_m')} "
                f"orientation_error_rad={bound.get('orientation_error_rad')} "
                f"joint_margin_rad={bound.get('minimum_joint_margin_rad')} "
                f"waist_margin_rad={bound.get('waist_margin_rad')} "
                f"solver_status={bound.get('solver_return_status')} "
                f"failure={failure.get('failed_checks', [])}"
            )
    return 0 if value.get("valid", False) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="天工 2 Pro 工作空间标定向导；采集和扫描均不发布机器人命令")
    parser.add_argument("--stage", choices=("guide", "capture", "analyze", "scan", "report", "all"), default="all")
    parser.add_argument("--input", type=Path, default=None, help="capture.json、analysis.json 或 calibration.json")
    parser.add_argument("--output", type=Path, default=None, help="兼容参数：覆盖最终 calibration JSON 路径")
    parser.add_argument("--capture-output", type=Path, default=DEFAULT_OUTPUT / "raw")
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_OUTPUT / "calibration.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--capture-rate", type=float, default=30.0)
    parser.add_argument("--device-mode", choices=("glove", "controller", "auto"), default="glove")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--home-seconds", type=float, default=3.0)
    parser.add_argument("--move-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--min-direction-displacement", type=float, default=0.06)
    parser.add_argument("--max-direction-rms", type=float, default=0.055)
    parser.add_argument("--min-valid-rate", type=float, default=0.90)
    parser.add_argument("--min-direction-cosine", type=float, default=0.68)
    parser.add_argument("--max-opposite-direction-cosine", type=float, default=-0.40)
    parser.add_argument("--position-error-threshold", type=float, default=0.015)
    parser.add_argument("--orientation-error-threshold", type=float, default=0.20)
    parser.add_argument("--joint-margin-rad", type=float, default=0.0)
    parser.add_argument("--safe-boundary-ratio", type=float, default=0.65)
    parser.add_argument("--scan-step", type=float, default=0.01)
    parser.add_argument("--scan-max", type=float, default=0.5)
    parser.add_argument("--wrist-calibration", type=Path, required=False, help="有效 hand→anatomical wrist 标定；capture/all 阶段必需")
    parser.add_argument("--control-frame", choices=("wrist_roll",), default="wrist_roll")
    parser.add_argument("--robot-home-q", type=float, nargs=14, default=[0.0] * 14)
    parser.add_argument("--dry-run", action="store_true", help="标定阶段不连接机器人、不发布命令")
    parser.add_argument("--yes", action="store_true", help="跳过非运动类确认")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.output is not None:
        args.calibration_output = args.output
    args.capture_output = args.capture_output.expanduser()
    args.calibration_output = args.calibration_output.expanduser()
    if args.stage == "guide":
        print_guide()
        return 0
    if args.stage == "capture":
        run_capture(args)
        return 0
    if args.stage == "analyze":
        if args.input is None:
            raise ValueError("--stage analyze 需要 --input capture.json 或采集目录")
        analysis_path = analyze_capture(args)
        return 0 if load_json(analysis_path).get("valid", False) else 2
    if args.stage == "scan":
        if args.input is None:
            raise ValueError("--stage scan 需要 --input analysis.json")
        calibration_path = scan_workspace(args)
        return 0 if load_json(calibration_path).get("valid", False) else 2
    if args.stage == "report":
        if args.input is None:
            raise ValueError("--stage report 需要 --input calibration.json")
        return report(args.input)
    if args.stage == "all":
        capture_path = run_capture(args)
        analyze_values = dict(vars(args))
        analyze_values["input"] = capture_path
        analysis_path = analyze_capture(argparse.Namespace(**analyze_values))
        if not load_json(analysis_path).get("valid", False):
            return 2
        scan_values = dict(vars(args))
        scan_values["input"] = analysis_path
        calibration_path = scan_workspace(argparse.Namespace(**scan_values))
        report_code = report(calibration_path)
        return 0 if report_code == 0 and load_json(calibration_path).get("valid", False) else 2
    raise ValueError(f"未知标定阶段: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))