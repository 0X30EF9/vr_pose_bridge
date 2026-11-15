from __future__ import annotations

import argparse
import copy
import json
import math
import time
import threading
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import roslibpy
from scipy.spatial.transform import Rotation as R

from teleop_ik_adapters import build_rosbridge_motor_message, build_rosbridge_waist_message
from teleop_ik_types import IKCommand
from teleop_control_safety import command_time_step, feedback_is_fresh, joint_velocity_limits_from_urdf, require_fresh_feedback
from teleop_joint_schema import arm_joint_limits_from_urdf, arm_positions_from_status
from teleop_direct_retarget import (
    DirectRetargeter,
    add_direct_retarget_arguments,
    direct_retarget_config_from_args,
    skeleton_observation,
)
from teleop_hand_ik import (
    HandIkError,
    Revo2HandIkSidecar,
    LatestHandIkWorker,
    add_hand_ik_arguments,
    hand_ik_config_from_args,
    hand_ik_enabled,
)
from teleop_hand_control import (
    HandControlConfig,
    HandControlError,
    apply_hand_command_safety,
    build_hand_joint_state_message,
    hand_control_config_from_mapping,
    map_hand_radians,
    parse_hand_state_message,
)
from teleop_mocap_calibration import RelativeMocapTargetMapper, load_workspace_calibration
from teleop_wrist_calibration import load_wrist_calibration, reconstruct_wrist_pose_from_mapping
from teleop_wrist_safety import (
    WristBehaviorLimits,
    validate_wrist_behavior_candidate,
    wrist_position_errors,
)
from mocap_csv_format import load_replay_frames_from_csv
from teleop_mocap_protocol import QUAT_ORDER, to_jsonable
from teleop_robot_constants import (
    ARM_MOTOR_TEMPLATES,
    HAND_CMD_TOPICS,
    HAND_JOINT_STATE_TYPE,
    HAND_STATE_TOPICS,
    NATURAL_DOWN_Q,
    POSITION_ONLY_LOCKED_JOINT_INDICES,
    WAIST_CMD_TOPIC,
    WAIST_CMD_TYPE,
    WAIST_ERROR_DESCRIPTIONS,
    WAIST_MOTOR_ID,
    WAIST_JOINT_NAME,
    WAIST_MOTOR_TEMPLATE,
    WAIST_STATUS_TOPIC,
    WAIST_STATUS_TYPE,
    build_relaxed_joint_limits,
)

ARM_STATUS_TOPIC = "/arm/status"
ARM_STATUS_TYPE = "bodyctrl_msgs/msg/MotorStatusMsg"
ARM_CMD_TOPIC = "/arm/cmd_pos"
ARM_CMD_TYPE = "bodyctrl_msgs/msg/CmdSetMotorPosition"
ROSBRIDGE_HOST = "192.168.1.110"
ROSBRIDGE_PORT = 9091

CONTROL_RATE_HZ = 100.0
ARM_MOTOR_IDS = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
IK_ARM_JOINT_NAMES = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

T_BASE_MOCAP = np.eye(4)

DEFAULT_MAX_POSITION_STEP = 0.15
DEFAULT_MAX_ANGULAR_STEP = 0.8
DEFAULT_MAX_JOINT_DELTA_PER_STEP = 0.04
DEFAULT_MAX_COMMAND_DELTA_FROM_BASELINE = 0.5
DEFAULT_RESTORE_SECONDS = 3.0
MAX_ARM_STATUS_AGE_SECONDS = 0.5
MAX_WAIST_STATUS_AGE_SECONDS = 0.5


def make_transform(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 VR UDP 数采 frames.jsonl 转成 IK 命令，并可选择通过 rosbridge 安全回放给机器人",
    )
    parser.add_argument("input", type=Path, help="数采 session 目录或 frames.jsonl 文件")
    parser.add_argument("--host", default=ROSBRIDGE_HOST, help="rosbridge 地址")
    parser.add_argument("--port", type=int, default=ROSBRIDGE_PORT, help="rosbridge 端口")
    parser.add_argument("--control-side", choices=("left", "right", "both"), default="right", help="回放控制侧")
    parser.add_argument("--control-mode", choices=("skeleton", "direct", "ik", "raise_template"), default="skeleton", help="skeleton=自动对齐骨架并跟随手腕；direct=高级骨段方向映射；ik=高级末端位姿 IK；raise_template=姿态模板")
    parser.add_argument("--ik-backend", choices=("teleop_dls", "tiangong2pro_official", "ubtech_casadi"), default="teleop_dls", help="IK 求解后端；官方后端同时控制 31 号腰部关节")
    parser.add_argument("--ik-task", choices=("pose", "position_only"), default="pose", help="IK 任务类型：pose=官方完整 TCP 位姿 IK；position_only=实验性锁腕位置 IK")
    parser.add_argument("--rate", "--control-rate", dest="rate", type=float, default=CONTROL_RATE_HZ, help="回放手臂控制频率，默认 100Hz；需要相应的新鲜反馈")
    parser.add_argument("--max-joint-velocity", type=float, default=0.48, help="关节速度上限 rad/s，同时受 URDF 与单步上限约束")
    parser.add_argument("--hand-control-rate", type=float, default=20.0, help="独立手部异步 IK 最大计算频率 Hz")
    parser.add_argument("--max-hand-velocity", type=float, default=1.0, help="手部归一化位置每秒最大变化")
    parser.add_argument("--max-frames", type=int, default=None, help="最多读取多少帧")
    parser.add_argument("--stride", type=int, default=1, help="按帧间隔抽样，默认不抽样")
    parser.add_argument("--position-scale", type=float, default=1.0, help="有效标定安全边界内的总增益")
    parser.add_argument("--workspace-calibration", type=Path, default=None, help="可选工作空间标定 JSON；显式指定的文件必须有效")
    parser.add_argument("--wrist-calibration", type=Path, default=None, help="可选人体 hand→wrist 控制点标定 JSON；显式指定的文件必须有效，省略时使用 skeleton 腕点")
    parser.add_argument("--hand-only", action="store_true", help="只回放双手 IK，强制停发手臂和腰部且不要求工作空间标定")
    parser.add_argument("--orientation-scale", type=float, default=None, help="手腕旋转增益；skeleton 默认 1.0，其他模式默认 0.25")
    parser.add_argument("--startup-hold-seconds", type=float, default=2.0, help="skeleton 自动对齐前，人体自然下垂和机器人静止须同时保持的时间")
    parser.add_argument("--startup-human-vertical-cosine", type=float, default=0.78)
    parser.add_argument("--startup-human-min-wrist-drop", type=float, default=0.30)
    parser.add_argument("--startup-human-max-lateral-offset", type=float, default=0.16)
    parser.add_argument("--startup-max-home-rms", type=float, default=0.025)
    parser.add_argument("--max-position-step", type=float, default=DEFAULT_MAX_POSITION_STEP, help="单帧手部位置跳变阈值，单位 m")
    parser.add_argument("--max-angular-step", type=float, default=DEFAULT_MAX_ANGULAR_STEP, help="单帧手部姿态跳变阈值，单位 rad")
    parser.add_argument("--max-joint-delta-per-step", type=float, default=DEFAULT_MAX_JOINT_DELTA_PER_STEP, help="IK 命令单帧关节限幅")
    parser.add_argument("--max-command-delta-from-baseline", type=float, default=DEFAULT_MAX_COMMAND_DELTA_FROM_BASELINE, help="相对 baseline 的最大关节偏移")
    add_direct_retarget_arguments(parser)
    add_hand_ik_arguments(parser)
    parser.add_argument("--hand-control-json", default="{}", help="天工无疆手部控制 JSON 配置")
    parser.add_argument("--spd", type=float, default=0.7, help="/arm/cmd_pos 速度参数")
    parser.add_argument("--cur", type=float, default=2.0, help="/arm/cmd_pos 电流参数")
    parser.add_argument("--connect-timeout", type=float, default=8.0, help="rosbridge 连接超时")
    parser.add_argument("--status-timeout", type=float, default=5.0, help="等待 /arm/status 完整反馈超时")
    parser.add_argument("--restore-seconds", type=float, default=DEFAULT_RESTORE_SECONDS, help="恢复 baseline 后观察时间")
    parser.add_argument("--prepose", choices=("none", "natural_down"), default="natural_down", help="回放前先进入的机器人基线姿态")
    parser.add_argument("--prepose-seconds", type=float, default=3.0, help="进入回放基线姿态的等待时间")
    parser.add_argument("--prepose-steps", type=int, default=30, help="进入回放基线姿态的插值步数")
    parser.add_argument("--use-robot-status", action="store_true", help="dry-run 时也连接机器人读取真实 baseline，不发布命令")
    parser.add_argument("--allow-risk", action="store_true", help="允许数据跳变或命令偏移超过默认阈值；不建议真机使用")
    parser.add_argument("--streaming", action="store_true", help="execute+ik 时边算 IK 边发布；skeleton 执行始终按实时反馈逐帧求解")
    parser.add_argument("--execute", action="store_true", help="真正发布回放命令；不加则只做数据检查和 IK 预览")
    parser.add_argument("--yes", action="store_true", help="跳过二次确认，配合 --execute 使用")
    args = parser.parse_args()
    if args.orientation_scale is None:
        args.orientation_scale = 1.0 if args.control_mode == "skeleton" else 0.25
    return args


def resolve_input_path(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        csv_candidates = sorted(path.glob("vr_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if csv_candidates:
            return csv_candidates[0]
        jsonl_path = path / "frames.jsonl"
        if jsonl_path.is_file():
            return jsonl_path
    if not path.is_file():
        raise FileNotFoundError(f"找不到回放输入: {path}")
    return path


def load_jsonl_frames(path: Path, max_frames: int | None, stride: int) -> list[dict[str, Any]]:
    if stride <= 0:
        raise ValueError("--stride 必须 >= 1")

    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            if line_index % stride != 0:
                continue
            text = line.strip()
            if not text:
                continue
            frame = json.loads(text)
            validate_frame(frame, line_index + 1)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break

    if not frames:
        raise ValueError(f"没有可回放帧: {path}")
    return frames


def load_frames(path: Path, max_frames: int | None, stride: int) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return load_replay_frames_from_csv(path, max_frames, stride)
    return load_jsonl_frames(path, max_frames, stride)


def validate_frame(frame: dict[str, Any], line_number: int) -> None:
    for key in ("receive_time", "left_pos", "left_quat", "right_pos", "right_quat"):
        if key not in frame:
            raise ValueError(f"frames.jsonl 第 {line_number} 行缺少字段: {key}")
    for key, size in (("left_pos", 3), ("right_pos", 3), ("left_quat", 4), ("right_quat", 4)):
        values = np.asarray(frame[key], dtype=float)
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"frames.jsonl 第 {line_number} 行字段非法: {key}")


def quat_angle_delta(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def analyze_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    left_positions = np.asarray([frame["left_pos"] for frame in frames], dtype=float)
    right_positions = np.asarray([frame["right_pos"] for frame in frames], dtype=float)
    left_quats = np.asarray([frame["left_quat"] for frame in frames], dtype=float)
    right_quats = np.asarray([frame["right_quat"] for frame in frames], dtype=float)
    times = np.asarray([float(frame["receive_time"]) for frame in frames], dtype=float)

    left_steps = np.linalg.norm(np.diff(left_positions, axis=0), axis=1) if len(frames) > 1 else np.asarray([0.0])
    right_steps = np.linalg.norm(np.diff(right_positions, axis=0), axis=1) if len(frames) > 1 else np.asarray([0.0])
    left_angles = np.asarray([quat_angle_delta(left_quats[i - 1], left_quats[i]) for i in range(1, len(frames))]) if len(frames) > 1 else np.asarray([0.0])
    right_angles = np.asarray([quat_angle_delta(right_quats[i - 1], right_quats[i]) for i in range(1, len(frames))]) if len(frames) > 1 else np.asarray([0.0])

    return {
        "frame_count": len(frames),
        "duration_seconds": float(max(0.0, times[-1] - times[0])) if len(times) else 0.0,
        "left_position_range": (left_positions.max(axis=0) - left_positions.min(axis=0)).tolist(),
        "right_position_range": (right_positions.max(axis=0) - right_positions.min(axis=0)).tolist(),
        "left_max_position_step": float(left_steps.max()),
        "right_max_position_step": float(right_steps.max()),
        "left_max_angular_step": float(left_angles.max()),
        "right_max_angular_step": float(right_angles.max()),
    }


def print_json_line(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(to_jsonable(payload), ensure_ascii=False, separators=(',', ':'))}")


def data_risk_reasons(summary: dict[str, Any], max_position_step: float, max_angular_step: float) -> list[str]:
    reasons = []
    for key in ("left_max_position_step", "right_max_position_step"):
        if summary[key] > max_position_step:
            reasons.append(f"{key}={summary[key]:.6f}>{max_position_step:.6f}")
    for key in ("left_max_angular_step", "right_max_angular_step"):
        if summary[key] > max_angular_step:
            reasons.append(f"{key}={summary[key]:.6f}>{max_angular_step:.6f}")
    return reasons


def build_transform_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    quat_xyzw = quat if QUAT_ORDER == "xyzw" else np.asarray([quat[1], quat[2], quat[3], quat[0]])
    return make_transform(pos, R.from_quat(quat_xyzw).as_matrix())


def replay_anatomical_wrist_pose(frame: dict[str, Any], side: str, wrist_calibration) -> np.ndarray:
    if wrist_calibration is not None and side in wrist_calibration.sides:
        return T_BASE_MOCAP @ reconstruct_wrist_pose_from_mapping(
            frame,
            side,
            wrist_calibration.sides[side],
        )
    skeleton = frame.get("skeleton", {}).get(side, {})
    wrist_position = skeleton.get("wrist", {}).get("pos")
    hand_quaternion = frame.get(f"{side}_hand_quat", frame.get(f"{side}_quat"))
    if wrist_position is None or hand_quaternion is None:
        raise ValueError(f"回放帧缺少 {side} wrist/hand 观测")
    return T_BASE_MOCAP @ build_transform_matrix(
        np.asarray(wrist_position, dtype=float),
        np.asarray(hand_quaternion, dtype=float),
    )


def replay_mocap_reference_position(
    frame: dict[str, Any],
    workspace_calibration,
) -> np.ndarray | None:
    if workspace_calibration is None or workspace_calibration.position_reference not in {
        "hand_minus_chest",
        "wrist_minus_chest",
    }:
        return None
    mocap_frame = frame.get("mocap_frame")
    if mocap_frame is not None:
        chest = mocap_frame.joint("chest")
        if not chest.tracked:
            raise ValueError("回放帧胸口参考点未跟踪")
        chest_position = np.asarray(chest.position, dtype=float)
    else:
        chest_row = frame.get("skeleton", {}).get("torso", {}).get("chest", {})
        if not chest_row.get("tracked", False):
            raise ValueError("回放帧缺少已跟踪的胸口参考点")
        chest_position = np.asarray(chest_row.get("pos"), dtype=float)
    if chest_position.shape != (3,) or not np.all(np.isfinite(chest_position)):
        raise ValueError("回放帧胸口参考点无效")
    return np.asarray((T_BASE_MOCAP @ np.append(chest_position, 1.0))[:3], dtype=float)


def select_ik_targets(control_side: str, left_mocap_wrist: np.ndarray, right_mocap_wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Mapper 已把未控制侧固定在当前机器人 FK Home，不能回退到旧静态 Home。
    return left_mocap_wrist, right_mocap_wrist


def replay_wrist_behavior_reasons(
    solver: Any,
    result: Any,
    target_mapping: Any,
    reference_q: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, tuple[str, ...]]:
    candidate_q = np.asarray(result.command.positions, dtype=float)
    actual_wrist_poses = solver.wrist_effector_poses(candidate_q)
    position_errors = wrist_position_errors(
        actual_wrist_poses,
        {
            "left": target_mapping.meta["left_control_target"],
            "right": target_mapping.meta["right_control_target"],
        },
        args.control_side,
    )
    return validate_wrist_behavior_candidate(
        candidate_q=candidate_q,
        reference_q=reference_q,
        joint_limits=solver.joint_limits,
        control_side=args.control_side,
        wrist_position_error_by_side=position_errors,
        limits=WristBehaviorLimits(
            max_wrist_step_rad=args.max_joint_delta_per_step,
            max_elbow_step_rad=args.max_joint_delta_per_step,
        ),
    )


def official_ik_controls_waist(args: argparse.Namespace) -> bool:
    return args.control_mode == "ik" and args.ik_backend in {"tiangong2pro_official", "ubtech_casadi"}


def controlled_indices(control_side: str) -> list[int]:
    if control_side == "left":
        return list(range(7))
    if control_side == "right":
        return list(range(7, 14))
    return list(range(14))


def joint_delta_report(
    q: np.ndarray,
    reference_q: np.ndarray,
    indices: list[int],
    limit: int = 5,
) -> list[dict[str, Any]]:
    q_safe = np.asarray(q, dtype=float).reshape(-1)
    ref_safe = np.asarray(reference_q, dtype=float).reshape(-1)
    rows = []
    for joint_index in indices:
        delta = float(q_safe[joint_index] - ref_safe[joint_index])
        rows.append(
            {
                "joint_index": joint_index,
                "joint_name": IK_ARM_JOINT_NAMES[joint_index],
                "motor_id": ARM_MOTOR_IDS[joint_index],
                "delta": delta,
                "abs_delta": abs(delta),
                "baseline": float(ref_safe[joint_index]),
                "command": float(q_safe[joint_index]),
            }
        )
    return sorted(rows, key=lambda item: item["abs_delta"], reverse=True)[:limit]


def max_joint_delta(q: np.ndarray, reference_q: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    q_safe = np.asarray(q, dtype=float).reshape(-1)
    ref_safe = np.asarray(reference_q, dtype=float).reshape(-1)
    return float(np.max(np.abs(q_safe[indices] - ref_safe[indices])))


def make_motor_templates(control_side: str, spd: float, cur: float) -> list[dict[str, Any]]:
    if control_side == "left":
        templates = copy.deepcopy(ARM_MOTOR_TEMPLATES[:7])
    elif control_side == "right":
        templates = copy.deepcopy(ARM_MOTOR_TEMPLATES[7:])
    else:
        templates = copy.deepcopy(ARM_MOTOR_TEMPLATES)

    for motor in templates:
        motor["spd"] = float(spd)
        motor["cur"] = float(cur)
    return templates


def first_present(item: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        if key in item:
            return item[key]
    return None


def normalize_status(msg: dict[str, Any]) -> dict[int, float]:
    positions = arm_positions_from_status(msg.get("status"))
    return dict(zip(ARM_MOTOR_IDS, positions))


class ArmStatusBuffer:
    def __init__(self):
        self.sample: tuple[float, np.ndarray] | None = None
        self.lock = threading.Lock()

    def callback(self, msg):
        try:
            positions = arm_positions_from_status(msg.get("status"))
            sample = (time.monotonic(), np.asarray(positions, dtype=float))
        except (AttributeError, TypeError, ValueError):
            sample = None
        with self.lock:
            self.sample = sample

    def latest_complete(self) -> tuple[float, np.ndarray] | None:
        with self.lock:
            if self.sample is None:
                return None
            return self.sample[0], self.sample[1].copy()

    def wait_complete(self, timeout: float) -> tuple[float, np.ndarray] | None:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            sample = self.latest_complete()
            if sample is not None and feedback_is_fresh(sample[0], time.monotonic(), MAX_ARM_STATUS_AGE_SECONDS):
                return sample
            time.sleep(0.05)
        return None


class WaistStatusBuffer:
    def __init__(self):
        self.sample: tuple[float, float] | None = None
        self.error: int | None = None
        self.lock = threading.Lock()

    def callback(self, msg):
        try:
            raw_items = msg.get("status", [])
            if not isinstance(raw_items, list):
                raise ValueError("腰部状态必须是数组")
            matches = []
            for entry in raw_items:
                if not isinstance(entry, dict):
                    raise ValueError("腰部状态项无效")
                raw_id = first_present(entry, ("name", "id", "motor_id", "motorId"))
                motor_id = int(raw_id)
                if isinstance(raw_id, bool) or (isinstance(raw_id, float) and raw_id != motor_id):
                    raise ValueError("腰部电机 ID 无效")
                if motor_id == WAIST_MOTOR_ID:
                    matches.append(entry)
            if len(matches) != 1:
                raise ValueError("腰部状态缺少或重复电机 ID")
            item = matches[0]
            raw_error = float(item.get("error", 0))
            error = int(raw_error)
            raw_position = first_present(item, ("pos", "position", "angle", "q"))
            position = float(raw_position)
            if isinstance(raw_position, bool) or not math.isfinite(raw_error) or raw_error != 0 or not np.isfinite(position):
                raise ValueError("腰部位置无效或存在电机告警")
        except (AttributeError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.sample = None
                self.error = None
            return
        with self.lock:
            self.error = error
            self.sample = (time.monotonic(), position)

    def latest_valid(self) -> tuple[float, float] | None:
        with self.lock:
            return self.sample

    def wait_valid(self, timeout: float) -> tuple[float, float] | None:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            sample = self.latest_valid()
            if sample is not None and feedback_is_fresh(sample[0], time.monotonic(), MAX_WAIST_STATUS_AGE_SECONDS):
                return sample
            time.sleep(0.05)
        return None

    def error_description(self) -> str | None:
        with self.lock:
            if self.error in (None, 0):
                return None
            return f"error={self.error} ({WAIST_ERROR_DESCRIPTIONS.get(self.error, '未知错误')})"


class HandStatusBuffer:
    def __init__(self, side: str):
        self.side = side
        self.sample: tuple[float, tuple[float, ...]] | None = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def callback(self, msg: dict[str, Any]) -> None:
        try:
            if any(isinstance(value, bool) for value in msg.get("position", ())):
                raise HandControlError("手部反馈位置不能是布尔值")
            positions = parse_hand_state_message(msg)
        except (AttributeError, TypeError, ValueError) as exc:
            with self.lock:
                self.sample = None
                self.error = str(exc)
            return
        with self.lock:
            self.sample = (time.monotonic(), positions)
            self.error = None

    def latest_valid(self) -> tuple[float, tuple[float, ...]] | None:
        with self.lock:
            return self.sample

    def wait_valid(self, timeout: float) -> tuple[float, tuple[float, ...]] | None:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            sample = self.latest_valid()
            if sample is not None:
                return sample
            time.sleep(0.05)
        return None

    def error_description(self) -> str | None:
        with self.lock:
            return self.error


def active_hand_sides(control_side: str) -> tuple[str, ...]:
    return ("left", "right") if control_side == "both" else (control_side,)


def connect_rosbridge(host: str, port: int, timeout: float) -> roslibpy.Ros:
    client = roslibpy.Ros(host=host, port=port)
    client.run()
    start = time.monotonic()
    while not client.is_connected:
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"连接 rosbridge 超时: ws://{host}:{port}")
        time.sleep(0.05)
    return client


def make_ik_solver(args: argparse.Namespace, home_q: np.ndarray | None = None):
    from teleop_ik_types import TeleopIKConfig

    joint_limit_count = 15 if args.ik_backend in {"tiangong2pro_official", "ubtech_casadi"} else 14
    ik_config = TeleopIKConfig(
        joint_names=IK_ARM_JOINT_NAMES,
        motor_ids=ARM_MOTOR_IDS,
        joint_limits=build_relaxed_joint_limits(joint_limit_count),
        control_dt=1.0 / args.rate,
        task_mode=args.ik_task,
        max_iterations=8,
        damping=0.03,
        max_joint_delta_per_step=replay_step_cap(args),
        locked_joint_indices=POSITION_ONLY_LOCKED_JOINT_INDICES if args.ik_task == "position_only" else (),
        home_q=None if home_q is None else np.asarray(home_q, dtype=float),
        enable_local_log=False,
        log_to_console=False,
        log_ik_result=False,
    )
    if args.ik_backend == "teleop_dls":
        from teleop_ik_core import TeleopArmIK

        return TeleopArmIK(ik_config)
    if args.ik_backend in {"tiangong2pro_official", "ubtech_casadi"}:
        from teleop_ik_ubtech_casadi import OfficialTiangong2ProIK

        return OfficialTiangong2ProIK(ik_config)
    raise ValueError(f"Unsupported ik_backend: {args.ik_backend}")


def apply_waist_replay_safety(
    target_position: float | None,
    reference_position: float,
    baseline_position: float,
    waist_limit: tuple[float, float],
    max_step: float,
    max_from_baseline: float,
) -> float | None:
    if target_position is None:
        return None
    values = np.asarray([target_position, reference_position, baseline_position], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("腰部目标、反馈或 baseline 包含 NaN/Inf")
    target = float(np.clip(target_position, *waist_limit))
    target = float(np.clip(target, baseline_position - max_from_baseline, baseline_position + max_from_baseline))
    return float(np.clip(target, reference_position - max_step, reference_position + max_step))


def plan_ik_replay_commands(
    frames: list[dict[str, Any]],
    args: argparse.Namespace,
    baseline_q: np.ndarray,
    baseline_waist: float = 0.0,
) -> tuple[list[IKCommand], dict[str, Any]]:
    from teleop_ik_core import apply_joint_safety
    from teleop_ik_types import TeleopIKInput

    solver = make_ik_solver(args, baseline_q)
    controls_waist = getattr(solver, "controls_waist", False)
    if controls_waist:
        solver.last_waist_position = float(baseline_waist)
    if not hasattr(solver, "wrist_effector_poses") or not hasattr(solver, "wrist_to_tcp_transforms"):
        raise RuntimeError("当前 IK 后端未提供 wrist-roll 控制点 FK，无法建立机器人手腕 Home")
    left_robot_home, right_robot_home = solver.wrist_effector_poses(baseline_q)
    left_wrist_to_tcp, right_wrist_to_tcp = solver.wrist_to_tcp_transforms()
    workspace_calibration = load_workspace_calibration(
        args.workspace_calibration, args.control_side, optional=True
    )
    wrist_calibration = load_wrist_calibration(
        args.wrist_calibration, args.control_side, optional=args.wrist_calibration is None
    )
    mapper = RelativeMocapTargetMapper(
        left_robot_home=left_robot_home,
        right_robot_home=right_robot_home,
        mode="relative",
        position_scale=args.position_scale,
        orientation_scale=args.orientation_scale,
        workspace_calibration=workspace_calibration,
        control_side=args.control_side,
        left_control_to_target=left_wrist_to_tcp,
        right_control_to_target=right_wrist_to_tcp,
        control_frame="wrist_roll",
    )
    last_command_q: np.ndarray | None = None
    commands: list[IKCommand] = []
    error_norms: list[float] = []
    limited_count = 0
    max_step = 0.0
    max_from_baseline = 0.0
    max_waist_step = 0.0
    max_waist_from_baseline = 0.0
    last_waist_command = float(baseline_waist)
    worst_joint_deltas: list[dict[str, Any]] = []
    indices = controlled_indices(args.control_side)

    for frame_index, frame in enumerate(frames):
        current_q = baseline_q if last_command_q is None else last_command_q
        left_mocap_wrist = replay_anatomical_wrist_pose(frame, "left", wrist_calibration)
        right_mocap_wrist = replay_anatomical_wrist_pose(frame, "right", wrist_calibration)
        mocap_reference_position = replay_mocap_reference_position(frame, workspace_calibration)
        if frame_index == 0:
            mapper.set_runtime_mocap_home(
                left_mocap_wrist,
                right_mocap_wrist,
                mocap_reference_position=mocap_reference_position,
            )
        target_mapping = mapper.map_targets(
            left_mocap_wrist,
            right_mocap_wrist,
            mocap_reference_position=mocap_reference_position,
        )
        left_target, right_target = select_ik_targets(args.control_side, target_mapping.left_target, target_mapping.right_target)

        result = solver.solve(
            TeleopIKInput(
                timestamp=float(frame["receive_time"]),
                control_side=args.control_side,
                target_pose_l=left_target,
                target_pose_r=right_target,
                current_q=current_q,
                last_q_cmd=last_command_q,
                current_waist_position=last_waist_command if controls_waist else None,
            )
        )

        reference_q = current_q if last_command_q is None else last_command_q
        if result.reason != "ok":
            raise RuntimeError(f"回放第 {frame_index + 1} 帧 IK 失败: {result.reason}")
        behavior_reasons = replay_wrist_behavior_reasons(
            solver,
            result,
            target_mapping,
            reference_q,
            args,
        )
        if behavior_reasons:
            raise RuntimeError(f"回放第 {frame_index + 1} 帧腕部行为门禁拒绝候选: {behavior_reasons}")
        safe_q, limited = apply_joint_safety(
            q_next=np.asarray(result.command.positions, dtype=float),
            q_ref=reference_q,
            limits=solver.joint_limits,
            max_step=replay_step_cap(args),
            smoothing_alpha=1.0,
        )
        safe_waist = None
        if controls_waist:
            safe_waist = apply_waist_replay_safety(
                result.command.waist_position,
                last_waist_command,
                baseline_waist,
                solver.waist_limit,
                replay_step_cap(args),
                args.max_command_delta_from_baseline,
            )
        command = replace(result.command, positions=safe_q, waist_position=safe_waist)
        commands.append(command)

        if safe_waist is not None:
            max_waist_step = max(max_waist_step, abs(safe_waist - last_waist_command))
            max_waist_from_baseline = max(max_waist_from_baseline, abs(safe_waist - baseline_waist))
            last_waist_command = safe_waist

        if last_command_q is not None:
            max_step = max(max_step, float(np.max(np.abs(safe_q[indices] - last_command_q[indices]))))
        current_from_baseline = max_joint_delta(safe_q, baseline_q, indices)
        if current_from_baseline > max_from_baseline:
            max_from_baseline = current_from_baseline
            worst_joint_deltas = joint_delta_report(safe_q, baseline_q, indices, limit=5)
        limited_count += int(limited or result.limited)
        error_norms.append(float(result.error_norm))
        last_command_q = safe_q.copy()

    summary = {
        "command_count": len(commands),
        "control_side": args.control_side,
        "control_mode": "ik",
        "ik_backend": args.ik_backend,
        "ik_task": args.ik_task,
        "max_joint_step": max_step,
        "max_command_delta_from_baseline": max_from_baseline,
        "waist_motor_id": WAIST_MOTOR_ID if controls_waist else None,
        "baseline_waist_position": baseline_waist if controls_waist else None,
        "max_waist_step": max_waist_step,
        "max_waist_delta_from_baseline": max_waist_from_baseline,
        "worst_joint_deltas": worst_joint_deltas,
        "limited_count": limited_count,
        "max_error_norm": max(error_norms) if error_norms else 0.0,
        "mean_error_norm": float(np.mean(error_norms)) if error_norms else 0.0,
    }
    return commands, summary


def classify_skeleton_stage(frame: dict[str, Any], config: Any) -> str:
    observation, _ = skeleton_observation(frame, config)
    if observation is None:
        return "invalid"
    left = observation.arms["left"].upper_direction_local
    right = observation.arms["right"].upper_direction_local
    if left[2] < -0.70 and right[2] < -0.70:
        return "stand"
    if left[1] > 0.55 and right[1] < -0.55:
        return "side_raise"
    if left[0] > 0.55 and right[0] > 0.55:
        return "front_raise"
    return "transition"


def plan_direct_replay_commands(
    frames: list[dict[str, Any]],
    args: argparse.Namespace,
    baseline_q: np.ndarray,
) -> tuple[list[IKCommand], dict[str, Any]]:
    effective_args = copy.copy(args)
    effective_args.max_joint_delta_per_step = replay_step_cap(args)
    direct_config = direct_retarget_config_from_args(
        effective_args,
        joint_names=IK_ARM_JOINT_NAMES,
        motor_ids=ARM_MOTOR_IDS,
    )
    retargeter = DirectRetargeter(direct_config)
    commands: list[IKCommand] = []
    indices = controlled_indices(args.control_side)
    max_step = 0.0
    max_from_baseline = 0.0
    worst_joint_deltas: list[dict[str, Any]] = []
    limited_count = 0
    source_counts: dict[str, int] = {}
    upper_residuals: list[float] = []
    forearm_residuals: list[float] = []
    stage_positions: dict[str, list[np.ndarray]] = {}
    stage_counts: dict[str, int] = {}
    first_calibrated_frame: int | None = None
    first_calibrated_time: float | None = None
    urdf_boundary_frame_count = 0
    last_q: np.ndarray | None = None

    for frame_index, frame in enumerate(frames):
        command = retargeter.command_from_frame(frame, baseline_q)
        commands.append(command)
        stage = classify_skeleton_stage(frame, direct_config)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        stage_positions.setdefault(stage, []).append(command.positions.copy())
        if bool(command.meta.get("calibrated", False)) and first_calibrated_frame is None:
            first_calibrated_frame = frame_index
            first_calibrated_time = float(frame["receive_time"])
        at_urdf_boundary = any(
            abs(float(command.positions[index]) - float(retargeter.solver.joint_limits[index][bound])) <= 1e-6
            for index in indices
            for bound in (0, 1)
        )
        urdf_boundary_frame_count += int(at_urdf_boundary)
        if last_q is not None:
            max_step = max(max_step, float(np.max(np.abs(command.positions[indices] - last_q[indices]))))
        current_from_baseline = max_joint_delta(command.positions, baseline_q, indices)
        if current_from_baseline > max_from_baseline:
            max_from_baseline = current_from_baseline
            worst_joint_deltas = joint_delta_report(command.positions, baseline_q, indices, limit=5)
        limited_count += int(bool(command.meta.get("limited", False)))
        for source in command.meta.get("source_by_side", {}).values():
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
        solver_meta = command.meta.get("solver_meta", {})
        if isinstance(solver_meta, dict):
            residuals = solver_meta.get("residual_by_side", {})
            if isinstance(residuals, dict):
                for residual in residuals.values():
                    if isinstance(residual, dict):
                        upper_residuals.append(float(residual.get("upper_angle_rad", 0.0)))
                        forearm_residuals.append(float(residual.get("forearm_angle_rad", 0.0)))
        last_q = command.positions.copy()

    stage_joint_medians = {
        stage: {
            name: float(value)
            for name, value in zip(IK_ARM_JOINT_NAMES, np.median(np.asarray(positions), axis=0))
        }
        for stage, positions in stage_positions.items()
        if positions
    }
    calibration_seconds = None
    if first_calibrated_time is not None:
        calibration_seconds = first_calibrated_time - float(frames[0]["receive_time"])
    final_return_error = 0.0 if not commands else max_joint_delta(commands[-1].positions, baseline_q, indices)

    summary = {
        "command_count": len(commands),
        "control_side": args.control_side,
        "control_mode": "direct",
        "max_joint_step": max_step,
        "max_command_delta_from_baseline": max_from_baseline,
        "worst_joint_deltas": worst_joint_deltas,
        "limited_count": limited_count,
        "urdf_boundary_frame_count": urdf_boundary_frame_count,
        "first_calibrated_frame": first_calibrated_frame,
        "calibration_seconds": calibration_seconds,
        "stage_counts": stage_counts,
        "stage_joint_medians_rad": stage_joint_medians,
        "final_return_error_rad": final_return_error,
        "source_counts": source_counts,
        "max_upper_direction_residual_rad": max(upper_residuals) if upper_residuals else 0.0,
        "mean_upper_direction_residual_rad": float(np.mean(upper_residuals)) if upper_residuals else 0.0,
        "max_forearm_direction_residual_rad": max(forearm_residuals) if forearm_residuals else 0.0,
        "mean_forearm_direction_residual_rad": float(np.mean(forearm_residuals)) if forearm_residuals else 0.0,
    }
    return commands, summary


def plan_raise_template_replay_commands(
    frames: list[dict[str, Any]],
    args: argparse.Namespace,
    baseline_q: np.ndarray,
) -> tuple[list[IKCommand], dict[str, Any]]:
    from teleop_raise_template import RaiseTemplateConfig, RaiseTemplateRetargeter

    retargeter = RaiseTemplateRetargeter(
        RaiseTemplateConfig(
            joint_names=IK_ARM_JOINT_NAMES,
            motor_ids=ARM_MOTOR_IDS,
            control_side=args.control_side,
            max_joint_delta_per_step=replay_step_cap(args),
            smoothing_alpha=1.0,
        )
    )
    commands: list[IKCommand] = []
    indices = controlled_indices(args.control_side)
    max_step = 0.0
    max_from_baseline = 0.0
    worst_joint_deltas: list[dict[str, Any]] = []
    limited_count = 0
    pose_counts: dict[str, int] = {}
    last_q: np.ndarray | None = None

    for frame in frames:
        command = retargeter.command_from_frame(frame, baseline_q)
        commands.append(command)
        if last_q is not None:
            max_step = max(max_step, float(np.max(np.abs(command.positions[indices] - last_q[indices]))))
        current_from_baseline = max_joint_delta(command.positions, baseline_q, indices)
        if current_from_baseline > max_from_baseline:
            max_from_baseline = current_from_baseline
            worst_joint_deltas = joint_delta_report(command.positions, baseline_q, indices, limit=5)
        limited_count += int(bool(command.meta.get("limited", False)))
        pose = str(command.meta.get("pose", "unknown"))
        pose_counts[pose] = pose_counts.get(pose, 0) + 1
        last_q = command.positions.copy()

    summary = {
        "command_count": len(commands),
        "control_side": args.control_side,
        "control_mode": "raise_template",
        "max_joint_step": max_step,
        "max_command_delta_from_baseline": max_from_baseline,
        "worst_joint_deltas": worst_joint_deltas,
        "limited_count": limited_count,
        "pose_counts": pose_counts,
    }
    return commands, summary


def replay_step_cap(args: argparse.Namespace) -> float:
    velocity = getattr(args, "max_joint_velocity", 0.48)
    if not math.isfinite(velocity) or velocity <= 0 or not math.isfinite(args.rate) or args.rate <= 0:
        raise ValueError("rate 和 max_joint_velocity 必须为有限正数")
    return min(args.max_joint_delta_per_step, velocity / args.rate)


def validate_skeleton_replay_options(args: argparse.Namespace) -> None:
    if args.workspace_calibration is not None or args.wrist_calibration is not None:
        raise ValueError("skeleton 自动对齐模式不接受 workspace/wrist 人工标定文件；请移除这些参数或显式选择高级 ik 模式")
    if args.ik_backend != "teleop_dls" or args.ik_task != "pose":
        raise ValueError("skeleton 模式使用 walker teleop_dls pose 求解；其他后端或任务请显式选择高级 ik 模式")


def make_skeleton_retargeter(args: argparse.Namespace):
    validate_skeleton_replay_options(args)
    from teleop_skeleton_retarget import SkeletonRetargeter

    effective_args = copy.copy(args)
    effective_args.max_joint_delta_per_step = replay_step_cap(args)
    config = direct_retarget_config_from_args(effective_args, joint_names=IK_ARM_JOINT_NAMES, motor_ids=ARM_MOTOR_IDS)
    return SkeletonRetargeter(config, orientation_scale=args.orientation_scale)


def make_skeleton_startup_gate(args: argparse.Namespace):
    from teleop_startup_pose import NaturalDownStartupConfig, NaturalDownStartupGate

    return NaturalDownStartupGate(NaturalDownStartupConfig(
        hold_seconds=getattr(args, "startup_hold_seconds", 2.0),
        human_vertical_cosine=getattr(args, "startup_human_vertical_cosine", 0.78),
        human_min_wrist_drop_m=getattr(args, "startup_human_min_wrist_drop", 0.30),
        human_max_lateral_offset_m=getattr(args, "startup_human_max_lateral_offset", 0.16),
        max_home_rms_m=getattr(args, "startup_max_home_rms", 0.025),
    ))


def observe_skeleton_startup(gate, now: float, frame: dict[str, Any], current_q: np.ndarray):
    positions = []
    for side in ("left", "right"):
        try:
            positions.append(np.asarray(frame["skeleton"][side]["wrist"]["pos"], dtype=float))
        except (KeyError, TypeError, ValueError):
            positions.append(np.full(3, np.nan))
    return gate.observe(
        now=now, frame=frame, current_q=current_q,
        left_home_position=positions[0], right_home_position=positions[1],
    )


def plan_skeleton_replay_commands(
    frames: list[dict[str, Any]], args: argparse.Namespace, baseline_q: np.ndarray,
) -> tuple[list[IKCommand], dict[str, Any]]:
    """Preview one automatic alignment window; simulate following only for preview."""
    retargeter = make_skeleton_retargeter(args)
    gate = make_skeleton_startup_gate(args)
    current_q = ReplayFeedbackGate.arm_vector(baseline_q)
    startup_baseline = current_q.copy()
    aligned = False
    commands: list[IKCommand] = []
    max_step = 0.0
    max_from_baseline = 0.0
    last_command = None
    startup_frames = 0
    for index, frame in enumerate(frames):
        timestamp = float(frame["receive_time"])
        if not math.isfinite(timestamp):
            raise ValueError("骨架回放时间必须为有限值")
        if not aligned:
            status = observe_skeleton_startup(gate, timestamp, frame, current_q)
            if not status.ready:
                continue
            startup_baseline = ReplayFeedbackGate.arm_vector(status.robot_baseline_q)
            retargeter.align(status.window_frames, startup_baseline)
            aligned = True
            startup_frames = index + 1
        command = retargeter.command_from_frame(frame, current_q, last_q_cmd=last_command, timestamp=timestamp)
        if command.waist_position is not None:
            raise RuntimeError("skeleton 回放不得产生腰部命令")
        commands.append(command)
        indices = controlled_indices(args.control_side)
        max_step = max(max_step, max_joint_delta(command.positions, current_q, indices))
        max_from_baseline = max(max_from_baseline, max_joint_delta(command.positions, startup_baseline, indices))
        current_q = ReplayFeedbackGate.arm_vector(command.positions)
        last_command = current_q.copy()
    if not aligned:
        raise ValueError("骨架回放未取得稳定自然下垂与机器人静止窗口，未生成运动命令")
    return commands, {
        "control_mode": "skeleton", "control_side": args.control_side,
        "command_count": len(commands), "startup_frame_count": startup_frames,
        "max_joint_step": max_step, "max_command_delta_from_baseline": max_from_baseline,
        "baseline_source": "automatic_startup_window", "feedback_source": "simulated_preview",
    }


def plan_replay_commands(
    frames: list[dict[str, Any]],
    args: argparse.Namespace,
    baseline_q: np.ndarray,
    baseline_waist: float = 0.0,
) -> tuple[list[IKCommand], dict[str, Any]]:
    if args.control_mode == "skeleton":
        return plan_skeleton_replay_commands(frames, args, baseline_q)
    if args.control_mode == "direct":
        return plan_direct_replay_commands(frames, args, baseline_q)
    if args.control_mode == "raise_template":
        return plan_raise_template_replay_commands(frames, args, baseline_q)
    commands, summary = plan_ik_replay_commands(frames, args, baseline_q, baseline_waist)
    summary["control_mode"] = "ik"
    summary["ik_backend"] = args.ik_backend
    return commands, summary


def plan_hand_ik_outputs(
    frames: list[dict[str, Any]],
    args: argparse.Namespace,
    config: HandControlConfig,
) -> tuple[dict[str, Any], list[dict[str, tuple[float, ...]] | None]]:
    if not hand_ik_enabled(args):
        return {"enabled": False, "published": False}, [None] * len(frames)

    solved_count = 0
    failed_count = 0
    both_calibrated_count = 0
    first_both_calibrated_sequence: int | None = None
    max_joint_step = 0.0
    max_left_solve_ms = 0.0
    max_right_solve_ms = 0.0
    last_positions: np.ndarray | None = None
    last_result = None
    latest_mapped: dict[str, tuple[float, ...]] | None = None
    outputs: list[dict[str, tuple[float, ...]] | None] = []
    errors: list[str] = []
    try:
        with Revo2HandIkSidecar(hand_ik_config_from_args(args)) as sidecar:
            for frame in frames:
                try:
                    result = sidecar.solve_replay_frame(frame)
                except HandIkError as exc:
                    outputs.append(None)
                    failed_count += 1
                    if len(errors) < 3:
                        errors.append(str(exc))
                    continue
                mapped: dict[str, tuple[float, ...]] = {}
                if result.left_calibrated:
                    mapped["left"] = map_hand_radians(result.joint_positions, "left", config)
                if result.right_calibrated:
                    mapped["right"] = map_hand_radians(result.joint_positions, "right", config)
                outputs.append(mapped or None)
                latest_mapped = mapped or latest_mapped
                positions = np.asarray(result.joint_positions, dtype=float)
                if last_positions is not None:
                    max_joint_step = max(max_joint_step, float(np.max(np.abs(positions - last_positions))))
                last_positions = positions
                last_result = result
                solved_count += 1
                max_left_solve_ms = max(max_left_solve_ms, result.left_solve_time_ms)
                max_right_solve_ms = max(max_right_solve_ms, result.right_solve_time_ms)
                if result.left_calibrated and result.right_calibrated:
                    both_calibrated_count += 1
                    if first_both_calibrated_sequence is None:
                        first_both_calibrated_sequence = result.sequence
    except Exception as exc:
        errors.append(f"sidecar:{type(exc).__name__}: {exc}")
        outputs.extend([None] * (len(frames) - len(outputs)))

    summary = {
        "enabled": True,
        "transport": "local_subprocess_pipe",
        "published": False,
        "input_count": len(frames),
        "solved_count": solved_count,
        "failed_count": failed_count,
        "both_calibrated_count": both_calibrated_count,
        "first_both_calibrated_sequence": first_both_calibrated_sequence,
        "max_joint_step_rad": max_joint_step,
        "max_left_solve_time_ms": max_left_solve_ms,
        "max_right_solve_time_ms": max_right_solve_ms,
        "latest": None if last_result is None else last_result.to_summary(),
        "latest_mapped_open_ratio": latest_mapped,
        "errors": errors,
        "diagnostic_csv": None if not args.hand_ik_diagnostic_csv else str(args.hand_ik_diagnostic_csv),
    }
    return summary, outputs


def command_risk_reasons(summary: dict[str, Any], max_command_delta_from_baseline: float) -> list[str]:
    if summary.get("control_mode") == "raise_template":
        return []
    reasons = []
    if summary["max_command_delta_from_baseline"] > max_command_delta_from_baseline:
        reasons.append(
            "max_command_delta_from_baseline="
            f"{summary['max_command_delta_from_baseline']:.6f}>{max_command_delta_from_baseline:.6f}"
        )
    waist_delta = float(summary.get("max_waist_delta_from_baseline", 0.0))
    if waist_delta > max_command_delta_from_baseline:
        reasons.append(
            f"max_waist_delta_from_baseline={waist_delta:.6f}>{max_command_delta_from_baseline:.6f}"
        )
    return reasons


def confirm_execution(args: argparse.Namespace, command_summary: dict[str, Any]) -> None:
    if not args.execute or args.yes:
        return
    if command_summary.get("control_mode") == "hand-only":
        print("即将只回放双手命令；手臂和腰部命令发布数量固定为 0。")
    else:
        print("即将把数采动作回放到机器人 /arm/cmd_pos；官方 IK 同时发布 /waist/cmd_pos。")
    print(f"frames={command_summary['command_count']} side={args.control_side} rate={args.rate}Hz")
    print("请确认人员在机器人旁、急停可用、手臂周围无干涉，并且已先检查 dry-run 输出。")
    answer = input("输入 EXECUTE_MOCAP_REPLAY 继续: ").strip()
    if answer != "EXECUTE_MOCAP_REPLAY":
        raise SystemExit("用户未确认，取消发布")


class ReplayFeedbackGate:
    """Validate measured feedback again at every command publication boundary."""

    def __init__(
        self,
        args: argparse.Namespace,
        client: roslibpy.Ros,
        arm_status: ArmStatusBuffer | None,
        waist_status: WaistStatusBuffer | None,
        hand_status: dict[str, HandStatusBuffer],
        hand_config: HandControlConfig,
        baseline_q: np.ndarray | None = None,
        baseline_waist: float = 0.0,
    ):
        self.args = args
        self.client = client
        self.arm_status = arm_status
        self.waist_status = waist_status
        self.hand_status = hand_status
        self.hand_config = hand_config
        self.hand_worker = None
        self.baseline_q = None if baseline_q is None else self.arm_vector(baseline_q)
        self.baseline_waist = float(baseline_waist)
        self.last_arm_command: np.ndarray | None = None
        self.last_waist_command: float | None = None
        self.last_arm_publish_time: float | None = None
        self.last_waist_publish_time: float | None = None
        self.last_hand_publish_time: dict[str, float] = {}
        self.joint_velocity_limits = np.full(14, np.inf)
        self.waist_velocity_limit = float("inf")
        self.joint_limits: np.ndarray | None = None
        self.waist_limit: tuple[float, float] | None = None
        for name in ("rate", "max_joint_delta_per_step", "max_command_delta_from_baseline"):
            value = float(getattr(args, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        for name in ("spd", "cur"):
            value = float(getattr(args, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限非负数")
        for name, default in (("max_joint_velocity", 0.48), ("hand_control_rate", 20.0), ("max_hand_velocity", 1.0)):
            value = float(getattr(args, name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not math.isfinite(self.baseline_waist):
            raise ValueError("腰部 baseline 必须为有限数值")
        if arm_status is not None:
            source = Path(__file__).resolve().parent
            if official_ik_controls_waist(args):
                model_path = source.parents[1] / "third_party" / "tiangong2pro_ik" / "tiangong2pro_ik" / "tienkung2_pro" / "urdf" / "tiangong2.0_pro.urdf"
                joint = ET.parse(model_path).getroot().find(f"joint[@name='{WAIST_JOINT_NAME}']/limit")
                if joint is None:
                    raise ValueError("官方 URDF 缺少腰部机械限位")
                self.waist_limit = (float(joint.attrib["lower"]), float(joint.attrib["upper"]))
                self.waist_velocity_limit = joint_velocity_limits_from_urdf(model_path, (WAIST_JOINT_NAME,))[0]
                if not all(math.isfinite(value) for value in self.waist_limit) or self.waist_limit[0] >= self.waist_limit[1]:
                    raise ValueError("官方 URDF 腰部机械限位无效")
            else:
                model_path = source / "models" / "walker_tienkung_ei_description" / "walker_tienkung_ei.urdf"
            self.joint_limits = np.asarray(arm_joint_limits_from_urdf(model_path), dtype=float)
            self.joint_velocity_limits = np.asarray(joint_velocity_limits_from_urdf(model_path, IK_ARM_JOINT_NAMES))

    @staticmethod
    def arm_vector(positions: Any) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise RuntimeError("手臂位置必须包含完整、有限的 14 轴")
        return values.copy()

    def snapshot(self) -> tuple[np.ndarray | None, float | None]:
        if self.hand_worker is not None:
            self.hand_worker.check_health()
        if not self.client.is_connected:
            raise RuntimeError("rosbridge 已断开，停止发送控制命令")
        arm_q = None
        waist_position = None
        received_times = []
        if self.arm_status is not None:
            sample = self.arm_status.latest_complete()
            require_fresh_feedback(None if sample is None else sample[0], time.monotonic(), MAX_ARM_STATUS_AGE_SECONDS, "arm/status")
            arm_q = self.arm_vector(sample[1])
            received_times.append((sample[0], MAX_ARM_STATUS_AGE_SECONDS, "arm/status"))
        if self.waist_limit is not None:
            sample = None if self.waist_status is None else self.waist_status.latest_valid()
            require_fresh_feedback(None if sample is None else sample[0], time.monotonic(), MAX_WAIST_STATUS_AGE_SECONDS, "waist/status")
            waist_position = float(sample[1])
            if not math.isfinite(waist_position):
                raise RuntimeError("腰部反馈包含 NaN/Inf")
            received_times.append((sample[0], MAX_WAIST_STATUS_AGE_SECONDS, "waist/status"))
        if self.hand_config.publish_enabled:
            for side in active_hand_sides(self.args.control_side):
                sample = self.hand_sample(side)
                received_times.append((sample[0], self.hand_config.state_timeout_s, f"{side} hand feedback"))
        now = time.monotonic()
        for received_at, timeout, label in received_times:
            require_fresh_feedback(received_at, now, timeout, label)
        return arm_q, waist_position

    def hand_sample(self, side: str) -> tuple[float, tuple[float, ...]]:
        buffer = self.hand_status.get(side)
        sample = None if buffer is None else buffer.latest_valid()
        require_fresh_feedback(None if sample is None else sample[0], time.monotonic(), self.hand_config.state_timeout_s, f"{side} hand feedback")
        if len(sample[1]) != 6 or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in sample[1]):
            raise RuntimeError(f"{side} 手部反馈无效")
        return sample

    def prepare_arm(self, command: IKCommand, *, enforce_baseline: bool, submitted_arm: bool = False) -> IKCommand:
        if self.arm_status is None or self.joint_limits is None:
            raise RuntimeError("发布手臂命令需要实测状态")
        if tuple(command.motor_ids) != tuple(ARM_MOTOR_IDS) or tuple(command.joint_names) != tuple(IK_ARM_JOINT_NAMES):
            raise RuntimeError("手臂命令的电机 ID 或关节名与协议不匹配")
        if command.control_side != self.args.control_side:
            raise RuntimeError("手臂命令控制侧与回放配置不匹配")
        target = self.arm_vector(command.positions)
        current_q, current_waist = self.snapshot()
        indices = controlled_indices(command.control_side)
        now = time.monotonic()
        dt = command_time_step(self.args.rate, now, None if submitted_arm else self.last_arm_publish_time)
        max_step = np.minimum(self.args.max_joint_delta_per_step, np.minimum(getattr(self.args, "max_joint_velocity", 0.48), self.joint_velocity_limits) * dt)
        measured_step = np.minimum(self.args.max_joint_delta_per_step, np.minimum(getattr(self.args, "max_joint_velocity", 0.48), self.joint_velocity_limits) / self.args.rate)
        lower = np.maximum(self.joint_limits[:, 0], current_q - measured_step)
        upper = np.minimum(self.joint_limits[:, 1], current_q + measured_step)
        if self.last_arm_command is not None:
            lower = np.maximum(lower, self.last_arm_command - max_step)
            upper = np.minimum(upper, self.last_arm_command + max_step)
        if enforce_baseline and self.baseline_q is not None:
            lower = np.maximum(lower, self.baseline_q - self.args.max_command_delta_from_baseline)
            upper = np.minimum(upper, self.baseline_q + self.args.max_command_delta_from_baseline)
        if np.any(lower[indices] > upper[indices]):
            raise RuntimeError("实测手臂状态与命令安全范围不相容")
        safe_q = current_q.copy()
        safe_q[indices] = np.clip(target[indices], lower[indices], upper[indices])
        safe_waist = command.waist_position
        if self.waist_limit is not None:
            if safe_waist is None or not math.isfinite(float(safe_waist)):
                raise RuntimeError("官方 IK 回放需要有限的腰部命令")
            waist_dt = command_time_step(self.args.rate, now, self.last_waist_publish_time)
            waist_step = min(self.args.max_joint_delta_per_step, getattr(self.args, "max_joint_velocity", 0.48) * waist_dt, self.waist_velocity_limit * waist_dt)
            measured_waist_step = min(self.args.max_joint_delta_per_step, min(getattr(self.args, "max_joint_velocity", 0.48), self.waist_velocity_limit) / self.args.rate)
            lower_waist = max(self.waist_limit[0], current_waist - measured_waist_step)
            upper_waist = min(self.waist_limit[1], current_waist + measured_waist_step)
            if self.last_waist_command is not None:
                lower_waist = max(lower_waist, self.last_waist_command - waist_step)
                upper_waist = min(upper_waist, self.last_waist_command + waist_step)
            if enforce_baseline:
                lower_waist = max(lower_waist, self.baseline_waist - self.args.max_command_delta_from_baseline)
                upper_waist = min(upper_waist, self.baseline_waist + self.args.max_command_delta_from_baseline)
            if lower_waist > upper_waist:
                raise RuntimeError("实测腰部状态与命令安全范围不相容")
            safe_waist = float(np.clip(float(safe_waist), lower_waist, upper_waist))
        elif safe_waist is not None:
            raise RuntimeError("当前回放模式未启用腰部反馈，拒绝腰部命令")
        return replace(command, positions=safe_q, waist_position=safe_waist)


def publish_command(
    arm_topic: roslibpy.Topic,
    command: IKCommand,
    templates: list[dict[str, Any]],
    waist_topic: roslibpy.Topic | None = None,
    *,
    safety: ReplayFeedbackGate,
    enforce_baseline: bool = True,
    mocap_received_at: float | None = None,
) -> IKCommand:
    expected_ids = [ARM_MOTOR_IDS[index] for index in controlled_indices(command.control_side)]
    if [template.get("name") for template in templates] != expected_ids:
        raise RuntimeError("发布模板的电机 ID 与控制侧不匹配")
    for template in templates:
        for field in ("spd", "cur"):
            value = float(template[field])
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"电机 {field} 必须为有限非负数")
    command = safety.prepare_arm(command, enforce_baseline=enforce_baseline)
    if command.waist_position is not None:
        if waist_topic is None:
            raise RuntimeError("IKCommand 包含腰部命令，但 /waist/cmd_pos 发布器未初始化")
    if mocap_received_at is not None:
        require_fresh_feedback(mocap_received_at, time.monotonic(), 0.3, "replay mocap input")
    arm_topic.publish(build_rosbridge_motor_message(command, templates))
    safety.last_arm_command = command.positions.copy()
    safety.last_arm_publish_time = time.monotonic()
    if command.waist_position is not None:
        checked = safety.prepare_arm(command, enforce_baseline=enforce_baseline, submitted_arm=True)
        command = replace(command, waist_position=checked.waist_position)
        if mocap_received_at is not None:
            require_fresh_feedback(mocap_received_at, time.monotonic(), 0.3, "replay mocap input")
        waist_topic.publish(build_rosbridge_waist_message(command, WAIST_MOTOR_TEMPLATE))
        safety.last_waist_command = command.waist_position
        safety.last_waist_publish_time = time.monotonic()
    return command


def advertise_hand_command_topics(
    client: roslibpy.Ros,
    args: argparse.Namespace,
    config: HandControlConfig,
) -> dict[str, roslibpy.Topic]:
    if not config.publish_enabled:
        return {}
    topics: dict[str, roslibpy.Topic] = {}
    for side in active_hand_sides(args.control_side):
        topic = roslibpy.Topic(client, HAND_CMD_TOPICS[side], HAND_JOINT_STATE_TYPE)
        topic.advertise()
        topics[side] = topic
    return topics


def validate_hand_output(
    output: dict[str, tuple[float, ...]] | None,
    topics: dict[str, roslibpy.Topic],
) -> None:
    if output is not None:
        for side in topics:
            if side in output:
                build_hand_joint_state_message(output[side])


def publish_hand_output(
    output: dict[str, tuple[float, ...]] | None,
    topics: dict[str, roslibpy.Topic],
    status_buffers: dict[str, HandStatusBuffer],
    last_commands: dict[str, tuple[float, ...] | None],
    config: HandControlConfig,
    *,
    safety: ReplayFeedbackGate,
    mocap_received_at: float | None = None,
) -> None:
    safety.snapshot()
    validate_hand_output(output, topics)
    if output is None:
        return
    for side, topic in topics.items():
        target = output.get(side)
        if target is None:
            continue
        safety.snapshot()
        if mocap_received_at is not None:
            require_fresh_feedback(mocap_received_at, time.monotonic(), 0.3, "hand IK input")
        sample = safety.hand_sample(side)
        dt = command_time_step(getattr(safety.args, "hand_control_rate", 20.0), time.monotonic(), safety.last_hand_publish_time.get(side))
        budget = min(config.max_step, getattr(safety.args, "max_hand_velocity", 1.0) * dt)
        measured_budget = min(config.max_step, getattr(safety.args, "max_hand_velocity", 1.0) / getattr(safety.args, "hand_control_rate", 20.0))
        if budget <= 0:
            continue
        limited_config = replace(config, max_step=budget)
        safe = apply_hand_command_safety(target, sample[1], None, side, limited_config)
        previous = last_commands.get(side)
        if previous is not None:
            safe = tuple(float(value) for value in np.clip(safe, np.asarray(previous) - budget, np.asarray(previous) + budget))
        if any(abs(value - measured) > measured_budget + 1e-9 for value, measured in zip(safe, sample[1])):
            raise RuntimeError(f"{side} 手部命令与最新反馈的变化过大")
        topic.publish(build_hand_joint_state_message(safe))
        last_commands[side] = safe
        safety.last_hand_publish_time[side] = time.monotonic()


def restore_baselines(
    args: argparse.Namespace,
    safety: ReplayFeedbackGate,
    hand_topics: dict[str, roslibpy.Topic],
    hand_baselines: dict[str, tuple[float, ...]],
    last_hand_commands: dict[str, tuple[float, ...] | None],
    arm_topic: roslibpy.Topic | None = None,
    templates: list[dict[str, Any]] | None = None,
    baseline_q: np.ndarray | None = None,
    waist_topic: roslibpy.Topic | None = None,
    baseline_waist: float | None = None,
    duration: float | None = None,
) -> None:
    """Return toward measured baselines with a deadline and a finite step budget."""
    interval = 1.0 / args.rate
    duration = args.restore_seconds if duration is None else duration
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("恢复时间必须为有限非负数")
    duration = max(interval, duration)
    steps = min(1000, max(1, math.ceil(duration / interval)))
    deadline = time.monotonic() + duration
    if baseline_q is not None:
        baseline_q = safety.arm_vector(baseline_q)
        indices = controlled_indices(args.control_side)
        if np.any(baseline_q[indices] < safety.joint_limits[indices, 0]) or np.any(baseline_q[indices] > safety.joint_limits[indices, 1]):
            raise RuntimeError("手臂恢复目标超出机械限位")
    else:
        indices = []
    for side in hand_topics:
        if side not in hand_baselines:
            raise RuntimeError(f"缺少 {side} 手部恢复目标")
        build_hand_joint_state_message(hand_baselines[side])
    if safety.waist_limit is not None and (
        baseline_waist is None or not math.isfinite(baseline_waist) or
        not safety.waist_limit[0] <= baseline_waist <= safety.waist_limit[1]
    ):
        raise RuntimeError("腰部恢复目标超出机械限位")

    for step in range(steps + 1):
        measured_q, measured_waist = safety.snapshot()
        # Feedback can lag at home while an accepted target is still away from
        # home. Replace that outstanding target before reporting restoration.
        arm_done = baseline_q is None or (
            max_joint_delta(baseline_q, measured_q, indices) <= 1e-3 and
            (safety.last_arm_command is None or max_joint_delta(baseline_q, safety.last_arm_command, indices) <= 1e-3)
        )
        waist_done = baseline_waist is None or (
            abs(baseline_waist - measured_waist) <= 1e-3 and
            (safety.last_waist_command is None or abs(baseline_waist - safety.last_waist_command) <= 1e-3)
        )
        hands_done = all(
            max(abs(target - measured) for target, measured in zip(hand_baselines[side], safety.hand_sample(side)[1])) <= 1e-3 and
            (last_hand_commands.get(side) is None or
             max(abs(target - submitted) for target, submitted in zip(hand_baselines[side], last_hand_commands[side])) <= 1e-3)
            for side in hand_topics
        )
        if arm_done and waist_done and hands_done:
            return
        if step == steps or time.monotonic() >= deadline:
            raise RuntimeError("恢复未在有限步数和时限内得到实测确认，停止发送")
        if arm_topic is not None and (not arm_done or not waist_done):
            publish_command(
                arm_topic, restore_command(args, baseline_q, baseline_waist), templates,
                waist_topic, safety=safety, enforce_baseline=False,
            )
        if hand_topics and not hands_done:
            publish_hand_output(
                hand_baselines, hand_topics, safety.hand_status, last_hand_commands,
                safety.hand_config, safety=safety,
            )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))


def restore_command(
    args: argparse.Namespace,
    baseline_q: np.ndarray,
    waist_position: float | None = None,
) -> IKCommand:
    return IKCommand(
        timestamp=time.time(),
        control_side=args.control_side,
        joint_names=IK_ARM_JOINT_NAMES,
        positions=baseline_q.copy(),
        motor_ids=ARM_MOTOR_IDS,
        waist_position=waist_position,
    )


def resolve_replay_baseline(args: argparse.Namespace, status_q: np.ndarray | None) -> tuple[np.ndarray, str]:
    if args.control_mode == "skeleton":
        if status_q is not None:
            return np.asarray(status_q, dtype=float).copy(), "robot_status_before_automatic_alignment"
        return NATURAL_DOWN_Q.copy(), "natural_down_offline_preview"
    if args.prepose == "natural_down":
        return NATURAL_DOWN_Q.copy(), "natural_down"
    if status_q is not None:
        return np.asarray(status_q, dtype=float).copy(), "robot_status"
    return np.zeros(14), "zero_offline_preview"


def prepose_to_baseline(
    args: argparse.Namespace,
    topic: roslibpy.Topic,
    templates: list[dict[str, Any]],
    baseline_q: np.ndarray,
    *,
    safety: ReplayFeedbackGate,
    waist_topic: roslibpy.Topic | None = None,
    baseline_waist: float | None = None,
) -> None:
    if args.prepose == "none":
        return
    restore_baselines(
        args, safety, {}, {}, {}, topic, templates, baseline_q, waist_topic,
        baseline_waist, duration=args.prepose_seconds,
    )
    print_json_line("PREPOSE_BASELINE", {"mode": args.prepose, "seconds": args.prepose_seconds, "positions": baseline_q})


def replay_control_ticks(frames, rate: float, timestamps=None):
    """Keep recorded time while sampling the latest available source at rate Hz.

    Repeated held frames retain their original scheduled monotonic receipt time.
    A delayed callback skips obsolete ticks rather than speeding up the movie.
    """
    if not frames:
        return
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("回放控制频率必须为有限正数")
    if timestamps is None:
        timestamps = [frame.get("receive_time", index / rate) for index, frame in enumerate(frames)]
    times = np.asarray(timestamps, dtype=float)
    if times.shape != (len(frames),) or not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("录制帧时间必须有限并按时间顺序排列")
    offsets = times - times[0]
    duration = float(offsets[-1])
    began = time.monotonic()
    source_index, tick = 0, 0
    while True:
        deadline = began + min(tick / rate, duration)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elapsed = time.monotonic() - began
        while source_index + 1 < len(frames) and offsets[source_index + 1] <= elapsed + 1e-9:
            source_index += 1
        yield source_index, frames[source_index], began + float(offsets[source_index])
        if elapsed + 1e-9 >= duration and source_index == len(frames) - 1:
            # Keep the final sample for one control period before recovery.
            time.sleep(1.0 / rate)
            return
        tick = max(tick + 1, int(math.floor(elapsed * rate + 1e-9)) + 1)


def poll_async_hand_output(worker, frame, received_at: float, config: HandControlConfig):
    if worker is None:
        return None, None
    worker.submit(frame, received_at)
    completed = worker.poll()
    if completed is None or not feedback_is_fresh(completed.received_at, time.monotonic(), 0.3):
        return None, None
    result = completed.result
    output = {}
    if result.left_calibrated:
        output["left"] = map_hand_radians(result.joint_positions, "left", config)
    if result.right_calibrated:
        output["right"] = map_hand_radians(result.joint_positions, "right", config)
    return output, completed.received_at


def execute_skeleton_replay(
    args: argparse.Namespace, client: roslibpy.Ros, status_buffer: ArmStatusBuffer,
    frames: list[dict[str, Any]], baseline_q: np.ndarray,
    hand_status_buffers: dict[str, HandStatusBuffer],
    hand_baselines: dict[str, tuple[float, ...]], hand_control_config: HandControlConfig,
) -> int:
    """Align once without prepose, then solve every frame from measured feedback."""
    validate_skeleton_replay_options(args)
    safety = ReplayFeedbackGate(args, client, status_buffer, None, hand_status_buffers, hand_control_config, baseline_q)
    retargeter = make_skeleton_retargeter(args)
    startup = make_skeleton_startup_gate(args)
    templates = make_motor_templates(args.control_side, args.spd, args.cur)
    cmd_topic = None
    hand_topics: dict[str, roslibpy.Topic] = {}
    last_hands: dict[str, tuple[float, ...] | None] = {}
    sidecar = None
    aligned = False
    try:
        if hand_control_config.publish_enabled and hand_ik_enabled(args):
            sidecar = LatestHandIkWorker(Revo2HandIkSidecar(hand_ik_config_from_args(args)), getattr(args, "hand_control_rate", 20.0))
            safety.hand_worker = sidecar
        for index, frame, source_received_at in replay_control_ticks(frames, args.rate):
            solve_started_at = time.monotonic()
            require_fresh_feedback(source_received_at, solve_started_at, 0.3, "replay mocap input")
            current_q, _ = safety.snapshot()
            hand_output, hand_received_at = poll_async_hand_output(sidecar, frame, source_received_at, hand_control_config)
            current_q, _ = safety.snapshot()
            if not aligned:
                status = observe_skeleton_startup(startup, source_received_at, frame, current_q)
                if status.ready:
                    baseline_q = safety.arm_vector(status.robot_baseline_q)
                    retargeter.align(status.window_frames, baseline_q)
                    safety.baseline_q = baseline_q.copy()
                    current_q, _ = safety.snapshot()
                    cmd_topic = roslibpy.Topic(client, ARM_CMD_TOPIC, ARM_CMD_TYPE)
                    cmd_topic.advertise()
                    hand_topics = advertise_hand_command_topics(client, args, hand_control_config)
                    last_hands = {side: None for side in hand_topics}
                    aligned = True
                    print_json_line("SKELETON_AUTO_ALIGNED", {"frame": index + 1, "window_frames": len(status.window_frames), "baseline": baseline_q})
            if aligned:
                current_q, _ = safety.snapshot()
                command = retargeter.command_from_frame(
                    frame, current_q, last_q_cmd=safety.last_arm_command,
                    timestamp=float(frame["receive_time"]),
                )
                if command.waist_position is not None:
                    raise RuntimeError("skeleton 回放不得产生腰部命令")
                validate_hand_output(hand_output, hand_topics)
                require_fresh_feedback(solve_started_at, time.monotonic(), MAX_ARM_STATUS_AGE_SECONDS, "skeleton IK result")
                publish_command(cmd_topic, command, templates, safety=safety, mocap_received_at=source_received_at)
                publish_hand_output(hand_output, hand_topics, hand_status_buffers, last_hands, hand_control_config, safety=safety, mocap_received_at=hand_received_at)
        if not aligned:
            raise RuntimeError("骨架回放未取得稳定启动窗口，运动命令保持停发")
        restore_baselines(args, safety, hand_topics, hand_baselines, last_hands, cmd_topic, templates, baseline_q)
        print("PASS skeleton replay published and baseline restored")
        return 0
    except KeyboardInterrupt:
        print("ABORT_REPLAY interrupted; further motion commands stopped")
        return 130
    except Exception as exc:
        print(f"ABORT_REPLAY {exc}; further motion commands stopped")
        return 14
    finally:
        for topic in ([cmd_topic] if cmd_topic is not None else []) + list(hand_topics.values()):
            try:
                topic.unadvertise()
            except Exception:
                pass
        if sidecar is not None:
            try:
                sidecar.close()
            except Exception:
                pass


def execute_ik_replay_streaming(
    args: argparse.Namespace,
    client: roslibpy.Ros,
    status_buffer: ArmStatusBuffer,
    waist_status_buffer: WaistStatusBuffer | None,
    frames: list[dict[str, Any]],
    baseline_q: np.ndarray,
    baseline_waist: float,
    hand_status_buffers: dict[str, HandStatusBuffer],
    hand_baselines: dict[str, tuple[float, ...]],
    hand_control_config: HandControlConfig,
) -> int:
    # An explicitly requested bad calibration must fail before prepose can move.
    workspace_calibration = load_workspace_calibration(
        args.workspace_calibration, args.control_side, optional=args.workspace_calibration is None
    )
    wrist_calibration = load_wrist_calibration(
        args.wrist_calibration, args.control_side, optional=args.wrist_calibration is None
    )
    from teleop_ik_core import apply_joint_safety
    from teleop_ik_types import TeleopIKInput

    templates = make_motor_templates(args.control_side, args.spd, args.cur)
    cmd_topic = roslibpy.Topic(client, ARM_CMD_TOPIC, ARM_CMD_TYPE)
    cmd_topic.advertise()
    controls_waist = official_ik_controls_waist(args)
    waist_cmd_topic = None
    if controls_waist:
        if waist_status_buffer is None:
            raise RuntimeError("官方 IK 流式回放缺少 /waist/status 缓存")
        waist_cmd_topic = roslibpy.Topic(client, WAIST_CMD_TOPIC, WAIST_CMD_TYPE)
        waist_cmd_topic.advertise()
    hand_cmd_topics = advertise_hand_command_topics(client, args, hand_control_config)
    last_hand_commands: dict[str, tuple[float, ...] | None] = {
        side: None for side in hand_cmd_topics
    }
    safety = ReplayFeedbackGate(
        args, client, status_buffer, waist_status_buffer, hand_status_buffers,
        hand_control_config, baseline_q, baseline_waist,
    )
    hand_sidecar = None
    if hand_cmd_topics and hand_ik_enabled(args):
        try:
            hand_sidecar = LatestHandIkWorker(Revo2HandIkSidecar(hand_ik_config_from_args(args)), getattr(args, "hand_control_rate", 20.0))
            safety.hand_worker = hand_sidecar
        except Exception as exc:
            raise RuntimeError(f"手部 IK 启动失败，取消回放: {exc}") from exc
    time.sleep(0.2)

    prepose_to_baseline(
        args, cmd_topic, templates, baseline_q, safety=safety,
        waist_topic=waist_cmd_topic,
        baseline_waist=baseline_waist if controls_waist else None,
    )
    post_prepose_sample = status_buffer.latest_complete()
    if post_prepose_sample is not None:
        _, post_prepose_q = post_prepose_sample
        print_json_line(
            "PREPOSE_FEEDBACK",
            {
                "max_error_from_baseline": max_joint_delta(post_prepose_q, baseline_q, controlled_indices(args.control_side)),
                "top_joint_errors": joint_delta_report(post_prepose_q, baseline_q, controlled_indices(args.control_side), limit=3),
            },
        )

    print_json_line(
        "STREAM_IK_START",
        {
            "frame_count": len(frames),
            "control_side": args.control_side,
            "ik_backend": args.ik_backend,
            "ik_task": args.ik_task,
            "rate": args.rate,
            "position_scale": args.position_scale,
            "orientation_scale": args.orientation_scale,
            "max_joint_delta_per_step": args.max_joint_delta_per_step,
            "max_command_delta_from_baseline": args.max_command_delta_from_baseline,
        },
    )

    solver = make_ik_solver(args, baseline_q)
    if controls_waist:
        solver.last_waist_position = float(baseline_waist)
    if not hasattr(solver, "wrist_effector_poses") or not hasattr(solver, "wrist_to_tcp_transforms"):
        raise RuntimeError("当前 IK 后端未提供 wrist-roll 控制点 FK，无法建立机器人手腕 Home")
    left_robot_home, right_robot_home = solver.wrist_effector_poses(baseline_q)
    left_wrist_to_tcp, right_wrist_to_tcp = solver.wrist_to_tcp_transforms()
    mapper = RelativeMocapTargetMapper(
        left_robot_home=left_robot_home,
        right_robot_home=right_robot_home,
        mode="relative",
        position_scale=args.position_scale,
        orientation_scale=args.orientation_scale,
        workspace_calibration=workspace_calibration,
        control_side=args.control_side,
        left_control_to_target=left_wrist_to_tcp,
        right_control_to_target=right_wrist_to_tcp,
        control_frame="wrist_roll",
    )

    indices = controlled_indices(args.control_side)
    interval = 1.0 / args.rate
    last_command_q: np.ndarray | None = None
    error_norms: list[float] = []
    limited_count = 0
    overrun_count = 0
    max_step = 0.0
    max_from_baseline = 0.0
    max_waist_step = 0.0
    max_waist_from_baseline = 0.0
    last_waist_command = float(baseline_waist)
    max_solve_seconds = 0.0

    try:
        for index, frame, source_received_at in replay_control_ticks(frames, args.rate):
            frame_start = time.perf_counter()
            solve_started_at = time.monotonic()
            require_fresh_feedback(source_received_at, solve_started_at, 0.3, "replay mocap input")
            current_q, current_waist = safety.snapshot()
            left_mocap_wrist = replay_anatomical_wrist_pose(frame, "left", wrist_calibration)
            right_mocap_wrist = replay_anatomical_wrist_pose(frame, "right", wrist_calibration)
            mocap_reference_position = replay_mocap_reference_position(frame, workspace_calibration)
            if index == 0:
                mapper.set_runtime_mocap_home(
                    left_mocap_wrist,
                    right_mocap_wrist,
                    mocap_reference_position=mocap_reference_position,
                )
            target_mapping = mapper.map_targets(
                left_mocap_wrist,
                right_mocap_wrist,
                mocap_reference_position=mocap_reference_position,
            )
            left_target, right_target = select_ik_targets(
                args.control_side,
                target_mapping.left_target,
                target_mapping.right_target,
            )

            result = solver.solve(
                TeleopIKInput(
                    timestamp=float(frame["receive_time"]),
                    control_side=args.control_side,
                    target_pose_l=left_target,
                    target_pose_r=right_target,
                    current_q=current_q,
                    last_q_cmd=last_command_q,
                    current_waist_position=current_waist if controls_waist else None,
                )
            )
            current_q, current_waist = safety.snapshot()
            if result.reason != "ok":
                behavior_reasons: dict[str, tuple[str, ...]] = {
                    "solver": (result.reason,)
                }
            else:
                behavior_reasons = replay_wrist_behavior_reasons(
                    solver,
                    result,
                    target_mapping,
                    current_q,
                    args,
                )
            if behavior_reasons:
                print_json_line(
                    "FAIL_WRIST_BEHAVIOR",
                    {"frame": index + 1, "reasons": behavior_reasons},
                )
                return 16
            safe_q, limited = apply_joint_safety(
                q_next=np.asarray(result.command.positions, dtype=float),
                q_ref=current_q,
                limits=solver.joint_limits,
                max_step=replay_step_cap(args),
                smoothing_alpha=1.0,
            )
            safe_waist = None
            if controls_waist:
                safe_waist = apply_waist_replay_safety(
                    result.command.waist_position,
                    current_waist,
                    baseline_waist,
                    solver.waist_limit,
                    replay_step_cap(args),
                    args.max_command_delta_from_baseline,
                )

            command_delta_from_baseline = max_joint_delta(safe_q, baseline_q, indices)
            if command_delta_from_baseline > args.max_command_delta_from_baseline and not args.allow_risk:
                print_json_line(
                    "FAIL_COMMAND_RISK",
                    {
                        "frame": index + 1,
                        "max_command_delta_from_baseline": command_delta_from_baseline,
                        "threshold": args.max_command_delta_from_baseline,
                        "error_norm": result.error_norm,
                        "limited": bool(limited or result.limited),
                        "reason": result.reason,
                        "top_joint_deltas": joint_delta_report(safe_q, baseline_q, indices, limit=5),
                    },
                )
                return 13

            command = replace(result.command, positions=safe_q, waist_position=safe_waist)
            hand_output, hand_received_at = poll_async_hand_output(hand_sidecar, frame, source_received_at, hand_control_config)
            validate_hand_output(hand_output, hand_cmd_topics)
            require_fresh_feedback(solve_started_at, time.monotonic(), MAX_ARM_STATUS_AGE_SECONDS, "IK result")
            command = publish_command(cmd_topic, command, templates, waist_cmd_topic, safety=safety, mocap_received_at=source_received_at)
            safe_q, safe_waist = command.positions, command.waist_position
            publish_hand_output(
                hand_output or None, hand_cmd_topics, hand_status_buffers,
                last_hand_commands, hand_control_config, safety=safety,
                mocap_received_at=hand_received_at,
            )

            solve_seconds = time.perf_counter() - frame_start
            max_solve_seconds = max(max_solve_seconds, solve_seconds)
            if last_command_q is not None:
                max_step = max(max_step, float(np.max(np.abs(safe_q[indices] - last_command_q[indices]))))
            max_from_baseline = max(max_from_baseline, command_delta_from_baseline)
            limited_count += int(limited or result.limited)
            error_norms.append(float(result.error_norm))
            last_command_q = safe_q.copy()
            if safe_waist is not None:
                max_waist_step = max(max_waist_step, abs(safe_waist - last_waist_command))
                max_waist_from_baseline = max(max_waist_from_baseline, abs(safe_waist - baseline_waist))
                last_waist_command = safe_waist

            frame_number = index + 1
            if frame_number == 1 or frame_number % 10 == 0 or frame_number == len(frames):
                print_json_line(
                    "STREAM_IK_PROGRESS",
                    {
                        "frame": frame_number,
                        "total": len(frames),
                        "progress_percent": round(frame_number * 100.0 / max(1, len(frames)), 2),
                        "solve_seconds": solve_seconds,
                        "error_norm": result.error_norm,
                        "limited": bool(limited or result.limited),
                        "frame_command_delta_from_baseline": command_delta_from_baseline,
                        "max_command_delta_from_baseline": max_from_baseline,
                        "waist_position": safe_waist,
                        "max_waist_delta_from_baseline": max_waist_from_baseline,
                        "worst_joint": joint_delta_report(safe_q, baseline_q, indices, limit=1)[0],
                        "overrun_count": overrun_count,
                    },
                )

            if float(result.error_norm) > 0.12 and (frame_number == 1 or frame_number % 10 == 0):
                print_json_line(
                    "WARN_IK_ERROR",
                    {
                        "frame": frame_number,
                        "error_norm": result.error_norm,
                        "reason": result.reason,
                        "top_joint_deltas": joint_delta_report(safe_q, baseline_q, indices, limit=3),
                    },
                )

            if solve_seconds > interval:
                overrun_count += 1
                print_json_line(
                    "WARN_STREAM_OVERRUN",
                    {
                        "frame": frame_number,
                        "solve_seconds": solve_seconds,
                        "period_seconds": interval,
                        "lag_seconds": solve_seconds - interval,
                    },
                )


        print_json_line(
            "STREAM_IK_SUMMARY",
            {
                "command_count": len(frames),
                "control_side": args.control_side,
                "control_mode": "ik",
                "ik_backend": args.ik_backend,
                "max_joint_step": max_step,
                "max_command_delta_from_baseline": max_from_baseline,
                "max_waist_step": max_waist_step,
                "max_waist_delta_from_baseline": max_waist_from_baseline,
                "limited_count": limited_count,
                "overrun_count": overrun_count,
                "max_solve_seconds": max_solve_seconds,
                "max_error_norm": max(error_norms) if error_norms else 0.0,
                "mean_error_norm": float(np.mean(error_norms)) if error_norms else 0.0,
            },
        )

        after_move = status_buffer.latest_complete()
        if after_move is not None:
            _, moved_q = after_move
            print(f"MOVE_FEEDBACK_MAX_DELTA {float(np.max(np.abs(moved_q[indices] - baseline_q[indices]))):.6f}")

        restore_baselines(
            args, safety, hand_cmd_topics, hand_baselines, last_hand_commands,
            cmd_topic, templates, baseline_q, waist_cmd_topic,
            baseline_waist if controls_waist else None,
        )
        print("PUBLISHED_RESTORE_BASELINE")
        if hand_cmd_topics:
            print("PUBLISHED_RESTORE_HAND_BASELINES")

        restored = status_buffer.latest_complete()
        if restored is not None:
            _, restored_q = restored
            print(f"RESTORE_MAX_ERROR {float(np.max(np.abs(restored_q[indices] - baseline_q[indices]))):.6f}")

        print("PASS mocap replay streaming command sequence published")
        return 0
    except KeyboardInterrupt:
        print("ABORT_REPLAY interrupted; further motion commands stopped")
        return 130
    except Exception as exc:
        print(f"ABORT_REPLAY {exc}; further motion commands stopped")
        return 14
    finally:
        try:
            cmd_topic.unadvertise()
        except Exception:
            pass
        if waist_cmd_topic is not None:
            try:
                waist_cmd_topic.unadvertise()
            except Exception:
                pass
        for hand_topic in hand_cmd_topics.values():
            try:
                hand_topic.unadvertise()
            except Exception:
                pass
        if hand_sidecar is not None:
            hand_sidecar.close()



def execute_hand_only_replay(
    args: argparse.Namespace,
    client: roslibpy.Ros,
    frames: list[dict[str, Any]],
    hand_outputs: list[dict[str, tuple[float, ...]] | None],
    hand_status_buffers: dict[str, HandStatusBuffer],
    hand_baselines: dict[str, tuple[float, ...]],
    hand_control_config: HandControlConfig,
) -> int:
    hand_cmd_topics = advertise_hand_command_topics(client, args, hand_control_config)
    last_hand_commands: dict[str, tuple[float, ...] | None] = {
        side: None for side in hand_cmd_topics
    }
    safety = ReplayFeedbackGate(
        args, client, None, None, hand_status_buffers, hand_control_config,
    )
    try:
        for index, _, source_received_at in replay_control_ticks(frames, args.rate):
            require_fresh_feedback(source_received_at, time.monotonic(), 0.3, "replay mocap input")
            hand_output = hand_outputs[index] if index < len(hand_outputs) else None
            publish_hand_output(
                hand_output,
                hand_cmd_topics,
                hand_status_buffers,
                last_hand_commands,
                hand_control_config,
                safety=safety,
                mocap_received_at=source_received_at,
            )

        restore_baselines(args, safety, hand_cmd_topics, hand_baselines, last_hand_commands)
        if hand_cmd_topics:
            print("PUBLISHED_RESTORE_HAND_BASELINES")
        print("PASS hand-only replay published; arm_and_waist_commands=0")
        return 0
    except KeyboardInterrupt:
        print("ABORT_REPLAY interrupted; further motion commands stopped")
        return 130
    except Exception as exc:
        print(f"ABORT_REPLAY {exc}; further motion commands stopped")
        return 14
    finally:
        for hand_topic in hand_cmd_topics.values():
            try:
                hand_topic.unadvertise()
            except Exception:
                pass



def execute_replay(
    args: argparse.Namespace,
    client: roslibpy.Ros,
    status_buffer: ArmStatusBuffer,
    commands: list[IKCommand],
    baseline_q: np.ndarray,
    baseline_waist: float,
    hand_outputs: list[dict[str, tuple[float, ...]] | None],
    hand_status_buffers: dict[str, HandStatusBuffer],
    hand_baselines: dict[str, tuple[float, ...]],
    hand_control_config: HandControlConfig,
    waist_status_buffer: WaistStatusBuffer | None = None,
) -> int:
    templates = make_motor_templates(args.control_side, args.spd, args.cur)
    cmd_topic = roslibpy.Topic(client, ARM_CMD_TOPIC, ARM_CMD_TYPE)
    cmd_topic.advertise()
    controls_waist = official_ik_controls_waist(args)
    waist_cmd_topic = None
    if controls_waist:
        waist_cmd_topic = roslibpy.Topic(client, WAIST_CMD_TOPIC, WAIST_CMD_TYPE)
        waist_cmd_topic.advertise()
    hand_cmd_topics = advertise_hand_command_topics(client, args, hand_control_config)
    last_hand_commands: dict[str, tuple[float, ...] | None] = {
        side: None for side in hand_cmd_topics
    }
    safety = ReplayFeedbackGate(
        args, client, status_buffer, waist_status_buffer, hand_status_buffers,
        hand_control_config, baseline_q, baseline_waist,
    )
    time.sleep(0.2)

    prepose_to_baseline(
        args, cmd_topic, templates, baseline_q, safety=safety,
        waist_topic=waist_cmd_topic,
        baseline_waist=baseline_waist if controls_waist else None,
    )

    try:
        for index, command, source_received_at in replay_control_ticks(
            commands, args.rate, [command.timestamp for command in commands],
        ):
            require_fresh_feedback(source_received_at, time.monotonic(), 0.3, "replay command input")
            hand_output = hand_outputs[index] if index < len(hand_outputs) else None
            validate_hand_output(hand_output, hand_cmd_topics)
            publish_command(cmd_topic, command, templates, waist_cmd_topic, safety=safety, mocap_received_at=source_received_at)
            publish_hand_output(
                hand_output,
                hand_cmd_topics,
                hand_status_buffers,
                last_hand_commands,
                hand_control_config,
                safety=safety,
                mocap_received_at=source_received_at,
            )

        after_move = status_buffer.latest_complete()
        if after_move is not None:
            _, moved_q = after_move
            indices = controlled_indices(args.control_side)
            print(f"MOVE_FEEDBACK_MAX_DELTA {float(np.max(np.abs(moved_q[indices] - baseline_q[indices]))):.6f}")

        restore_baselines(
            args, safety, hand_cmd_topics, hand_baselines, last_hand_commands,
            cmd_topic, templates, baseline_q, waist_cmd_topic,
            baseline_waist if controls_waist else None,
        )
        print("PUBLISHED_RESTORE_BASELINE")
        if hand_cmd_topics:
            print("PUBLISHED_RESTORE_HAND_BASELINES")

        restored = status_buffer.latest_complete()
        if restored is not None:
            _, restored_q = restored
            indices = controlled_indices(args.control_side)
            print(f"RESTORE_MAX_ERROR {float(np.max(np.abs(restored_q[indices] - baseline_q[indices]))):.6f}")

        print("PASS mocap replay command sequence published")
        return 0
    except KeyboardInterrupt:
        print("ABORT_REPLAY interrupted; further motion commands stopped")
        return 130
    except Exception as exc:
        print(f"ABORT_REPLAY {exc}; further motion commands stopped")
        return 14
    finally:
        try:
            cmd_topic.unadvertise()
        except Exception:
            pass
        if waist_cmd_topic is not None:
            try:
                waist_cmd_topic.unadvertise()
            except Exception:
                pass
        for hand_topic in hand_cmd_topics.values():
            try:
                hand_topic.unadvertise()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    replay_step_cap(args)
    for name, default in (("hand_control_rate", 20.0), ("max_hand_velocity", 1.0)):
        value = float(getattr(args, name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    if args.control_mode == "skeleton":
        validate_skeleton_replay_options(args)
    try:
        hand_control_raw = json.loads(args.hand_control_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--hand-control-json 不是有效 JSON: {exc}") from exc
    hand_control_config = hand_control_config_from_mapping(hand_control_raw or None)
    if hand_control_config.publish_enabled and not hand_ik_enabled(args):
        raise ValueError("hand_control.publish=true 需要同时启用 hand_ik.compute")
    if args.hand_only and not hand_ik_enabled(args):
        raise ValueError("--hand-only 需要启用 hand_ik.compute")
    if args.hand_only and args.execute and not hand_control_config.publish_enabled:
        raise ValueError("--hand-only --execute 需要 hand_control.publish=true")
    if args.rate <= 0:
        raise ValueError("--rate 必须 > 0")
    if args.max_joint_delta_per_step <= 0:
        raise ValueError("--max-joint-delta-per-step 必须 > 0")
    if not 0.0 <= args.orientation_scale <= 1.0:
        raise ValueError("--orientation-scale 必须在 [0, 1] 范围内")
    if (
        not args.hand_only
        and args.control_mode == "ik"
    ):
        workspace_calibration = load_workspace_calibration(
            args.workspace_calibration, args.control_side, optional=args.workspace_calibration is None
        )
        wrist_calibration = load_wrist_calibration(
            args.wrist_calibration, args.control_side, optional=args.wrist_calibration is None
        )
        if workspace_calibration is not None:
            print(f"REPLAY_CALIBRATION_INFO 已加载 workspace 参考标定: {workspace_calibration.source_path}")
        else:
            print("REPLAY_CALIBRATION_INFO workspace 参考标定不可用，使用默认相对映射")
        if wrist_calibration is not None:
            print(f"REPLAY_CALIBRATION_INFO 已加载 wrist 参考标定: {wrist_calibration.source_path}")
        else:
            print("REPLAY_CALIBRATION_INFO wrist 参考标定不可用，使用 skeleton 腕点")

    input_path = resolve_input_path(args.input)
    frames = load_frames(input_path, args.max_frames, args.stride)
    frame_summary = analyze_frames(frames)
    print(f"MOCAP_REPLAY_INPUT {input_path}")
    print_json_line("MOCAP_DATA_SUMMARY", frame_summary)

    reasons = data_risk_reasons(frame_summary, args.max_position_step, args.max_angular_step)
    if reasons:
        print("WARN_DATA_RISK " + ";".join(reasons))
        if not args.allow_risk:
            print("FAIL_DATA_RISK add --allow-risk only after confirming the capture is safe")
            return 12

    client = None
    status_topic = None
    waist_status_topic = None
    hand_status_topics: dict[str, roslibpy.Topic] = {}
    status_buffer = None
    waist_status_buffer = None
    hand_status_buffers: dict[str, HandStatusBuffer] = {}
    hand_baselines: dict[str, tuple[float, ...]] = {}
    status_q: np.ndarray | None = None
    baseline_q = np.zeros(14)
    baseline_waist = 0.0

    need_robot_status = args.execute or args.use_robot_status
    try:
        if need_robot_status:
            try:
                client = connect_rosbridge(args.host, args.port, args.connect_timeout)
            except Exception as exc:
                print(f"FAIL_CONNECT rosbridge=ws://{args.host}:{args.port} error={exc}")
                return 10
            print(f"ROSBRIDGE_CONNECTED ws://{args.host}:{args.port}")

            if not args.hand_only:
                status_buffer = ArmStatusBuffer()
                status_topic = roslibpy.Topic(client, ARM_STATUS_TOPIC, ARM_STATUS_TYPE)
                status_topic.subscribe(status_buffer.callback)
                sample = status_buffer.wait_complete(args.status_timeout)
                if sample is None:
                    print(f"FAIL_STATUS_TIMEOUT topic={ARM_STATUS_TOPIC} timeout={args.status_timeout}s expected_ids={ARM_MOTOR_IDS}")
                    return 11
                _, status_q = sample
                print_json_line("ROBOT_STATUS_Q", {"motor_ids": ARM_MOTOR_IDS, "positions": status_q})

            if not args.hand_only and official_ik_controls_waist(args):
                waist_status_buffer = WaistStatusBuffer()
                waist_status_topic = roslibpy.Topic(client, WAIST_STATUS_TOPIC, WAIST_STATUS_TYPE)
                waist_status_topic.subscribe(waist_status_buffer.callback)
                waist_sample = waist_status_buffer.wait_valid(args.status_timeout)
                if waist_sample is None:
                    error = waist_status_buffer.error_description()
                    print(
                        f"FAIL_STATUS_TIMEOUT topic={WAIST_STATUS_TOPIC} timeout={args.status_timeout}s "
                        f"expected_id={WAIST_MOTOR_ID} error={error or 'missing'}"
                    )
                    return 14
                _, baseline_waist = waist_sample
                print_json_line(
                    "ROBOT_WAIST_STATUS",
                    {"motor_id": WAIST_MOTOR_ID, "position": baseline_waist},
                )

            if hand_control_config.publish_enabled:
                for side in active_hand_sides(args.control_side):
                    hand_buffer = HandStatusBuffer(side)
                    hand_topic = roslibpy.Topic(
                        client,
                        HAND_STATE_TOPICS[side],
                        HAND_JOINT_STATE_TYPE,
                    )
                    hand_topic.subscribe(hand_buffer.callback)
                    hand_status_buffers[side] = hand_buffer
                    hand_status_topics[side] = hand_topic
                for side, hand_buffer in hand_status_buffers.items():
                    hand_sample = hand_buffer.wait_valid(args.status_timeout)
                    if hand_sample is None:
                        print(
                            f"FAIL_STATUS_TIMEOUT topic={HAND_STATE_TOPICS[side]} "
                            f"timeout={args.status_timeout}s error={hand_buffer.error_description() or 'missing'}"
                        )
                        return 15
                    hand_baselines[side] = hand_sample[1]
                    print_json_line(
                        "ROBOT_HAND_STATUS",
                        {"side": side, "positions": hand_sample[1]},
                    )
        else:
            print("ROBOT_STATUS_SOURCE offline_preview")

        if args.hand_only:
            hand_summary, hand_outputs = plan_hand_ik_outputs(frames, args, hand_control_config)
            print_json_line(
                "HAND_ONLY_REPLAY_SUMMARY",
                {
                    **hand_summary,
                    "arm_commands": 0,
                    "waist_commands": 0,
                    "execute": bool(args.execute),
                },
            )
            if not args.execute:
                print("DRY_RUN_ONLY hand-only no arm/waist/hand command published; add --execute after checking summaries")
                return 0
            if client is None:
                raise RuntimeError("hand-only execute requires rosbridge")
            confirm_execution(args, {"command_count": len(frames), "control_mode": "hand-only", "arm_commands": 0, "waist_commands": 0})
            return execute_hand_only_replay(
                args,
                client,
                frames,
                hand_outputs,
                hand_status_buffers,
                hand_baselines,
                hand_control_config,
            )

        baseline_q, baseline_source = resolve_replay_baseline(args, status_q)
        print_json_line("REPLAY_BASELINE_Q", {"source": baseline_source, "motor_ids": ARM_MOTOR_IDS, "positions": baseline_q})
        if official_ik_controls_waist(args):
            print_json_line(
                "REPLAY_BASELINE_WAIST",
                {
                    "source": "robot_status" if need_robot_status else "zero_offline_preview",
                    "motor_id": WAIST_MOTOR_ID,
                    "position": baseline_waist,
                },
            )

        if args.execute and args.control_mode == "skeleton":
            if status_buffer is None or client is None:
                raise RuntimeError("skeleton execute requires rosbridge status buffer")
            confirm_execution(args, {"command_count": len(frames), "control_side": args.control_side, "control_mode": "skeleton"})
            return execute_skeleton_replay(
                args, client, status_buffer, frames, baseline_q,
                hand_status_buffers, hand_baselines, hand_control_config,
            )

        if args.execute and args.streaming and args.control_mode == "ik":
            if status_buffer is None or client is None:
                raise RuntimeError("streaming execute requires rosbridge status buffer")
            confirm_execution(
                args,
                {
                    "command_count": len(frames),
                    "control_side": args.control_side,
                    "control_mode": "ik",
                    "ik_backend": args.ik_backend,
                },
            )
            return execute_ik_replay_streaming(
                args,
                client,
                status_buffer,
                waist_status_buffer,
                frames,
                baseline_q,
                baseline_waist,
                hand_status_buffers,
                hand_baselines,
                hand_control_config,
            )

        commands, command_summary = plan_replay_commands(frames, args, baseline_q, baseline_waist)
        print_json_line("IK_REPLAY_SUMMARY", command_summary)
        hand_summary, hand_outputs = plan_hand_ik_outputs(frames, args, hand_control_config)
        print_json_line("HAND_IK_SUMMARY", hand_summary)

        command_reasons = command_risk_reasons(command_summary, args.max_command_delta_from_baseline)
        if command_reasons:
            print("WARN_COMMAND_RISK " + ";".join(command_reasons))
            if not args.allow_risk:
                print("FAIL_COMMAND_RISK add --allow-risk only after reducing scale or confirming the motion")
                return 13

        if not args.execute:
            print("DRY_RUN_ONLY no command published; add --execute after checking summaries")
            return 0

        if status_buffer is None or client is None:
            raise RuntimeError("execute requires rosbridge status buffer")
        confirm_execution(args, command_summary)
        return execute_replay(
            args,
            client,
            status_buffer,
            commands,
            baseline_q,
            baseline_waist,
            hand_outputs,
            hand_status_buffers,
            hand_baselines,
            hand_control_config,
            waist_status_buffer=waist_status_buffer,
        )
    finally:
        if status_topic is not None:
            try:
                status_topic.unsubscribe()
            except Exception:
                pass
        if waist_status_topic is not None:
            try:
                waist_status_topic.unsubscribe()
            except Exception:
                pass
        for hand_status_topic in hand_status_topics.values():
            try:
                hand_status_topic.unsubscribe()
            except Exception:
                pass
        if client is not None:
            try:
                client.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
