#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT_DIR / "src"
CONFIG_PATH = ROOT_DIR / "run_ik_config.json"
RECORDER_ENTRY = SOURCE_DIR / "mocap_udp_recorder.py"
TELEOP_ENTRY = SOURCE_DIR / "mocap_ros_2_bridge_node.py"
ARM_TEST_ENTRY = SOURCE_DIR / "arm_link_test.py"
REPLAY_ENTRY = SOURCE_DIR / "mocap_replay_to_robot.py"
PROTOCOL_ENTRY = SOURCE_DIR / "teleop_mocap_protocol.py"
VISUALIZER_ENTRY = SOURCE_DIR / "visualize_joint_commands.py"
DEFAULT_CAPTURE_DIR = ROOT_DIR / "data" / "vr_udp_capture"

REQUIRED_MODULES = ("numpy", "scipy", "roslibpy")
TELEOP_REQUIRED_MODULES = ("rclpy", "pinocchio")

DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "python": None,
        "conda_env": None,
        "pause_after_action": True,
        "blas_threads": 1,
    },
    "rosbridge": {
        "host": "192.168.1.110",
        "port": 9091,
        "connect_timeout": 8.0,
        "status_timeout": 5.0,
    },
    "environment": {
        "basic_include_teleop": True,
        "self_test_port": 15002,
        "self_test_output": "data/launcher_test/self_test",
    },
    "record": {
        "device_mode": "glove",
        "ip": "0.0.0.0",
        "port": 15000,
        "duration": 60.0,
        "output": "data/vr_udp_capture",
        "flush_every": 20,
        "diagnostic_samples": 5,
        "diagnostic_bytes": 256,
        "save_raw": False,
        "save_jsonl": False,
    },
    "workspace_calibration": {
        "device_mode": "glove",
        "ip": "0.0.0.0",
        "port": 15000,
        "capture_output": "data/workspace_calibration/raw",
        "calibration_output": "data/workspace_calibration/calibration.json",
        "runtime_path": None,
        "side": "both",
        "capture_rate": 30.0,
        "home_seconds": 3.0,
        "move_seconds": 2.0,
        "hold_seconds": 2.0,
        "min_direction_displacement": 0.06,
        "max_direction_rms": 0.055,
        "min_valid_rate": 0.90,
        "min_direction_cosine": 0.68,
        "max_opposite_direction_cosine": -0.40,
        "position_error_threshold": 0.015,
        "orientation_error_threshold": 0.20,
        "joint_margin_rad": 0.0,
        "safe_boundary_ratio": 0.65,
        "scan_step": 0.01,
        "scan_max": 0.50,
        "control_frame": "wrist_roll",
        "overwrite": False,
        "robot_home_q": [0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "wrist_calibration": {
        "device_mode": "glove",
        "ip": "0.0.0.0",
        "port": 15000,
        "side": "both",
        "group_seconds": 10.0,
        "capture_output": "data/wrist_calibration/raw",
        "calibration_output": "data/wrist_calibration/calibration.json",
        "runtime_path": None,
        "min_samples": 60,
        "min_rotation_excitation": 0.25,
        "min_rotation_rank": 2,
        "max_outlier_fraction": 0.15,
        "max_pivot_rms": 0.010,
        "max_pivot_p95": 0.020,
        "max_wrist_validation_rms": 0.020,
        "huber_delta": 0.010,
    },
    "replay": {
        "input": None,
        "use_latest_capture": True,
        "control_side": "both",
        "control_mode": "skeleton",
        "ik_backend": "teleop_dls",
        "ik_task": "pose",
        "rate": 100.0,
        "max_joint_velocity": 0.48,
        "hand_control_rate": 20.0,
        "max_hand_velocity": 1.0,
        "max_frames": 200,
        "stride": 1,
        "position_scale": 1.0,
        "orientation_scale": 1.0,
        "max_position_step": 0.25,
        "max_angular_step": 0.8,
        "max_joint_delta_per_step": 0.06,
        "max_command_delta_from_baseline": 1.4,
        "calibration_frames": 45,
        "calibration_max_direction_rms": 0.055,
        "calibration_max_length_cv": 0.025,
        "max_segment_length_ratio": 1.18,
        "max_skeleton_joint_step": 0.10,
        "upper_direction_weight": 1.0,
        "forearm_direction_weight": 0.85,
        "direction_damping": 0.08,
        "direction_iterations": 6,
        "spd": 0.7,
        "cur": 2.0,
        "restore_seconds": 3.0,
        "prepose": "none",
        "prepose_seconds": 3.0,
        "prepose_steps": 30,
        "use_robot_status": False,
        "allow_risk": False,
        "streaming": False,
        "hand_only": False,
        "execute": False,
        "yes": True,
    },
    "arm_test": {
        "side": "both",
        "scale": 0.35,
        "min_delta": 0.08,
        "max_delta": 0.35,
        "spd": 0.7,
        "cur": 2.0,
        "hold_seconds": 3.0,
        "restore_seconds": 3.0,
        "finish_pose": "initial",
        "feedback_threshold": 0.03,
        "hand_delta": 0.20,
        "hand_feedback_threshold": 0.02,
        "arm_only": False,
        "execute": False,
        "yes": False,
    },
    "teleop": {
        "device_mode": "glove",
        "skip_check": False,
        "control_side": "both",
        "control_mode": "skeleton",
        "ik_backend": "teleop_dls",
        "ik_task": "pose",
        "position_scale": 1.0,
        "orientation_scale": 1.0,
        "safe_teleop": True,
        "control_rate": 100.0,
        "safe_rate": 100.0,
        "max_joint_velocity": 0.48,
        "hand_control_rate": 20.0,
        "max_hand_velocity": 1.0,
        "log_rate": 1.0,
        "max_joint_delta_per_step": 0.06,
        "max_command_delta_from_baseline": 1.4,
        "startup_hold_seconds": 2.0,
        "startup_human_vertical_cosine": 0.78,
        "startup_human_min_wrist_drop": 0.30,
        "startup_human_max_lateral_offset": 0.16,
        "startup_max_home_rms": 0.025,
        "calibration_frames": 45,
        "calibration_max_direction_rms": 0.055,
        "calibration_max_length_cv": 0.025,
        "max_segment_length_ratio": 1.18,
        "max_skeleton_joint_step": 0.10,
        "upper_direction_weight": 1.0,
        "forearm_direction_weight": 0.85,
        "direction_damping": 0.08,
        "direction_iterations": 6,
        "robot_command": False,
        "hand_only": False,
    },
    "simulation": {
        "output": "data/simulation",
        "control_hz": 100.0,
        "physics_substeps": 4,
        "duration": 4.0,
        "profile": "skeleton-100hz",
        "max_joint_velocity": 0.48,
    },
    "hand_ik": {
        "compute": True,
        "sdk_root": "../third_party/revo2_hand_ik_sdk",
        "sdk_config": "../third_party/revo2_hand_ik_sdk/bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json",
        "library": "../third_party/revo2_hand_ik_sdk/lib/librevo2_ik_sdk.so.0.10.35",
        "timeout": 5.0,
        "diagnostic_enabled": False,
        "diagnostic_csv": "data/diagnostics/revo2_hand_ik.csv",
    },
    "hand_control": {
        "publish": False,
        "state_timeout": 0.5,
        "max_step": 0.05,
        "thumb_rotation_mode": "hold_feedback",
        "calibration": {
            side: {
                "little": {"open_rad": 0.0, "closed_rad": 1.41, "enabled": True},
                "ring": {"open_rad": 0.0, "closed_rad": 1.41, "enabled": True},
                "middle": {"open_rad": 0.0, "closed_rad": 1.41, "enabled": True},
                "fore": {"open_rad": 0.0, "closed_rad": 1.41, "enabled": True},
                "thumb_bend": {"open_rad": 0.0, "closed_rad": 1.03, "enabled": True},
                "thumb_rotation": {"open_rad": 0.0, "closed_rad": 1.57, "enabled": True},
            }
            for side in ("left", "right")
        },
    },
}


SKELETON_CONFIG_KEYS = (
    "calibration_frames",
    "calibration_max_direction_rms",
    "calibration_max_length_cv",
    "max_segment_length_ratio",
    "max_skeleton_joint_step",
    "upper_direction_weight",
    "forearm_direction_weight",
    "direction_damping",
    "direction_iterations",
)

TELEOP_STARTUP_CONFIG_KEYS = (
    "startup_hold_seconds",
    "startup_human_vertical_cosine",
    "startup_human_min_wrist_drop",
    "startup_human_max_lateral_offset",
    "startup_max_home_rms",
)


def add_skeleton_retarget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-frames", type=int, default=None, help="direct 稳定 home 标定帧数")
    parser.add_argument("--calibration-max-direction-rms", type=float, default=None, help="标定窗口最大骨段方向 RMS，单位 rad")
    parser.add_argument("--calibration-max-length-cv", type=float, default=None, help="标定窗口最大骨长变异系数")
    parser.add_argument("--max-segment-length-ratio", type=float, default=None, help="相对 home 的最大骨长比")
    parser.add_argument("--max-skeleton-joint-step", type=float, default=None, help="单帧人体关节点最大跳变，单位 m")
    parser.add_argument("--upper-direction-weight", type=float, default=None, help="机器人上臂方向任务权重")
    parser.add_argument("--forearm-direction-weight", type=float, default=None, help="机器人前臂方向任务权重")
    parser.add_argument("--direction-damping", type=float, default=None, help="骨段方向 DLS 阻尼")
    parser.add_argument("--direction-iterations", type=int, default=None, help="每帧骨段方向 DLS 迭代次数")


def apply_skeleton_retarget_config(args: argparse.Namespace, config: dict[str, Any], section: str) -> None:
    for key in SKELETON_CONFIG_KEYS:
        setattr(args, key, coalesce(getattr(args, key, None), cfg_get(config, section, key)))


def append_skeleton_retarget_cli(command: list[str], args: argparse.Namespace) -> None:
    for key in SKELETON_CONFIG_KEYS:
        command.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))


def add_teleop_startup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--startup-hold-seconds", type=float, default=None, help="人体自然下垂连续稳定保持秒数")
    parser.add_argument("--startup-human-vertical-cosine", type=float, default=None, help="人体手臂与躯干向下方向最小余弦")
    parser.add_argument("--startup-human-min-wrist-drop", type=float, default=None, help="人体手腕相对肩部最小向下距离，单位 m")
    parser.add_argument("--startup-human-max-lateral-offset", type=float, default=None, help="人体手腕相对肩部最大横向偏移，单位 m")
    parser.add_argument("--startup-max-home-rms", type=float, default=None, help="自然下垂保持窗口最大手部位置 RMS，单位 m")


def apply_teleop_startup_config(args: argparse.Namespace, config: dict[str, Any], section: str = "teleop") -> None:
    for key in TELEOP_STARTUP_CONFIG_KEYS:
        default = config.get(section, {}).get(key, cfg_get(config, "teleop", key))
        setattr(args, key, coalesce(getattr(args, key, None), default))


def append_teleop_startup_cli(command: list[str], args: argparse.Namespace) -> None:
    for key in TELEOP_STARTUP_CONFIG_KEYS:
        command.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))


def add_hand_ik_launcher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hand-ik-compute", choices=("true", "false"), default=None, help="是否在隔离子进程计算 Revo2 双手 12 路弧度角")
    parser.add_argument("--hand-ik-sdk-root", type=Path, default=None, help="Revo2 SDK 根目录")
    parser.add_argument("--hand-ik-sdk-config", type=Path, default=None, help="Revo2 双手 direct API 配置")
    parser.add_argument("--hand-ik-library", type=Path, default=None, help="Revo2 native 动态库")
    parser.add_argument("--hand-ik-timeout", type=float, default=None, help="手部 sidecar 单帧超时秒数")
    parser.add_argument("--hand-ik-diagnostic-csv", type=Path, default=None, help="可选本地诊断 CSV；空值表示不记录")


def apply_hand_ik_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    compute = coalesce(args.hand_ik_compute, cfg_get(config, "hand_ik", "compute"))
    args.hand_ik_compute = "true" if config_bool(compute) else "false"
    args.hand_ik_sdk_root = resolve_path(coalesce(args.hand_ik_sdk_root, cfg_get(config, "hand_ik", "sdk_root")))
    args.hand_ik_sdk_config = resolve_path(coalesce(args.hand_ik_sdk_config, cfg_get(config, "hand_ik", "sdk_config")))
    args.hand_ik_library = resolve_path(coalesce(args.hand_ik_library, cfg_get(config, "hand_ik", "library")))
    args.hand_ik_timeout = coalesce(args.hand_ik_timeout, cfg_get(config, "hand_ik", "timeout"))
    diagnostic_enabled = config_bool(cfg_get(config, "hand_ik", "diagnostic_enabled"))
    configured_diagnostic = resolve_path(cfg_get(config, "hand_ik", "diagnostic_csv")) if diagnostic_enabled else None
    args.hand_ik_diagnostic_csv = resolve_path(coalesce(args.hand_ik_diagnostic_csv, configured_diagnostic))


def append_hand_ik_cli(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        (
            "--hand-ik-compute", str(args.hand_ik_compute),
            "--hand-ik-sdk-root", str(args.hand_ik_sdk_root),
            "--hand-ik-sdk-config", str(args.hand_ik_sdk_config),
            "--hand-ik-library", str(args.hand_ik_library),
            "--hand-ik-timeout", str(args.hand_ik_timeout),
        )
    )
    if args.hand_ik_diagnostic_csv is not None:
        command.extend(("--hand-ik-diagnostic-csv", str(args.hand_ik_diagnostic_csv)))


def add_timing_args(parser: argparse.ArgumentParser, *, live: bool) -> None:
    parser.add_argument("--max-joint-velocity", type=float, default=None, help="关节角速度上限 rad/s，同时受 URDF 限制")
    parser.add_argument("--hand-control-rate", type=float, default=None, help="独立手部计算/控制频率，默认 20Hz")
    parser.add_argument("--max-hand-velocity", type=float, default=None, help="手部归一化通道每秒最大变化")
    if live:
        parser.add_argument("--control-rate", type=float, default=None, help="手臂计算周期 Hz，默认 100")
        parser.add_argument("--log-rate", type=float, default=None, help="例行帧日志频率 Hz，0 关闭；故障仍立即记录")


def apply_timing_config(args: argparse.Namespace, config: dict[str, Any], section: str) -> None:
    keys = {"max_joint_velocity": 0.48, "hand_control_rate": 20.0, "max_hand_velocity": 1.0}
    if section == "teleop":
        keys.update(control_rate=100.0, log_rate=1.0)
    for key, fallback in keys.items():
        value = coalesce(getattr(args, key, None), config.get(section, {}).get(key, fallback))
        value = float(value)
        if not math.isfinite(value) or value < 0 or (key != "log_rate" and value == 0):
            raise ValueError(f"{key} 必须为有限的有效频率/速度")
        setattr(args, key, value)


def append_timing_cli(command: list[str], args: argparse.Namespace, *, live: bool) -> None:
    fields = {"max_joint_velocity": 0.48, "hand_control_rate": 20.0, "max_hand_velocity": 1.0}
    if live:
        fields.update(control_rate=100.0, log_rate=1.0)
    for name, fallback in fields.items():
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name, fallback))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IK 工程统一启动入口：参数从 run_ik_config.json 读取，CLI 只负责选择功能",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="启动配置文件路径")
    parser.add_argument("--python", default=None, help="覆盖配置里的 runtime.python")
    parser.add_argument("--conda-env", default=None, help="覆盖配置里的 runtime.conda_env")
    parser.add_argument("--blas-threads", type=int, default=None, help="小矩阵 IK 的 BLAS 线程数，默认 1")
    parser.add_argument("--no-pause", action="store_true", default=None, help="菜单动作结束后不等待回车")

    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("menu", help="打开交互式 CLI 菜单")

    combined_parser = subparsers.add_parser("calibrate", aliases=("legacy-calibrate",), help="legacy 高级工具：腕点与工作空间人工标定，skeleton 模式不使用")
    calibration_parsers = []
    for name, stages in (
        ("wrist-calibrate", ("all", "capture", "analyze", "validate")),
        ("workspace-calibrate", ("guide", "all", "capture", "analyze", "scan", "report")),
    ):
        calibration_parser = subparsers.add_parser(name, aliases=("legacy-" + name,), help="legacy 高级人工标定工具，skeleton 模式不使用")
        calibration_parser.add_argument("--stage", choices=stages, default="all")
        calibration_parser.add_argument("--input", type=Path, default=None)
        calibration_parser.add_argument("--capture-output", type=Path, default=None)
        calibration_parser.add_argument("--calibration-output", type=Path, default=None)
        calibration_parsers.append(calibration_parser)
    for calibration_parser in (combined_parser, *calibration_parsers):
        calibration_parser.add_argument("--device-mode", choices=("glove", "controller", "auto"), default=None)
        calibration_parser.add_argument("--ip", default=None)
        calibration_parser.add_argument("--port", type=int, default=None)
        calibration_parser.add_argument("--side", choices=("left", "right", "both"), default=None)
    workspace_parser = calibration_parsers[1]
    workspace_parser.add_argument("--wrist-calibration", type=Path, default=None)
    workspace_parser.add_argument("--overwrite", action="store_true", default=None)

    check_parser = subparsers.add_parser("check", help="检查目录、入口文件和 Python 依赖")
    check_parser.add_argument("--teleop", action="store_true", default=None, help="同时检查遥操作所需 ROS2/Pinocchio 依赖")
    check_parser.add_argument("--ik-backend", choices=("teleop_dls", "tiangong2pro_official", "ubtech_casadi"), default=None, help="检查指定后端；仅 legacy 官方后端需要 CasADi")

    test_parser = subparsers.add_parser("test", help="执行编译检查和最小 UDP 数采自测")
    test_parser.add_argument("--port", type=int, default=None, help="自测使用的本机 UDP 端口")
    test_parser.add_argument("--output", type=Path, default=None, help="自测输出目录")

    simulation_parser = subparsers.add_parser("simulate", help="基于机器人 URDF 的独立 PyBullet IK 仿真，无需 ROS/机器人")
    simulation_parser.add_argument("--output", type=Path, default=None)
    simulation_parser.add_argument("--control-hz", type=float, default=None)
    simulation_parser.add_argument("--physics-substeps", type=int, default=None)
    simulation_parser.add_argument("--duration", type=float, default=None, help="每场景仿真秒数")
    simulation_parser.add_argument("--profile", choices=("skeleton-current", "skeleton-100hz"), default=None)
    simulation_parser.add_argument("--realtime", action="store_true", help="按真实墙钟节拍运行并测量调度")
    simulation_parser.add_argument("--scenario", action="append", choices=("reachable", "wrist_twist", "left_only", "near_limits", "unreachable", "tracking_loss", "limit_guard", "mocap_end_to_end"), help="只运行指定场景，可重复传入")
    simulation_parser.add_argument("--max-joint-velocity", type=float, default=None, help="仿真操作速度上限 rad/s，默认 0.48")

    visualize_parser = subparsers.add_parser("visualize", help="从遥操作日志生成离线关节角曲线")
    visualize_parser.add_argument("input", nargs="?", type=Path, default=ROOT_DIR / "data" / "logs" / "mocap_ik_bridge.log", help="遥操作日志或纯 JSONL")
    visualize_parser.add_argument("--output", "-o", type=Path, default=ROOT_DIR / "data" / "visualization" / "joint_commands.html", help="输出的单文件 HTML")
    visualize_parser.add_argument("--session", choices=("latest", "all"), default="latest", help="只看最近会话或全部日志")
    visualize_parser.add_argument("--max-points", type=int, default=5000, help="每条轨迹最多保留的绘图点数")
    visualize_parser.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")

    record_parser = subparsers.add_parser("record", help="启动 VR UDP 数采")
    record_parser.add_argument("--device-mode", choices=("glove", "controller", "auto"), default=None, help="VR 设备模式：glove=手套，controller=手柄，auto=按已验证帧头分派")
    record_parser.add_argument("--ip", default=None, help="UDP 监听地址")
    record_parser.add_argument("--port", type=int, default=None, help="UDP 监听端口")
    record_parser.add_argument("--duration", type=float, default=None, help="采集秒数；配置为 null 则持续采集到 Ctrl+C")
    record_parser.add_argument("--output", type=Path, default=None, help="数采输出根目录")
    record_parser.add_argument("--flush-every", type=int, default=None, help="每多少个有效包 flush 一次")
    record_parser.add_argument("--diagnostic-samples", type=int, default=None, help="最多保存多少个无法解码的数据报；0=不保存")
    record_parser.add_argument("--diagnostic-bytes", type=int, default=None, help="无效包日志最多显示的头部字节数")
    record_parser.add_argument("--no-raw", action="store_true", default=None, help="不保存 raw_packets.bin")
    record_parser.add_argument("--no-jsonl", action="store_true", default=None, help="不保存 frames.jsonl")

    teleop_parser = subparsers.add_parser("teleop", help="启动动捕遥操作节点")
    teleop_parser.add_argument("--device-mode", choices=("glove", "controller", "auto"), default=None, help="VR 设备模式：glove=手套，controller=手柄，auto=按已验证帧头分派")
    teleop_parser.add_argument("--skip-check", action="store_true", default=None, help="跳过启动前依赖检查")
    teleop_parser.add_argument("--control-side", choices=("left", "right", "both"), default=None, help="实时动捕控制侧")
    teleop_parser.add_argument("--control-mode", choices=("skeleton", "direct", "ik", "raise_template"), default=None, help="skeleton=默认骨架自动映射；direct/ik/raise_template=legacy 高级兼容")
    teleop_parser.add_argument("--ik-backend", choices=("teleop_dls", "tiangong2pro_official", "ubtech_casadi"), default=None, help="IK 求解后端")
    teleop_parser.add_argument("--ik-task", choices=("pose", "position_only"), default=None, help="IK 任务类型：pose=官方完整 TCP 位姿 IK；position_only=实验性锁腕位置 IK")
    teleop_parser.add_argument("--position-scale", type=float, default=None, help="骨段方向映射增益，默认 1.0；legacy ik 中表示位置增益")
    teleop_parser.add_argument("--workspace-calibration", type=Path, default=None, help="仅 legacy ik 模式可用；skeleton 拒绝人工标定文件")
    teleop_parser.add_argument("--wrist-calibration", type=Path, default=None, help="仅 legacy ik 模式可用；skeleton 直接读取骨架腕点")
    teleop_parser.add_argument("--hand-only", action="store_true", default=None, help="只运行双手 IK，强制停发手臂和腰部")
    teleop_parser.add_argument("--orientation-scale", type=float, default=None, help="相对手部旋转增益，默认 1.0")
    teleop_parser.add_argument("--safe-teleop", choices=("true", "false"), default=None, help="是否启用低速安全遥操作门控")
    teleop_parser.add_argument("--safe-rate", type=float, default=None, help="安全模式下最大命令发布频率")
    teleop_parser.add_argument("--max-joint-delta-per-step", type=float, default=None, help="实时遥操作单次下发最大关节变化")
    teleop_parser.add_argument("--max-command-delta-from-baseline", type=float, default=None, help="实时遥操作相对启动姿态的最大关节偏移")
    add_teleop_startup_args(teleop_parser)
    add_timing_args(teleop_parser, live=True)
    add_skeleton_retarget_args(teleop_parser)
    add_hand_ik_launcher_args(teleop_parser)
    teleop_parser.add_argument("--robot-command", choices=("true", "false"), default=None, help="是否向 /arm/cmd_pos 下发命令")

    arm_test_parser = subparsers.add_parser(
        "arm-test",
        aliases=("robot-test",),
        help="手臂与双灵巧手 rosbridge 闭环测试",
    )
    arm_test_parser.add_argument("--host", default=None, help="rosbridge 地址")
    arm_test_parser.add_argument("--port", type=int, default=None, help="rosbridge 端口")
    arm_test_parser.add_argument("--side", choices=("left", "right", "both"), default=None, help="测试手臂")
    arm_test_parser.add_argument("--scale", type=float, default=None, help="参考 init_position 目标的插值比例")
    arm_test_parser.add_argument("--min-delta", type=float, default=None, help="至少制造的单关节目标变化")
    arm_test_parser.add_argument("--max-delta", type=float, default=None, help="单关节最大目标变化限幅")
    arm_test_parser.add_argument("--spd", type=float, default=None, help="电机速度参数")
    arm_test_parser.add_argument("--cur", type=float, default=None, help="电机电流参数")
    arm_test_parser.add_argument("--hold-seconds", type=float, default=None, help="动作目标保持时间")
    arm_test_parser.add_argument("--restore-seconds", type=float, default=None, help="结束姿态发布后的观察时间")
    arm_test_parser.add_argument("--finish-pose", choices=("initial", "baseline", "natural_down"), default=None, help="测试结束后固定到的姿态；initial/baseline=测试开始时 /arm/status 读到的位置")
    arm_test_parser.add_argument("--connect-timeout", type=float, default=None, help="rosbridge 连接超时秒数")
    arm_test_parser.add_argument("--status-timeout", type=float, default=None, help="等待 /arm/status 完整反馈的超时秒数")
    arm_test_parser.add_argument("--feedback-threshold", type=float, default=None, help="判定手臂有动作的最小反馈变化")
    arm_test_parser.add_argument("--hand-delta", type=float, default=None, help="双手 1..5 号通道的小幅握紧变化")
    arm_test_parser.add_argument("--hand-feedback-threshold", type=float, default=None, help="判定灵巧手有动作的最小反馈变化")
    arm_test_parser.add_argument("--arm-only", action="store_true", default=None, help="仅执行旧版手臂测试")
    arm_test_parser.add_argument("--execute", action="store_true", default=None, help="真正发布控制命令")
    arm_test_parser.add_argument("--dry-run", action="store_true", default=None, help="只预览目标，不发布控制命令")
    arm_test_parser.add_argument("--yes", action="store_true", default=None, help="跳过二次确认，配合 --execute 使用")

    replay_parser = subparsers.add_parser("replay", help="回放 VR UDP 数采动作")
    replay_parser.add_argument("input", nargs="?", type=Path, default=None, help="数采 CSV 文件或目录；不传则走配置")
    replay_parser.add_argument("--host", default=None, help="rosbridge 地址")
    replay_parser.add_argument("--port", type=int, default=None, help="rosbridge 端口")
    replay_parser.add_argument("--control-side", choices=("left", "right", "both"), default=None, help="回放控制侧")
    replay_parser.add_argument("--control-mode", choices=("skeleton", "direct", "ik", "raise_template"), default=None, help="skeleton=默认骨架自动映射；direct/ik/raise_template=legacy 高级兼容")
    replay_parser.add_argument("--ik-backend", choices=("teleop_dls", "tiangong2pro_official", "ubtech_casadi"), default=None, help="IK 求解后端")
    replay_parser.add_argument("--ik-task", choices=("pose", "position_only"), default=None, help="IK 任务类型：pose=官方完整 TCP 位姿 IK；position_only=实验性锁腕位置 IK")
    replay_parser.add_argument("--rate", "--control-rate", type=float, default=None, help="回放频率，默认 100Hz")
    replay_parser.add_argument("--max-frames", type=int, default=None, help="最多回放/预览多少帧")
    replay_parser.add_argument("--stride", type=int, default=None, help="按帧间隔抽样")
    replay_parser.add_argument("--position-scale", type=float, default=None, help="骨段方向映射增益，默认 1.0；legacy ik 中表示位置增益")
    replay_parser.add_argument("--workspace-calibration", type=Path, default=None, help="仅 legacy ik 模式可用；skeleton 拒绝人工标定文件")
    replay_parser.add_argument("--wrist-calibration", type=Path, default=None, help="仅 legacy ik 模式可用；skeleton 直接读取骨架腕点")
    replay_parser.add_argument("--hand-only", action="store_true", default=None, help="只回放双手 IK，强制停发手臂和腰部")
    replay_parser.add_argument("--orientation-scale", type=float, default=None, help="相对手部旋转增益，默认 1.0")
    replay_parser.add_argument("--max-position-step", type=float, default=None, help="单帧手部位置跳变阈值，单位 m")
    replay_parser.add_argument("--max-angular-step", type=float, default=None, help="单帧手部姿态跳变阈值，单位 rad")
    replay_parser.add_argument("--max-joint-delta-per-step", type=float, default=None, help="IK 命令单帧关节限幅")
    replay_parser.add_argument("--max-command-delta-from-baseline", type=float, default=None, help="相对 baseline 的最大关节偏移")
    add_skeleton_retarget_args(replay_parser)
    add_teleop_startup_args(replay_parser)
    add_timing_args(replay_parser, live=False)
    add_hand_ik_launcher_args(replay_parser)
    replay_parser.add_argument("--spd", type=float, default=None, help="电机速度参数")
    replay_parser.add_argument("--cur", type=float, default=None, help="电机电流参数")
    replay_parser.add_argument("--connect-timeout", type=float, default=None, help="rosbridge 连接超时秒数")
    replay_parser.add_argument("--status-timeout", type=float, default=None, help="等待 /arm/status 完整反馈的超时秒数")
    replay_parser.add_argument("--restore-seconds", type=float, default=None, help="恢复 baseline 后观察时间")
    replay_parser.add_argument("--prepose", choices=("none", "natural_down"), default=None, help="回放前先进入的机器人基线姿态")
    replay_parser.add_argument("--prepose-seconds", type=float, default=None, help="进入回放基线姿态的等待时间")
    replay_parser.add_argument("--prepose-steps", type=int, default=None, help="进入回放基线姿态的插值步数")
    replay_parser.add_argument("--use-robot-status", action="store_true", default=None, help="dry-run 时也读取真实机器人 baseline，不发布命令")
    replay_parser.add_argument("--allow-risk", action="store_true", default=None, help="允许超过默认数据/命令风险阈值")
    replay_parser.add_argument("--streaming", action="store_true", default=None, help="execute+ik 时边算 IK 边发布")
    replay_parser.add_argument("--execute", action="store_true", default=None, help="真正发布回放命令")
    replay_parser.add_argument("--dry-run", action="store_true", default=None, help="仅离线预览，覆盖配置中的 execute/use_robot_status，不连接机器人")
    replay_parser.add_argument("--yes", action="store_true", default=None, help="跳过二次确认，配合 --execute 使用")

    args = parser.parse_args()
    if args.command and args.command.startswith("legacy-"):
        args.command = args.command.removeprefix("legacy-")
    try:
        return apply_config(args)
    except ValueError as exc:
        parser.error(str(exc))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with path.expanduser().open("r", encoding="utf-8") as file:
        user_config = json.load(file)
    if not isinstance(user_config, dict):
        raise ValueError(f"配置文件必须是 JSON object: {path}")
    return deep_merge(DEFAULT_CONFIG, user_config)


def cfg_get(config: dict[str, Any], section: str, key: str) -> Any:
    return config.get(section, {}).get(key, DEFAULT_CONFIG[section][key])


def coalesce(current: Any, default: Any) -> Any:
    return default if current is None else current


def resolve_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    args.config = config_path
    args.launcher_config = config

    threads = coalesce(getattr(args, "blas_threads", None), config["runtime"].get("blas_threads", 1))
    if isinstance(threads, bool) or int(threads) != threads or int(threads) < 1:
        raise ValueError("blas_threads 必须为正整数")
    args.blas_threads = int(threads)
    # The launcher is stdlib-only; child NumPy/Pinocchio imports see these caps.
    # This changes this process and its children, not the system environment.
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.blas_threads)

    args.python = coalesce(args.python, cfg_get(config, "runtime", "python")) or sys.executable
    args.conda_env = coalesce(args.conda_env, cfg_get(config, "runtime", "conda_env"))
    if args.no_pause is None:
        args.no_pause = not bool(cfg_get(config, "runtime", "pause_after_action"))

    command = args.command
    if command in (None, "menu"):
        return args

    if command in {"wrist-calibrate", "workspace-calibrate"}:
        apply_calibration_config(args, config)
        return args

    if command == "check":
        args.teleop = coalesce(args.teleop, cfg_get(config, "environment", "basic_include_teleop"))
        args.ik_backend = coalesce(args.ik_backend, cfg_get(config, "teleop", "ik_backend"))
        return args

    if command == "test":
        args.port = coalesce(args.port, cfg_get(config, "environment", "self_test_port"))
        args.output = resolve_path(coalesce(args.output, cfg_get(config, "environment", "self_test_output")))
        return args

    if command == "simulate":
        simulation = config["simulation"]
        supplied_output = getattr(args, "output", None)
        args.output = Path(supplied_output).expanduser().resolve() if supplied_output is not None else resolve_path(simulation["output"])
        for name in ("control_hz", "physics_substeps", "duration", "profile", "max_joint_velocity"):
            setattr(args, name, coalesce(getattr(args, name, None), simulation[name]))
        args.realtime = bool(getattr(args, "realtime", False))
        return args

    if command == "visualize":
        args.input = args.input.expanduser().resolve()
        args.output = args.output.expanduser().resolve()
        return args

    if command == "record":
        args.device_mode = coalesce(args.device_mode, cfg_get(config, "record", "device_mode"))
        args.ip = coalesce(args.ip, cfg_get(config, "record", "ip"))
        args.port = coalesce(args.port, cfg_get(config, "record", "port"))
        args.duration = coalesce(args.duration, cfg_get(config, "record", "duration"))
        args.output = resolve_path(coalesce(args.output, cfg_get(config, "record", "output")))
        args.flush_every = coalesce(args.flush_every, cfg_get(config, "record", "flush_every"))
        args.diagnostic_samples = coalesce(args.diagnostic_samples, cfg_get(config, "record", "diagnostic_samples"))
        args.diagnostic_bytes = coalesce(args.diagnostic_bytes, cfg_get(config, "record", "diagnostic_bytes"))
        args.no_raw = coalesce(args.no_raw, not bool(cfg_get(config, "record", "save_raw")))
        args.no_jsonl = coalesce(args.no_jsonl, not bool(cfg_get(config, "record", "save_jsonl")))
        return args

    if command == "teleop":
        apply_teleop_config(args, config)
        return args

    if command in {"arm-test", "robot-test"}:
        apply_arm_test_config(args, config)
        return args

    if command == "replay":
        apply_replay_config(args, config)
        return args

    return args


def config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def apply_calibration_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    section = "workspace_calibration" if args.command == "workspace-calibrate" else "wrist_calibration"
    for key in DEFAULT_CONFIG[section]:
        if key != "runtime_path":
            setattr(args, key, coalesce(getattr(args, key, None), cfg_get(config, section, key)))
    for name in ("input", "capture_output", "calibration_output"):
        setattr(args, name, resolve_path(getattr(args, name, None)))
    if args.command == "workspace-calibrate":
        args.wrist_calibration = resolve_path(coalesce(
            getattr(args, "wrist_calibration", None),
            cfg_get(config, "wrist_calibration", "calibration_output"),
        ))


def apply_teleop_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.device_mode = coalesce(args.device_mode, cfg_get(config, "teleop", "device_mode"))
    args.skip_check = coalesce(args.skip_check, cfg_get(config, "teleop", "skip_check"))
    args.control_side = coalesce(args.control_side, cfg_get(config, "teleop", "control_side"))
    args.control_mode = coalesce(args.control_mode, cfg_get(config, "teleop", "control_mode"))
    args.ik_backend = coalesce(args.ik_backend, cfg_get(config, "teleop", "ik_backend"))
    args.ik_task = coalesce(args.ik_task, cfg_get(config, "teleop", "ik_task"))
    args.position_scale = coalesce(args.position_scale, cfg_get(config, "teleop", "position_scale"))
    args.workspace_calibration = resolve_path(
        coalesce(args.workspace_calibration, cfg_get(config, "workspace_calibration", "runtime_path"))
    )
    args.wrist_calibration = resolve_path(
        coalesce(args.wrist_calibration, cfg_get(config, "wrist_calibration", "runtime_path"))
    )
    args.hand_only = config_bool(coalesce(args.hand_only, cfg_get(config, "teleop", "hand_only")))
    args.orientation_scale = coalesce(args.orientation_scale, cfg_get(config, "teleop", "orientation_scale"))
    safe_teleop = coalesce(args.safe_teleop, cfg_get(config, "teleop", "safe_teleop"))
    args.safe_teleop = "true" if config_bool(safe_teleop) else "false"
    args.safe_rate = coalesce(args.safe_rate, cfg_get(config, "teleop", "safe_rate"))
    apply_timing_config(args, config, "teleop")
    args.max_joint_delta_per_step = coalesce(args.max_joint_delta_per_step, cfg_get(config, "teleop", "max_joint_delta_per_step"))
    args.max_command_delta_from_baseline = coalesce(
        args.max_command_delta_from_baseline,
        cfg_get(config, "teleop", "max_command_delta_from_baseline"),
    )
    apply_teleop_startup_config(args, config)
    apply_skeleton_retarget_config(args, config, "teleop")
    apply_hand_ik_config(args, config)
    robot_command = coalesce(args.robot_command, cfg_get(config, "teleop", "robot_command"))
    args.robot_command = "true" if config_bool(robot_command) else "false"
    validate_mapping_options(args)


def apply_arm_test_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    args.host = coalesce(args.host, cfg_get(config, "rosbridge", "host"))
    args.port = coalesce(args.port, cfg_get(config, "rosbridge", "port"))
    args.side = coalesce(args.side, cfg_get(config, "arm_test", "side"))
    args.scale = coalesce(args.scale, cfg_get(config, "arm_test", "scale"))
    args.min_delta = coalesce(args.min_delta, cfg_get(config, "arm_test", "min_delta"))
    args.max_delta = coalesce(args.max_delta, cfg_get(config, "arm_test", "max_delta"))
    args.spd = coalesce(args.spd, cfg_get(config, "arm_test", "spd"))
    args.cur = coalesce(args.cur, cfg_get(config, "arm_test", "cur"))
    args.hold_seconds = coalesce(args.hold_seconds, cfg_get(config, "arm_test", "hold_seconds"))
    args.restore_seconds = coalesce(args.restore_seconds, cfg_get(config, "arm_test", "restore_seconds"))
    args.finish_pose = coalesce(args.finish_pose, cfg_get(config, "arm_test", "finish_pose"))
    args.connect_timeout = coalesce(args.connect_timeout, cfg_get(config, "rosbridge", "connect_timeout"))
    args.status_timeout = coalesce(args.status_timeout, cfg_get(config, "rosbridge", "status_timeout"))
    args.feedback_threshold = coalesce(args.feedback_threshold, cfg_get(config, "arm_test", "feedback_threshold"))
    args.hand_delta = coalesce(args.hand_delta, cfg_get(config, "arm_test", "hand_delta"))
    args.hand_feedback_threshold = coalesce(
        args.hand_feedback_threshold,
        cfg_get(config, "arm_test", "hand_feedback_threshold"),
    )
    args.arm_only = coalesce(args.arm_only, cfg_get(config, "arm_test", "arm_only"))
    args.execute = False if getattr(args, "dry_run", None) else coalesce(args.execute, cfg_get(config, "arm_test", "execute"))
    args.yes = coalesce(args.yes, cfg_get(config, "arm_test", "yes"))


def apply_replay_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    configured_input = resolve_path(cfg_get(config, "replay", "input"))
    args.input = coalesce(args.input, configured_input)
    args.host = coalesce(args.host, cfg_get(config, "rosbridge", "host"))
    args.port = coalesce(args.port, cfg_get(config, "rosbridge", "port"))
    args.control_side = coalesce(args.control_side, cfg_get(config, "replay", "control_side"))
    args.control_mode = coalesce(args.control_mode, cfg_get(config, "replay", "control_mode"))
    args.ik_backend = coalesce(args.ik_backend, cfg_get(config, "replay", "ik_backend"))
    args.ik_task = coalesce(args.ik_task, cfg_get(config, "replay", "ik_task"))
    args.rate = coalesce(args.rate, cfg_get(config, "replay", "rate"))
    apply_timing_config(args, config, "replay")
    args.max_frames = coalesce(args.max_frames, cfg_get(config, "replay", "max_frames"))
    args.stride = coalesce(args.stride, cfg_get(config, "replay", "stride"))
    args.position_scale = coalesce(args.position_scale, cfg_get(config, "replay", "position_scale"))
    args.workspace_calibration = resolve_path(
        coalesce(args.workspace_calibration, cfg_get(config, "workspace_calibration", "runtime_path"))
    )
    args.wrist_calibration = resolve_path(
        coalesce(args.wrist_calibration, cfg_get(config, "wrist_calibration", "runtime_path"))
    )
    args.hand_only = config_bool(coalesce(args.hand_only, cfg_get(config, "replay", "hand_only")))
    args.orientation_scale = coalesce(args.orientation_scale, cfg_get(config, "replay", "orientation_scale"))
    args.max_position_step = coalesce(args.max_position_step, cfg_get(config, "replay", "max_position_step"))
    args.max_angular_step = coalesce(args.max_angular_step, cfg_get(config, "replay", "max_angular_step"))
    args.max_joint_delta_per_step = coalesce(args.max_joint_delta_per_step, cfg_get(config, "replay", "max_joint_delta_per_step"))
    args.max_command_delta_from_baseline = coalesce(args.max_command_delta_from_baseline, cfg_get(config, "replay", "max_command_delta_from_baseline"))
    apply_skeleton_retarget_config(args, config, "replay")
    apply_hand_ik_config(args, config)
    args.spd = coalesce(args.spd, cfg_get(config, "replay", "spd"))
    args.cur = coalesce(args.cur, cfg_get(config, "replay", "cur"))
    args.connect_timeout = coalesce(args.connect_timeout, cfg_get(config, "rosbridge", "connect_timeout"))
    args.status_timeout = coalesce(args.status_timeout, cfg_get(config, "rosbridge", "status_timeout"))
    args.restore_seconds = coalesce(args.restore_seconds, cfg_get(config, "replay", "restore_seconds"))
    args.prepose = coalesce(args.prepose, cfg_get(config, "replay", "prepose"))
    args.prepose_seconds = coalesce(args.prepose_seconds, cfg_get(config, "replay", "prepose_seconds"))
    args.prepose_steps = coalesce(args.prepose_steps, cfg_get(config, "replay", "prepose_steps"))
    args.use_robot_status = coalesce(args.use_robot_status, cfg_get(config, "replay", "use_robot_status"))
    args.allow_risk = coalesce(args.allow_risk, cfg_get(config, "replay", "allow_risk"))
    args.streaming = coalesce(args.streaming, cfg_get(config, "replay", "streaming"))
    args.execute = coalesce(args.execute, cfg_get(config, "replay", "execute"))
    if getattr(args, "dry_run", False):
        args.execute = False
        args.use_robot_status = False
    args.yes = coalesce(args.yes, cfg_get(config, "replay", "yes"))
    apply_teleop_startup_config(args, config, "replay")
    validate_mapping_options(args)


def validate_mapping_options(args: argparse.Namespace) -> None:
    if args.control_mode != "skeleton":
        return
    if args.workspace_calibration is not None or args.wrist_calibration is not None:
        raise ValueError("skeleton 模式自动读取完整骨架，不接受人工标定路径；移除 runtime_path/标定参数，或显式选择 legacy --control-mode ik")
    if args.ik_backend != "teleop_dls":
        raise ValueError("skeleton 模式使用纯 Pinocchio teleop_dls 后端；旧官方后端请显式选择 legacy --control-mode ik")
    if args.ik_task != "pose":
        raise ValueError("skeleton 模式联合映射骨段方向与手部姿态，必须使用 --ik-task pose")


def command_prefix(args: argparse.Namespace) -> list[str]:
    if args.conda_env:
        current_env = os.environ.get("CONDA_DEFAULT_ENV")
        if current_env == str(args.conda_env):
            return [sys.executable]
        return ["conda", "run", "-n", str(args.conda_env), "python"]
    return [str(args.python)]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少入口文件: {path}")


def check_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_environment(include_teleop: bool = False, ik_backend: str = "teleop_dls") -> int:
    print(f"ROOT_DIR={ROOT_DIR}")
    print(f"SOURCE_DIR={SOURCE_DIR}")
    print(f"PYTHON={sys.executable}")
    print(f"CONFIG={CONFIG_PATH}")

    ok = True
    for path in (
        SOURCE_DIR,
        RECORDER_ENTRY,
        TELEOP_ENTRY,
        ARM_TEST_ENTRY,
        REPLAY_ENTRY,
        PROTOCOL_ENTRY,
        VISUALIZER_ENTRY,
    ):
        exists = path.exists()
        print(f"CHECK_PATH {'OK' if exists else 'FAIL'} {path}")
        ok = ok and exists

    modules = list(REQUIRED_MODULES)
    if include_teleop:
        modules.extend(TELEOP_REQUIRED_MODULES)
        if ik_backend in {"tiangong2pro_official", "ubtech_casadi"}:
            modules.append("casadi")

    for module_name in modules:
        exists = check_import(module_name)
        print(f"CHECK_MODULE {'OK' if exists else 'FAIL'} {module_name}")
        ok = ok and exists

    if not ok:
        print("CHECK_RESULT FAIL")
        return 1

    print("CHECK_RESULT OK")
    return 0


def run_child(command: list[str]) -> int:
    print("RUN", " ".join(str(part) for part in command))
    return subprocess.call(command, cwd=SOURCE_DIR)


def run_record(args: argparse.Namespace) -> int:
    require_file(RECORDER_ENTRY)
    command = command_prefix(args) + [
        str(RECORDER_ENTRY),
        "--device-mode",
        str(args.device_mode),
        "--ip",
        str(args.ip),
        "--port",
        str(args.port),
        "--output",
        str(args.output),
        "--flush-every",
        str(args.flush_every),
        "--diagnostic-samples",
        str(args.diagnostic_samples),
        "--diagnostic-bytes",
        str(args.diagnostic_bytes),
    ]
    if args.duration is not None:
        command += ["--duration", str(args.duration)]
    if args.no_raw:
        command.append("--no-raw")
    if args.no_jsonl:
        command.append("--no-jsonl")
    return run_child(command)


def run_visualize(args: argparse.Namespace) -> int:
    require_file(VISUALIZER_ENTRY)
    command = [
        sys.executable,
        str(VISUALIZER_ENTRY),
        str(args.input),
        "--output", str(args.output),
        "--session", str(args.session),
        "--max-points", str(args.max_points),
    ]
    if args.open:
        command.append("--open")
    return run_child(command)


def effective_hand_control_json(args: argparse.Namespace) -> str:
    settings = copy.deepcopy(args.launcher_config["hand_control"])
    if args.hand_ik_compute != "true":
        settings["publish"] = False
    return json.dumps(settings, separators=(",", ":"))


def run_teleop(args: argparse.Namespace) -> int:
    validate_mapping_options(args)
    require_file(TELEOP_ENTRY)
    if args.device_mode == "controller":
        print(
            "TELEOP_UNSUPPORTED_DEVICE_MODE controller: 已识别现场 C8 FF 帧，"
            "但厂商字段布局尚未验证。为避免生成错误关节命令，实时遥操作已拒绝启动。"
        )
        print(
            "请先执行: python run_ik.py record --device-mode controller --duration 10"
        )
        return 2
    if not args.skip_check:
        check_code = check_environment(include_teleop=True, ik_backend=args.ik_backend)
        if check_code != 0:
            print("遥操作依赖检查未通过。若你已在 ROS2 环境中启动，可在配置 teleop.skip_check=true 跳过。")
            return check_code
    command = command_prefix(args) + [
        str(TELEOP_ENTRY),
        "--device-mode", str(args.device_mode),
        "--control-side", str(args.control_side),
        "--control-mode", str(args.control_mode),
        "--ik-backend", str(args.ik_backend),
        "--ik-task", str(args.ik_task),
        "--position-scale", str(args.position_scale),
        "--orientation-scale", str(args.orientation_scale),
        "--safe-teleop", str(args.safe_teleop),
        "--safe-rate", str(args.safe_rate),
        "--max-joint-delta-per-step", str(args.max_joint_delta_per_step),
        "--max-command-delta-from-baseline", str(args.max_command_delta_from_baseline),
        "--hand-control-json", effective_hand_control_json(args),
        "--robot-command", str(args.robot_command),
    ]
    if args.workspace_calibration is not None:
        require_file(args.workspace_calibration)
        command.extend(("--workspace-calibration", str(args.workspace_calibration)))
    if args.wrist_calibration is not None:
        require_file(args.wrist_calibration)
        command.extend(("--wrist-calibration", str(args.wrist_calibration)))
    append_teleop_startup_cli(command, args)
    append_timing_cli(command, args, live=True)
    append_skeleton_retarget_cli(command, args)
    append_hand_ik_cli(command, args)
    if args.hand_only:
        command.append("--hand-only")
    return run_child(command)


def run_arm_test(args: argparse.Namespace) -> int:
    require_file(ARM_TEST_ENTRY)
    command = command_prefix(args) + [
        str(ARM_TEST_ENTRY),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--side",
        str(args.side),
        "--scale",
        str(args.scale),
        "--min-delta",
        str(args.min_delta),
        "--max-delta",
        str(args.max_delta),
        "--spd",
        str(args.spd),
        "--cur",
        str(args.cur),
        "--hold-seconds",
        str(args.hold_seconds),
        "--restore-seconds",
        str(args.restore_seconds),
        "--finish-pose",
        str(args.finish_pose),
        "--connect-timeout",
        str(args.connect_timeout),
        "--status-timeout",
        str(args.status_timeout),
        "--feedback-threshold",
        str(args.feedback_threshold),
        "--hand-delta",
        str(args.hand_delta),
        "--hand-feedback-threshold",
        str(args.hand_feedback_threshold),
    ]
    if args.arm_only:
        command.append("--arm-only")
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return run_child(command)


def is_replay_candidate(path: Path) -> bool:
    parts = set(path.resolve().parts)
    return "self_test" not in parts


def latest_capture_session(args: argparse.Namespace) -> Path | None:
    config = args.launcher_config
    roots = [
        resolve_path(cfg_get(config, "record", "output")),
        ROOT_DIR / "data" / "launcher_test",
        DEFAULT_CAPTURE_DIR,
    ]
    candidates: list[Path] = []
    for root in roots:
        if root is None or not root.exists() or not is_replay_candidate(root):
            continue
        candidates.extend(path for path in root.glob("vr_*.csv") if path.is_file() and is_replay_candidate(path))
        for path in root.iterdir():
            if path.is_dir() and is_replay_candidate(path):
                csv_files = sorted(
                    (item for item in path.glob("vr_*.csv") if is_replay_candidate(item)),
                    key=lambda item: item.stat().st_mtime,
                    reverse=True,
                )
                if csv_files:
                    candidates.append(csv_files[0])
                elif (path / "frames.jsonl").is_file():
                    candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def resolve_replay_input(args: argparse.Namespace) -> Path | None:
    if args.input is not None:
        return Path(args.input).expanduser().resolve()
    if bool(cfg_get(args.launcher_config, "replay", "use_latest_capture")):
        selected = latest_capture_session(args)
        if selected is not None:
            print(f"使用最新数采 session: {selected}")
        return selected
    return None


def run_replay(args: argparse.Namespace) -> int:
    validate_mapping_options(args)
    require_file(REPLAY_ENTRY)
    input_path = resolve_replay_input(args)
    if input_path is None:
        print("未配置 replay.input，也没有找到可回放的 vr_*.csv。请先数采，或在 run_ik_config.json 里设置 replay.input。")
        return 1

    command = command_prefix(args) + [
        str(REPLAY_ENTRY),
        str(input_path),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--control-side",
        str(args.control_side),
        "--control-mode",
        str(args.control_mode),
        "--ik-backend",
        str(args.ik_backend),
        "--ik-task",
        str(args.ik_task),
        "--rate",
        str(args.rate),
        "--stride",
        str(args.stride),
        "--position-scale",
        str(args.position_scale),
        "--orientation-scale",
        str(args.orientation_scale),
        "--max-position-step",
        str(args.max_position_step),
        "--max-angular-step",
        str(args.max_angular_step),
        "--max-joint-delta-per-step",
        str(args.max_joint_delta_per_step),
        "--max-command-delta-from-baseline",
        str(args.max_command_delta_from_baseline),
        "--hand-control-json",
        effective_hand_control_json(args),
        "--spd",
        str(args.spd),
        "--cur",
        str(args.cur),
        "--connect-timeout",
        str(args.connect_timeout),
        "--status-timeout",
        str(args.status_timeout),
        "--restore-seconds",
        str(args.restore_seconds),
        "--prepose",
        str(args.prepose),
        "--prepose-seconds",
        str(args.prepose_seconds),
        "--prepose-steps",
        str(args.prepose_steps),
    ]
    if args.workspace_calibration is not None:
        require_file(args.workspace_calibration)
        command.extend(("--workspace-calibration", str(args.workspace_calibration)))
    if args.wrist_calibration is not None:
        require_file(args.wrist_calibration)
        command.extend(("--wrist-calibration", str(args.wrist_calibration)))
    append_skeleton_retarget_cli(command, args)
    append_teleop_startup_cli(command, args)
    append_timing_cli(command, args, live=False)
    append_hand_ik_cli(command, args)
    if args.hand_only:
        command.append("--hand-only")
    if args.max_frames is not None:
        command += ["--max-frames", str(args.max_frames)]
    if args.use_robot_status:
        command.append("--use-robot-status")
    if args.allow_risk:
        command.append("--allow-risk")
    if args.streaming:
        command.append("--streaming")
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return run_child(command)


def append_calibration_arguments(command: list[str], args: argparse.Namespace, names: Iterable[str]) -> None:
    for name in names:
        value = getattr(args, name, None)
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
        elif value is not None:
            command.extend((flag, str(value)))


def run_workspace_calibration(args: argparse.Namespace) -> int:
    entry = SOURCE_DIR / "teleop_workspace_calibration.py"
    require_file(entry)
    command = command_prefix(args) + [str(entry), "--stage", args.stage]
    names = [name for name in DEFAULT_CONFIG["workspace_calibration"] if name != "runtime_path"]
    append_calibration_arguments(command, args, [*names, "input", "wrist_calibration"])
    return run_child(command)


def run_wrist_calibration(args: argparse.Namespace) -> int:
    entry = SOURCE_DIR / "calibrate_wrist.py"
    require_file(entry)
    command = command_prefix(args) + [str(entry), args.stage]
    if args.stage in {"analyze", "validate"}:
        if args.input is None:
            raise ValueError("腕点 analyze/validate 阶段需要 --input")
        command.append(str(args.input))
    if args.stage in {"all", "capture"}:
        append_calibration_arguments(command, args, ("device_mode", "ip", "port", "side", "group_seconds"))
        command.extend(("--output", str(args.capture_output)))
    if args.stage in {"all", "analyze"}:
        append_calibration_arguments(command, args, (
            "calibration_output", "min_samples", "min_rotation_excitation", "min_rotation_rank",
            "max_outlier_fraction", "max_pivot_rms", "max_pivot_p95", "max_wrist_validation_rms", "huber_delta",
        ))
    if args.stage == "validate":
        append_calibration_arguments(command, args, ("side",))
    return run_child(command)


def run_combined_calibration(args: argparse.Namespace) -> int:
    wrist_args = argparse.Namespace(**vars(args))
    wrist_args.command = "wrist-calibrate"
    wrist_args.stage = "all"
    apply_calibration_config(wrist_args, args.launcher_config)
    wrist_path = wrist_args.calibration_output
    if wrist_path.is_file():
        try:
            payload = json.loads(wrist_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            payload = {}
        if (
            isinstance(payload, dict)
            and payload.get("valid") is True
            and payload.get("control_point") == "anatomical_wrist"
        ):
            wrist_args.stage = "validate"
            wrist_args.input = wrist_path
    code = run_wrist_calibration(wrist_args)
    if code != 0:
        return code

    workspace_args = argparse.Namespace(**vars(args))
    workspace_args.command = "workspace-calibrate"
    workspace_args.stage = "all"
    apply_calibration_config(workspace_args, args.launcher_config)
    workspace_args.wrist_calibration = wrist_path
    final_path = workspace_args.calibration_output
    final_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = final_path.with_name(f".{final_path.stem}.{time.time_ns()}.candidate{final_path.suffix}")
    workspace_args.calibration_output = candidate_path
    try:
        code = run_workspace_calibration(workspace_args)
        if code != 0:
            return code
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("valid") is not True:
            return 2
        candidate_path.replace(final_path)
        print(f"COMBINED_CALIBRATION_OUTPUT {final_path}")
        return 0
    finally:
        candidate_path.unlink(missing_ok=True)


def run_simulation(args: argparse.Namespace) -> int:
    entry = SOURCE_DIR / "teleop_simulation.py"
    require_file(entry)
    command = command_prefix(args) + [str(entry), "--output", str(args.output),
        "--control-hz", str(args.control_hz), "--physics-substeps", str(args.physics_substeps),
        "--duration", str(args.duration), "--profile", str(args.profile),
        "--max-joint-velocity", str(args.max_joint_velocity)]
    if args.realtime:
        command.append("--realtime")
    for scenario in getattr(args, "scenario", None) or []:
        command.extend(("--scenario", scenario))
    return run_child(command)


def run_compile_check(args: argparse.Namespace) -> int:
    files = [
        PROTOCOL_ENTRY,
        SOURCE_DIR / "mocap_csv_format.py",
        RECORDER_ENTRY,
        TELEOP_ENTRY,
        ARM_TEST_ENTRY,
        REPLAY_ENTRY,
        SOURCE_DIR / "teleop_direct_retarget.py",
        SOURCE_DIR / "teleop_control_safety.py",
        SOURCE_DIR / "teleop_hand_ik.py",
        SOURCE_DIR / "teleop_hand_control.py",
        SOURCE_DIR / "revo2_hand_ik_worker.py",
        SOURCE_DIR / "teleop_ik_adapters.py",
        SOURCE_DIR / "teleop_ik_core.py",
        SOURCE_DIR / "teleop_skeleton_retarget.py",
        SOURCE_DIR / "teleop_simulation.py",
        SOURCE_DIR / "teleop_ik_ubtech_casadi.py",
        SOURCE_DIR / "teleop_ik_types.py",
        SOURCE_DIR / "teleop_mocap_calibration.py",
        SOURCE_DIR / "teleop_wrist_calibration.py",
        SOURCE_DIR / "teleop_startup_pose.py",
        SOURCE_DIR / "teleop_raise_template.py",
        SOURCE_DIR / "teleop_robot_constants.py",
        SOURCE_DIR / "teleop_joint_schema.py",
        VISUALIZER_ENTRY,
    ]
    command = command_prefix(args) + ["-m", "py_compile", *[str(path) for path in files]]
    return run_child(command)


def build_fake_mocap_packet() -> bytes:
    module_name = "_launcher_self_test_protocol"
    protocol = sys.modules.get(module_name)
    if protocol is None:
        spec = importlib.util.spec_from_file_location(module_name, PROTOCOL_ENTRY)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 64 节点协议自测")
        protocol = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = protocol
        spec.loader.exec_module(protocol)
    positions = {
        "left_shoulder": (0.0, 0.22, 1.45), "right_shoulder": (0.0, -0.22, 1.45),
        "left_elbow": (0.0, 0.22, 1.12), "right_elbow": (0.0, -0.22, 1.12),
        "left_wrist": (0.0, 0.22, 0.78), "right_wrist": (0.0, -0.22, 0.78),
        "left_hand": (0.0, 0.22, 0.72), "right_hand": (0.0, -0.22, 0.72),
        "left_hip": (0.0, 0.14, 0.95), "right_hip": (0.0, -0.14, 0.95),
        "chest": (0.0, 0.0, 1.30), "waist": (0.0, 0.0, 0.95),
    }
    timestamp_us = time.time_ns() // 1000
    frame = protocol.MocapFrame(
        timestamp_us / 1e6, timestamp_us, timestamp_us / 1e6,
        tuple(protocol.JointPose(positions.get(name, (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0, 1.0))
              for name in protocol.MOCAP_JOINT_NAMES),
        "synthetic_localhost_self_test",
    )
    return protocol.encode_mocap_g2_frame(frame)


def send_fake_udp(port: int) -> None:
    time.sleep(0.5)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _ in range(30):
            packet = build_fake_mocap_packet()
            sock.sendto(packet, ("127.0.0.1", port))
            time.sleep(0.1)
        print(f"TEST_UDP_SENT bytes={len(packet)} port={port}")
    finally:
        sock.close()


def latest_session_meta(output: Path) -> dict[str, Any] | None:
    root = output.expanduser().resolve()
    if not root.exists():
        return None
    meta_files = sorted(root.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not meta_files:
        return None
    with meta_files[0].open("r", encoding="utf-8") as file:
        return json.load(file)


def run_udp_record_test(args: argparse.Namespace) -> int:
    thread = threading.Thread(target=send_fake_udp, args=(args.port,), daemon=True)
    thread.start()
    command = command_prefix(args) + [
        str(RECORDER_ENTRY),
        "--ip",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--duration",
        "4.0",
        "--output",
        str(args.output),
        "--flush-every",
        "1",
    ]
    code = run_child(command)
    if code != 0:
        return code

    meta = latest_session_meta(args.output)
    valid_packets = 0 if meta is None else int(meta.get("valid_packets", 0))
    if valid_packets <= 0:
        print("TEST_RESULT FAIL udp_record_valid_packets=0")
        return 1
    print(f"TEST_UDP_VALID_PACKETS {valid_packets}")
    return 0


def run_test(args: argparse.Namespace) -> int:
    check_code = check_environment(include_teleop=False)
    if check_code != 0:
        return check_code

    compile_code = run_compile_check(args)
    if compile_code != 0:
        print("TEST_RESULT FAIL compile")
        return compile_code

    record_code = run_udp_record_test(args)
    if record_code != 0:
        print("TEST_RESULT FAIL udp_record")
        return record_code

    print("TEST_RESULT OK")
    return 0


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except EOFError:
        if default is not None:
            print(str(default))
            return str(default)
        return ""
    if value == "" and default is not None:
        return str(default)
    return value


def prompt_device_mode(default: str) -> str:
    labels = {"1": "glove", "2": "controller", "3": "auto"}
    reverse = {value: key for key, value in labels.items()}
    while True:
        print("设备模式: 1=手套 glove，2=手柄 controller，3=自动识别 auto")
        value = prompt_text("选择设备模式", reverse.get(default, "1"))
        mode = labels.get(value, value.strip().lower())
        if mode in labels.values():
            return mode
        print("设备模式必须是 1/2/3 或 glove/controller/auto。")


def make_menu_args(base: argparse.Namespace, command: str, **values: Any) -> argparse.Namespace:
    args = argparse.Namespace(
        command=command,
        config=base.config,
        launcher_config=base.launcher_config,
        python=base.python,
        conda_env=base.conda_env,
        no_pause=base.no_pause,
        **values,
    )

    def ensure_none_fields(*names: str) -> None:
        for name in names:
            if not hasattr(args, name):
                setattr(args, name, None)

    if command == "arm-test":
        ensure_none_fields(
            "host",
            "port",
            "side",
            "scale",
            "min_delta",
            "max_delta",
            "spd",
            "cur",
            "hold_seconds",
            "restore_seconds",
            "finish_pose",
            "connect_timeout",
            "status_timeout",
            "feedback_threshold",
            "hand_delta",
            "hand_feedback_threshold",
            "arm_only",
            "execute",
            "dry_run",
            "yes",
        )
        apply_arm_test_config(args, base.launcher_config)
    elif command == "replay":
        ensure_none_fields(
            "input",
            "host",
            "port",
            "control_side",
            "control_mode",
            "ik_backend",
            "ik_task",
            "rate",
            "max_frames",
            "stride",
            "position_scale",
            "workspace_calibration",
            "wrist_calibration",
            "hand_only",
            "orientation_scale",
            "max_position_step",
            "max_angular_step",
            "max_joint_delta_per_step",
            "max_command_delta_from_baseline",
            "spd",
            "cur",
            "connect_timeout",
            "status_timeout",
            "restore_seconds",
            "prepose",
            "prepose_seconds",
            "prepose_steps",
            "use_robot_status",
            "allow_risk",
            "streaming",
            "execute",
            "dry_run",
            "yes",
        )
        ensure_none_fields(*SKELETON_CONFIG_KEYS)
        ensure_none_fields(*TELEOP_STARTUP_CONFIG_KEYS)
        ensure_none_fields(
            "hand_ik_compute",
            "hand_ik_sdk_root",
            "hand_ik_sdk_config",
            "hand_ik_library",
            "hand_ik_timeout",
            "hand_ik_diagnostic_csv",
        )
        apply_replay_config(args, base.launcher_config)
    elif command == "record":
        args.device_mode = coalesce(
            getattr(args, "device_mode", None),
            cfg_get(base.launcher_config, "record", "device_mode"),
        )
        args.ip = cfg_get(base.launcher_config, "record", "ip")
        args.port = cfg_get(base.launcher_config, "record", "port")
        args.duration = cfg_get(base.launcher_config, "record", "duration")
        args.output = resolve_path(cfg_get(base.launcher_config, "record", "output"))
        args.flush_every = cfg_get(base.launcher_config, "record", "flush_every")
        args.diagnostic_samples = cfg_get(base.launcher_config, "record", "diagnostic_samples")
        args.diagnostic_bytes = cfg_get(base.launcher_config, "record", "diagnostic_bytes")
        args.no_raw = not bool(cfg_get(base.launcher_config, "record", "save_raw"))
        args.no_jsonl = not bool(cfg_get(base.launcher_config, "record", "save_jsonl"))
    elif command == "teleop":
        ensure_none_fields(
            "device_mode",
            "skip_check",
            "control_side",
            "control_mode",
            "ik_backend",
            "ik_task",
            "position_scale",
            "workspace_calibration",
            "wrist_calibration",
            "hand_only",
            "orientation_scale",
            "safe_teleop",
            "safe_rate",
            "max_joint_delta_per_step",
            "max_command_delta_from_baseline",
            "robot_command",
        )
        ensure_none_fields(*TELEOP_STARTUP_CONFIG_KEYS)
        ensure_none_fields(*SKELETON_CONFIG_KEYS)
        ensure_none_fields(
            "hand_ik_compute",
            "hand_ik_sdk_root",
            "hand_ik_sdk_config",
            "hand_ik_library",
            "hand_ik_timeout",
            "hand_ik_diagnostic_csv",
        )
        apply_teleop_config(args, base.launcher_config)
    elif command == "test":
        args.port = cfg_get(base.launcher_config, "environment", "self_test_port")
        args.output = resolve_path(cfg_get(base.launcher_config, "environment", "self_test_output"))
    return args


def run_menu_teleop(base: argparse.Namespace) -> int:
    config = base.launcher_config
    device_mode = prompt_device_mode(str(cfg_get(config, "teleop", "device_mode")))
    print("遥操作范围: 1=双臂+双手，2=仅双手")
    hand_only_choice = prompt_text("选择遥操作模式", "2" if cfg_get(config, "teleop", "hand_only") else "1")
    if hand_only_choice not in {"1", "2"}:
        print("遥操作模式必须是 1 或 2。")
        return 1
    hand_only = hand_only_choice == "2"
    print(
        "实时动捕配置: "
        f"device_mode={device_mode}, "
        f"hand_only={hand_only}, "
        f"side={cfg_get(config, 'teleop', 'control_side')}, "
        f"control_mode={cfg_get(config, 'teleop', 'control_mode')}, "
        f"ik_backend={cfg_get(config, 'teleop', 'ik_backend')}, "
        f"ik_task={cfg_get(config, 'teleop', 'ik_task')}, "
        f"position_scale={cfg_get(config, 'teleop', 'position_scale')}, "
        f"orientation_scale={cfg_get(config, 'teleop', 'orientation_scale')}, "
        f"safe={cfg_get(config, 'teleop', 'safe_teleop')}, "
        f"safe_rate={cfg_get(config, 'teleop', 'safe_rate')}, "
        f"max_step={cfg_get(config, 'teleop', 'max_joint_delta_per_step')}, "
        f"max_baseline_delta={cfg_get(config, 'teleop', 'max_command_delta_from_baseline')}, "
        f"publish={cfg_get(config, 'teleop', 'robot_command')}, "
        f"rosbridge={cfg_get(config, 'rosbridge', 'host')}:{cfg_get(config, 'rosbridge', 'port')}"
    )
    return run_teleop(
        make_menu_args(base, "teleop", device_mode=device_mode, hand_only=hand_only)
    )


def run_menu_record(base: argparse.Namespace) -> int:
    config = base.launcher_config
    device_mode = prompt_device_mode(str(cfg_get(config, "record", "device_mode")))
    print(
        "数采配置: "
        f"device_mode={device_mode}, "
        f"UDP={cfg_get(config, 'record', 'ip')}:{cfg_get(config, 'record', 'port')}, "
        f"duration={cfg_get(config, 'record', 'duration')}, "
        f"output={resolve_path(cfg_get(config, 'record', 'output'))}"
    )
    return run_record(make_menu_args(base, "record", device_mode=device_mode))


def run_menu_replay(base: argparse.Namespace) -> int:
    config = base.launcher_config
    mode = "真机回放" if cfg_get(config, "replay", "execute") else "数据检查 + 命令预览"
    print(
        "回放配置: "
        f"mode={mode}, side={cfg_get(config, 'replay', 'control_side')}, "
        f"control_mode={cfg_get(config, 'replay', 'control_mode')}, "
        f"ik_backend={cfg_get(config, 'replay', 'ik_backend')}, "
        f"ik_task={cfg_get(config, 'replay', 'ik_task')}, "
        f"streaming={cfg_get(config, 'replay', 'streaming')}, "
        f"position_scale={cfg_get(config, 'replay', 'position_scale')}, "
        f"orientation_scale={cfg_get(config, 'replay', 'orientation_scale')}, "
        f"rate={cfg_get(config, 'replay', 'rate')}, max_frames={cfg_get(config, 'replay', 'max_frames')}, "
        f"rosbridge={cfg_get(config, 'rosbridge', 'host')}:{cfg_get(config, 'rosbridge', 'port')}"
    )
    return run_replay(make_menu_args(base, "replay"))


def run_menu_visualize(base: argparse.Namespace) -> int:
    return run_visualize(
        make_menu_args(
            base,
            "visualize",
        input=ROOT_DIR / "data" / "logs" / "mocap_ik_bridge.log",
            output=ROOT_DIR / "data" / "visualization" / "joint_commands.html",
            session="latest",
            max_points=5000,
            open=True,
        )
    )


def run_menu_arm_test(base: argparse.Namespace) -> int:
    config = base.launcher_config
    print(
        "手臂与双灵巧手链路检测会顺序发布小幅测试动作，并分别恢复测试开始时读取的基线。"
        f" side={cfg_get(config, 'arm_test', 'side')}, "
        f"finish_pose={cfg_get(config, 'arm_test', 'finish_pose')}, "
        f"rosbridge={cfg_get(config, 'rosbridge', 'host')}:{cfg_get(config, 'rosbridge', 'port')}"
    )
    return run_arm_test(make_menu_args(base, "arm-test"))


def print_menu(base: argparse.Namespace) -> None:
    print("\n=== IK 动捕遥操作控制台 ===")
    print(f"配置文件: {base.config}")
    print("默认：64 节点骨架自动映射；双臂自然下垂静止启动，无需转腕/六方向标定。")
    print("1. 环境检测")
    print("2. VR UDP 数采")
    print("3. 数采动作回放")
    print("4. 启动实时动捕遥操作")
    print("5. 查看最近一次关节角曲线")
    print("6. URDF 离线仿真与 IK 验证（无机器人连接）")
    print("0. 退出")


def print_environment_menu() -> None:
    print("\n=== 环境检测 ===")
    print("1. 基础环境检查")
    print("2. 完整自测：编译 + 模拟 UDP 数采")
    print("3. 手臂与双灵巧手链路检测：顺序执行小幅动作并自动复位")
    print("0. 返回主菜单")


def wait_after_action(base: argparse.Namespace) -> None:
    if not getattr(base, "no_pause", False):
        input("\n操作结束，按回车返回菜单...")


def run_environment_menu(base: argparse.Namespace) -> int:
    while True:
        print_environment_menu()
        choice = prompt_text("选择检测项", "0")
        if choice == "0":
            return 0
        if choice == "1":
            code = check_environment(
                include_teleop=cfg_get(base.launcher_config, "environment", "basic_include_teleop"),
                ik_backend=cfg_get(base.launcher_config, "teleop", "ik_backend"),
            )
        elif choice == "2":
            code = run_test(make_menu_args(base, "test"))
        elif choice == "3":
            code = run_menu_arm_test(base)
        else:
            print("未知选项。")
            continue

        print(f"MENU_ACTION_EXIT_CODE {code}")
        wait_after_action(base)


def run_interactive_menu(base: argparse.Namespace) -> int:
    while True:
        print_menu(base)
        choice = prompt_text("选择操作", "0")
        if choice == "0":
            return 0
        if choice == "1":
            code = run_environment_menu(base)
        elif choice == "2":
            code = run_menu_record(base)
        elif choice == "3":
            code = run_menu_replay(base)
        elif choice == "4":
            code = run_menu_teleop(base)
        elif choice == "5":
            code = run_menu_visualize(base)
        elif choice == "6":
            code = run_simulation(make_menu_args(base, "simulate"))
        else:
            print("未知选项。")
            continue

        print(f"MENU_ACTION_EXIT_CODE {code}")
        wait_after_action(base)


def main() -> int:
    args = parse_args()
    if args.command is None or args.command == "menu":
        return run_interactive_menu(args)
    if args.command == "check":
        return check_environment(include_teleop=args.teleop, ik_backend=args.ik_backend)
    if args.command == "test":
        return run_test(args)
    if args.command == "simulate":
        return run_simulation(args)
    if args.command == "visualize":
        return run_visualize(args)
    if args.command == "record":
        return run_record(args)
    if args.command == "teleop":
        return run_teleop(args)
    if args.command in {"arm-test", "robot-test"}:
        return run_arm_test(args)
    if args.command == "replay":
        return run_replay(args)
    if args.command == "calibrate":
        return run_combined_calibration(args)
    if args.command == "workspace-calibrate":
        return run_workspace_calibration(args)
    if args.command == "wrist-calibrate":
        return run_wrist_calibration(args)
    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n操作已停止。")
        raise SystemExit(130) from None
