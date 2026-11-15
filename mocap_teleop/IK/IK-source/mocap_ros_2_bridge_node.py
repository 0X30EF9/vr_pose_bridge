import argparse
import copy
import json
import logging
import socket
import time
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import roslibpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# 全局配置参数：调试和部署时优先只改这里
# ==============================================================================

# UDP 动捕接收参数
UDP_IP = "0.0.0.0"
UDP_PORT = 15000
UDP_PACKET_SIZE = 65535
MOCAP_DEVICE_MODE = "glove"
CONTROL_RATE_HZ = 20.0
TIME_WARNING_THRESHOLD_MS = (1.0 / CONTROL_RATE_HZ) * 1000.0 * 0.8

# 控制模式：direct 走骨架直映射演示包络；ik 走末端 IK；raise_template 走固定平举动作模板。
CONTROL_MODE = "ik"  # "direct" | "ik" | "raise_template"
IK_SOLVER_BACKEND = "tiangong2pro_official"  # "teleop_dls" | "tiangong2pro_official"
IK_TASK_MODE = "pose"  # "pose" | "position_only"
CONTROL_SIDE = "both"  # "left" | "right" | "both"

from teleop_mocap_protocol import (
    MocapPacketError,
    QUAT_ORDER,
    parse_mocap_packet,
    snapshot_mocap_data,
    to_jsonable,
)
from teleop_direct_retarget import (
    DirectRetargeter,
    add_direct_retarget_arguments,
    direct_retarget_config_from_args,
)
from teleop_hand_ik import (
    HandIkError,
    Revo2HandIkSidecar,
    add_hand_ik_arguments,
    hand_ik_config_from_args,
    hand_ik_enabled,
)
from teleop_hand_control import (
    HandControlError,
    apply_hand_command_safety,
    build_hand_joint_state_message,
    hand_control_config_from_mapping,
    map_hand_radians,
    parse_hand_state_message,
)
from teleop_ik_adapters import (
    build_rosbridge_motor_message,
    build_rosbridge_waist_message,
    select_command_slice,
)
from teleop_ik_types import IKCommand
from teleop_joint_schema import ARM_JOINT_NAMES as SCHEMA_ARM_JOINT_NAMES
from teleop_joint_schema import ARM_MOTOR_IDS as SCHEMA_ARM_MOTOR_IDS
from teleop_joint_schema import arm_joint_limits_from_urdf, arm_positions_from_status
from teleop_control_safety import feedback_is_fresh, require_fresh_feedback
from teleop_mocap_calibration import RelativeMocapTargetMapper, load_workspace_calibration
from teleop_wrist_calibration import (
    load_wrist_calibration,
    reconstruct_wrist_pose_from_mapping,
)
from teleop_wrist_safety import (
    WristBehaviorLimits,
    validate_wrist_behavior_candidate,
    wrist_position_errors,
)
from teleop_startup_pose import NaturalDownStartupConfig, NaturalDownStartupGate
from teleop_robot_constants import (
    ARM_MOTOR_TEMPLATES,
    HAND_CMD_TOPICS,
    HAND_JOINT_STATE_TYPE,
    HAND_STATE_TOPICS,
    POSITION_ONLY_LOCKED_JOINT_INDICES,
    WAIST_CMD_TOPIC,
    WAIST_CMD_TYPE,
    WAIST_ERROR_DESCRIPTIONS,
    WAIST_MOTOR_ID,
    WAIST_MOTOR_TEMPLATE,
    WAIST_STATUS_TOPIC,
    WAIST_STATUS_TYPE,
    build_relaxed_joint_limits,
)

# 仅表示动捕数据进入公共映射前的固定坐标变换；绝对世界外参不在现有设备能力范围内。
T_BASE_MOCAP = np.eye(4)

# 目标映射模式。relative 使用第一帧动捕手腕作为人体 home，只把后续相对增量叠加到机器人 home wrist。
MOCAP_TARGET_MODE = "relative"  # "relative" | "absolute"
MOCAP_RELATIVE_POSITION_SCALE = 1.0
MOCAP_RELATIVE_ORIENTATION_SCALE = 0.25
WORKSPACE_CALIBRATION_PATH: Path | None = None
WRIST_CALIBRATION_PATH: Path | None = None

# 机器人通信。真机链路固定使用 rosbridge，不通过 ROS2 命令话题直接下发。
COMMAND_BACKEND = "rosbridge"
ROSBRIDGE_HOST = "192.168.1.110"
ROSBRIDGE_PORT = 9091
ARM_STATUS_TOPIC = "/arm/status"
ARM_STATUS_TYPE = "bodyctrl_msgs/msg/MotorStatusMsg"
ARM_CMD_TOPIC = "/arm/cmd_pos"
ARM_CMD_TYPE = "bodyctrl_msgs/msg/CmdSetMotorPosition"
ROS2_JOINT_STATE_TOPIC = "/joint_states"
ROS2_JOINT_COMMAND_TOPIC = "/set_joint_positions"

# 安全参数。关节顺序需与 UBTECH_TainGong_ArmIK reduced robot 输出保持一致。
IK_ARM_JOINT_NAMES = list(SCHEMA_ARM_JOINT_NAMES)
ARM_MOTOR_IDS = list(SCHEMA_ARM_MOTOR_IDS)
MAX_JOINT_DELTA_PER_STEP = 0.08
SAFE_TELEOP_MODE = True
SAFE_COMMAND_RATE_HZ = 3.0
MAX_COMMAND_DELTA_FROM_BASELINE = 1.0
STARTUP_HOLD_SECONDS = 2.0
STARTUP_HUMAN_VERTICAL_COSINE = 0.78
STARTUP_HUMAN_MIN_WRIST_DROP_M = 0.30
STARTUP_HUMAN_MAX_LATERAL_OFFSET_M = 0.16
STARTUP_MAX_HOME_RMS_M = 0.025
DIRECT_RETARGET_RUNTIME_ARGS: argparse.Namespace | None = None
HAND_CONTROL_CONFIG = hand_control_config_from_mapping(None)

# 运行保护
REQUIRE_ARM_STATE_BEFORE_IK = True
DROP_STALE_MOCAP_SECONDS = 0.3
ARM_STATE_TIMEOUT_SECONDS = 0.5
WAIST_STATE_TIMEOUT_SECONDS = 0.5
ENABLE_IK_ERROR_LOG = False
ENABLE_ROBOT_COMMAND = True
HAND_ONLY_MODE = False

# 本地日志。三类核心日志：接收到的动捕数据、IK 解算结果、发送给机器人的数据。
ENABLE_LOCAL_LOG = True
LOG_TO_CONSOLE = False
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_FILE_NAME = "mocap_ik_bridge.log"
LOG_MOCAP_DATA = True
LOG_IK_RESULT = True
LOG_ROBOT_COMMAND = True
LOG_EVERY_N_FRAMES = 1  # 20Hz 长时间运行时可改成 5/10 减少日志量
# ==============================================================================


def make_transform(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform


class MocapBridgeNode(Node):
    def __init__(self):
        super().__init__('mocap_bridge_node')

        if CONTROL_SIDE not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported CONTROL_SIDE: {CONTROL_SIDE}")

        if CONTROL_MODE not in {"direct", "ik", "raise_template"}:
            raise ValueError(f"Unsupported CONTROL_MODE: {CONTROL_MODE}")

        self.ik_solver = None
        self.direct_retargeter = None
        self.raise_template_retargeter = None
        self.direct_baseline_q = None
        self.raise_template_baseline_q = None
        if HAND_ONLY_MODE:
            pass
        elif CONTROL_MODE == "ik":
            from teleop_ik_types import TeleopIKConfig

            joint_limit_count = 15 if IK_SOLVER_BACKEND in {"tiangong2pro_official", "ubtech_casadi"} else 14
            ik_config = TeleopIKConfig(
                joint_names=IK_ARM_JOINT_NAMES,
                motor_ids=ARM_MOTOR_IDS,
                joint_limits=build_relaxed_joint_limits(joint_limit_count),
                task_mode=IK_TASK_MODE,
                control_dt=1.0 / CONTROL_RATE_HZ,
                max_iterations=8,
                damping=0.03,
                max_joint_delta_per_step=MAX_JOINT_DELTA_PER_STEP,
                locked_joint_indices=POSITION_ONLY_LOCKED_JOINT_INDICES if IK_TASK_MODE == "position_only" else (),
            )
            if IK_SOLVER_BACKEND == "teleop_dls":
                from teleop_ik_core import TeleopArmIK

                self.ik_solver = TeleopArmIK(ik_config)
            elif IK_SOLVER_BACKEND in {"tiangong2pro_official", "ubtech_casadi"}:
                from teleop_ik_ubtech_casadi import OfficialTiangong2ProIK

                self.ik_solver = OfficialTiangong2ProIK(ik_config)
            else:
                raise ValueError(f"Unsupported IK_SOLVER_BACKEND: {IK_SOLVER_BACKEND}")
        elif CONTROL_MODE == "raise_template":
            from teleop_raise_template import RaiseTemplateConfig, RaiseTemplateRetargeter

            self.raise_template_retargeter = RaiseTemplateRetargeter(
                RaiseTemplateConfig(
                    joint_names=IK_ARM_JOINT_NAMES,
                    motor_ids=ARM_MOTOR_IDS,
                    control_side=CONTROL_SIDE,
                    max_joint_delta_per_step=MAX_JOINT_DELTA_PER_STEP,
                    smoothing_alpha=1.0,
                )
            )
        else:
            if DIRECT_RETARGET_RUNTIME_ARGS is None:
                raise RuntimeError("direct 模式运行参数尚未初始化")
            self.direct_retargeter = DirectRetargeter(
                direct_retarget_config_from_args(
                    DIRECT_RETARGET_RUNTIME_ARGS,
                    joint_names=IK_ARM_JOINT_NAMES,
                    motor_ids=ARM_MOTOR_IDS,
                )
            )

        self.command_joint_limits = None
        if not HAND_ONLY_MODE:
            if self.ik_solver is not None:
                self.command_joint_limits = list(self.ik_solver.joint_limits)
            elif self.direct_retargeter is not None:
                self.command_joint_limits = list(self.direct_retargeter.solver.joint_limits)
            else:
                self.command_joint_limits = arm_joint_limits_from_urdf(
                    Path(__file__).resolve().parent / "models" /
                    "walker_tienkung_ei_description" / "walker_tienkung_ei.urdf"
                )

        self.hand_ik_sidecar = None
        self.latest_hand_ik_result = None
        self.last_hand_mocap_timestamp_us = None
        self.hand_feedback: dict[str, tuple[float, ...] | None] = {"left": None, "right": None}
        self.hand_feedback_time: dict[str, float | None] = {"left": None, "right": None}
        self.last_hand_command: dict[str, tuple[float, ...] | None] = {"left": None, "right": None}
        self._warned_hand_state: dict[str, bool] = {"left": False, "right": False}
        if DIRECT_RETARGET_RUNTIME_ARGS is not None and hand_ik_enabled(DIRECT_RETARGET_RUNTIME_ARGS):
            try:
                self.hand_ik_sidecar = Revo2HandIkSidecar(
                    hand_ik_config_from_args(DIRECT_RETARGET_RUNTIME_ARGS)
                )
            except Exception as exc:
                self.get_logger().error(f"Revo2 手部 IK sidecar 启动失败；双臂管线继续运行: {exc}")

        self.rosbridge_client = None
        self.arm_status_sub = None
        self.waist_status_sub = None
        self.arm_cmd_pub = None
        self.waist_cmd_pub = None
        self.hand_status_subs: dict[str, roslibpy.Topic] = {}
        self.hand_cmd_pubs: dict[str, roslibpy.Topic] = {}
        self.ros2_cmd_pub = None
        self.ros2_joint_sub = None

        self.current_lr_arm_motor_q = None
        self.current_waist_position = None
        self.last_arm_state_time = None
        self.last_waist_state_time = None
        self.last_command_q = None
        self.last_waist_command = None
        self.safe_baseline_q = None
        self.safe_baseline_waist = None
        self.last_command_publish_time = 0.0
        self.command_faulted = False
        self._warned_missing_arm_state = False
        self._warned_arm_error = False
        self._warned_missing_waist_state = False
        self._warned_waist_error = False
        self._warned_command_disabled = False
        self.log_frame_index = 0
        self.local_logger = self._init_local_logger()
        self.state_lock = threading.Lock()

        self.left_motors = copy.deepcopy(ARM_MOTOR_TEMPLATES[:7])
        self.right_motors = copy.deepcopy(ARM_MOTOR_TEMPLATES[7:])
        self.all_motors = copy.deepcopy(ARM_MOTOR_TEMPLATES)
        self.waist_motor = copy.deepcopy(WAIST_MOTOR_TEMPLATE)
        self.workspace_calibration = None
        self.wrist_calibration = None
        self.mocap_target_mapper = None
        self.startup_pose_gate = NaturalDownStartupGate(
            NaturalDownStartupConfig(
                hold_seconds=STARTUP_HOLD_SECONDS,
                human_vertical_cosine=STARTUP_HUMAN_VERTICAL_COSINE,
                human_min_wrist_drop_m=STARTUP_HUMAN_MIN_WRIST_DROP_M,
                human_max_lateral_offset_m=STARTUP_HUMAN_MAX_LATERAL_OFFSET_M,
                max_home_rms_m=STARTUP_MAX_HOME_RMS_M,
            )
        )
        self.last_startup_pose_log_time = 0.0
        if CONTROL_MODE == "ik" and not HAND_ONLY_MODE:
            self.workspace_calibration = load_workspace_calibration(
                WORKSPACE_CALIBRATION_PATH, CONTROL_SIDE, optional=True
            )
            self.wrist_calibration = load_wrist_calibration(
                WRIST_CALIBRATION_PATH, CONTROL_SIDE, optional=WRIST_CALIBRATION_PATH is None
            )
            if self.workspace_calibration is None:
                self.get_logger().info("CALIBRATION_STATUS workspace 参考标定不可用，使用默认相对映射")
            else:
                self.get_logger().info(
                    f"CALIBRATION_STATUS 已加载 workspace 参考标定: {self.workspace_calibration.source_path}"
                )
            if self.wrist_calibration is None:
                self.get_logger().info("CALIBRATION_STATUS wrist 参考标定不可用，使用 skeleton 腕点")
            else:
                self.get_logger().info(
                    f"CALIBRATION_STATUS 已加载 wrist 参考标定: {self.wrist_calibration.source_path}"
                )
        self.startup_pose_locked = False
        self.robot_home_initialized = False

        self._init_robot_io()

        self.udp_ip = UDP_IP
        self.udp_port = UDP_PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self._socket_closed = False

        self.latest_mocap_data = None
        self.data_lock = threading.Lock()
        self.invalid_mocap_packets = 0
        self.last_invalid_mocap_log_time = 0.0

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.udp_thread = threading.Thread(target=self.udp_receive_loop, daemon=True)
        self.udp_thread.start()

        self.get_logger().info(
            f"动捕桥接节点已启动，UDP={self.udp_port}, device_mode={MOCAP_DEVICE_MODE}, "
            f"side={CONTROL_SIDE}, "
            f"mode={CONTROL_MODE}, ik_backend={IK_SOLVER_BACKEND}, ik_task={IK_TASK_MODE}, "
            f"position_scale={MOCAP_RELATIVE_POSITION_SCALE}, orientation_scale={MOCAP_RELATIVE_ORIENTATION_SCALE}, "
            f"backend={COMMAND_BACKEND}, publish={ENABLE_ROBOT_COMMAND}, hand_only={HAND_ONLY_MODE}, "
            f"safe={SAFE_TELEOP_MODE}, "
            f"safe_rate={SAFE_COMMAND_RATE_HZ}Hz, max_step={MAX_JOINT_DELTA_PER_STEP}, "
            f"max_baseline_delta={MAX_COMMAND_DELTA_FROM_BASELINE}"
        )
        if not HAND_ONLY_MODE:
            self.get_logger().info(
                "STARTUP_HOME_WAIT 请保持人体双臂自然下垂且静止；"
                "机器人只需保持当前姿态并提供有效关节反馈；"
                f"连续通过 {STARTUP_HOLD_SECONDS:.1f}s 后才启用手臂和腰部命令"
            )

    def _init_local_logger(self) -> logging.Logger | None:
        if not ENABLE_LOCAL_LOG:
            return None

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("mocap_ik_bridge")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        log_path = LOG_DIR / LOG_FILE_NAME
        if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(file_handler)

        self.get_logger().info(f"本地日志写入: {log_path}")
        return logger

    def _init_robot_io(self):
        if COMMAND_BACKEND == "rosbridge":
            self.rosbridge_client = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
            self.rosbridge_client.run()

            if not HAND_ONLY_MODE:
                self.arm_status_sub = roslibpy.Topic(
                    self.rosbridge_client,
                    ARM_STATUS_TOPIC,
                    ARM_STATUS_TYPE,
                )
                self.arm_status_sub.subscribe(self.arm_status_callback)

                if CONTROL_MODE == "ik" and getattr(self.ik_solver, "controls_waist", False):
                    self.waist_status_sub = roslibpy.Topic(
                        self.rosbridge_client,
                        WAIST_STATUS_TOPIC,
                        WAIST_STATUS_TYPE,
                    )
                    self.waist_status_sub.subscribe(self.waist_status_callback)

                self.arm_cmd_pub = roslibpy.Topic(
                    self.rosbridge_client,
                    ARM_CMD_TOPIC,
                    ARM_CMD_TYPE,
                )
                self.arm_cmd_pub.advertise()
                if CONTROL_MODE == "ik" and getattr(self.ik_solver, "controls_waist", False):
                    self.waist_cmd_pub = roslibpy.Topic(
                        self.rosbridge_client,
                        WAIST_CMD_TOPIC,
                        WAIST_CMD_TYPE,
                    )
                    self.waist_cmd_pub.advertise()

            if HAND_CONTROL_CONFIG.publish_enabled:
                active_sides = ("left", "right") if CONTROL_SIDE == "both" else (CONTROL_SIDE,)
                for side in active_sides:
                    state_sub = roslibpy.Topic(
                        self.rosbridge_client,
                        HAND_STATE_TOPICS[side],
                        HAND_JOINT_STATE_TYPE,
                    )
                    state_sub.subscribe(lambda msg, hand_side=side: self.hand_state_callback(hand_side, msg))
                    self.hand_status_subs[side] = state_sub
                    command_pub = roslibpy.Topic(
                        self.rosbridge_client,
                        HAND_CMD_TOPICS[side],
                        HAND_JOINT_STATE_TYPE,
                    )
                    command_pub.advertise()
                    self.hand_cmd_pubs[side] = command_pub
            return

        if COMMAND_BACKEND == "ros2_joint_state":
            self.ros2_cmd_pub = self.create_publisher(JointState, ROS2_JOINT_COMMAND_TOPIC, 10)
            self.ros2_joint_sub = self.create_subscription(
                JointState,
                ROS2_JOINT_STATE_TOPIC,
                self.ros2_joint_state_callback,
                10,
            )
            return

        raise ValueError(f"Unsupported COMMAND_BACKEND: {COMMAND_BACKEND}")

    def udp_receive_loop(self):
        """后台线程只接收和规范化最新一帧，控制计算留给定时器线程。"""
        while rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(UDP_PACKET_SIZE)
                receive_time = time.time()
                receive_monotonic = time.monotonic()
                frame = parse_mocap_packet(
                    data,
                    receive_time,
                    device_mode=MOCAP_DEVICE_MODE,
                )
                mocap_data = frame.to_replay_dict()
                # 实时新鲜度只使用本机接收时钟，不能依赖设备时钟是否同步。
                mocap_data["timestamp"] = receive_time
                mocap_data["receive_time"] = receive_time
                mocap_data["receive_monotonic"] = receive_monotonic

                with self.data_lock:
                    self.latest_mocap_data = mocap_data

                if self.should_log() and LOG_MOCAP_DATA:
                    self.log_section(
                        "mocap_data",
                        {
                            "packet_timestamp_us": frame.timestamp_us,
                            "wire": frame.wire,
                            "left_pos": mocap_data["left_pos"],
                            "left_quat": mocap_data["left_quat"],
                            "right_pos": mocap_data["right_pos"],
                            "right_quat": mocap_data["right_quat"],
                        },
                    )
            except MocapPacketError as exc:
                self.invalid_mocap_packets += 1
                now = time.time()
                if self.invalid_mocap_packets == 1 or now - self.last_invalid_mocap_log_time >= 5.0:
                    self.get_logger().warn(
                        "丢弃无效动捕 UDP 包: "
                        f"mode={MOCAP_DEVICE_MODE} count={self.invalid_mocap_packets} "
                        f"len={len(data)} head={data[:32].hex(' ')} error={exc}"
                    )
                    self.last_invalid_mocap_log_time = now
            except OSError:
                if not self._socket_closed:
                    self.get_logger().error("UDP socket 异常关闭")
                return
            except Exception as e:
                self.get_logger().error(f"UDP 接收错误: {e}")

    def arm_status_callback(self, msg):
        """rosbridge /arm/status 回调，只接受完整、无告警的 14 轴反馈。"""
        try:
            status = msg.get("status", [])
            alarm_items = [
                (int(item.get("name", -1)), int(item.get("error", 0)))
                for item in status
                if int(item.get("error", 0)) != 0
            ]
            if alarm_items:
                with self.state_lock:
                    self.current_lr_arm_motor_q = None
                    self.last_arm_state_time = None
                if not self._warned_arm_error:
                    self.get_logger().warn(
                        f"{ARM_STATUS_TOPIC} 存在电机告警 {alarm_items}，停止手臂和腰部控制"
                    )
                    self._warned_arm_error = True
                return
            q = self.extract_arm_q_from_status(status)
            with self.state_lock:
                self.current_lr_arm_motor_q = q
                self.last_arm_state_time = time.monotonic()
                self._warned_missing_arm_state = False
                self._warned_arm_error = False
        except Exception as e:
            with self.state_lock:
                self.current_lr_arm_motor_q = None
                self.last_arm_state_time = None
            self.get_logger().warn(f"解析 {ARM_STATUS_TOPIC} 失败: {e}")

    def waist_status_callback(self, msg):
        """读取官方 `/waist/status` 的 31 号腰部关节位置，并拒绝带告警的样本。"""
        try:
            status = msg.get("status", [])
            items = [entry for entry in status if float(entry.get("name", -1)) == WAIST_MOTOR_ID]
            if len(items) != 1:
                raise ValueError(f"腰部电机 ID={WAIST_MOTOR_ID} 缺失或重复")
            item = items[0]
            error = float(item.get("error", 0))
            if error != 0:
                with self.state_lock:
                    self.current_waist_position = None
                    self.last_waist_state_time = None
                if not self._warned_waist_error:
                    description = WAIST_ERROR_DESCRIPTIONS.get(error, "未知错误")
                    self.get_logger().warn(
                        f"{WAIST_STATUS_TOPIC} 腰部告警 error={error} ({description})，暂停官方 IK 和腰部命令"
                    )
                    self._warned_waist_error = True
                return
            position = float(item["pos"])
            if not np.isfinite(position):
                raise ValueError("腰部位置包含 NaN/Inf")
            with self.state_lock:
                self.current_waist_position = position
                self.last_waist_state_time = time.monotonic()
                self._warned_missing_waist_state = False
                self._warned_waist_error = False
        except Exception as exc:
            with self.state_lock:
                self.current_waist_position = None
                self.last_waist_state_time = None
            self.get_logger().warn(f"解析 {WAIST_STATUS_TOPIC} 失败: {exc}")

    def hand_state_callback(self, side: str, msg: dict[str, Any]) -> None:
        try:
            feedback = parse_hand_state_message(msg)
            with self.state_lock:
                self.hand_feedback[side] = feedback
                self.hand_feedback_time[side] = time.monotonic()
                self._warned_hand_state[side] = False
        except HandControlError as exc:
            with self.state_lock:
                self.hand_feedback[side] = None
                self.hand_feedback_time[side] = None
            if not self._warned_hand_state[side]:
                self.get_logger().warn(f"解析 {HAND_STATE_TOPICS[side]} 失败，停止该手命令: {exc}")
                self._warned_hand_state[side] = True

    def ros2_joint_state_callback(self, msg: JointState):
        """ROS2 JointState 备选后端，按 IK_ARM_JOINT_NAMES 顺序提取关节角。"""
        try:
            if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
                raise ValueError("JointState 的名称/位置维度错误或名称重复")
            name_to_index = {name: index for index, name in enumerate(msg.name)}
            q = [msg.position[name_to_index[name]] for name in IK_ARM_JOINT_NAMES]
            if not np.all(np.isfinite(q)):
                raise ValueError("JointState 包含 NaN/Inf")
            with self.state_lock:
                self.current_lr_arm_motor_q = np.asarray(q, dtype=float)
                self.last_arm_state_time = time.monotonic()
                self._warned_missing_arm_state = False
        except Exception as e:
            with self.state_lock:
                self.current_lr_arm_motor_q = None
                self.last_arm_state_time = None
            self.get_logger().warn(f"解析 {ROS2_JOINT_STATE_TOPIC} 失败: {e}")

    @staticmethod
    def extract_arm_q_from_status(status: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(arm_positions_from_status(status), dtype=float)

    def control_loop(self):
        """定时器回调：取最新动捕和最新关节状态，执行 IK 并发布命令。"""
        with self.data_lock:
            mocap_data = snapshot_mocap_data(self.latest_mocap_data)

        if mocap_data is None:
            return

        self.log_frame_index += 1

        if not feedback_is_fresh(
            mocap_data.get("receive_monotonic"), time.monotonic(), DROP_STALE_MOCAP_SECONDS
        ):
            self.get_logger().warn("动捕数据超时，跳过本帧控制")
            return

        self.process_hand_ik(mocap_data)

        if HAND_ONLY_MODE:
            return

        with self.state_lock:
            current_q = None if self.current_lr_arm_motor_q is None else self.current_lr_arm_motor_q.copy()
            last_arm_state_time = self.last_arm_state_time
            current_waist = self.current_waist_position
            last_waist_state_time = self.last_waist_state_time

        current_q_source = "robot_feedback"
        arm_state_stale = (
            last_arm_state_time is None
            or not feedback_is_fresh(last_arm_state_time, time.monotonic(), ARM_STATE_TIMEOUT_SECONDS)
        )
        if REQUIRE_ARM_STATE_BEFORE_IK and (current_q is None or arm_state_stale):
            if not self._warned_missing_arm_state:
                self.get_logger().warn(
                    f"{ARM_STATUS_TOPIC} 状态缺失或超过 {ARM_STATE_TIMEOUT_SECONDS:.2f}s，"
                    "手臂和腰部保持停发"
                )
                self._warned_missing_arm_state = True
            return

        if current_q is None:
            current_q = np.zeros(14)
            current_q_source = "zero_fallback_no_feedback"

        requires_waist = CONTROL_MODE == "ik" and getattr(self.ik_solver, "controls_waist", False)
        waist_state_stale = (
            last_waist_state_time is None
            or not feedback_is_fresh(last_waist_state_time, time.monotonic(), WAIST_STATE_TIMEOUT_SECONDS)
        )
        if requires_waist and (current_waist is None or waist_state_stale):
            if not self._warned_missing_waist_state:
                self.get_logger().warn(
                    f"{WAIST_STATUS_TOPIC} 的 31 号关节状态缺失或超过 "
                    f"{WAIST_STATE_TIMEOUT_SECONDS:.2f}s，暂不执行官方 IK"
                )
                self._warned_missing_waist_state = True
            return

        mocap_data["current_q"] = current_q
        mocap_data["current_waist_position"] = current_waist
        mocap_data["current_q_source"] = current_q_source
        self.process_mocap_data(mocap_data)

    def process_hand_ik(self, mocap_data: dict[str, Any]) -> None:
        if self.hand_ik_sidecar is None:
            return
        mocap_frame = mocap_data.get("mocap_frame")
        timestamp_us = getattr(mocap_frame, "timestamp_us", None)
        if timestamp_us is not None and timestamp_us == self.last_hand_mocap_timestamp_us:
            return
        try:
            result = self.hand_ik_sidecar.solve_replay_frame(mocap_data)
        except HandIkError as exc:
            self.get_logger().warn(f"Revo2 手部 IK 失败；本帧双臂管线继续运行: {exc}")
            return
        self.latest_hand_ik_result = result
        self.last_hand_mocap_timestamp_us = timestamp_us
        if self.should_log() and LOG_IK_RESULT:
            self.log_section("hand_ik_local_result", result.to_summary())
        self.publish_hand_result(result, mocap_data.get("receive_monotonic"))

    def publish_hand_result(self, result, mocap_received_at: float | None = None) -> None:
        if self.command_faulted or not HAND_CONTROL_CONFIG.publish_enabled or not ENABLE_ROBOT_COMMAND:
            return
        active_sides = ("left", "right") if CONTROL_SIDE == "both" else (CONTROL_SIDE,)
        for side in active_sides:
            now = time.monotonic()
            if not feedback_is_fresh(mocap_received_at, now, DROP_STALE_MOCAP_SECONDS):
                self.get_logger().warn("手部 IK 结果对应的动捕数据已过期，停止本帧发送")
                return
            calibrated = result.left_calibrated if side == "left" else result.right_calibrated
            if not calibrated:
                continue
            with self.state_lock:
                feedback = self.hand_feedback[side]
                feedback_time = self.hand_feedback_time[side]
                last_command = self.last_hand_command[side]
            if (
                feedback is None
                or not feedback_is_fresh(feedback_time, now, HAND_CONTROL_CONFIG.state_timeout_s)
            ):
                if not self._warned_hand_state[side]:
                    self.get_logger().warn(
                        f"{HAND_STATE_TOPICS[side]} 状态缺失或超过 "
                        f"{HAND_CONTROL_CONFIG.state_timeout_s:.2f}s，停止该手命令"
                    )
                    self._warned_hand_state[side] = True
                continue
            try:
                target = map_hand_radians(result.joint_positions, side, HAND_CONTROL_CONFIG)
                safe = apply_hand_command_safety(
                    target,
                    feedback,
                    last_command,
                    side,
                    HAND_CONTROL_CONFIG,
                )
                message = build_hand_joint_state_message(safe)
                with self.state_lock:
                    feedback = self.hand_feedback[side]
                    feedback_time = self.hand_feedback_time[side]
                    last_command = self.last_hand_command[side]
                now = time.monotonic()
                require_fresh_feedback(mocap_received_at, now, DROP_STALE_MOCAP_SECONDS, "手部动捕")
                require_fresh_feedback(feedback_time, now, HAND_CONTROL_CONFIG.state_timeout_s, "手部反馈")
                if feedback is None:
                    raise HandControlError("手部反馈失效")
                for reference in (feedback, last_command):
                    if reference is None:
                        continue
                    values = np.asarray(reference, dtype=float)
                    if values.shape != (6,) or not np.all(np.isfinite(values)):
                        raise HandControlError("手部反馈或上一命令无效")
                    if np.any(np.abs(np.asarray(safe) - values) > HAND_CONTROL_CONFIG.max_step + 1e-9):
                        raise HandControlError("手部命令超出实测位置或上一已提交命令的单步范围")
            except (HandControlError, KeyError) as exc:
                self.get_logger().warn(f"生成 {side} 手命令失败；该手本帧停发: {exc}")
                continue
            except RuntimeError as exc:
                self.get_logger().warn(str(exc))
                continue
            try:
                self.require_command_connection()
                self.hand_cmd_pubs[side].publish(message)
            except Exception as exc:
                self.command_faulted = True
                self.get_logger().error(f"手部控制发送异常，已停发本次会话，需检查并重新启动: {exc}")
                return
            with self.state_lock:
                self.last_hand_command[side] = safe
            if self.should_log() and LOG_ROBOT_COMMAND:
                self.log_section(
                    "hand_robot_command",
                    {
                        "side": side,
                        "topic": HAND_CMD_TOPICS[side],
                        "feedback": feedback,
                        "target": target,
                        "positions": safe,
                    },
                )

    def process_mocap_data(self, mocap_data):
        """核心处理管线：direct 走骨架直映射；ik 走末端位姿求解。"""
        start_time = time.perf_counter()

        try:
            left_mocap_wrist = self.mocap_anatomical_wrist_pose(mocap_data, "left")
            right_mocap_wrist = self.mocap_anatomical_wrist_pose(mocap_data, "right")
        except (KeyError, ValueError) as exc:
            self.get_logger().warn(f"动捕 hand/wrist 控制点无效，手臂和腰部保持停发: {exc}")
            return

        try:
            mocap_reference_position = self.mocap_reference_position(mocap_data)
            if not self.ensure_natural_down_startup(
                mocap_data,
                left_mocap_wrist,
                right_mocap_wrist,
                mocap_reference_position,
            ):
                return
        except (KeyError, RuntimeError, ValueError) as exc:
            self.get_logger().warn(f"动捕启动零位或胸口参考无效，手臂和腰部保持停发: {exc}")
            return

        if CONTROL_MODE == "direct":
            self.process_mocap_data_direct(mocap_data)
            cost_ms = (time.perf_counter() - start_time) * 1000.0
            if cost_ms > TIME_WARNING_THRESHOLD_MS:
                self.get_logger().warn(f"性能警告: 直映射管线耗时 {cost_ms:.2f} ms，系统即将跟不上 {CONTROL_RATE_HZ}Hz 帧率")
            return

        if CONTROL_MODE == "raise_template":
            self.process_mocap_data_raise_template(mocap_data)
            cost_ms = (time.perf_counter() - start_time) * 1000.0
            if cost_ms > TIME_WARNING_THRESHOLD_MS:
                self.get_logger().warn(f"性能警告: 平举模板管线耗时 {cost_ms:.2f} ms，系统即将跟不上 {CONTROL_RATE_HZ}Hz 帧率")
            return

        try:
            if self.mocap_target_mapper is None:
                raise RuntimeError("运行时 Home 尚未初始化")
            target_mapping = self.mocap_target_mapper.map_targets(
                left_mocap_wrist,
                right_mocap_wrist,
                mocap_reference_position=mocap_reference_position,
            )
        except (RuntimeError, ValueError) as exc:
            self.get_logger().warn(f"运行时零位映射失败，手臂和腰部保持停发: {exc}")
            return
        left_wrist, right_wrist = self.select_ik_targets(target_mapping.left_target, target_mapping.right_target)

        try:
            from teleop_ik_types import TeleopIKInput

            if self.ik_solver is None:
                raise RuntimeError("CONTROL_MODE=ik 但 IK 求解器未初始化")
            result = self.ik_solver.solve(
                TeleopIKInput(
                    timestamp=mocap_data["timestamp"],
                    control_side=CONTROL_SIDE,
                    target_pose_l=left_wrist,
                    target_pose_r=right_wrist,
                    current_q=mocap_data["current_q"],
                    last_q_cmd=self.last_command_q,
                    current_waist_position=mocap_data.get("current_waist_position"),
                )
            )
            if result.reason != "ok":
                raise RuntimeError(result.reason)
            sol_q = np.asarray(result.command.positions, dtype=float)
            reference_q = (
                np.asarray(mocap_data["current_q"], dtype=float)
                if self.last_command_q is None
                else self.last_command_q
            )
            actual_wrist_poses = self.ik_solver.wrist_effector_poses(sol_q)
            wrist_errors = wrist_position_errors(
                actual_wrist_poses,
                {
                    "left": target_mapping.meta["left_control_target"],
                    "right": target_mapping.meta["right_control_target"],
                },
                CONTROL_SIDE,
            )
            behavior_reasons = validate_wrist_behavior_candidate(
                candidate_q=sol_q,
                reference_q=reference_q,
                joint_limits=self.ik_solver.joint_limits,
                control_side=CONTROL_SIDE,
                wrist_position_error_by_side=wrist_errors,
                limits=WristBehaviorLimits(
                    max_wrist_step_rad=MAX_JOINT_DELTA_PER_STEP,
                    max_elbow_step_rad=MAX_JOINT_DELTA_PER_STEP,
                ),
            )
            if behavior_reasons:
                raise RuntimeError(f"腕部行为门禁拒绝候选: {behavior_reasons}")
        except Exception as e:
            self.get_logger().warn(f"遥操作 IK 求解失败，跳过本帧: {e}")
            return

        if self.should_log() and LOG_IK_RESULT:
            self.log_section(
                "ik_result",
                {
                    "control_side": CONTROL_SIDE,
                    "current_q_source": mocap_data.get("current_q_source", "unknown"),
                    "current_q": mocap_data["current_q"],
                    "current_waist_position": mocap_data.get("current_waist_position"),
                    "left_mocap_wrist": left_mocap_wrist,
                    "right_mocap_wrist": right_mocap_wrist,
                    "left_wrist": left_wrist,
                    "right_wrist": right_wrist,
                    "target_mapping": target_mapping.meta,
                    "sol_q": sol_q,
                    "sol_waist_position": result.command.waist_position,
                    "teleop_ik": {
                        "converged": result.converged,
                        "error_norm": result.error_norm,
                        "iteration_count": result.iteration_count,
                        "limited": result.limited,
                        "reason": result.reason,
                    },
                },
            )

        safe_q = self.apply_safety(sol_q, mocap_data["current_q"])
        if safe_q is None:
            return

        safe_waist = self.apply_waist_safety(
            result.command.waist_position,
            mocap_data.get("current_waist_position"),
        )
        if result.command.waist_position is not None and safe_waist is None:
            return

        self.publish_to_robot(
            replace(result.command, positions=safe_q, waist_position=safe_waist),
            mocap_received_at=mocap_data.get("receive_monotonic"),
        )

        cost_ms = (time.perf_counter() - start_time) * 1000.0
        if cost_ms > TIME_WARNING_THRESHOLD_MS:
            self.get_logger().warn(f"性能警告: IK及管线耗时 {cost_ms:.2f} ms！系统即将跟不上 {CONTROL_RATE_HZ}Hz 帧率！")

    def process_mocap_data_direct(self, mocap_data: dict[str, Any]) -> None:
        if self.direct_retargeter is None:
            self.get_logger().warn("CONTROL_MODE=direct 但直映射器未初始化")
            return
        if self.direct_baseline_q is None:
            self.direct_baseline_q = np.asarray(mocap_data["current_q"], dtype=float).copy()
            self.get_logger().info("已用当前 /arm/status 初始化直映射 baseline_q")

        try:
            command = self.direct_retargeter.command_from_frame(
                mocap_data,
                baseline_q=self.direct_baseline_q,
                timestamp=mocap_data["timestamp"],
            )
        except Exception as e:
            self.get_logger().warn(f"直映射生成命令失败，跳过本帧: {e}")
            return

        safe_q = self.apply_safety(command.positions, mocap_data["current_q"])
        if safe_q is None:
            return
        safe_command = replace(command, positions=safe_q)

        if self.should_log() and LOG_IK_RESULT:
            self.log_section(
                "direct_retarget_result",
                {
                    "control_side": CONTROL_SIDE,
                    "current_q_source": mocap_data.get("current_q_source", "unknown"),
                    "baseline_q": self.direct_baseline_q,
                    "positions": safe_command.positions,
                    "raw_positions": command.positions,
                    "meta": command.meta,
                },
            )
        self.publish_to_robot(safe_command, mocap_received_at=mocap_data.get("receive_monotonic"))

    def process_mocap_data_raise_template(self, mocap_data: dict[str, Any]) -> None:
        if self.raise_template_retargeter is None:
            self.get_logger().warn("CONTROL_MODE=raise_template 但平举模板映射器未初始化")
            return
        if self.raise_template_baseline_q is None:
            self.raise_template_baseline_q = np.asarray(mocap_data["current_q"], dtype=float).copy()
            self.get_logger().info("已用当前 /arm/status 初始化平举模板 baseline_q")

        try:
            command = self.raise_template_retargeter.command_from_frame(
                mocap_data,
                baseline_q=self.raise_template_baseline_q,
                timestamp=mocap_data["timestamp"],
            )
        except Exception as e:
            self.get_logger().warn(f"平举模板生成命令失败，跳过本帧: {e}")
            return

        safe_q = self.apply_safety(command.positions, mocap_data["current_q"])
        if safe_q is None:
            return
        safe_command = replace(command, positions=safe_q)

        if self.should_log() and LOG_IK_RESULT:
            self.log_section(
                "raise_template_result",
                {
                    "control_side": CONTROL_SIDE,
                    "current_q_source": mocap_data.get("current_q_source", "unknown"),
                    "baseline_q": self.raise_template_baseline_q,
                    "positions": safe_command.positions,
                    "raw_positions": command.positions,
                    "meta": command.meta,
                },
            )
        self.publish_to_robot(safe_command, mocap_received_at=mocap_data.get("receive_monotonic"))

    def mocap_reference_position(self, mocap_data: dict[str, Any]) -> np.ndarray | None:
        if (
            self.workspace_calibration is None
            or self.workspace_calibration.position_reference == "absolute_hand_legacy"
        ):
            return None
        frame = mocap_data.get("mocap_frame")
        if frame is None:
            raise ValueError("实时工作空间映射缺少 mocap_frame")
        chest = frame.joint("chest")
        if not chest.tracked:
            raise ValueError("胸口参考点未跟踪")
        chest_homogeneous = np.append(np.asarray(chest.position, dtype=float), 1.0)
        return np.asarray((T_BASE_MOCAP @ chest_homogeneous)[:3], dtype=float)

    def ensure_natural_down_startup(
        self,
        mocap_data: dict[str, Any],
        left_mocap_wrist: np.ndarray,
        right_mocap_wrist: np.ndarray,
        mocap_reference_position: np.ndarray | None,
    ) -> bool:
        if self.startup_pose_locked:
            return True
        frame = mocap_data.get("mocap_frame")
        if frame is None:
            return False
        reference = np.zeros(3, dtype=float) if mocap_reference_position is None else mocap_reference_position
        status = self.startup_pose_gate.observe(
            now=float(mocap_data["receive_monotonic"]),
            frame=frame,
            current_q=np.asarray(mocap_data["current_q"], dtype=float),
            left_home_position=left_mocap_wrist[:3, 3] - reference,
            right_home_position=right_mocap_wrist[:3, 3] - reference,
        )
        if not status.ready:
            now = time.time()
            if now - self.last_startup_pose_log_time >= 1.0:
                self.get_logger().warn(
                    "启动姿态门禁未通过，手臂和腰部保持停发: "
                    f"held={status.held_seconds:.2f}s reasons={list(status.reasons)} "
                    f"metrics={to_jsonable(status.metrics)}"
                )
                self.last_startup_pose_log_time = now
            return False
        current_q = np.asarray(mocap_data["current_q"], dtype=float)
        if CONTROL_MODE == "ik":
            self.ensure_robot_home_from_current_q(
                current_q=current_q,
                current_waist_position=mocap_data.get("current_waist_position"),
                left_mocap_wrist=left_mocap_wrist,
                right_mocap_wrist=right_mocap_wrist,
                mocap_reference_position=mocap_reference_position,
            )
        else:
            self.safe_baseline_q = current_q.copy()
            self.startup_pose_locked = True
            self.get_logger().info(
                "STARTUP_HOME_READY 人体双臂自然下垂已稳定，"
                f"已锁定 {CONTROL_MODE} 本次运行基线"
            )
        return True

    def ensure_robot_home_from_current_q(
        self,
        current_q: np.ndarray,
        current_waist_position: float | None,
        left_mocap_wrist: np.ndarray,
        right_mocap_wrist: np.ndarray,
        mocap_reference_position: np.ndarray | None,
    ) -> None:
        """用稳定的人体自然下垂姿态和机器人当前反馈建立本次遥操作零位。"""

        if self.robot_home_initialized:
            return
        if self.ik_solver is None:
            raise RuntimeError("CONTROL_MODE=ik 需要先初始化 IK 求解器")
        if current_waist_position is not None and hasattr(self.ik_solver, "last_waist_position"):
            self.ik_solver.last_waist_position = float(current_waist_position)
        if not hasattr(self.ik_solver, "wrist_effector_poses") or not hasattr(
            self.ik_solver, "wrist_to_tcp_transforms"
        ):
            raise RuntimeError("当前 IK 后端未提供 wrist-roll 控制点 FK，无法建立机器人手腕 Home")
        left_robot_home, right_robot_home = self.ik_solver.wrist_effector_poses(current_q)
        left_wrist_to_tcp, right_wrist_to_tcp = self.ik_solver.wrist_to_tcp_transforms()
        self.ik_solver.set_home_q(current_q)
        self.mocap_target_mapper = RelativeMocapTargetMapper(
            left_robot_home=left_robot_home,
            right_robot_home=right_robot_home,
            mode=MOCAP_TARGET_MODE,
            position_scale=MOCAP_RELATIVE_POSITION_SCALE,
            orientation_scale=MOCAP_RELATIVE_ORIENTATION_SCALE,
            workspace_calibration=self.workspace_calibration,
            control_side=CONTROL_SIDE,
            left_control_to_target=left_wrist_to_tcp,
            right_control_to_target=right_wrist_to_tcp,
            control_frame="wrist_roll",
        )
        self.mocap_target_mapper.set_runtime_mocap_home(
            left_mocap_wrist=left_mocap_wrist,
            right_mocap_wrist=right_mocap_wrist,
            mocap_reference_position=mocap_reference_position,
        )
        self.safe_baseline_q = np.asarray(current_q, dtype=float).copy()
        self.safe_baseline_waist = None if current_waist_position is None else float(current_waist_position)
        self.robot_home_initialized = True
        self.startup_pose_locked = True
        self.get_logger().info(
            "STARTUP_HOME_READY 人体双臂自然下垂已稳定，"
            "机器人当前姿态已设为运行时 Home，方向映射已建立"
        )

    def mocap_anatomical_wrist_pose(self, mocap_data: dict[str, Any], side: str) -> np.ndarray:
        if self.wrist_calibration is not None and side in self.wrist_calibration.sides:
            mocap_wrist = reconstruct_wrist_pose_from_mapping(
                mocap_data,
                side,
                self.wrist_calibration.sides[side],
            )
            return T_BASE_MOCAP @ mocap_wrist

        skeleton = mocap_data.get("skeleton", {}).get(side, {})
        wrist = skeleton.get("wrist", {})
        hand_quat = mocap_data.get(f"{side}_hand_quat", mocap_data.get(f"{side}_quat"))
        if wrist.get("pos") is None or hand_quat is None:
            raise ValueError(f"当前帧缺少 {side} wrist/hand 观测")
        return T_BASE_MOCAP @ self.build_transform_matrix(
            np.asarray(wrist["pos"], dtype=float),
            np.asarray(hand_quat, dtype=float),
        )

    @staticmethod
    def build_transform_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        quat_xyzw = quat if QUAT_ORDER == "xyzw" else np.asarray([quat[1], quat[2], quat[3], quat[0]])
        return make_transform(pos, R.from_quat(quat_xyzw).as_matrix())

    @staticmethod
    def select_ik_targets(left_mocap_wrist: np.ndarray, right_mocap_wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # RelativeMocapTargetMapper 已把未控制侧固定在本次机器人 Home；这里不能再退回静态旧 Home。
        return left_mocap_wrist, right_mocap_wrist

    def apply_safety(self, sol_q: list[float], current_q: np.ndarray) -> np.ndarray | None:
        target_q = np.asarray(sol_q, dtype=float)
        current_q = np.asarray(current_q, dtype=float)
        if target_q.shape != (14,):
            self.get_logger().warn(f"IK 输出维度错误: {target_q.shape}")
            return None
        if current_q.shape != (14,) or not np.all(np.isfinite(current_q)):
            self.get_logger().warn("当前机器人手臂状态无效，跳过本帧")
            return None
        if not np.all(np.isfinite(target_q)):
            self.get_logger().warn("目标关节命令包含 NaN/Inf，跳过本帧")
            return None

        if self.safe_baseline_q is None:
            self.safe_baseline_q = current_q.copy()
            self.get_logger().info("已用当前 /arm/status 初始化安全遥操作 baseline_q")

        if SAFE_TELEOP_MODE and SAFE_COMMAND_RATE_HZ > 0:
            now = time.monotonic()
            min_interval = 1.0 / SAFE_COMMAND_RATE_HZ
            if now - self.last_command_publish_time < min_interval:
                return None

        reference_q = self.last_command_q if self.last_command_q is not None else current_q
        from teleop_ik_core import apply_joint_safety

        joint_limits = np.asarray(self.command_joint_limits, dtype=float).copy()
        if SAFE_TELEOP_MODE and CONTROL_MODE != "raise_template":
            joint_limits[:, 0] = np.maximum(
                joint_limits[:, 0], self.safe_baseline_q - MAX_COMMAND_DELTA_FROM_BASELINE)
            joint_limits[:, 1] = np.minimum(
                joint_limits[:, 1], self.safe_baseline_q + MAX_COMMAND_DELTA_FROM_BASELINE)
        try:
            safe_q, limited = apply_joint_safety(
                q_next=target_q,
                q_ref=reference_q,
                limits=list(map(tuple, joint_limits)),
                max_step=MAX_JOINT_DELTA_PER_STEP,
                smoothing_alpha=1.0,
            )
        except ValueError as exc:
            self.get_logger().warn(f"关节安全边界无效，跳过本帧: {exc}")
            return None

        if not np.all(np.isfinite(safe_q)):
            self.get_logger().warn("安全层输出包含 NaN/Inf，跳过本帧")
            return None
        if limited:
            self.get_logger().debug("下发前安全层已限制本帧关节命令")

        return safe_q

    def apply_waist_safety(
        self,
        target_position: float | None,
        current_position: float | None,
    ) -> float | None:
        if target_position is None:
            return None
        if current_position is None or not np.isfinite(current_position) or not np.isfinite(target_position):
            self.get_logger().warn("腰部目标或反馈无效，跳过本帧")
            return None
        if self.ik_solver is None or not hasattr(self.ik_solver, "waist_limit"):
            self.get_logger().warn("当前求解器未提供官方腰部限位，跳过本帧")
            return None

        lower, upper = self.ik_solver.waist_limit
        if self.safe_baseline_waist is None:
            self.safe_baseline_waist = float(current_position)
            self.get_logger().info(f"已用 {WAIST_STATUS_TOPIC} 初始化腰部安全 baseline")
        reference = float(current_position if self.last_waist_command is None else self.last_waist_command)
        target = float(np.clip(target_position, lower, upper))
        if SAFE_TELEOP_MODE:
            target = float(
                np.clip(
                    target,
                    self.safe_baseline_waist - MAX_COMMAND_DELTA_FROM_BASELINE,
                    self.safe_baseline_waist + MAX_COMMAND_DELTA_FROM_BASELINE,
                )
            )
        safe_position = float(
            np.clip(
                target,
                reference - MAX_JOINT_DELTA_PER_STEP,
                reference + MAX_JOINT_DELTA_PER_STEP,
            )
        )
        return float(np.clip(safe_position, lower, upper))

    def validate_command_for_publish(
        self, command: IKCommand, mocap_received_at: float | None, *, check_rate: bool = True
    ) -> None:
        if self.command_faulted:
            raise RuntimeError("本次控制会话已有发送异常，检查机器人并重新启动后再启用")
        if command.control_side != CONTROL_SIDE:
            raise ValueError("命令控制侧与当前会话不一致")
        if tuple(command.motor_ids) != SCHEMA_ARM_MOTOR_IDS or tuple(command.joint_names) != SCHEMA_ARM_JOINT_NAMES:
            raise ValueError("关节命令的名称或电机 ID 顺序无效")
        positions = np.asarray(command.positions, dtype=float)
        limits = np.asarray(self.command_joint_limits, dtype=float)
        if positions.shape != (14,) or not np.all(np.isfinite(positions)):
            raise ValueError("关节命令必须是有限的 14 维向量")
        if limits.shape != (14, 2) or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("会话机械限位无效")
        if np.any(positions < limits[:, 0] - 1e-9) or np.any(positions > limits[:, 1] + 1e-9):
            raise ValueError("关节命令超出机械限位")
        with self.state_lock:
            current_q = None if self.current_lr_arm_motor_q is None else self.current_lr_arm_motor_q.copy()
            arm_received_at = self.last_arm_state_time
            current_waist = self.current_waist_position
            waist_received_at = self.last_waist_state_time
        now = time.monotonic()
        require_fresh_feedback(mocap_received_at, now, DROP_STALE_MOCAP_SECONDS, "动捕数据")
        require_fresh_feedback(arm_received_at, now, ARM_STATE_TIMEOUT_SECONDS, "手臂反馈")
        if current_q is None or current_q.shape != (14,) or not np.all(np.isfinite(current_q)):
            raise ValueError("手臂反馈无效")
        indices = slice(None) if CONTROL_SIDE == "both" else slice(0, 7) if CONTROL_SIDE == "left" else slice(7, 14)
        if np.any(np.abs(positions[indices] - current_q[indices]) > MAX_JOINT_DELTA_PER_STEP + 1e-9):
            raise ValueError("关节命令超出当前实测位置的单步范围")
        if self.last_command_q is not None and np.any(
            np.abs(positions[indices] - self.last_command_q[indices]) > MAX_JOINT_DELTA_PER_STEP + 1e-9
        ):
            raise ValueError("关节命令超出上一条已提交命令的单步范围")
        if SAFE_TELEOP_MODE and CONTROL_MODE != "raise_template":
            if self.safe_baseline_q is None or not np.all(np.isfinite(self.safe_baseline_q)):
                raise ValueError("会话基线无效")
            if np.any(np.abs(positions[indices] - self.safe_baseline_q[indices]) > MAX_COMMAND_DELTA_FROM_BASELINE + 1e-9):
                raise ValueError("关节命令超出会话基线范围")
        if check_rate and SAFE_TELEOP_MODE and SAFE_COMMAND_RATE_HZ > 0 and self.last_command_q is not None:
            if now - self.last_command_publish_time < 1.0 / SAFE_COMMAND_RATE_HZ:
                raise RuntimeError("命令发布频率超过会话限制")
        if command.waist_position is not None:
            require_fresh_feedback(waist_received_at, now, WAIST_STATE_TIMEOUT_SECONDS, "腰部反馈")
            target = float(command.waist_position)
            lower, upper = self.ik_solver.waist_limit
            if current_waist is None or not np.isfinite(current_waist) or not np.isfinite(target):
                raise ValueError("腰部命令或反馈无效")
            if not lower <= target <= upper or abs(target - current_waist) > MAX_JOINT_DELTA_PER_STEP + 1e-9:
                raise ValueError("腰部命令超出机械或实测单步范围")
            if self.last_waist_command is not None and abs(target - self.last_waist_command) > MAX_JOINT_DELTA_PER_STEP + 1e-9:
                raise ValueError("腰部命令超出上一条已提交命令的单步范围")
            if SAFE_TELEOP_MODE and (
                self.safe_baseline_waist is None or not np.isfinite(self.safe_baseline_waist)
                or abs(target - self.safe_baseline_waist) > MAX_COMMAND_DELTA_FROM_BASELINE + 1e-9
            ):
                raise ValueError("腰部命令超出会话基线范围")

    def require_command_connection(self) -> None:
        if COMMAND_BACKEND == "rosbridge" and (
            self.rosbridge_client is None or not self.rosbridge_client.is_connected
        ):
            self.command_faulted = True
            raise RuntimeError("rosbridge 已断开，需检查并重新启动控制会话")

    def publish_to_robot(self, command: IKCommand, *, mocap_received_at: float | None = None):
        """将统一 IKCommand 适配成当前后端消息。

        求解层始终输出 14 维关节角；这里才根据控制侧切成左臂、右臂或双臂，
        避免下发协议反向污染遥操作 IK 的输入输出契约。
        """
        if HAND_ONLY_MODE:
            raise RuntimeError("hand-only 模式禁止进入手臂/腰部发布路径")

        if CONTROL_SIDE == "left":
            motor_templates = self.left_motors
        elif CONTROL_SIDE == "right":
            motor_templates = self.right_motors
        else:
            motor_templates = self.all_motors

        names, command_positions, _ = select_command_slice(command)

        if ENABLE_ROBOT_COMMAND:
            try:
                self.validate_command_for_publish(command, mocap_received_at)
            except (RuntimeError, ValueError) as exc:
                self.get_logger().warn(f"发送门禁拒绝本帧: {exc}")
                return

        if COMMAND_BACKEND == "rosbridge":
            msg = build_rosbridge_motor_message(command, motor_templates)
            waist_msg = None
            if command.waist_position is not None:
                if self.waist_cmd_pub is None:
                    self.get_logger().warn("IK 包含腰部命令，但 /waist/cmd_pos 发布器未初始化")
                    return
                waist_msg = build_rosbridge_waist_message(command, self.waist_motor)
            if ENABLE_ROBOT_COMMAND:
                try:
                    self.validate_command_for_publish(command, mocap_received_at)
                except (RuntimeError, ValueError) as exc:
                    self.get_logger().warn(f"发送门禁拒绝本帧: {exc}")
                    return
                try:
                    self.require_command_connection()
                    self.arm_cmd_pub.publish(msg)
                    self.last_command_q = np.asarray(command.positions, dtype=float).copy()
                    self.last_command_publish_time = time.monotonic()
                    if waist_msg is not None:
                        self.validate_command_for_publish(command, mocap_received_at, check_rate=False)
                        self.require_command_connection()
                        self.waist_cmd_pub.publish(waist_msg)
                        self.last_waist_command = float(command.waist_position)
                except Exception as exc:
                    self.command_faulted = True
                    self.get_logger().error(f"控制发送异常，已停发本次会话，需检查并重新启动: {exc}")
                    return
            elif not self._warned_command_disabled:
                self.get_logger().warn("ENABLE_ROBOT_COMMAND=False，仅计算 IK，不向机器人下发")
                self._warned_command_disabled = True
            if self.should_log() and LOG_ROBOT_COMMAND:
                self.log_section(
                    "robot_command",
                    {
                        "backend": COMMAND_BACKEND,
                        "topic": ARM_CMD_TOPIC,
                        "enabled": ENABLE_ROBOT_COMMAND,
                        "control_side": CONTROL_SIDE,
                        "positions": command_positions,
                        "message": msg,
                        "waist_topic": WAIST_CMD_TOPIC if waist_msg is not None else None,
                        "waist_message": waist_msg,
                    },
                )
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = [float(value) for value in command_positions]
        if ENABLE_ROBOT_COMMAND:
            try:
                self.validate_command_for_publish(command, mocap_received_at)
            except (RuntimeError, ValueError) as exc:
                self.get_logger().warn(f"发送门禁拒绝本帧: {exc}")
                return
            try:
                self.ros2_cmd_pub.publish(msg)
                self.last_command_q = np.asarray(command.positions, dtype=float).copy()
                self.last_command_publish_time = time.monotonic()
            except Exception as exc:
                self.command_faulted = True
                self.get_logger().error(f"控制发送异常，已停发本次会话，需检查并重新启动: {exc}")
                return
        elif not self._warned_command_disabled:
            self.get_logger().warn("ENABLE_ROBOT_COMMAND=False，仅计算 IK，不向机器人下发")
            self._warned_command_disabled = True
        if self.should_log() and LOG_ROBOT_COMMAND:
            self.log_section(
                "robot_command",
                {
                    "backend": COMMAND_BACKEND,
                    "topic": ROS2_JOINT_COMMAND_TOPIC,
                    "enabled": ENABLE_ROBOT_COMMAND,
                    "control_side": CONTROL_SIDE,
                    "names": msg.name,
                    "positions": msg.position,
                },
            )

    def should_log(self) -> bool:
        return LOG_EVERY_N_FRAMES <= 1 or self.log_frame_index % LOG_EVERY_N_FRAMES == 0

    def log_section(self, section: str, payload: dict[str, Any]):
        record = {
            "section": section,
            "frame": self.log_frame_index,
            "time": time.time(),
            "data": payload,
        }
        text = json.dumps(to_jsonable(record), ensure_ascii=False)

        if self.local_logger is not None:
            self.local_logger.info(text)

        if LOG_TO_CONSOLE:
            self.get_logger().info(text)

    def destroy_node(self):
        self._socket_closed = True
        try:
            self.sock.close()
        except Exception:
            pass

        if self.arm_status_sub is not None:
            try:
                self.arm_status_sub.unsubscribe()
            except Exception:
                pass

        if self.waist_status_sub is not None:
            try:
                self.waist_status_sub.unsubscribe()
            except Exception:
                pass

        if self.arm_cmd_pub is not None:
            try:
                self.arm_cmd_pub.unadvertise()
            except Exception:
                pass

        if self.waist_cmd_pub is not None:
            try:
                self.waist_cmd_pub.unadvertise()
            except Exception:
                pass

        for status_sub in self.hand_status_subs.values():
            try:
                status_sub.unsubscribe()
            except Exception:
                pass
        self.hand_status_subs.clear()

        for command_pub in self.hand_cmd_pubs.values():
            try:
                command_pub.unadvertise()
            except Exception:
                pass
        self.hand_cmd_pubs.clear()

        if self.rosbridge_client is not None:
            try:
                self.rosbridge_client.terminate()
            except Exception:
                pass

        if self.hand_ik_sidecar is not None:
            try:
                self.hand_ik_sidecar.close()
            except Exception:
                pass
            self.hand_ik_sidecar = None

        super().destroy_node()


def parse_runtime_args(args=None):
    parser = argparse.ArgumentParser(description="实时动捕 IK 遥操作节点")
    parser.add_argument("--device-mode", choices=("glove", "controller", "auto"), default=MOCAP_DEVICE_MODE, help="VR 设备协议模式")
    parser.add_argument("--control-side", choices=("left", "right", "both"), default=CONTROL_SIDE, help="控制左臂、右臂或双臂")
    parser.add_argument("--control-mode", choices=("direct", "ik", "raise_template"), default=CONTROL_MODE, help="direct=骨架直映射；ik=末端位姿 IK；raise_template=立正/侧平举/前平举模板映射")
    parser.add_argument("--ik-backend", choices=("teleop_dls", "tiangong2pro_official", "ubtech_casadi"), default=IK_SOLVER_BACKEND, help="IK 求解后端；官方天工 2 Pro IK 默认含腰部 15 轴")
    parser.add_argument("--ik-task", choices=("pose", "position_only"), default=IK_TASK_MODE, help="IK 任务类型：pose=官方完整 TCP 位姿 IK；position_only=实验性锁腕位置 IK")
    parser.add_argument("--position-scale", type=float, default=MOCAP_RELATIVE_POSITION_SCALE, help="有效标定安全边界内的总增益")
    parser.add_argument("--workspace-calibration", type=Path, default=None, help="可选 wrist_roll 工作空间标定；显式指定的文件必须有效")
    parser.add_argument("--wrist-calibration", type=Path, default=None, help="可选人体 hand→wrist 标定；省略时使用 skeleton 腕点，显式文件必须有效")
    parser.add_argument("--hand-only", action="store_true", help="只运行并发布双手 IK；不创建手臂/腰部发布器，也不要求工作空间标定")
    parser.add_argument("--orientation-scale", type=float, default=MOCAP_RELATIVE_ORIENTATION_SCALE, help="pose 模式的世界系相对手腕旋转增益；默认 0.25 优先保证 TCP 位置")
    parser.add_argument("--safe-teleop", choices=("true", "false"), default="true" if SAFE_TELEOP_MODE else "false", help="是否启用低速安全遥操作门控")
    parser.add_argument("--safe-rate", type=float, default=SAFE_COMMAND_RATE_HZ, help="安全模式下最大命令发布频率")
    parser.add_argument("--max-joint-delta-per-step", type=float, default=MAX_JOINT_DELTA_PER_STEP, help="单次下发最大关节变化，单位 rad")
    parser.add_argument("--max-command-delta-from-baseline", type=float, default=MAX_COMMAND_DELTA_FROM_BASELINE, help="相对启动时手臂姿态的最大偏移，单位 rad")
    parser.add_argument("--startup-hold-seconds", type=float, default=STARTUP_HOLD_SECONDS, help="人体自然下垂需连续稳定保持的秒数")
    parser.add_argument("--startup-human-vertical-cosine", type=float, default=STARTUP_HUMAN_VERTICAL_COSINE, help="人体上臂和前臂与躯干向下方向的最小余弦")
    parser.add_argument("--startup-human-min-wrist-drop", type=float, default=STARTUP_HUMAN_MIN_WRIST_DROP_M, help="人体手腕相对肩部的最小向下距离，单位 m")
    parser.add_argument("--startup-human-max-lateral-offset", type=float, default=STARTUP_HUMAN_MAX_LATERAL_OFFSET_M, help="人体手腕相对肩部的最大横向偏移，单位 m")
    parser.add_argument("--startup-max-home-rms", type=float, default=STARTUP_MAX_HOME_RMS_M, help="自然下垂保持窗口最大手部位置 RMS，单位 m")
    add_direct_retarget_arguments(parser)
    add_hand_ik_arguments(parser)
    parser.add_argument("--hand-control-json", default="{}", help="天工无疆手部控制 JSON 配置")
    parser.add_argument("--robot-command", choices=("true", "false"), default="true" if ENABLE_ROBOT_COMMAND else "false", help="是否向机器人控制话题下发命令")
    parsed, remaining = parser.parse_known_args(args)
    return parsed, remaining


def apply_runtime_args(parsed) -> None:
    global MOCAP_DEVICE_MODE, CONTROL_SIDE, CONTROL_MODE, IK_SOLVER_BACKEND, IK_TASK_MODE
    global MOCAP_RELATIVE_POSITION_SCALE, MOCAP_RELATIVE_ORIENTATION_SCALE, ENABLE_ROBOT_COMMAND
    global WORKSPACE_CALIBRATION_PATH, WRIST_CALIBRATION_PATH
    global HAND_ONLY_MODE
    global SAFE_TELEOP_MODE, SAFE_COMMAND_RATE_HZ, MAX_JOINT_DELTA_PER_STEP, MAX_COMMAND_DELTA_FROM_BASELINE
    global STARTUP_HOLD_SECONDS, STARTUP_HUMAN_VERTICAL_COSINE
    global STARTUP_HUMAN_MIN_WRIST_DROP_M, STARTUP_HUMAN_MAX_LATERAL_OFFSET_M, STARTUP_MAX_HOME_RMS_M
    global DIRECT_RETARGET_RUNTIME_ARGS, HAND_CONTROL_CONFIG
    MOCAP_DEVICE_MODE = parsed.device_mode
    CONTROL_SIDE = parsed.control_side
    CONTROL_MODE = parsed.control_mode
    IK_SOLVER_BACKEND = parsed.ik_backend
    IK_TASK_MODE = parsed.ik_task
    MOCAP_RELATIVE_POSITION_SCALE = float(parsed.position_scale)
    MOCAP_RELATIVE_ORIENTATION_SCALE = float(parsed.orientation_scale)
    WORKSPACE_CALIBRATION_PATH = parsed.workspace_calibration
    WRIST_CALIBRATION_PATH = parsed.wrist_calibration
    HAND_ONLY_MODE = bool(parsed.hand_only)
    SAFE_TELEOP_MODE = parsed.safe_teleop == "true"
    SAFE_COMMAND_RATE_HZ = float(parsed.safe_rate)
    MAX_JOINT_DELTA_PER_STEP = float(parsed.max_joint_delta_per_step)
    MAX_COMMAND_DELTA_FROM_BASELINE = float(parsed.max_command_delta_from_baseline)
    for name, value in (
        ("safe_rate", SAFE_COMMAND_RATE_HZ),
        ("max_joint_delta_per_step", MAX_JOINT_DELTA_PER_STEP),
        ("max_command_delta_from_baseline", MAX_COMMAND_DELTA_FROM_BASELINE),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须是有限非负数")
    STARTUP_HOLD_SECONDS = float(parsed.startup_hold_seconds)
    STARTUP_HUMAN_VERTICAL_COSINE = float(parsed.startup_human_vertical_cosine)
    STARTUP_HUMAN_MIN_WRIST_DROP_M = float(parsed.startup_human_min_wrist_drop)
    STARTUP_HUMAN_MAX_LATERAL_OFFSET_M = float(parsed.startup_human_max_lateral_offset)
    STARTUP_MAX_HOME_RMS_M = float(parsed.startup_max_home_rms)
    DIRECT_RETARGET_RUNTIME_ARGS = parsed
    try:
        hand_control_raw = json.loads(parsed.hand_control_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--hand-control-json 不是有效 JSON: {exc}") from exc
    HAND_CONTROL_CONFIG = hand_control_config_from_mapping(hand_control_raw or None)
    ENABLE_ROBOT_COMMAND = parsed.robot_command == "true"
    if HAND_ONLY_MODE and not hand_ik_enabled(parsed):
        raise ValueError("--hand-only 需要启用 hand_ik.compute")
    if HAND_ONLY_MODE and ENABLE_ROBOT_COMMAND and not HAND_CONTROL_CONFIG.publish_enabled:
        raise ValueError("--hand-only 真机模式需要 hand_control.publish=true")


def main(args=None):
    parsed, remaining = parse_runtime_args(args)
    apply_runtime_args(parsed)
    rclpy.init(args=remaining)
    node = None
    try:
        node = MocapBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("节点正在退出...")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
