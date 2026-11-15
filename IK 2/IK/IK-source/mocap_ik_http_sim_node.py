import json
import logging
import socket
import struct
import sys
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# 仿真节点全局配置：调试和部署时优先只改这里
# ==============================================================================

# UDP 动捕接收参数
UDP_IP = "0.0.0.0"
UDP_PORT = 15000
UDP_PACKET_SIZE = 4096
EXPECTED_UDP_PACKET_SIZE = 1366
MOCAP_STRUCT_FORMAT = "<BBHHI 3f 188f 6H 6H 141f I"
CONTROL_RATE_HZ = 20.0
TIME_WARNING_THRESHOLD_MS = (1.0 / CONTROL_RATE_HZ) * 1000.0 * 0.8

# 动捕协议字段索引
LEFT_HAND_INDEX = 5
RIGHT_HAND_INDEX = 9
ROTATION_START_INDEX = 8
POSITION_START_INDEX = 208
ROTATION_STRIDE = 4
POSITION_STRIDE = 3
QUAT_ORDER = "xyzw"
MOCAP_POSITION_SCALE = 1.0

# 控制模式：第一版建议只控单臂，另一只手保持初始位姿。
CONTROL_SIDE = "right"  # "left" | "right" | "both"

# IK 源码路径和导入
IK_SRC_DIR = Path(__file__).resolve().parent / "Ability_script" / "src"
if str(IK_SRC_DIR) not in sys.path:
    sys.path.append(str(IK_SRC_DIR))

from IK_sdk.UBTECH_IK import UBTECH_TainGong_ArmIK  # noqa: E402
from IKAbility_utils import (  # noqa: E402
    left_init_position,
    left_init_rotation,
    right_init_position,
    right_init_rotation,
)

# 坐标系标定矩阵。正式使用前需要替换为实测标定。
T_BASE_MOCAP = np.eye(4)
T_LEFT_MARKER_EE = np.eye(4)
T_RIGHT_MARKER_EE = np.eye(4)

# 仿真 HTTP 输出参数
SIM_HTTP_URL = "http://127.0.0.1:8000/api/joint_command"
SIM_HTTP_TIMEOUT_SECONDS = 0.05
SIM_HTTP_RETRY = 0
SIM_SEND_JOINT_SCOPE = "full"  # "full" | "controlled_side"
SIM_PAYLOAD_FORMAT = "joint_positions"  # "joint_positions" | "motor_commands"
ENABLE_SIM_COMMAND = False  # 离线验证通过后再改 True

# IK 输出关节顺序
IK_ARM_JOINT_NAMES = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "wrist_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "wrist_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]
ARM_MOTOR_IDS = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
ARM_JOINT_LIMITS = [(-np.pi, np.pi)] * 14
MAX_JOINT_DELTA_PER_STEP = 0.08
INITIAL_ARM_Q = np.zeros(14)

# 运行保护
DROP_STALE_MOCAP_SECONDS = 0.3
ENABLE_IK_ERROR_LOG = False

# 本地日志
ENABLE_LOCAL_LOG = True
LOG_TO_CONSOLE = False
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_FILE_NAME = "mocap_ik_http_sim_node.log"
LOG_MOCAP_DATA = True
LOG_IK_RESULT = True
LOG_HTTP_COMMAND = True
LOG_EVERY_N_FRAMES = 1
# ==============================================================================


def make_transform(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform


LEFT_INIT_WRIST = make_transform(left_init_position, left_init_rotation)
RIGHT_INIT_WRIST = make_transform(right_init_position, right_init_rotation)


class ConsoleLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, message: str):
        self.logger.info(message)

    def warn(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)


class MocapIkHttpSimNode:
    def __init__(self):
        self.console_logger = self._init_console_logger()

        if CONTROL_SIDE not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported CONTROL_SIDE: {CONTROL_SIDE}")
        if SIM_SEND_JOINT_SCOPE not in {"full", "controlled_side"}:
            raise ValueError(f"Unsupported SIM_SEND_JOINT_SCOPE: {SIM_SEND_JOINT_SCOPE}")
        if SIM_PAYLOAD_FORMAT not in {"joint_positions", "motor_commands"}:
            raise ValueError(f"Unsupported SIM_PAYLOAD_FORMAT: {SIM_PAYLOAD_FORMAT}")

        self.ik_solver = UBTECH_TainGong_ArmIK()
        self.http_session = requests.Session()

        self.latest_mocap_data = None
        self.data_lock = threading.Lock()
        self.last_command_q = INITIAL_ARM_Q.copy()
        self.log_frame_index = 0
        self._warned_command_disabled = False
        self._socket_closed = False
        self.stop_event = threading.Event()
        self.local_logger = self._init_local_logger()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)

        self.udp_thread = threading.Thread(target=self.udp_receive_loop, daemon=True)
        self.udp_thread.start()

        self.get_logger().info(
            f"仿真 HTTP IK 节点已启动，UDP={UDP_PORT}, side={CONTROL_SIDE}, "
            f"url={SIM_HTTP_URL}, send={ENABLE_SIM_COMMAND}"
        )

    def _init_console_logger(self) -> ConsoleLogger:
        logger = logging.getLogger("mocap_ik_http_sim_node_console")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return ConsoleLogger(logger)

    def get_logger(self) -> ConsoleLogger:
        return self.console_logger

    def _init_local_logger(self) -> logging.Logger | None:
        if not ENABLE_LOCAL_LOG:
            return None

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("mocap_ik_http_sim_node")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        log_path = LOG_DIR / LOG_FILE_NAME
        if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(file_handler)

        self.get_logger().info(f"本地日志写入: {log_path}")
        return logger

    def udp_receive_loop(self):
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(UDP_PACKET_SIZE)
                if len(data) != EXPECTED_UDP_PACKET_SIZE:
                    continue

                unpacked = struct.unpack(MOCAP_STRUCT_FORMAT, data)
                left_pos, left_quat = self.extract_mocap_pose(unpacked, LEFT_HAND_INDEX)
                right_pos, right_quat = self.extract_mocap_pose(unpacked, RIGHT_HAND_INDEX)

                with self.data_lock:
                    self.latest_mocap_data = {
                        "timestamp": time.time(),
                        "packet_timestamp_ms": int(unpacked[4]),
                        "left_pos": left_pos,
                        "left_quat": left_quat,
                        "right_pos": right_pos,
                        "right_quat": right_quat,
                    }

                if self.should_log() and LOG_MOCAP_DATA:
                    self.log_section(
                        "mocap_data",
                        {
                            "packet_timestamp_ms": int(unpacked[4]),
                            "left_pos": left_pos,
                            "left_quat": left_quat,
                            "right_pos": right_pos,
                            "right_quat": right_quat,
                        },
                    )
            except OSError:
                if not self._socket_closed:
                    self.get_logger().error("UDP socket 异常关闭")
                return
            except Exception as e:
                self.get_logger().error(f"UDP 接收错误: {e}")

    @staticmethod
    def extract_mocap_pose(unpacked: tuple[Any, ...], joint_index: int) -> tuple[np.ndarray, np.ndarray]:
        rot_start = ROTATION_START_INDEX + joint_index * ROTATION_STRIDE
        pos_start = POSITION_START_INDEX + joint_index * POSITION_STRIDE
        quat = np.asarray(unpacked[rot_start: rot_start + ROTATION_STRIDE], dtype=float)
        pos = np.asarray(unpacked[pos_start: pos_start + POSITION_STRIDE], dtype=float)
        pos *= MOCAP_POSITION_SCALE
        return pos, normalize_quaternion(quat)

    def control_loop(self):
        with self.data_lock:
            mocap_data = copy_mocap_data(self.latest_mocap_data)

        if mocap_data is None:
            return

        self.log_frame_index += 1

        if time.time() - mocap_data["timestamp"] > DROP_STALE_MOCAP_SECONDS:
            self.get_logger().warn("动捕数据超时，跳过本帧仿真控制")
            return

        self.process_mocap_data(mocap_data)

    def process_mocap_data(self, mocap_data: dict[str, Any]):
        start_time = time.perf_counter()

        left_mocap_wrist = self.transform_mocap_to_robot(
            mocap_data["left_pos"],
            mocap_data["left_quat"],
            T_LEFT_MARKER_EE,
        )
        right_mocap_wrist = self.transform_mocap_to_robot(
            mocap_data["right_pos"],
            mocap_data["right_quat"],
            T_RIGHT_MARKER_EE,
        )

        left_wrist, right_wrist = self.select_ik_targets(left_mocap_wrist, right_mocap_wrist)
        current_q = self.last_command_q.copy()

        try:
            sol_q, _ = self.ik_solver.solve_ik(
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                current_lr_arm_motor_q=current_q,
                print_error=ENABLE_IK_ERROR_LOG,
            )
        except Exception as e:
            self.get_logger().warn(f"IK 求解失败，跳过本帧: {e}")
            return

        if self.should_log() and LOG_IK_RESULT:
            self.log_section(
                "ik_result",
                {
                    "control_side": CONTROL_SIDE,
                    "current_q": current_q,
                    "left_wrist": left_wrist,
                    "right_wrist": right_wrist,
                    "sol_q": sol_q,
                },
            )

        safe_q = self.apply_safety(sol_q, current_q)
        if safe_q is None:
            return

        payload = self.build_http_payload(safe_q)
        self.send_http_command(payload)

        cost_ms = (time.perf_counter() - start_time) * 1000.0
        if cost_ms > TIME_WARNING_THRESHOLD_MS:
            self.get_logger().warn(f"性能警告: IK及HTTP管线耗时 {cost_ms:.2f} ms，可能跟不上 {CONTROL_RATE_HZ}Hz")

    @staticmethod
    def build_transform_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        quat_xyzw = quat if QUAT_ORDER == "xyzw" else np.asarray([quat[1], quat[2], quat[3], quat[0]])
        return make_transform(pos, R.from_quat(quat_xyzw).as_matrix())

    def transform_mocap_to_robot(self, pos: np.ndarray, quat: np.ndarray, marker_to_ee: np.ndarray) -> np.ndarray:
        mocap_marker = self.build_transform_matrix(pos, quat)
        return T_BASE_MOCAP @ mocap_marker @ marker_to_ee

    @staticmethod
    def select_ik_targets(left_mocap_wrist: np.ndarray, right_mocap_wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if CONTROL_SIDE == "left":
            return left_mocap_wrist, RIGHT_INIT_WRIST
        if CONTROL_SIDE == "right":
            return LEFT_INIT_WRIST, right_mocap_wrist
        return left_mocap_wrist, right_mocap_wrist

    def apply_safety(self, sol_q: list[float], current_q: np.ndarray) -> np.ndarray | None:
        target_q = np.asarray(sol_q, dtype=float)
        if target_q.shape[0] != 14:
            self.get_logger().warn(f"IK 输出维度错误: {target_q.shape}")
            return None
        if not np.all(np.isfinite(target_q)):
            self.get_logger().warn("IK 输出包含 NaN/Inf，跳过本帧")
            return None

        lower = np.asarray([limit[0] for limit in ARM_JOINT_LIMITS], dtype=float)
        upper = np.asarray([limit[1] for limit in ARM_JOINT_LIMITS], dtype=float)
        safe_q = np.clip(target_q, lower, upper)

        if MAX_JOINT_DELTA_PER_STEP is not None and MAX_JOINT_DELTA_PER_STEP > 0:
            safe_q = np.clip(
                safe_q,
                current_q - MAX_JOINT_DELTA_PER_STEP,
                current_q + MAX_JOINT_DELTA_PER_STEP,
            )

        self.last_command_q = safe_q.copy()
        return safe_q

    def build_http_payload(self, safe_q: np.ndarray) -> dict[str, Any]:
        names, positions, motor_ids = self.select_command_joints(safe_q)

        if SIM_PAYLOAD_FORMAT == "motor_commands":
            command = [
                {"name": int(motor_id), "pos": float(position)}
                for motor_id, position in zip(motor_ids, positions)
            ]
            return {
                "timestamp": time.time(),
                "control_side": CONTROL_SIDE,
                "format": SIM_PAYLOAD_FORMAT,
                "cmds": command,
            }

        return {
            "timestamp": time.time(),
            "control_side": CONTROL_SIDE,
            "format": SIM_PAYLOAD_FORMAT,
            "joint_names": names,
            "positions": [float(value) for value in positions],
        }

    @staticmethod
    def select_command_joints(safe_q: np.ndarray) -> tuple[list[str], np.ndarray, list[int]]:
        if SIM_SEND_JOINT_SCOPE == "full" or CONTROL_SIDE == "both":
            return IK_ARM_JOINT_NAMES, safe_q, ARM_MOTOR_IDS
        if CONTROL_SIDE == "left":
            return IK_ARM_JOINT_NAMES[:7], safe_q[:7], ARM_MOTOR_IDS[:7]
        return IK_ARM_JOINT_NAMES[7:], safe_q[7:], ARM_MOTOR_IDS[7:]

    def send_http_command(self, payload: dict[str, Any]):
        if self.should_log() and LOG_HTTP_COMMAND:
            self.log_section(
                "http_command",
                {
                    "url": SIM_HTTP_URL,
                    "enabled": ENABLE_SIM_COMMAND,
                    "payload": payload,
                },
            )

        if not ENABLE_SIM_COMMAND:
            if not self._warned_command_disabled:
                self.get_logger().warn("ENABLE_SIM_COMMAND=False，仅计算 IK，不向仿真机发送 HTTP")
                self._warned_command_disabled = True
            return

        attempts = SIM_HTTP_RETRY + 1
        for attempt in range(attempts):
            try:
                response = self.http_session.post(
                    SIM_HTTP_URL,
                    json=payload,
                    timeout=SIM_HTTP_TIMEOUT_SECONDS,
                )
                if response.ok:
                    return
                self.get_logger().warn(f"仿真 HTTP 返回异常: {response.status_code} {response.text}")
            except requests.RequestException as e:
                if attempt == attempts - 1:
                    self.get_logger().warn(f"仿真 HTTP 发送失败: {e}")

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
        self.stop_event.set()
        self._socket_closed = True
        try:
            self.sock.close()
        except Exception:
            pass
        self.http_session.close()

    def spin(self):
        period = 1.0 / CONTROL_RATE_HZ
        next_tick = time.perf_counter()

        while not self.stop_event.is_set():
            self.control_loop()
            next_tick += period
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("动捕四元数模长过小")
    return quat / norm


def copy_mocap_data(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    return {
        "timestamp": data["timestamp"],
        "packet_timestamp_ms": data["packet_timestamp_ms"],
        "left_pos": data["left_pos"].copy(),
        "left_quat": data["left_quat"].copy(),
        "right_pos": data["right_pos"].copy(),
        "right_quat": data["right_quat"].copy(),
    }


def to_jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def main(args=None):
    node = MocapIkHttpSimNode()

    try:
        node.spin()
    except KeyboardInterrupt:
        node.get_logger().info("仿真节点正在退出...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
