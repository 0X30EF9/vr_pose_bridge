import copy
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
UDP_PACKET_SIZE = 4096
EXPECTED_UDP_PACKET_SIZE = 1366
MOCAP_STRUCT_FORMAT = "<BBHHI 3f 188f 6H 6H 141f I"
CONTROL_RATE_HZ = 20.0
TIME_WARNING_THRESHOLD_MS = (1.0 / CONTROL_RATE_HZ) * 1000.0 * 0.8

# 动捕协议字段索引。当前协议中 jointRotations 从 unpacked[8] 开始，
# jointPositions 从 unpacked[208] 开始。
LEFT_HAND_INDEX = 5
RIGHT_HAND_INDEX = 9
ROTATION_START_INDEX = 8
POSITION_START_INDEX = 208
ROTATION_STRIDE = 4
POSITION_STRIDE = 3
QUAT_ORDER = "xyzw"  # scipy 默认顺序为 x, y, z, w
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
    motors,
    right_init_position,
    right_init_rotation,
)

# 坐标系标定矩阵。正式上真机前必须替换为实测标定：
# T_base_ee = T_BASE_MOCAP @ T_mocap_marker @ T_MARKER_EE
T_BASE_MOCAP = np.eye(4)
T_LEFT_MARKER_EE = np.eye(4)
T_RIGHT_MARKER_EE = np.eye(4)

# 机器人通信。默认对齐 IKAbility.py 中的 rosbridge 真机链路。
COMMAND_BACKEND = "rosbridge"  # "rosbridge" | "ros2_joint_state"
ROSBRIDGE_HOST = "127.0.0.1"
ROSBRIDGE_PORT = 9090
ARM_STATUS_TOPIC = "/arm/status"
ARM_STATUS_TYPE = "bodyctrl_msgs/msg/MotorStatusMsg"
ARM_CMD_TOPIC = "/arm/cmd_pos"
ARM_CMD_TYPE = "bodyctrl_msgs/msg/CmdSetMotorPosition"

# ROS2 JointState 仅作为备选后端，不作为默认真机链路。
ROS2_JOINT_STATE_TOPIC = "/joint_states"
ROS2_JOINT_COMMAND_TOPIC = "/set_joint_positions"

# 安全参数。关节顺序需与 UBTECH_TainGong_ArmIK reduced robot 输出保持一致。
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

# 运行保护
REQUIRE_ARM_STATE_BEFORE_IK = True
DROP_STALE_MOCAP_SECONDS = 0.3
ENABLE_IK_ERROR_LOG = False
ENABLE_ROBOT_COMMAND = False  # 离线验证通过后再改 True

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


LEFT_INIT_WRIST = make_transform(left_init_position, left_init_rotation)
RIGHT_INIT_WRIST = make_transform(right_init_position, right_init_rotation)


class MocapBridgeNode(Node):
    def __init__(self):
        super().__init__('mocap_bridge_node')

        if CONTROL_SIDE not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported CONTROL_SIDE: {CONTROL_SIDE}")

        self.ik_solver = UBTECH_TainGong_ArmIK()

        self.rosbridge_client = None
        self.arm_status_sub = None
        self.arm_cmd_pub = None
        self.ros2_cmd_pub = None
        self.ros2_joint_sub = None

        self.current_lr_arm_motor_q = None
        self.last_arm_state_time = None
        self.last_command_q = None
        self._warned_missing_arm_state = False
        self._warned_command_disabled = False
        self.log_frame_index = 0
        self.local_logger = self._init_local_logger()
        self.state_lock = threading.Lock()

        self.left_motors = copy.deepcopy(motors[:7])
        self.right_motors = copy.deepcopy(motors[7:])
        self.all_motors = copy.deepcopy(motors)

        self._init_robot_io()

        self.udp_ip = UDP_IP
        self.udp_port = UDP_PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self._socket_closed = False

        self.latest_mocap_data = None
        self.data_lock = threading.Lock()

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.udp_thread = threading.Thread(target=self.udp_receive_loop, daemon=True)
        self.udp_thread.start()

        self.get_logger().info(
            f"动捕 IK 桥接节点已启动，UDP={self.udp_port}, side={CONTROL_SIDE}, "
            f"backend={COMMAND_BACKEND}, publish={ENABLE_ROBOT_COMMAND}"
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

            self.arm_status_sub = roslibpy.Topic(
                self.rosbridge_client,
                ARM_STATUS_TOPIC,
                ARM_STATUS_TYPE,
            )
            self.arm_status_sub.subscribe(self.arm_status_callback)

            self.arm_cmd_pub = roslibpy.Topic(
                self.rosbridge_client,
                ARM_CMD_TOPIC,
                ARM_CMD_TYPE,
            )
            self.arm_cmd_pub.advertise()
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
        """后台线程只接收和解析最新一帧，避免在这里做 IK 阻塞 UDP 缓冲。"""
        while rclpy.ok():
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

    def arm_status_callback(self, msg):
        """rosbridge /arm/status 回调，提取 14 个手臂关节角作为 IK 初值。"""
        try:
            status = msg.get("status", [])
            q = self.extract_arm_q_from_status(status)
            with self.state_lock:
                self.current_lr_arm_motor_q = q
                self.last_arm_state_time = time.time()
                self._warned_missing_arm_state = False
        except Exception as e:
            self.get_logger().warn(f"解析 {ARM_STATUS_TOPIC} 失败: {e}")

    def ros2_joint_state_callback(self, msg: JointState):
        """ROS2 JointState 备选后端，按 IK_ARM_JOINT_NAMES 顺序提取关节角。"""
        try:
            name_to_index = {name: index for index, name in enumerate(msg.name)}
            q = [msg.position[name_to_index[name]] for name in IK_ARM_JOINT_NAMES]
            with self.state_lock:
                self.current_lr_arm_motor_q = np.asarray(q, dtype=float)
                self.last_arm_state_time = time.time()
                self._warned_missing_arm_state = False
        except Exception as e:
            self.get_logger().warn(f"解析 {ROS2_JOINT_STATE_TOPIC} 失败: {e}")

    @staticmethod
    def extract_arm_q_from_status(status: list[dict[str, Any]]) -> np.ndarray:
        if len(status) < 14:
            raise ValueError(f"arm status length < 14: {len(status)}")

        if all("name" in item for item in status):
            by_id = {int(item["name"]): float(item["pos"]) for item in status}
            if all(motor_id in by_id for motor_id in ARM_MOTOR_IDS):
                return np.asarray([by_id[motor_id] for motor_id in ARM_MOTOR_IDS], dtype=float)

        return np.asarray([float(item["pos"]) for item in status[:14]], dtype=float)

    def control_loop(self):
        """定时器回调：取最新动捕和最新关节状态，执行 IK 并发布命令。"""
        with self.data_lock:
            mocap_data = copy.deepcopy(self.latest_mocap_data)

        if mocap_data is None:
            return

        self.log_frame_index += 1

        if time.time() - mocap_data["timestamp"] > DROP_STALE_MOCAP_SECONDS:
            self.get_logger().warn("动捕数据超时，跳过本帧控制")
            return

        with self.state_lock:
            current_q = None if self.current_lr_arm_motor_q is None else self.current_lr_arm_motor_q.copy()

        if current_q is None and REQUIRE_ARM_STATE_BEFORE_IK:
            if not self._warned_missing_arm_state:
                self.get_logger().warn(f"尚未收到 {ARM_STATUS_TOPIC}，暂不执行 IK")
                self._warned_missing_arm_state = True
            return

        if current_q is None:
            current_q = np.zeros(14)

        mocap_data["current_q"] = current_q
        self.process_mocap_data(mocap_data)

    def process_mocap_data(self, mocap_data):
        """核心处理管线：坐标转换 -> IK -> 限幅 -> 下发。"""
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

        try:
            sol_q, _ = self.ik_solver.solve_ik(
                left_wrist=left_wrist,
                right_wrist=right_wrist,
                current_lr_arm_motor_q=mocap_data["current_q"],
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
                    "current_q": mocap_data["current_q"],
                    "left_wrist": left_wrist,
                    "right_wrist": right_wrist,
                    "sol_q": sol_q,
                },
            )

        safe_q = self.apply_safety(sol_q, mocap_data["current_q"])
        if safe_q is None:
            return

        self.publish_to_robot(safe_q)

        cost_ms = (time.perf_counter() - start_time) * 1000.0
        if cost_ms > TIME_WARNING_THRESHOLD_MS:
            self.get_logger().warn(f"性能警告: IK及管线耗时 {cost_ms:.2f} ms！系统即将跟不上 {CONTROL_RATE_HZ}Hz 帧率！")

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

        reference_q = self.last_command_q if self.last_command_q is not None else current_q
        if MAX_JOINT_DELTA_PER_STEP is not None and MAX_JOINT_DELTA_PER_STEP > 0:
            safe_q = np.clip(
                safe_q,
                reference_q - MAX_JOINT_DELTA_PER_STEP,
                reference_q + MAX_JOINT_DELTA_PER_STEP,
            )

        self.last_command_q = safe_q.copy()
        return safe_q

    def publish_to_robot(self, safe_q: np.ndarray):
        """将 14 关节 IK 结果按当前控制侧写入机器人命令。"""
        if CONTROL_SIDE == "left":
            command_motors = copy.deepcopy(self.left_motors)
            command_positions = safe_q[:7]
        elif CONTROL_SIDE == "right":
            command_motors = copy.deepcopy(self.right_motors)
            command_positions = safe_q[7:]
        else:
            command_motors = copy.deepcopy(self.all_motors)
            command_positions = safe_q

        if COMMAND_BACKEND == "rosbridge":
            for motor, value in zip(command_motors, command_positions):
                motor["pos"] = float(value)

            msg = {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "cmds": command_motors,
            }
            if ENABLE_ROBOT_COMMAND:
                self.arm_cmd_pub.publish(msg)
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
                    },
                )
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if CONTROL_SIDE == "left":
            msg.name = IK_ARM_JOINT_NAMES[:7]
        elif CONTROL_SIDE == "right":
            msg.name = IK_ARM_JOINT_NAMES[7:]
        else:
            msg.name = IK_ARM_JOINT_NAMES
        msg.position = [float(value) for value in command_positions]
        if ENABLE_ROBOT_COMMAND:
            self.ros2_cmd_pub.publish(msg)
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

        if self.arm_cmd_pub is not None:
            try:
                self.arm_cmd_pub.unadvertise()
            except Exception:
                pass

        if self.rosbridge_client is not None:
            try:
                self.rosbridge_client.terminate()
            except Exception:
                pass

        super().destroy_node()


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("动捕四元数模长过小")
    return quat / norm


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
    rclpy.init(args=args)
    node = MocapBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点正在退出...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()