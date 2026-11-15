from __future__ import annotations

import copy

import numpy as np

# 遥操作链路的机器人侧固定数据。
# 这里替代旧 Ability 包里的大工具文件，只保留真机/仿真入口实际依赖的常量。
LEFT_INIT_POSITION = np.asarray([0.0250, 0.2670, -0.3909])
RIGHT_INIT_POSITION = np.asarray([0.0250, -0.2670, -0.3909])
LEFT_INIT_ROTATION = np.asarray(
    [
        [1.0, 0.0, -0.0],
        [-0.0, 0.99876, -0.1566],
        [0.0, 0.1566, 0.9876],
    ]
)
RIGHT_INIT_ROTATION = np.asarray(
    [
        [1.0, 0.0, -0.0],
        [-0.0, 0.99876, 0.1566],
        [0.0, -0.1566, 0.9876],
    ]
)

# URDF 零位作为自然下垂基准，只在 shoulder_roll 上做左右镜像外扩。
NATURAL_DOWN_OUTWARD_ROLL = 0.12
NATURAL_DOWN_Q = np.asarray(
    [
        0.0,
        NATURAL_DOWN_OUTWARD_ROLL,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -NATURAL_DOWN_OUTWARD_ROLL,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

# position-only IK 将双腕 pitch/roll 固定为本次启动反馈；其余关节不设置人为 Home 边界。
POSITION_ONLY_LOCKED_JOINT_INDICES = (5, 6, 12, 13)

MOTOR_COMMAND_TEMPLATE = {"name": 0.0, "pos": 0.0, "spd": 0.7, "cur": 2.0}
ARM_MOTOR_TEMPLATES = [copy.deepcopy(MOTOR_COMMAND_TEMPLATE) for _ in range(14)]
WAIST_JOINT_NAME = "body_yaw_joint"
WAIST_MOTOR_ID = 31
WAIST_STATUS_TOPIC = "/waist/status"
WAIST_STATUS_TYPE = "bodyctrl_msgs/msg/MotorStatusMsg"
WAIST_CMD_TOPIC = "/waist/cmd_pos"
WAIST_CMD_TYPE = "bodyctrl_msgs/msg/CmdSetMotorPosition"
WAIST_MOTOR_TEMPLATE = {"name": WAIST_MOTOR_ID, "pos": 0.0, "spd": 0.2, "cur": 8.0}
WAIST_ERROR_DESCRIPTIONS = {
    1: "关节电机过温",
    2: "过流",
    3: "电压过低",
    4: "关节 mos 过温",
    5: "堵转",
    6: "电压过高",
    7: "缺相",
    8: "编码器错误",
    33072: "设备掉线",
    33073: "关节位置超限",
}

HAND_JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
HAND_MOTOR_IDS = (1, 2, 3, 4, 5, 6)
HAND_MOTOR_NAMES = tuple(str(motor_id) for motor_id in HAND_MOTOR_IDS)
HAND_CHANNEL_ORDER = (
    "little",
    "ring",
    "middle",
    "fore",
    "thumb_bend",
    "thumb_rotation",
)
HAND_STATE_TOPICS = {
    "left": "/inspire_hand/state/left_hand",
    "right": "/inspire_hand/state/right_hand",
}
HAND_CMD_TOPICS = {
    "left": "/inspire_hand/ctrl/left_hand",
    "right": "/inspire_hand/ctrl/right_hand",
}
HAND_ANGLE_SERVICE_TYPE = "bodyctrl_msgs/srv/set_angle_flexible"
HAND_FORCE_SERVICE_TYPE = "bodyctrl_msgs/srv/set_force"
HAND_SPEED_SERVICE_TYPE = "bodyctrl_msgs/srv/set_speed"
HAND_CLEAR_ERROR_SERVICE_TYPE = "bodyctrl_msgs/srv/set_clear_error"
HAND_ANGLE_SERVICE_TOPICS = {
    "left": "/inspire_hand/set_angle_flexible/left_hand",
    "right": "/inspire_hand/set_angle_flexible/right_hand",
}
HAND_FORCE_SERVICE_TOPICS = {
    "left": "/inspire_hand/set_force/left_hand",
    "right": "/inspire_hand/set_force/right_hand",
}
HAND_SPEED_SERVICE_TOPICS = {
    "left": "/inspire_hand/set_speed/left_hand",
    "right": "/inspire_hand/set_speed/right_hand",
}
HAND_CLEAR_ERROR_SERVICE_TOPICS = {
    "left": "/inspire_hand/set_clear_error/left_hand",
    "right": "/inspire_hand/set_clear_error/right_hand",
}

RELAXED_JOINT_LIMIT = (-6.0, 6.0)


def build_relaxed_joint_limits(
    count: int,
    joint_limit: tuple[float, float] = RELAXED_JOINT_LIMIT,
) -> list[tuple[float, float]]:
    lower, upper = map(float, joint_limit)
    if lower >= upper:
        raise ValueError("joint_limit 下界必须小于上界")
    return [(lower, upper) for _ in range(int(count))]


for motor_index, motor in enumerate(ARM_MOTOR_TEMPLATES, 11):
    motor["name"] = motor_index if motor_index < 18 else motor_index + 3