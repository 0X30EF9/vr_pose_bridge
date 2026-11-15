# Mocap IK Bridge 使用说明

本文档说明 `mocap_ros_2_bridge_node.py` 的作用、启动方式、输入输出、订阅/发布话题，以及文件顶部每个参数的含义。

该脚本用于把动捕系统通过 UDP 发来的左右手位姿转换为机器人手腕目标位姿，再调用工程中已有的 `UBTECH_TainGong_ArmIK` 求解 14 个手臂关节角，最后按机器人底层控制话题格式发送动作指令。

## 1. 数据流

```text
动捕 UDP 数据包
    -> 解析左手/右手 marker position + quaternion
    -> 动捕坐标系转换到机器人 base 坐标系
    -> 构造 left_wrist / right_wrist 4x4 齐次矩阵
    -> 调用 UBTECH_TainGong_ArmIK.solve_ik()
    -> 得到 14 个手臂关节角
    -> 关节限幅、单步变化限制、NaN 检查
    -> 发布到 /arm/cmd_pos
```

默认配置下，脚本只控制右臂，左臂保持初始位姿。这样便于先跑通单臂遥操作闭环。

## 2. 启动前准备

### 2.1 确认依赖

脚本依赖以下 Python 包和 ROS 环境：

- `rclpy`
- `sensor_msgs`
- `numpy`
- `scipy`
- `roslibpy`
- `pinocchio`
- `casadi`
- `meshcat`
- `IPOPT/CasADi` 运行环境

可以使用仓库内的依赖文件一次性安装 pip 依赖：

```bash
cd IK/IK-source
python -m pip install -r requirements_mocap_ik.txt
```

注意：`rclpy` 和 `sensor_msgs` 属于 ROS 2 环境，不通过 pip 安装。只有运行真机 ROS 节点 `mocap_ros_2_bridge_node.py` 时才需要它们；仿真 HTTP 节点 `mocap_ik_http_sim_node.py` 不依赖 ROS 2。

其中 IK 求解器来自：

```text
IK/IK-source/Ability_script/src/IK_sdk/UBTECH_IK.py
```

`UBTECH_IK.py` 内部会加载机器人 URDF，并使用 Pinocchio + CasADi + IPOPT 求解逆运动学。

### 2.2 确认机器人 rosbridge

默认真机链路使用 rosbridge：

```python
COMMAND_BACKEND = "rosbridge"
ROSBRIDGE_HOST = "127.0.0.1"
ROSBRIDGE_PORT = 9090
```

上真机前需要把 `ROSBRIDGE_HOST` 和 `ROSBRIDGE_PORT` 改成机器人实际的 rosbridge 地址。

### 2.3 确认动捕 UDP

脚本默认监听：

```python
UDP_IP = "0.0.0.0"
UDP_PORT = 15000
```

动捕端需要向运行脚本的机器发送 UDP 数据包。当前代码期望 UDP 包长度为 `1366` 字节，并使用如下结构解析：

```python
MOCAP_STRUCT_FORMAT = "<BBHHI 3f 188f 6H 6H 141f I"
```

如果动捕协议变化，需要同步修改顶部参数。

## 3. 启动方式

进入工程目录：

```bash
cd IK/IK-source
```

启动节点：

```bash
python mocap_ros_2_bridge_node.py
```

如果在 ROS 2 环境中运行，先 source 对应环境：

```bash
source /opt/ros/<distro>/setup.bash
python mocap_ros_2_bridge_node.py
```

Windows PowerShell 下可直接在已配置好 Python/ROS 环境的终端中运行：

```powershell
python .\mocap_ros_2_bridge_node.py
```

启动后日志会显示：

```text
动捕 IK 桥接节点已启动，UDP=15000, side=right, backend=rosbridge, publish=False
```

其中 `publish=False` 表示当前只计算 IK，不真正下发机器人命令。

## 4. 上真机前的安全开关

文件顶部默认：

```python
ENABLE_ROBOT_COMMAND = False
```

该配置表示：

- 会接收动捕数据
- 会订阅机器人当前关节状态
- 会执行 IK 求解和安全检查
- 不会向机器人发布动作命令

建议先保持 `False` 做离线验证。确认 IK 输出连续、坐标系正确、关节角合理后，再改为：

```python
ENABLE_ROBOT_COMMAND = True
```

## 5. 输入数据

### 5.1 动捕输入

脚本从 UDP 数据包中提取左右手 marker 位姿：

```python
left_pos
left_quat
right_pos
right_quat
```

含义：

- `left_pos`: 左手 marker 在动捕坐标系下的位置 `[x, y, z]`
- `left_quat`: 左手 marker 姿态四元数 `[x, y, z, w]`
- `right_pos`: 右手 marker 在动捕坐标系下的位置 `[x, y, z]`
- `right_quat`: 右手 marker 姿态四元数 `[x, y, z, w]`

当前动捕字段索引默认如下：

```python
LEFT_HAND_INDEX = 5
RIGHT_HAND_INDEX = 9
ROTATION_START_INDEX = 8
POSITION_START_INDEX = 208
ROTATION_STRIDE = 4
POSITION_STRIDE = 3
QUAT_ORDER = "xyzw"
```

### 5.2 机器人当前状态输入

默认订阅：

```python
ARM_STATUS_TOPIC = "/arm/status"
ARM_STATUS_TYPE = "bodyctrl_msgs/msg/MotorStatusMsg"
```

脚本会从消息中提取当前 14 个手臂关节角，作为 IK 的 warm start 初值。

当前默认电机 ID 顺序：

```python
ARM_MOTOR_IDS = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
```

含义：

- `11` 到 `17`: 左臂 7 个关节
- `21` 到 `27`: 右臂 7 个关节

如果 `/arm/status` 的消息顺序或电机 ID 定义发生变化，需要修改 `ARM_MOTOR_IDS` 或 `extract_arm_q_from_status()` 的解析逻辑。

## 6. 输出数据

### 6.1 IK 输出

`UBTECH_TainGong_ArmIK.solve_ik()` 输出 14 个关节角：

```python
sol_q[:7]   # 左臂
sol_q[7:]   # 右臂
```

关节顺序由 `IK_ARM_JOINT_NAMES` 描述：

```python
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
```

### 6.2 机器人命令输出

默认发布：

```python
ARM_CMD_TOPIC = "/arm/cmd_pos"
ARM_CMD_TYPE = "bodyctrl_msgs/msg/CmdSetMotorPosition"
```

消息结构：

```python
{
    "header": {
        "stamp": {"sec": 0, "nanosec": 0},
        "frame_id": ""
    },
    "cmds": [
        {"name": 21, "pos": 0.0, "spd": 0.3, "cur": 4.0},
        ...
    ]
}
```

`cmds` 来自 `IKAbility_utils.py` 中的 `motors` 定义。脚本会根据 `CONTROL_SIDE` 选择发送左臂、右臂或双臂。

## 7. 坐标系转换

动捕位姿不能直接送入 IK。脚本使用下面的转换关系：

```python
T_base_ee = T_BASE_MOCAP @ T_mocap_marker @ T_MARKER_EE
```

含义：

- `T_mocap_marker`: 动捕输出的手部 marker 位姿
- `T_BASE_MOCAP`: 动捕世界坐标系到机器人 base 坐标系的变换
- `T_MARKER_EE`: marker 坐标系到机器人 IK 末端坐标系的固定偏移
- `T_base_ee`: IK 求解器需要的手腕末端目标位姿

顶部默认：

```python
T_BASE_MOCAP = np.eye(4)
T_LEFT_MARKER_EE = np.eye(4)
T_RIGHT_MARKER_EE = np.eye(4)
```

这只是占位值。正式使用前必须标定，否则会出现方向反、左右错、位置偏移或 IK 目标不可达。

## 8. 顶部参数说明

### 8.1 UDP 参数

`UDP_IP`

监听的本机 IP。`"0.0.0.0"` 表示监听所有网卡。

`UDP_PORT`

动捕 UDP 数据接收端口。

`UDP_PACKET_SIZE`

单次 `recvfrom()` 最大读取字节数。

`EXPECTED_UDP_PACKET_SIZE`

期望收到的数据包长度。长度不匹配的数据包会被丢弃。

`MOCAP_STRUCT_FORMAT`

`struct.unpack()` 使用的动捕数据格式。

`CONTROL_RATE_HZ`

IK 解算和控制循环频率。建议先用 `5` 或 `10` 做真机低频测试，再提高到 `20`。

`TIME_WARNING_THRESHOLD_MS`

性能告警阈值。默认是控制周期的 80%。

### 8.2 动捕字段参数

`LEFT_HAND_INDEX`

左手 marker 在动捕 joint 数组中的索引。

`RIGHT_HAND_INDEX`

右手 marker 在动捕 joint 数组中的索引。

`ROTATION_START_INDEX`

动捕包中旋转数组在 `unpacked` 里的起始索引。

`POSITION_START_INDEX`

动捕包中位置数组在 `unpacked` 里的起始索引。

`ROTATION_STRIDE`

每个 joint 的四元数长度，通常为 `4`。

`POSITION_STRIDE`

每个 joint 的位置长度，通常为 `3`。

`QUAT_ORDER`

四元数顺序。当前是 `"xyzw"`，与 `scipy.spatial.transform.Rotation.from_quat()` 默认一致。

`MOCAP_POSITION_SCALE`

动捕位置缩放系数。如果动捕单位是毫米、而 IK 使用米，应设置为 `0.001`。

### 8.3 控制模式参数

`CONTROL_SIDE`

控制哪只手臂：

- `"left"`: 只控左臂，右臂保持初始位姿
- `"right"`: 只控右臂，左臂保持初始位姿
- `"both"`: 同时控制双臂

建议第一版使用：

```python
CONTROL_SIDE = "right"
```

### 8.4 IK 路径参数

`IK_SRC_DIR`

IK 源码目录。脚本会把它加入 `sys.path`，以导入：

```python
from IK_sdk.UBTECH_IK import UBTECH_TainGong_ArmIK
from IKAbility_utils import ...
```

### 8.5 机器人通信参数

`COMMAND_BACKEND`

通信后端：

- `"rosbridge"`: 默认真机链路，使用 `/arm/status` 和 `/arm/cmd_pos`
- `"ros2_joint_state"`: 备选调试链路，使用 ROS2 `JointState`

`ROSBRIDGE_HOST`

机器人 rosbridge IP。

`ROSBRIDGE_PORT`

机器人 rosbridge 端口。

`ARM_STATUS_TOPIC`

当前手臂状态话题，默认 `/arm/status`。

`ARM_STATUS_TYPE`

当前手臂状态消息类型，默认 `bodyctrl_msgs/msg/MotorStatusMsg`。

`ARM_CMD_TOPIC`

手臂命令话题，默认 `/arm/cmd_pos`。

`ARM_CMD_TYPE`

手臂命令消息类型，默认 `bodyctrl_msgs/msg/CmdSetMotorPosition`。

`ROS2_JOINT_STATE_TOPIC`

ROS2 备选后端的当前关节状态话题。

`ROS2_JOINT_COMMAND_TOPIC`

ROS2 备选后端的关节命令话题。

### 8.6 安全参数

`IK_ARM_JOINT_NAMES`

IK 输出的 14 个手臂关节名，用于 ROS2 `JointState` 备选后端和人工检查。

`ARM_MOTOR_IDS`

`/arm/status` 和 `/arm/cmd_pos` 使用的电机 ID 顺序。

`ARM_JOINT_LIMITS`

每个关节的安全限幅，当前默认 `[-pi, pi]`。建议根据真实机器人硬件限制替换为更严格的范围。

`MAX_JOINT_DELTA_PER_STEP`

单个控制周期允许的最大关节变化量，单位 rad。用于避免 IK 跳解导致突然大动作。

### 8.7 运行保护参数

`REQUIRE_ARM_STATE_BEFORE_IK`

为 `True` 时，必须先收到机器人当前关节状态，才开始 IK。建议上真机保持 `True`。

`DROP_STALE_MOCAP_SECONDS`

动捕数据超时时间。超过该时间未收到新数据，则跳过本帧控制。

`ENABLE_IK_ERROR_LOG`

是否打印 IK 位姿误差。调试 IK 精度时可设为 `True`。

`ENABLE_ROBOT_COMMAND`

是否真正向机器人发布动作命令。建议流程：

1. 离线验证时保持 `False`
2. 确认坐标和 IK 输出正确
3. 低频小范围测试
4. 改为 `True`

## 9. 推荐调试流程

### 9.1 离线验证

保持：

```python
ENABLE_ROBOT_COMMAND = False
CONTROL_RATE_HZ = 5.0
CONTROL_SIDE = "right"
```

确认：

- 脚本能收到 UDP 数据
- 脚本能收到 `/arm/status`
- 没有四元数异常
- IK 输出没有 NaN/Inf
- IK 耗时低于控制周期

### 9.2 单臂低频真机测试

设置：

```python
CONTROL_SIDE = "right"
CONTROL_RATE_HZ = 5.0
ENABLE_ROBOT_COMMAND = True
```

建议动捕手柄只做很小幅度移动，确认右臂运动方向正确。

### 9.3 坐标系标定

如果出现以下现象，优先检查 `T_BASE_MOCAP` 和 `T_MARKER_EE`：

- 手向前，机器人向后
- 左右方向反了
- 手腕姿态旋转 90 度或 180 度
- IK 经常失败或关节跳变
- 末端位置整体偏移很大

### 9.4 提高频率

低频稳定后，再逐步提高：

```python
CONTROL_RATE_HZ = 10.0
CONTROL_RATE_HZ = 20.0
```

如果日志频繁出现性能告警，说明 IK 或通信链路耗时接近控制周期，需要降低频率或优化 IK。

## 10. 常见问题

### 10.1 启动后提示尚未收到 `/arm/status`

检查：

- `ROSBRIDGE_HOST` 是否正确
- `ROSBRIDGE_PORT` 是否正确
- 机器人 rosbridge 是否已启动
- `/arm/status` 是否存在并发布 `bodyctrl_msgs/msg/MotorStatusMsg`

### 10.2 收不到动捕数据

检查：

- 动捕端是否向本机 IP 和 `UDP_PORT` 发送
- 防火墙是否拦截 UDP
- UDP 包长度是否等于 `EXPECTED_UDP_PACKET_SIZE`
- `MOCAP_STRUCT_FORMAT` 是否和动捕协议一致

### 10.3 IK 求解失败

可能原因：

- 目标点超出机器人可达空间
- 坐标系标定错误
- 动捕单位错误，例如毫米没有转米
- 当前关节初值错误
- 姿态约束过强

### 10.4 机器人不动

检查：

- `ENABLE_ROBOT_COMMAND` 是否为 `True`
- `COMMAND_BACKEND` 是否为 `"rosbridge"`
- `/arm/cmd_pos` 是否是当前机器人实际接收的话题
- `cmds` 中电机 ID 是否与机器人一致

### 10.5 机器人动作方向不对

优先检查：

- `T_BASE_MOCAP`
- `T_LEFT_MARKER_EE`
- `T_RIGHT_MARKER_EE`
- `MOCAP_POSITION_SCALE`
- `QUAT_ORDER`

## 11. 与 IKAbility.py 的关系

`IKAbility.py` 是能力框架封装版本，通过 HTTP/任务接口触发一次 IK 和动作下发。

`mocap_ros_2_bridge_node.py` 是遥操作连续控制版本，不走 HTTP 能力接口，而是在 ROS timer 中持续执行：

```python
receive mocap -> solve IK -> publish command
```

两者共用同一个 IK 求解器：

```python
UBTECH_TainGong_ArmIK
```

也共用同一套真机控制话题：

```python
/arm/status
/arm/cmd_pos
```

## 12. 仿真 HTTP 节点

如果只需要控制仿真机，可以使用独立脚本：

```text
IK/IK-source/mocap_ik_http_sim_node.py
```

它和真机节点的前半段一致：

```text
动捕 UDP -> 解析左右手位姿 -> 坐标系转换 -> IK -> 安全限幅
```

区别是输出端不走 rosbridge，也不发布 ROS 话题，而是通过 HTTP POST 把关节角发送给仿真机。

启动方式：

```bash
cd IK/IK-source
python mocap_ik_http_sim_node.py
```

Windows PowerShell：

```powershell
python .\mocap_ik_http_sim_node.py
```

仿真相关顶部参数：

```python
SIM_HTTP_URL = "http://127.0.0.1:8000/api/joint_command"
SIM_HTTP_TIMEOUT_SECONDS = 0.05
SIM_HTTP_RETRY = 0
SIM_SEND_JOINT_SCOPE = "full"
SIM_PAYLOAD_FORMAT = "joint_positions"
ENABLE_SIM_COMMAND = False
```

参数说明：

- `SIM_HTTP_URL`: 仿真机接收关节角的 HTTP 接口地址。
- `SIM_HTTP_TIMEOUT_SECONDS`: 单次 HTTP 请求超时时间。
- `SIM_HTTP_RETRY`: HTTP 发送失败后的重试次数。
- `SIM_SEND_JOINT_SCOPE`: 发送范围，`"full"` 发送 14 个手臂关节，`"controlled_side"` 只发送当前控制侧。
- `SIM_PAYLOAD_FORMAT`: 请求体格式，`"joint_positions"` 发送关节名和角度，`"motor_commands"` 发送电机 ID 和角度。
- `ENABLE_SIM_COMMAND`: 是否真的向仿真机发送 HTTP。默认 `False`，离线确认后再改 `True`。

默认 `joint_positions` 请求体：

```json
{
  "timestamp": 0.0,
  "control_side": "right",
  "format": "joint_positions",
  "joint_names": [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint"
  ],
  "positions": [0.0, 0.0]
}
```

如果设置：

```python
SIM_PAYLOAD_FORMAT = "motor_commands"
```

请求体会变为：

```json
{
  "timestamp": 0.0,
  "control_side": "right",
  "format": "motor_commands",
  "cmds": [
    {"name": 21, "pos": 0.0}
  ]
}
```

仿真节点没有订阅 `/arm/status`，IK 的 warm start 使用上一帧发送给仿真的关节角；第一帧使用 `INITIAL_ARM_Q = np.zeros(14)`。

仿真节点也有本地日志，默认写入：

```text
IK/IK-source/log/mocap_ik_http_sim_node.log
```

