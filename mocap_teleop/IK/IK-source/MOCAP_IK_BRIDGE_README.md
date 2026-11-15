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
- `std_msgs`
- `numpy`
- `scipy`
- `requests`
- `roslibpy`
- `pinocchio`

可以使用仓库内的依赖文件一次性安装 pip 依赖：

```bash
cd IK/IK-source
python -m pip install -r requirements_mocap_ik.txt
```

注意：`rclpy`、`sensor_msgs` 和 `std_msgs` 属于 ROS 2 环境，不通过 pip 安装。真机 ROS 节点需要 `rclpy` 和 `sensor_msgs`；仿真 HTTP 节点只有在启用 ROS2 关节反馈时才需要 `rclpy` 和 `std_msgs`。

当前遥操作 IK 求解器来自：

```text
IK/IK-source/teleop_ik_core.py
```

机器人模型资产位于：

```text
IK/IK-source/models/walker_tienkung_ei_description/walker_tienkung_ei.urdf
```

`teleop_ik_core.py` 会加载该 URDF，并使用 Pinocchio FK/Jacobian + DLS 小步增量求解，不再依赖 CasADi、IPOPT 或旧 Ability 封装。

### 2.2 确认机器人 rosbridge

默认真机链路固定使用 rosbridge：

```python
COMMAND_BACKEND = "rosbridge"
ROSBRIDGE_HOST = "192.168.1.110"
ROSBRIDGE_PORT = 9091
```

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

进入 IK 根目录，打开统一 CLI 菜单：

```bash
cd /home/lenovo/dzy/mocap_teleop/IK
conda run -n mocap_ik python run_ik.py
```

菜单只负责选择功能，端口、时长、路径、控制侧、限幅、安全阈值等参数统一写在：

```text
/home/lenovo/dzy/mocap_teleop/IK/run_ik_config.json
```

主菜单入口：

```text
1. 环境检测
2. VR UDP 数采
3. 数采动作回放
4. 启动实时动捕遥操作
0. 退出
```

环境检测里包含二级菜单：

```text
1. 基础环境检查
2. 完整自测：编译 + 模拟 UDP 数采
3. 机械臂链路检测：直接执行测试动作并自动复位
0. 返回主菜单
```

参数作用域：

```text
runtime      # Python / conda 环境 / 菜单结束后是否暂停
rosbridge    # 固定真机链路，默认 192.168.1.110:9091
record       # VR UDP 数采参数
replay       # 数采动作回放参数，默认先做数据检查 + IK 预览，不直接上机器人
environment  # 基础检查和完整自测参数
arm_test     # 机械臂链路检测参数，菜单执行时直接动作并自动复位
teleop       # 实时动捕遥操作启动参数
```

数采入口只监听 VR UDP 并落盘，不连接 rosbridge，不发布 `/arm/cmd_pos`。后续录制主格式固定为 CSV，文件名按 `vr_时间戳.csv` 生成；CSV 列结构以样例文件为模板：

```text
/home/lenovo/dzy/mocap_teleop/IK/data/launcher_test/vr_20260710_113621.csv
```

回放入口优先读取 `vr_*.csv` 或包含该文件的数采目录；历史 `frames.jsonl` 仅作为兼容格式保留，不再作为后续默认格式。默认 `replay.execute=false`，用于先检查数据和 IK 输出；需要真实回放给机器人时，在配置文件里把 `replay.execute` 改成 `true`。

安全边界：样例 CSV 前 200 帧离线 IK 预览可通过，但相对零基线的命令最大偏移约 `1.65 rad`，会被默认 `max_command_delta_from_baseline=0.5 rad` 拦截。这是正确的保护行为。`--allow-risk` 只用于离线分析，不代表可以直接上机器人。

真机回放必须在能连接 `192.168.1.110:9091` 的环境中执行，并先通过 `/arm/status` 读取当前 14 个手臂电机 baseline；命令只通过 rosbridge 发布到 `/arm/cmd_pos`。

机械臂链路检测会直接发布 `/arm/cmd_pos` 测试动作并自动复位，不再 dry-run，也不再要求二次确认。

遥操作入口会启动 `mocap_ros_2_bridge_node.py`，当前默认 `ENABLE_ROBOT_COMMAND=True`，即会通过 rosbridge 向机器人下发命令。

启动后日志会显示：

```text
动捕 IK 桥接节点已启动，UDP=15000, side=right, backend=rosbridge, publish=True
```

其中 `publish=True` 表示会通过 rosbridge 向机器人发布动作命令。

## 4. 真机发布开关

文件顶部默认：

```python
ENABLE_ROBOT_COMMAND = True
```

该配置表示：

- 会接收动捕数据
- 会订阅机器人当前关节状态
- 会执行 IK 求解和安全检查
- 会向机器人发布动作命令

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

### 5.2 CSV 数采与回放输入

后续录制和回放的主格式是 CSV：

```text
vr_时间戳.csv
```

CSV 模板来源：

```text
/home/lenovo/dzy/mocap_teleop/IK/data/launcher_test/vr_20260710_113621.csv
```

CSV 适配边界集中在：

```text
IK/IK-source/mocap_csv_format.py
```

职责划分：

```text
完整骨架 CSV
    -> mocap_csv_format.py
    -> left_pos / left_quat / right_pos / right_quat
    -> RelativeMocapTargetMapper
    -> TeleopArmIK
    -> /arm/cmd_pos
```

录制器和回放器不直接理解复杂骨架列结构。后续 CSV 字段变化时，优先只改 `mocap_csv_format.py`，不要把列名解析扩散到控制链路。

当前样例 CSV 检查结果：

- 共 `2957` 帧、`473` 列。
- 时间戳单调。
- 左右手关键字段存在。
- 左右手四元数接近单位长度。
- 可解析成 IK 所需左右手目标帧。
- 可构造 `/arm/cmd_pos` 消息。
- 前 `5` 帧和前 `200` 帧离线 IK 预览通过。

注意：当前环境曾无法连接 `192.168.1.110:9091`，因此这里只能证明数据、IK 和命令构造链路成立；真实发送必须在 rosbridge 可达环境中验证。

### 5.3 机器人当前状态输入

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

每个关节的运行限幅。最终范围为模型 URDF 机械限位与配置限幅的交集；配置只能收紧机械限位，不能扩大它。非有限值、反向区间和空交集会被拒绝。

`MAX_JOINT_DELTA_PER_STEP`

单次命令相对最新实测反馈及上一条已提交命令允许的最大变化量，单位 rad。反馈保持不动时，命令不能逐帧累积越过这个范围；`0` 表示冻结。

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

实时桥接在发送前再次检查动捕、手臂、腰部及启用的手部反馈时效，使用本机单调时钟，不受系统时间校准影响。新收到的无效状态会使旧缓存失效。具名手臂反馈必须包含全部 14 个约定电机 ID，拒绝重复、缺失、混合命名及错误状态；保留恰好 14 项的旧式无名顺序反馈。

回放在普通、streaming 和手部模式下同样检查连接、最新实测反馈、URDF 限位和单步范围。预备姿态与结束恢复采用有限步数和超时的受限移动；反馈不跟进、过期或发送异常时停止，不会在异常后跳回基线。

命令状态只在本地 `publish()` 成功返回后更新。实时控制检测到断线或发送异常后停发本次会话；检查连接及实测姿态后重新启动。过期帧只跳过，不会锁住正常会话。停止发布不是控制器急停，也不会撤销已经接收的运动。连接检查与异步传输之间仍有竞态，本地发布成功也不等于控制器执行确认。

工作空间与手腕标定的运行路径由 `run_ik_config.json` 中各自的 `runtime_path` 显式启用，默认 `null`。生成输出不会自动投入控制。省略标定时使用既有相对映射及 skeleton 腕点；显式给出的缺失、无效或旧控制点文件会报错。旧 hand-frame 标定及原始采集保留，需重新采集 anatomical wrist 数据才能生成有效的 wrist_roll 标定，步骤见 [工作空间标定说明](../WORKSPACE_CALIBRATION.md)。

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

两者现在共用遥操作专用 IK 内核：

```python
TeleopArmIK
```

该内核定义在：

```text
IK/IK-source/teleop_ik_core.py
```

输入输出结构定义在：

```text
IK/IK-source/teleop_ik_types.py
```

下发适配器定义在：

```text
IK/IK-source/teleop_ik_adapters.py
```

真机控制话题仍然沿用：

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

## 13. 遥操作专用 IK 输入输出

遥操作 IK 不再以“全局最优一次性到位”为目标，而是每帧围绕上一帧指令做小步增量修正。
这种结构更适合动捕高频下发：耗时更稳定，关节输出更连续，也更容易在安全层截断异常目标。

### 13.1 输入结构

核心输入由 `TeleopIKInput` 描述：

```python
TeleopIKInput(
    timestamp=frame_time,
    control_side="right",
    target_pose_l=left_wrist_4x4,
    target_pose_r=right_wrist_4x4,
    current_q=current_14d_joint_position,
    last_q_cmd=last_14d_joint_command,
)
```

字段含义：

- `timestamp`: 动捕帧时间戳。
- `control_side`: 控制侧，支持 `"left"`、`"right"`、`"both"`。
- `target_pose_l`: 左手腕目标位姿，4x4 齐次矩阵，坐标系为机器人 base。
- `target_pose_r`: 右手腕目标位姿，4x4 齐次矩阵，坐标系为机器人 base。
- `current_q`: 当前 14 维手臂关节角。
- `last_q_cmd`: 上一帧 14 维下发关节角，用于做连续小步修正。

### 13.2 输出结构

核心输出由 `IKCommand` 描述：

```python
IKCommand(
    timestamp=frame_time,
    control_side="right",
    joint_names=[...],
    positions=np.ndarray(shape=(14,)),
    motor_ids=[...],
    frame_id="base",
    meta={"solver": "teleop_dls"},
)
```

输出约束：

- `positions` 始终是完整 14 维关节角。
- 下发到单臂、双臂、仿真 HTTP 或 rosbridge 时，由 `teleop_ik_adapters.py` 负责切片和格式转换。
- 求解层不直接依赖 ROS、HTTP、UDP，避免通信协议反向污染 IK 核心。

### 13.3 核心函数

主要函数位于 `teleop_ik_core.py`：

```python
TeleopArmIK.solve(ik_input)
```

职责：

- 输入一帧 `TeleopIKInput`。
- 输出一帧 `TeleopIKResult`。
- 内部执行少量 DLS 增量迭代。
- 单臂遥操作时冻结非控制侧，防止另一只手被最小二乘牵连。

```python
compute_pose_error(current_pose, target_pose, config)
```

职责：

- 输出 6 维末端误差：位置误差 + 姿态误差。
- 对位置和姿态误差限幅，避免动捕跳变直接变成大关节步长。

```python
compute_damped_increment(jacobian, error, damping)
```

职责：

- 使用 Damped Least Squares 求关节增量。
- 接近奇异位形时避免普通伪逆产生过大关节速度。

```python
apply_joint_safety(q_next, q_ref, limits, max_step, smoothing_alpha)
```

职责：

- 检查 NaN/Inf。
- 应用关节限位。
- 限制单帧最大关节变化量。
- 根据上一帧指令做平滑，保证遥操作手感连续。

### 13.4 下发适配器

主要函数位于 `teleop_ik_adapters.py`：

```python
build_rosbridge_motor_message(command, motor_templates)
```

把统一 `IKCommand` 转换为 `/arm/cmd_pos` 的 rosbridge 消息。

```python
build_http_payload(command, payload_format, send_scope)
```

把统一 `IKCommand` 转换为仿真 HTTP 请求体。

```python
select_command_slice(command, send_scope)
```

根据 `control_side` 和 `send_scope` 决定发送完整 14 维，还是只发送当前控制侧 7 维。

### 13.5 遥操作 IK 本地日志

`TeleopArmIK` 会在初始化和每帧求解时输出结构化日志，默认写入：

```text
IK/IK-source/log/teleop_ik_core.log
```

默认同时输出到控制台，便于观察模型加载耗时。相关配置在 `TeleopIKConfig` 中：

```python
TeleopIKConfig(
    enable_local_log=True,
    log_dir=None,
    log_file_name="teleop_ik_core.log",
    log_to_console=True,
    show_model_load_progress=True,
    model_load_progress_interval_seconds=5.0,
    log_ik_result=True,
    log_every_n_frames=1,
)
```

主要日志 section：

- `local_log_ready`: 本地日志文件路径。
- `model_load_start`: 开始加载 URDF 模型。
- `model_load_progress`: URDF 阻塞加载期间的周期性计时心跳，默认每 5 秒输出一次。
- `model_load_progress_done`: URDF 阻塞加载结束时的总耗时。
- `model_load_stage`: 模型加载分段耗时，包括完整模型构建、reduced model 构建、运行数据初始化。
- `model_load_done`: 模型加载总耗时。
- `ik_result`: 单帧 IK 求解结果，包括输入目标、输出 14 维关节角、耗时、误差、迭代次数和限幅状态。

如果控制频率较高且日志量过大，可以调大：

```python
log_every_n_frames=10
```

