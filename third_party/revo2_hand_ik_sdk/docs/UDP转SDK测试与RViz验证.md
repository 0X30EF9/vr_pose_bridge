# UDP 转 SDK 测试与 RViz 验证

## 1. 测试目的

该流程用于交付前验证 Python SDK，而不是改变 SDK 正式接口边界。

正式 SDK：

```text
业务程序 -> Revo2DualHandIkSdk.solve(...) -> joint_positions(rad)
```

测试链路：

```text
Formotion G2 UDP 或 CSV 回放 UDP
  -> python/examples/udp_to_sdk_test.py
  -> Revo2DualHandIkSdk.solve(...)
  -> tools/control_udp_to_rviz.py
  -> RViz
```

## 2. 一键启动 SDK 测试

```bash
./scripts/run_sdk_test.sh
```

默认行为：

- 监听 Formotion G2 UDP：`0.0.0.0:15000`
- 调用 Python SDK 接口
- 输出 `joint_positions(rad)`
- 向标准 `/joint_states` 发布主动关节和 mimic 可视关节
- 启动 `robot_state_publisher`
- 启动 RViz
- 发布 raw/mapped/task/FK 调试显示

## 3. 使用 CSV 回放

终端 1：

```bash
./scripts/run_sdk_test.sh
```

终端 2：

```bash
./tools/csv_udp_replay.py your_record.csv
```

首次使用新 CSV 时建议先检查格式：

```bash
./tools/csv_udp_replay.py your_record.csv --dry-run
```

默认端口匹配：

```text
csv_udp_replay.py -> 127.0.0.1:15000
run_sdk_test.sh  -> 0.0.0.0:15000
```

## 4. 常用参数

指定配置：

```bash
./scripts/run_sdk_test.sh bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json
```

关闭 legacy `id=0x0001` CRC 检查。新版 G2 Mocap Frame 正常不需要该参数：

```bash
./scripts/run_sdk_test.sh --no-crc
```

修改监听端口：

```bash
FSS_LISTEN_PORT=15001 ./scripts/run_sdk_test.sh
```

修改 ROS domain：

```bash
REVO2_RVIZ_ROS_DOMAIN_ID=125 ./scripts/run_sdk_test.sh
```

## 5. 输出日志

默认日志目录：

```text
bin/log/SDK测试运行/
```

关键日志：

| 文件 | 作用 |
|---|---|
| `udp_to_sdk_test.log` | UDP 接收、SDK 求解、输出统计 |
| `control_udp_to_rviz.log` | RViz JointState 与 overlay 转发 |
| `robot_state_publisher.log` | URDF 发布日志 |
| `rviz2.log` | RViz 启动日志 |

## 6. 正常现象

正常运行后应满足：

```text
1. RViz 能看到双手模型
2. CSV 回放或动捕启动后，raw/mapped 骨架随手部动作变化
3. green task 与 mapped 对应点基本一致
4. yellow FK 跟随 green task
5. SDK 输出字段 joint_unit 为 rad
```

## 7. 异常排查

### 7.1 线条会动，但 RobotModel 不动

同时观察两个日志：

```bash
tail -f bin/log/SDK测试运行/udp_to_sdk_test.log
tail -f bin/log/SDK测试运行/control_udp_to_rviz.log
```

SDK 日志正常应出现：

```text
calibrated=L1/R1
moving_frames > 0
joint_abs_max > 0
```

RViz 转发日志正常应出现：

```text
moving_packets > 0
joint_subscribers >= 1
```

含义：

| 日志现象 | 判断 |
|---|---|
| `calibrated=L0/R0` | 仍在开掌标定，或输入掌面几何无效 |
| `calibrated=L1/R1` 但 `moving_frames=0` | SDK 输出关节角没有变化，继续检查 IK/映射 |
| `moving_frames>0` 但 `moving_packets=0` | SDK 到 RViz 镜像 UDP 未接通 |
| `moving_packets>0` 但 `joint_subscribers=0` | `robot_state_publisher` 未订阅 `/joint_states` |
| 三项均正常但模型不动 | 检查 `robot_state_publisher.log` 和 `/tf` |

当前版本直接发布 `/joint_states`，不再依赖自定义话题 remap。

### 7.2 RViz 完全不动

检查：

```bash
tail -f bin/log/SDK测试运行/udp_to_sdk_test.log
```

如果没有接收帧，检查 UDP 端口和 CSV 是否正在回放。

### 7.3 raw 骨架异常

说明问题在动捕数据源或 CSV 文件，不在 SDK。检查 CSV 是否为新版 Formotion G2 原生位姿 CSV或完整 legacy FSS motion CSV。回放工具会自动识别并打包成新版 G2 UDP；IK 输出 CSV 和诊断 CSV不能作为输入。

### 7.4 mapped 与 raw 趋势不一致

检查：

- 左右手是否交换；
- 坐标系是否重复转换；
- 点序是否与已核实的 Formotion G2 64关节顺序一致，尤其是左手 15..34、右手 hand=39、右手手指 44..63；
- 手套硬件 IMU 是否存在装反或异常。

### 7.5 green task 正常但 yellow FK 不贴合

检查：

- 四指 `flex_gain`；
- 拇指极限保护参数；
- `max_command_step_rad`；
- `max_joint_velocity_rad_s`；
- URDF 关节方向和限位。

### 7.6 开掌标定状态和零位异常

v0.10.51 不再使用最先到达的 45 帧。启动后保持静态完全开掌约 2 秒，并实时查看：

```bash
tail -f bin/log/SDK测试运行/udp_to_sdk_test.log
```

正常过程会依次出现：

```text
warming_up
collecting_stable_open_hand:.../80
collecting_stable_thumb:.../60
calibrated=L1/R1
```

如果反复出现 `waiting_for_fully_open_finger`，说明至少一根四指仍有明显弯曲；如果反复出现 `restarted_after_unstable_frame`，说明手部在移动或输入点存在跳变；如果 `quality_fail` 增加，说明整段标定数据的点位或骨长离散过大。

标定完成后应满足：

```text
物理开掌 -> mapped 骨架贴合机器人 q_open
green task -> 落在 q_open 曲线上
四指输出 -> 接近初始开掌关节角
```

仍异常时检查 raw 骨架点序和骨段长度，不要用 `flex_gain` 修复零位。

## 8. 单位说明

SDK 测试链路发送到 RViz 的新版协议为：

```json
{
  "protocol": "revo2_dual_hand_ik_sdk_v2",
  "joint_unit": "rad",
  "joint_positions": []
}
```

RViz 工具收到该协议时直接按 rad 发布 `JointState`，不会做 `0..100` 归一化或反归一化。


## 9. ROS 话题检查

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --once
ros2 topic hz /tf
```

正常情况下 `/joint_states` 接近 100 Hz，消息中同时包含主动关节和 distal mimic 关节。

## 真实 UDP 无显示时的检查顺序

1. 实时查看 SDK UDP 日志：

```bash
tail -f "bin/log/SDK测试运行/udp_to_sdk_test.log"
```

2. 只检查目标端口为 15000 的数据：

```bash
sudo tcpdump -ni any -c 5 -vv 'udp dst port 15000'
```

3. 查看监听进程：

```bash
ss -H -lunp 'sport = :15000'
```

4. 限定真实设备来源 IP：

```bash
FSS_SOURCE_IP=<PICO或动捕设备IP> ./scripts/run_sdk_test.sh
```

首包会记录来源、长度和前 32 字节；协议解析失败时会直接记录大小端、长度和结构偏移判断结果。
