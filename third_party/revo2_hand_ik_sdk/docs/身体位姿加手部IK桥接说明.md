# Revo2 身体位姿 + 手部 IK 桥接说明

## 1. 概述

`fullbody_hand_ik_bridge.py` 是一个实时 UDP 桥接脚本，用于把 VR/动捕设备发送的新版 Formotion G2 数据转换成下游程序可直接消费的 JSON 数据。

输入：

```text
UDP 15000
Formotion G2 Mocap Frame
1860 字节
64 个输入关节位姿
```

输出：

```text
UDP 51000，或终端打印
身体 24 个关节位姿
Revo2 手部 12 路 IK 角度，单位 rad
```

这一版输出中不包含手指关节的位置和姿态。手部控制结果只使用 `hand_ik.joint_positions`。

## 2. 数据链路

```text
VR / 动捕设备
  ↓ UDP 15000
fullbody_hand_ik_bridge.py
  ↓
解析 Formotion G2 1860 字节动捕包
  ↓
保留身体 24 个关节位姿
  ↓
提取左右手 IK 所需点位
  ↓
调用 Revo2 IK SDK
  ↓
输出 12 路手部角度 rad
  ↓
输出 JSON 到 UDP 51000，或打印到终端
```

## 3. 目录位置

交付包解压后，主要文件位置：

```text
revo2_hand_ik_sdk/
├── BODY24_HAND12_BRIDGE_README.md
├── bin/
│   └── conf/
│       └── mocap_ik/
│           └── mocap_ik_revo2_dual_hand_sdk.json
├── docs/
│   └── 身体位姿加手部IK桥接说明.md
├── lib/
│   ├── librevo2_ik_sdk.so
│   └── 其他依赖 .so
└── python/
    ├── examples/
    │   └── fullbody_hand_ik_bridge.py
    └── revo2_hand_ik/
```

运行入口：

```text
python/examples/fullbody_hand_ik_bridge.py
```

## 4. 运行环境

建议环境：

```text
Ubuntu 20.04 / 22.04 / 24.04
Python 3.10 或更高版本
```

运行前需要让系统能找到 SDK 自带的 `.so` 动态库：

```bash
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
```

这不是编译动态库，也不是用户自己写动态库。它只是告诉 Linux：运行时优先从当前 SDK 目录的 `lib/` 里查找 SDK 自带库，例如：

```text
lib/librevo2_ik_sdk.so
lib/libstdc++.so.6
lib/libpinocchio_default.so
```

## 5. 启动桥接脚本

进入 SDK 目录：

```bash
cd revo2_hand_ik_sdk
```

设置动态库路径：

```bash
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
```

### 5.1 只验证输入和 IK 解算

该模式不发送输出 UDP，只在终端打印摘要：

```bash
python3 python/examples/fullbody_hand_ik_bridge.py \
  --listen-ip 0.0.0.0 \
  --listen-port 15000 \
  --print-only
```

如果收到有效数据，会看到类似：

```text
[fullbody-hand-bridge][FRAME] seq=1 timestamp_ms=... body_joints=24 hand_angles=12
```

含义：

```text
body_joints=24   输出身体 24 个关节位姿
hand_angles=12   输出 12 路手部 IK 角度
```

如果没有数据，会看到：

```text
waiting_for_udp=1
```

表示程序已启动，正在等待 VR/动捕设备向 `15000` 端口发送 UDP 数据。

### 5.2 打印完整 JSON

用于检查完整输出结构：

```bash
python3 python/examples/fullbody_hand_ik_bridge.py \
  --listen-ip 0.0.0.0 \
  --listen-port 15000 \
  --print-json
```

### 5.3 输出到下游 UDP

输出到下游 UDP：

```bash
python3 python/examples/fullbody_hand_ik_bridge.py \
  --listen-ip 0.0.0.0 \
  --listen-port 15000 \
  --output-ip <下游程序IP> \
  --output-port 51000
```

如果下游程序在另一台机器，把 `--output-ip` 改成下游机器 IP。

## 6. 输入协议说明

桥接脚本使用 SDK 内部的 `revo2_hand_ik.fss_udp` 解析输入 UDP。

输入格式：

```text
AA 55 | uint16 id=0x0000 | uint16 length | uint64 timestamp_us | float[64][7] joints_pose | tail
```

每个输入关节：

```text
x, y, z, w, qx, qy, qz
```

说明：

- 输入仍然是 64 个关节。
- 每个关节包含位置和四元数。
- 实测设备常见 UDP 总包长为 `1860` 字节。
- 实测设备常见 Length 字段为 `1850`。
- 输入中的手指关节仍用于 IK 解算，但不会作为手指位姿输出到 `body.joints`。

## 7. 输出 JSON 说明

顶层结构：

```json
{
  "protocol": "revo2_body_pose_plus_hand_ik_v1",
  "sequence": 1,
  "timestamp_ms": 123456,
  "source": {},
  "body": {},
  "hand_ik": {}
}
```

### 7.1 body

`body` 只包含身体 24 个关节：

```text
head
neck
chest
spine1
spine2
waist
left_collar
left_shoulder
left_elbow
left_wrist
left_hand
left_hip
left_knee
left_ankle
left_foot
right_collar
right_shoulder
right_elbow
right_wrist
right_hand
right_hip
right_knee
right_ankle
right_foot
```

示例：

```json
{
  "joint_count": 24,
  "joint_order": ["head", "neck", "chest"],
  "position_unit": "same_as_input",
  "quaternion_order": "wxyz",
  "joints": {
    "left_hand": {
      "pos": [0.0, 0.0, 0.0],
      "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

不会出现在 `body.joints` 中的手指位姿：

```text
left_thumb_*
left_index_*
left_middle_*
left_ring_*
left_pinky_*
right_thumb_*
right_index_*
right_middle_*
right_ring_*
right_pinky_*
```

### 7.2 hand_ik

`hand_ik` 包含 12 路手部 IK 角度，单位为 `rad`。

固定顺序：

```text
left_thumb_flex
left_thumb_abduct
left_index
left_middle
left_ring
left_pinky
right_thumb_flex
right_thumb_abduct
right_index
right_middle
right_ring
right_pinky
```

示例：

```json
{
  "joint_count": 12,
  "joint_unit": "rad",
  "joint_order": [
    "left_thumb_flex",
    "left_thumb_abduct",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb_flex",
    "right_thumb_abduct",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky"
  ],
  "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "joint_positions_named": {},
  "left_calibrated": true,
  "right_calibrated": true
}
```

## 8. 常用参数

```text
--listen-ip         输入 UDP 绑定地址，默认 0.0.0.0
--listen-port       输入 UDP 端口，默认 15000
--source-ip         只接受指定发送端 IP，默认不限制
--output-ip         输出 UDP 目标 IP
--output-port       输出 UDP 目标端口，默认 51000
--print-only        只打印摘要，不发送输出 UDP
--print-json        打印完整 JSON
--max-frames        解算 N 帧后退出，0 表示一直运行
--sdk-config        双手 IK 配置，默认 bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json
--library           可选，手动指定 librevo2_ik_sdk.so
```

## 9. 网络配置建议

VR/动捕设备需要把 UDP 数据发送到运行桥接脚本的 Ubuntu 主机：

```text
<Ubuntu主机IP>:15000
```

查看 Ubuntu 主机 IP：

```bash
hostname -I
```

或者：

```bash
ip addr
```

如果 Ubuntu 主机有多块网卡，请确认 VR/动捕设备和 Ubuntu 主机处于同一网络，并把数据发到正确的网卡 IP。

## 10. 常见问题

### 10.1 一直 waiting_for_udp=1

说明程序已启动，但没有收到 UDP 数据。检查：

- VR/动捕设备目标 IP 是否是 Ubuntu 主机 IP。
- 端口是否是 `15000`。
- Ubuntu 防火墙是否拦截 UDP。
- VR/动捕设备和 Ubuntu 主机是否在同一网络。

### 10.2 received 增加，但 dropped 增加

说明收到了 UDP，但协议解析失败。检查：

- 是否发送新版 Formotion G2 1860 字节协议。
- Header 是否是 `AA 55`。
- ID 是否是 `0x0000`。
- Length 字段是否符合 SDK 支持的写法。

### 10.3 ctypes ABI size mismatch

说明 Python 文件和 `lib/librevo2_ik_sdk.so` 不是同一个 SDK 版本。不要混用不同版本文件，重新干净解压同一个交付包。

### 10.4 找不到 librevo2_ik_sdk.so

确认在 `revo2_hand_ik_sdk` 目录执行过：

```bash
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
```

并确认文件存在：

```bash
ls lib/librevo2_ik_sdk.so
```

### 10.5 body_joints=24 是什么意思

输入协议仍然包含 64 个关节，但输出 JSON 的 `body.joints` 只保留 24 个身体关节，过滤掉了手指位姿。

### 10.6 hand_angles=12 是什么意思

表示 Revo2 IK SDK 输出了 12 路手部角度，单位是 `rad`。
