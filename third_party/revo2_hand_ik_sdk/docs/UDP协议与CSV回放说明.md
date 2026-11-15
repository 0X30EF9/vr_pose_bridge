# UDP 协议与 CSV 回放说明

## 1. UDP 在当前工程中的角色

UDP 仅用于工程测试、真实动捕验证和使用例程，不是 Python SDK 正式输入接口。

```text
真实 Formotion Glove G2 UDP / CSV 回放 UDP
  -> python/examples/udp_to_sdk_test.py
  -> Revo2DualHandIkSdk.solve(...)
  -> joint_positions(rad)
  -> RViz
```

正式 SDK 由业务程序直接构造输入参数并调用 `solve(...)`。

## 2. Formotion G2 Mocap Frame

测试例程接收的数据帧：

```text
AA 55 | uint16 id=0x0000 | uint16 length=1850
      | uint64 timestamp_us | float[64][7] joint_pose | reserved

完整 UDP 数据报长度：1860 字节
```

| 字段 | 类型 | 说明 |
|---|---|---|
| Header1 | `uint8_t` | `0xAA` |
| Header2 | `uint8_t` | `0x55` |
| ID | `uint16_t` | `0x0000` |
| Length | `uint16_t` | 真机实测为 `1850`；完整 UDP 数据报仍为 `1860` 字节 |
| Timestamp | `uint64_t` | 协议时间戳，统一编码为微秒 |
| Joint Pose | `float[64][7]` | 64 个关节的世界坐标位姿 |

真机日志已确认头部：

```text
aa 55 00 00 3a 07 ...
```

按小端解析：

```text
ID     = 0x0000
Length = 0x073A = 1850
UDP数据报长度 = 1860
```

每个关节的 7 个浮点数：

```text
x, y, z, quaternion_w, quaternion_x, quaternion_y, quaternion_z
```

## 3. 已按新版 CSV 核实的 64 关节顺序

新版录制文件包含 64 组完整位姿列，顺序如下：

```text
0  head                 32 left_pinky_mid
1  neck                 33 left_pinky_far
2  chest                34 left_pinky_end
3  spine1               35 right_collar
4  spine2               36 right_shoulder
5  waist                37 right_elbow
6  left_collar          38 right_wrist
7  left_shoulder        39 right_hand
8  left_elbow           40 right_hip
9  left_wrist           41 right_knee
10 left_hand            42 right_ankle
11 left_hip             43 right_foot
12 left_knee            44 right_thumb_near
13 left_ankle           45 right_thumb_mid
14 left_foot            46 right_thumb_far
15 left_thumb_near      47 right_thumb_end
16 left_thumb_mid       48 right_index_near
17 left_thumb_far       49 right_index_mid
18 left_thumb_end       50 right_index_far
19 left_index_near      51 right_index_end
20 left_index_mid       52 right_middle_near
21 left_index_far       53 right_middle_mid
22 left_index_end       54 right_middle_far
23 left_middle_near     55 right_middle_end
24 left_middle_mid      56 right_ring_near
25 left_middle_far      57 right_ring_mid
26 left_middle_end      58 right_ring_far
27 left_ring_near       59 right_ring_end
28 left_ring_mid        60 right_pinky_near
29 left_ring_far        61 right_pinky_mid
30 left_ring_end        62 right_pinky_far
31 left_pinky_near      63 right_pinky_end
```

SDK 内部命名与协议 CSV 的差异：

| 协议 CSV | SDK 内部 |
|---|---|
| `mid` | `middle` |
| `pinky` | `little` |
| `end` | 不参与当前 IK 输入 |

当前 SDK 使用的手部索引：

| SDK 点位 | 左手索引 | 右手索引 |
|---|---:|---:|
| hand | 10 | 39 |
| thumb near/middle/far | 15/16/17 | 44/45/46 |
| index near/middle/far | 19/20/21 | 48/49/50 |
| middle near/middle/far | 23/24/25 | 52/53/54 |
| ring near/middle/far | 27/28/29 | 56/57/58 |
| little near/middle/far | 31/32/33 | 60/61/62 |

实现位置：

```text
python/revo2_hand_ik/fss_udp.py
MOCAP_G2_JOINT_NAMES
MOCAP_G2_HAND_INDEX
```

## 4. CSV 回放工具

工具：

```text
tools/csv_udp_replay.py
```

支持两类输入：

1. 新版 Formotion G2 原生 CSV；
2. 旧版 legacy FSS 47 关节 CSV。

工具会自动识别格式，统一输出新版 G2 UDP。

### 4.1 新版 G2 CSV 必需列

新版 CSV 需包含 64 组：

```text
pos_<joint>_X
pos_<joint>_Y
pos_<joint>_Z
quat_<joint>_W
quat_<joint>_X
quat_<joint>_Y
quat_<joint>_Z
```

例如：

```text
pos_left_index_mid_X
quat_left_index_mid_W
pos_right_pinky_far_Z
quat_right_pinky_far_Y
```

`systemTimestampUs` 通常存在，但不是格式识别的必要条件。当前录制器虽然将列名写为 `systemTimestampUs`，实测数值为毫秒尺度；回放工具会自动转换为协议微秒。

### 4.2 格式检查

只检查 CSV，不发送 UDP：

```bash
./tools/csv_udp_replay.py your_record.csv --dry-run
```

正常输出示例：

```text
format=formotion_g2_csv_v1
frames=2957
dry-run passed
```

### 4.3 开始回放

```bash
./tools/csv_udp_replay.py your_record.csv
```

默认值：

| 项目 | 默认值 |
|---|---|
| 目标 IP | `127.0.0.1` |
| 目标端口 | `15000` |
| 回放频率 | `100 Hz` |
| 输出协议 | G2 Mocap Frame，`id=0x0000`，`Length=1850`，UDP 数据报 `1860` 字节 |
| 回放模式 | 循环 |

指定参数：

```bash
./tools/csv_udp_replay.py your_record.csv \
  --target-ip 127.0.0.1 \
  --target-port 15000 \
  --hz 100
```

只回放一轮：

```bash
./tools/csv_udp_replay.py your_record.csv --once
```

## 5. 与 SDK + RViz 测试连接

终端 1：

```bash
./scripts/run_sdk_test.sh
```

终端 2：

```bash
./tools/csv_udp_replay.py your_record.csv
```

连接关系：

```text
csv_udp_replay.py -> 127.0.0.1:15000
run_sdk_test.sh   -> 0.0.0.0:15000
```

## 6. 常见错误

### 6.1 `unsupported CSV schema`

说明 CSV 同时不满足新版 G2 和旧版 FSS 字段要求。先执行：

```bash
head -n 1 your_record.csv
```

确认它是动捕原始位姿 CSV，而不是 IK 输出或诊断 CSV。

### 6.2 RViz 手指错位

重点检查：

1. `mid` 是否映射为 SDK 的 `middle`；
2. `pinky` 是否映射为 SDK 的 `little`；
3. 左手 finger 区间是否为 `15..34`；
4. 右手 hand 是否为索引 `39`，finger 区间是否为 `44..63`；
5. 每个四元数是否按 `w,x,y,z` 打包。

## 7. 真实设备 UDP 与 CSV 回放的差异

CSV 回放工具由本工程生成协议包，字段字节序和结构偏移是确定的；真实设备可能存在以下实现差异：

- `uint16_t` 头字段采用网络字节序；
- C/C++ 结构体在 `uint64_t timestamp` 前插入 2 字节对齐；
- 同步头按 `0xAA55` 整数的小端内存排列发送为 `55 AA`；
- 真机固定出现 `Length=1850`、UDP 数据报长度 `1860`；旧测试包可能填写总长或 payload 长度。

v0.10.49 的解析器会优先识别真机 `Length=1850` 布局，并继续兼容上述旧布局。首个有效包的日志示例：

```text
[udp-to-sdk][OK] first_valid_packet ...
wire=formotion_g2_mocap_v1,header=little,pose=little,timestamp_offset=8,pose_offset=16,sync=AA55
```

## 8. 确认 15000 是目标端口

以下命令会同时匹配源端口和目标端口，不能证明 SDK 能收到：

```bash
sudo tcpdump -ni any 'udp port 15000'
```

应使用：

```bash
sudo tcpdump -ni any -c 5 -vv 'udp dst port 15000'
```

如果只有下面的命令有数据：

```bash
sudo tcpdump -ni any -c 5 -vv 'udp src port 15000'
```

说明 15000 是发送端源端口，必须查看实际目标端口，并通过环境变量修改 SDK 监听端口：

```bash
FSS_LISTEN_PORT=<实际目标端口> ./scripts/run_sdk_test.sh
```

## 9. 实时查看接收与解析状态

```bash
tail -f "bin/log/SDK测试运行/udp_to_sdk_test.log"
```

关键状态：

| 日志 | 含义 |
|---|---|
| `received=0` | Python UDP socket 没有收到目标端口数据 |
| `received>0 solved=0 dropped>0` | 收到了 UDP，但协议字段或结构布局仍不匹配 |
| `first_valid_packet` | 已正确识别真实协议布局 |
| `solved>0 moving_frames=0` | 已调用 SDK，但输入或输出没有变化 |
| `solved>0 moving_frames>0` | UDP、协议和 SDK 链路均正常 |

检查是否有其他程序占用 15000：

```bash
ss -H -lunp 'sport = :15000'
```

v0.10.49 默认不允许端口复用。若端口已被其他接收程序占用，测试进程会直接报错，而不是静默启动后收不到数据。
