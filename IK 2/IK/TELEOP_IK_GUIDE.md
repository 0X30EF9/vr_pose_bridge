# Formotion G2 全骨架动作模仿与 Revo2 手部 IK 说明

本文只描述当前工程实际入口，不适用于备份目录或历史副本。

- 工程根目录：`/home/lenovo/dzy/ik_update/mocap_teleop/IK`
- 源码目录：`/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source`
- 启动器：`run_ik.py`
- 配置文件：`run_ik_config.json`
- 固定回放数据：`data/vr_udp_capture/vr_20260711_175844.csv`
- 机器人模型：`IK-source/models/walker_tienkung_ei_description/walker_tienkung_ei.urdf`
- 推荐 Python：`/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10`

当前默认主路径是 `control_mode=direct`。这里的 `direct` 是命令行兼容名称，实际算法是“稳定窗口标定 + 人体骨段方向 + 机器人 URDF 骨段方向 DLS”，不再是旧的平举模板启发式映射。

## 1. 当前安全状态

`run_ik_config.json` 当前必须持续保持以下设置：

```json
{
  "replay": {
    "use_robot_status": false,
    "allow_risk": false,
    "execute": false
  },
  "teleop": {
    "robot_command": false
  },
  "arm_test": {
    "execute": false
  }
}
```

含义如下：

- 回放使用离线 baseline，不连接 rosbridge，不读取 `/arm/status`，不发布命令。
- `allow_risk=false` 会在数据或关节轨迹超过风险门限时拒绝继续。
- 实时入口禁止向 `/arm/cmd_pos` 发布。
- arm-test 不执行真机动作。
- `run_ik.py` 内置默认值也是 fail-closed；配置缺失时不会默认执行回放、arm-test 或实时发布。

注意：`replay.streaming=true` 本身不会发布。只有 `execute=true + control_mode=ik + streaming=true` 同时成立时才进入边计算边发布分支。

回放 dry-run 正常结束必须出现：

```text
DRY_RUN_ONLY no command published; add --execute after checking summaries
```

### 实时“禁止发布”不等于“完全离线”

当前实时节点即使设置 `robot_command=false`，仍会通过 rosbridge 订阅 `/arm/status`，因为双臂求解需要完整 14 轴 baseline。开发阶段若要求完全不连接机器人，不要启动 `run_ik.py teleop`；应使用固定 CSV dry-run 和本机 UDP loopback 测试。

## 2. 模块职责与数据流

| 模块 | 职责 | 明确不负责 |
| --- | --- | --- |
| `teleop_mocap_protocol.py` | G2/旧 FSS UDP 解码、64 节点规范帧、四元数边界转换 | 不做机器人 IK |
| `mocap_csv_format.py` | 历史 CSV 与 `MocapFrame` 互转、时间戳尺度识别 | 不定义另一套骨架协议 |
| `teleop_direct_retarget.py` | 人体局部坐标、稳定标定、骨段质量、方向目标、安全保持 | 不解析原始 UDP 字节 |
| `teleop_ik_core.py` | reduced URDF、FK、骨段方向 Jacobian/DLS、URDF 限位 | 不读取动捕文件格式 |
| `mocap_replay_to_robot.py` | CSV 时序、离线 baseline、诊断汇总、可选发布门控 | 不复制重定向算法 |
| `mocap_ros_2_bridge_node.py` | UDP 最新帧、实时状态门控、调用同一重定向器 | 不维护私有协议布局 |
| `teleop_hand_ik.py` | 构造手部输入、管理隔离 sidecar、保存最新本地结果 | 不创建网络发布接口 |
| `revo2_hand_ik_worker.py` | 在独立 Python 进程中加载 vendor `.so` 并求解 | 不导入天工双臂 IK |

```mermaid
flowchart LR
    Input[G2 UDP / 历史 CSV] --> Protocol[统一 MocapFrame\n64 节点]
    Protocol --> Quality[标定与骨架质量检查]
    Quality --> Human[人体局部系\nX前 Y左 Z上]
    Human --> Segments[上臂/前臂方向]
    Segments --> DLS[URDF 骨段方向 DLS]
    DLS --> Safety[URDF限位/步进/平滑/baseline包络]
    Safety --> Preview[离线关节轨迹与诊断]
    Preview -->|仅显式执行| Arm[/arm/cmd_pos 14轴]
    Protocol --> HandInput[双手节点提取]
    HandInput --> Sidecar[Revo2 本地子进程]
    Sidecar --> Local[双手合计12路结果\n仅进程内/本地CSV]
```

双臂与手部是并列支路：任一支路单帧失败不会使另一支路崩溃。实时入口会先计算本地手部结果，再检查 `/arm/status` 是否满足双臂控制门控，因此缺少机器人状态时仍可得到手部本地结果。

## 3. Formotion G2 协议

### 3.1 实测格式

- UDP 端口：`15000`
- 同步头：`0xAA 0x55`，解析器兼容反向同步头
- `Id=0x0000`
- 实测 UDP datagram：`1860` 字节
- 实测包内 `Length`：可能为 `1850`，不能要求它等于 datagram 长度
- 节点数：64
- 每节点：7 个 float
- 节点内容：世界位置 `(x,y,z)` + 世界四元数 `(w,x,y,z)`
- 新版紧凑布局：timestamp offset `6`、pose offset `14`
- 新版对齐布局：timestamp offset `8`、pose offset `16`
- 兼容旧 FSS `1366/1370` 字节包

解析器不会仅按包长选择新布局，而会对候选浮点字节序、位置有限性和多个节点的单位四元数可信度评分。这样可避免首节点损坏或结构体错位时误选布局。

### 3.2 规范帧

所有 UDP、CSV 回放和手部输入统一转换为 `MocapFrame`：

- 固定 64 个 `JointPose`
- 位置单位为米
- `timestamp_s` 单位为秒
- 保留原始 `timestamp_us`
- 内部四元数顺序统一为 `XYZW`
- `wire` 保存包长、Length、字节序、offset 和同步头信息

协议边界把 wire `WXYZ` 显式转换为内部 `XYZW`。Revo2 SDK 需要 `WXYZ` 时，由手部适配器在调用边界显式转回。

所有节点姿态均为世界全局姿态，不是父子局部姿态。需要局部旋转时必须计算：

```text
q_local = inverse(q_parent_world) * q_child_world
```

不能直接把世界欧拉角复制到机器人关节。

### 3.3 世界坐标

Formotion 定义：

```text
+X = 人体前方
+Y = 人体左侧
+Z = 上方
```

人体局部系使用：

```text
left    = normalize(left_shoulder - right_shoulder)
up      = normalize(shoulder_center - hip/waist_center)
forward = normalize(left × up)
R_world_to_body = rows(forward, left, up)
```

历史实现曾把左肩到右肩作为正横轴；该方向与 `+Y=左` 相反，当前实现已修正并有单元测试覆盖。

### 3.4 固定 CSV

`vr_20260711_175844.csv` 共 2101 帧，时长约 `22.239 s`，采样率约 `90 Hz`。字段名 `systemTimestampUs` 在该历史文件中实际呈毫秒尺度，CSV 适配器会根据数值和帧间隔识别尺度，不按字段名盲目除以一百万。

该文件动作顺序为：

```text
立正 → 侧平举 → 立正 → 前平举 → 立正
```

手腕与手节点位置重合，因此不能从两者位置差伪造掌心方向。

## 4. `direct` 骨段方向重定向

### 4.1 主任务

每侧从人体骨架提取：

```text
upper_direction   = normalize(elbow - shoulder)
forearm_direction = normalize(wrist - elbow)
```

方向转换到人体躯干局部坐标，并相对稳定 home 消偏。机器人侧从 URDF FK 得到对应上臂和前臂方向，通过方向 Jacobian 的阻尼最小二乘迭代求解 14 轴关节。

优先级为：

1. 上臂方向。
2. 前臂方向。
3. 连续性和 home/baseline 正则。
4. URDF 限位、单步限制、平滑和 baseline 包络。

当前 `direct` 主路径没有启用末端完整 pose IK，也没有启用末端位置 residual。`position_scale` 为兼容旧 CLI 保留，在 `direct` 中表示人体相对骨段转角缩放，不再表示手部位置增益。

腕 `pitch/roll` 不参与上臂/前臂方向主任务，默认由 home 正则和较小 envelope 保持。只有完成父子局部旋转、腕部语义和真机关节方向验证后，才应考虑增加低权重腕部跟随。

### 4.2 稳定窗口标定

首帧不会直接成为 home。重定向器持续收集稳定站姿窗口：

- 位置取中位数
- 骨段方向使用稳定窗口统计
- 骨长变化必须低于阈值
- 窗口未满或不稳定时保持 baseline/上一安全命令
- 标定完成后 baseline 在同一实例生命周期内不得变化；需要更换 baseline 时必须 reset 并重新标定

当前显式参数：

| 配置项 | 当前值 | 作用 |
| --- | ---: | --- |
| `calibration_frames` | 45 | 稳定 home 窗口帧数 |
| `calibration_max_direction_rms` | 0.055 rad | 窗口骨段方向 RMS 上限 |
| `calibration_max_length_cv` | 0.025 | 窗口骨长变异系数上限 |
| `max_segment_length_ratio` | 1.18 | 相对 home 骨长变化上限 |
| `max_skeleton_joint_step` | 0.10 m | 人体关节点单帧跳变上限 |
| `upper_direction_weight` | 1.0 | 上臂方向任务权重 |
| `forearm_direction_weight` | 0.85 | 前臂方向任务权重 |
| `direction_damping` | 0.08 | 方向 DLS 阻尼 |
| `direction_iterations` | 6 | 每帧 DLS 最大迭代次数 |

这些参数由 `run_ik.py` 同名接入回放和实时入口，避免两条路径出现不同标定状态机。

### 4.3 失效保持与安全层

以下情况不会继续追踪坏数据，而是保持 baseline 或上一条安全命令：

- 躯干坐标无法建立
- 肩、肘、腕节点缺失或未跟踪
- 骨段过短或过长
- 相对 home 骨长突变
- 关节点单帧跳变过大
- 标定窗口未完成或不稳定
- 求解结果包含非有限值

有效解依次经过：

```text
方向 DLS
→ URDF 关节上下界
→ 每关节重定向 envelope
→ baseline 最大偏移
→ 单帧关节步进
→ 平滑/连续性约束
```

当前 URDF 的肩部限位使标准直臂水平前平举和侧平举不可达。软件必须保留方向残差并在边界停止，不能依赖 bodyctrl 被动截断，也不能通过扭腕或过度弯肘伪装为“达到目标”。

已知近似模型边界：

- 左肩 roll：`[-0.25, 0.58]`
- 右肩 roll：`[-0.8, 0.2]`
- 左肩 pitch 前举约：`-0.68`
- 右肩 pitch 前举约：`-0.70`

## 5. 动作阶段诊断

固定 CSV 回放会根据人体骨架额外标记：

- `stand`
- `side_raise`
- `front_raise`
- `transition`

该分类只用于输出诊断，不参与关节命令计算。阶段汇总包含：

- 每阶段帧数
- 标定首次完成的帧和实际耗时
- 各阶段 14 轴关节中位值
- 触达 URDF 边界的帧数
- 上臂/前臂方向 residual
- 末段回中误差

因此，看到阶段名称不表示程序退回了 `raise_template`。当前关节轨迹来源应为：

```text
skeleton_segment_direction_dls
```

## 6. Revo2 手部 IK sidecar

### 6.1 复用边界

Vendor SDK 位于：

```text
vendor/revo2_hand_ik_sdk/
```

原归档保留不变：

```text
revo2_hand_ik_sdk_0.10.51_body24_hand12_bridge.tar.xz
```

`librevo2_ik_sdk.so` 只用于 Revo2 手部/手指计算，不能输出天工双臂 14 轴。为隔离 SDK 自带 Pinocchio、`libstdc++` 与 conda 环境的 ABI，native 库只在独立子进程加载。

### 6.2 输入、输出和隔离

每帧从规范 `MocapFrame` 提取左右手的 hand 与五指关节点及 WXYZ 四元数。当前适配层输出双手合计 12 路弧度角，顺序为：

```text
left_thumb_flex, left_thumb_abduct,
left_index, left_middle, left_ring, left_pinky,
right_thumb_flex, right_thumb_abduct,
right_index, right_middle, right_ring, right_pinky
```

父子进程使用匿名管道和 JSON Lines 通信：

```text
transport=local_subprocess_pipe
published=false
```

代码中没有手部网络目标、ROS publisher 或 rosbridge publisher。它不会向 vendor 示例 UDP `51000`、`/arm/cmd_pos` 或任何猜测接口发送手部结果。`diagnostic_enabled=true` 只会写本地 CSV。

`hand_ik` 配置只允许：

- `compute`
- `sdk_root`
- `sdk_config`
- `library`
- `timeout`
- `diagnostic_enabled`
- `diagnostic_csv`

不存在 `publish`、host 或 port 配置。正式手部控制接口未取得前，不得增加网络发布器。

### 6.3 Vendor anchor warning

Native SDK 可能报告 `right_base_link` / `left_base_link` anchor warning。当前配置和左右手 URDF 中均存在这两个名称，路径和拼写已核对，完整 2101 帧求解不受阻。该信息目前作为 vendor 已知诊断保留，不通过篡改供应商配置来掩盖。

## 7. 回放与实时入口

### 7.1 推荐离线回放

使用明确的环境解释器运行，避免 `conda run` 输出缓冲干扰诊断：

```bash
CONDA_DEFAULT_ENV=mocap_ik \
/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10 \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
--no-pause replay
```

当前配置行为：

- 输入固定 2101 帧 CSV
- `control_mode=direct`
- 使用离线 baseline，不读取 `/arm/status`
- 运行骨段方向 DLS
- 运行 Revo2 手部本地 sidecar
- 输出动作阶段、方向 residual、限位、步进、回中和手部统计
- 不连接 rosbridge
- 不发布双臂或手部命令

命令行可临时覆盖骨架参数，例如：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10 \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
--no-pause replay \
--control-mode direct \
--calibration-frames 45 \
--upper-direction-weight 1.0 \
--forearm-direction-weight 0.85 \
--direction-damping 0.08 \
--direction-iterations 6
```

`run_ik.py replay` 是否发布最终由 `run_ik_config.json` 的 `replay.execute` 决定，不能只根据命令行中没有 `--execute` 就推断为 dry-run。执行前必须重新读取完整配置。

### 7.2 实时遥操作

实时入口：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10 \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py teleop \
--control-mode direct \
--robot-command false
```

实时节点只保留最新 UDP 帧，不排队追赶旧帧。控制循环默认 20 Hz；本机接收时间超过 `0.3 s` 的动捕帧会被丢弃。

双臂控制必须先收到完整 `/arm/status` 14 轴反馈。未完成骨架 home 标定时，`DirectRetargeter` 只保持 baseline，不生成动作增量。`robot_command=false` 禁止命令发布，但如前所述，当前节点仍会连接 rosbridge 读取状态。

### 7.3 录制

录制只负责接收、规范解析和落盘，不加载机器人 IK，也不连接 rosbridge：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10 \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py record
```

默认监听 `0.0.0.0:15000`。`save_raw=false`、`save_jsonl=false` 时仅保存配置要求的 CSV 产物。

### 7.4 离线查看关节角

实时遥操作结束后，从发布边界日志生成单文件 HTML：

```bash
python3 /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py visualize --open
```

默认读取 `IK-source/log/mocap_ik_bridge.log` 中最近一次连续会话，输出：

```text
/home/lenovo/dzy/ik_update/mocap_teleop/IK/data/visualization/joint_commands.html
```

脚本只使用 Python 标准库，HTML 不加载 CDN。页面包含两类互相同步的视图：

- 机器人 3D 姿态：从项目 URDF 提取双臂真实关节原点、安装角、旋转轴和上下界，在浏览器内做正向运动学。支持播放/暂停、逐帧、倍速、时间滑块、命令轨道切换、拖动旋转视角和滚轮缩放；鼠标悬停关节曲线时也会同步姿态。
- 关节曲线：按关节分别展示，支持角度/弧度切换、命令类型筛选、悬停查看时间和帧号。

为保持单文件和零依赖，3D 区域显示的是准确运动链骨架而不是 STL 外观网格；它用于判断手臂形态、肘腕朝向和限位状态，不等同于碰撞模型或真机安全仿真。黄色关节点表示该角度处于对应 URDF 行程边界的 2% 范围内。

颜色语义严格对应数据边界：

- 绿色“已发送”：`robot_command.enabled=true`，即记录在 `/arm/cmd_pos` 发布边界的最终安全关节角。
- 黄色“未发送”：`robot_command.enabled=false`，只计算但没有发布。
- 蓝色“本地未发送”：Revo2 手部 IK 结果；当前没有手部机器人发布接口。

查看指定日志或全部历史会话：

```bash
python3 /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py visualize \
/path/to/mocap_ik_bridge.log \
--session all \
--output /tmp/joint_commands.html
```

报告是离线历史视图，不连接 ROS、rosbridge 或机器人，也不会发送任何命令。

## 8. 兼容模式

以下模式仍保留，但不是当前全骨架动作模仿主路径：

| 模式 | 实现 | 当前定位 |
| --- | --- | --- |
| `direct` | URDF 骨段方向 DLS | 当前默认主路径 |
| `raise_template` | 立正/侧平举/前平举模板混合 | 历史固定动作兼容模式 |
| `ik + teleop_dls + position_only` | 通用末端位置 IK | 保留，非当前主控制 |
| `ik + teleop_dls + pose` | 通用末端完整位姿 IK | 需要外参与腕部语义标定 |
| `ik + ubtech_casadi + pose` | CasADi/IPOPT 完整位姿优化 | 主要用于离线检查 |

`ik_backend` 和 `ik_task` 只在 `control_mode=ik` 时生效。配置为 `control_mode=direct` 时，即使文件中保留 `ik_backend=teleop_dls`，运行的也是骨段方向重定向，不是通用末端位置 IK。

## 9. 可复现验证

测试目录不是 Python package，不要使用 `python -m unittest tests...`。使用绝对路径并显式设置 `PYTHONPATH`。

### 9.1 JSON 校验

```bash
python3 -m json.tool \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik_config.json \
>/dev/null
```

### 9.2 编译检查

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10 -m py_compile \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/teleop_direct_retarget.py \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/mocap_replay_to_robot.py \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/mocap_ros_2_bridge_node.py \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/teleop_hand_ik.py \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/revo2_hand_ik_worker.py \
/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/tests/test_udp_loopback.py
```

### 9.3 协议、骨架、手部和 loopback 测试

```bash
export PYTHONPATH=/home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source
PYTHON=/home/lenovo/miniconda3/envs/mocap_ik/bin/python3.10

$PYTHON /home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/tests/test_mocap_protocol.py
$PYTHON /home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/tests/test_skeleton_frame.py
$PYTHON /home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/tests/test_hand_ik.py
$PYTHON /home/lenovo/dzy/ik_update/mocap_teleop/IK/IK-source/tests/test_udp_loopback.py
```

`test_udp_loopback.py` 只绑定本机 `127.0.0.1` 临时 UDP 端口，使用内存中的 G2 测试包，不连接 `192.168.1.110`，不创建 rosbridge 客户端，也不发布机器人或手部命令。

### 9.4 最近一次离线回归基线

最近一次完整验证结果：

- JSON 校验通过
- 目标文件 `py_compile` 通过
- 协议、骨架、手部适配和 UDP loopback 共 17 项测试通过
- 固定 CSV 2101/2101 帧完成双臂 dry-run
- 前 44 帧处于骨架标定保持，随后进入 `skeleton_segment_direction_dls`
- 双臂最大单帧变化约 `0.02535 rad`
- 双臂最大 baseline 偏移约 `0.70 rad`
- 手部 2101/2101 帧求解成功，0 失败
- 双手在序号 479 完成 vendor 标定，之后 1622 帧保持已标定
- 手部 12 路最大单帧变化约 `0.095 rad`
- 单手 native 求解最大约 `0.22 ms`
- 输出确认 `transport=local_subprocess_pipe`、`published=false`
- 最终输出明确为 `DRY_RUN_ONLY`
- 未读取机器人状态、未连接 rosbridge、未发布双臂或手部命令

这些数值是固定样本和当前配置的回归基线，不是真机精度声明。

## 10. 真机接口与启用前置条件

当前唯一允许的真机双臂链路：

```text
rosbridge 192.168.1.110:9091
/arm/status  -> bodyctrl_msgs/msg/MotorStatusMsg
/arm/cmd_pos -> bodyctrl_msgs/msg/CmdSetMotorPosition
```

电机与 14 轴顺序：

```text
11..17 = left
21..27 = right

shoulder_pitch_l, shoulder_roll_l, shoulder_yaw_l,
elbow_pitch_l, elbow_yaw_l, wrist_pitch_l, wrist_roll_l,
shoulder_pitch_r, shoulder_roll_r, shoulder_yaw_r,
elbow_pitch_r, elbow_yaw_r, wrist_pitch_r, wrist_roll_r
```

代码中的关节名均带 `_joint` 后缀。

硬性限制：

- 禁止发布 `/leg/*`
- 禁止向 `/arm_6dof_left`、`/arm_6dof_right` 发布；它们是 `WrenchStamped` 诊断/力输入
- 不依赖不可用的 `/rosapi/topics`
- 不把 `--allow-risk` 作为真机方案
- 手部结果不得复用 `/arm/cmd_pos`
- 没有正式手部命令接口前，手部只能保留在进程内、日志或本地诊断 CSV

真机启用前必须重新完成以下人工确认，不能沿用旧会话状态：

1. 读取完整 `run_ik_config.json`，确认所有执行开关状态。
2. 确认急停可用、机器人周围空间清空、操作人员可立即断能。
3. 读取 `/arm/status` 的 14 轴当前位置并保存 baseline。
4. 逐关节小幅验证 URDF 正方向与真实电机方向；当前仍只是假设同方向。
5. 先验证保持当前位置，再验证单关节极小增量，再考虑低速骨架轨迹。
6. 任何 baseline 异常、方向不一致、边界触达或反馈缺失都必须停止，不得用 `allow_risk` 绕过。

历史上曾发生约 `1.646 rad` 的最大 baseline 偏移，恢复后仍有约 `1.106 rad` 残差。不得重复该执行路径，也不得把历史恢复结果视为当前安全状态。

OpenSSL legacy provider warning 不影响当前 IK、UDP 或 rosbridge 逻辑，可与 vendor anchor warning 一样作为已知诊断记录，但不能用它掩盖真正的求解、连接或发布错误。