# vr_pose_bridge

将 Formotion G2 的 64 节点实时 UDP 骨架映射到现有机器人双臂，并可通过独立 Revo2 SDK 驱动双手。

日常入口使用 `skeleton` 模式与纯 Pinocchio `teleop_dls` 后端：读取肩、肘、腕和躯干骨架，用机器人 URDF 与当前关节反馈建立映射。操作者只需双臂自然下垂、静止约 2 秒。无需在机器人侧执行转腕或六方向人工标定；动捕设备自身的佩戴、姿态或传感器校准仍按厂家要求完成。

## 支持范围

- 解析完整 64 节点数据；当前输出范围是双臂 14 轴和可选双手，不是全身机器人控制。头、腿等节点不会自动成为控制输出，默认骨架模式不驱动腰部。
- 按人体骨段方向和身体局部手部旋转重定向，默认 `position_scale=1.0`、`orientation_scale=1.0`。这些是映射增益，不表示人机关节角一一对应。
- 机器人 URDF 提供 FK、骨段方向和关节限位。命令仍受关节限位、单步变化、基线范围、反馈新鲜度等约束。
- Revo2 手部计算和发布分别配置；未安装 SDK 时可用 `--hand-ik-compute false` 仅运行双臂。当前手柄 `controller` 帧布局未验证，不能用于实时遥操作。

## 目录

整理后的目录如下，旧录制、标定和第三方代码保留，不放入日常源码目录：

```text
vr_pose_bridge/
├── run.py                       # 根入口
├── README.md
├── app/
│   ├── run_ik.py                 # 菜单与 CLI
│   ├── run_ik_config.json        # 运行配置
│   ├── src/                     # 协议、映射、IK、桥接、回放和 tests/
│   │   └── models/              # 活动机器人 URDF
│   └── data/                    # 新录制、诊断和图表
├── docs/
│   ├── bridge.md               # 当前使用说明
│   ├── device-guide.md         # 动捕设备原始使用说明
│   └── legacy-calibration.md   # legacy 高级标定说明
├── third_party/
│   ├── revo2_hand_ik_sdk/        # 可选双手 SDK
│   └── tiangong2pro_ik/          # 完整旧官方 IK 包，legacy 模式使用
├── examples/                   # 合成离线样本与生成脚本
└── archive/
    ├── IK_legacy/
    ├── ik_robot_legacy/
    └── history_data/            # 历史采集和旧标定产物
```

## 离线预览

离线回放不需要 ROS 2、机器人连接或手部 SDK。使用当前 Python 环境安装这些依赖后，从仓库根目录运行合成样本：

```bash
python -m pip install numpy scipy roslibpy pin pytest
python run.py replay examples/skeleton_demo.csv --dry-run --hand-ik-compute false
```

## 实时环境与启动

实时桥接运行在已配置 ROS 2 的 Linux 环境中，需 `rclpy`、ROS 消息包、NumPy、SciPy、roslibpy 和 Pinocchio。Pinocchio 的 pip 包名是 `pin`。默认骨架路径无需 CasADi 或旧官方求解器 `.so`。

在仓库根目录执行，先激活配置中的 Python 环境并加载本机 ROS 2 环境：

```bash
# 激活你选择的 Python 环境
# 按本机 ROS 2 安装位置加载 setup.bash
python -m pip install -r app/src/requirements_mocap_ik.txt
python run.py check --teleop
python run.py
```

编辑 `app/run_ik_config.json` 中的 rosbridge 地址、UDP 端口和 SDK 路径。`runtime.python=null` 与 `runtime.conda_env=null` 默认使用当前解释器。需要指定其他解释器或 conda 环境时再填写，仓库不依赖作者机器上的环境路径。

实时预览会读取机器人反馈和动捕数据，不发布控制命令：

```bash
python run.py teleop --robot-command false --hand-ik-compute false
```

启动实时控制：

```bash
python run.py teleop --robot-command true
```

保持双臂自然下垂且静止，等待启动零位就绪后动作。只有一个启动窗口，不需要再做 45 帧 direct 标定或转腕、六方向采集。设备未跟踪或反馈失效时停发命令；停发不等同机器人硬件急停。

失效后，人体和机器人姿态仍与上次目标相容时会重新等待静止对齐；若实测姿态已偏离上一已提交目标超过单步边界，或发生发送异常，本次会话停发，检查控制器后重新启动。

## 录制、预览与测试

```bash
python run.py record --duration 10
python run.py replay examples/skeleton_demo.csv --dry-run --hand-ik-compute false
python run.py test
python -m pytest -q app/src/tests
```

`examples/skeleton_demo.csv` 是合成的 64 节点示例，包含静止启动与小幅手臂动作，不来自真机采集，也不需要 SDK。`--dry-run` 明确覆盖配置中的执行开关，离线预览不连接机器人。记录需要含完整上半身骨架，并包含可通过自然下垂启动门禁的静止片段；仅含左右手位姿的旧数据不能直接用于新骨架路径。`test` 是编译与本机模拟 UDP 数采自测；离线测试通过不代表已完成真机验证。

需要向机器人执行已检查的录制时：

```bash
python run.py replay app/data/vr_udp_capture/你的录制.csv --execute --yes
```

仓库保留原有用户配置中的发布/执行开关。预览时请显式使用上面的 `--robot-command false` 或 `--dry-run`，不依赖当前配置值。

## 高级兼容

`--control-mode ik`、`direct`、`raise_template` 保留为 legacy 工具。人工标定命令只供这些历史流程维护，不出现在日常菜单，也不是新骨架路径的前置步骤：

```bash
python run.py legacy-workspace-calibrate --stage guide
python run.py legacy-wrist-calibrate --help
```

原 `calibrate`、`wrist-calibrate`、`workspace-calibrate` 名称仍可用于旧脚本。`skeleton` 显式拒绝 `--workspace-calibration`、`--wrist-calibration` 及对应非空 `runtime_path`，避免加载历史文件改变自动映射。新模式要求 `--ik-backend teleop_dls --ik-task pose`。

更多说明见 [当前桥接指南](docs/bridge.md) 和 [legacy 标定说明](docs/legacy-calibration.md)。
