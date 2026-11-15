# 双臂骨架桥接指南

默认入口为仓库根目录的 `python run.py`。配置位于 `app/run_ik_config.json`；日常使用 `control_mode=skeleton`、`ik_backend=teleop_dls`、`ik_task=pose`。手动转腕和工作空间六方向采集仅保留为 legacy 工具。

## 数据与输出

```text
Formotion G2 64 节点 UDP
  → 帧解析、有效性与跟踪检查
  → 肩/髋/躯干人体参考与上臂/前臂方向、手部局部旋转
  → 自然下垂静止窗口建立人体零位和机器人当前反馈零位
  → Pinocchio FK / Jacobian / DLS
  → URDF 限位与命令、反馈边界检查
  → rosbridge /arm/cmd_pos 和可选双手命令
```

输入是实时 UDP 骨架，不是 `.bvh` 文件。已验证的 G2 帧有 64 节点、每节点位置与四元数；解析器还保留旧格式兼容。当前 `controller` 手柄帧布局未验证，实时入口会拒绝启动；`auto` 也不会把未验证布局当成有效骨架。

虽然解析全身节点，当前只输出双臂 14 轴和可选 Revo2 双手。默认新骨架路径不驱动腰、腿或头部。Revo2 输出是 SDK 支持的手部通道，不是逐节点全身角度直传。

## 环境

离线骨架预览需要 NumPy、SciPy、roslibpy、Pinocchio，不需要 ROS 2 或手部 SDK：

```bash
python -m pip install numpy scipy roslibpy pin pytest
python run.py replay examples/skeleton_demo.csv --dry-run --hand-ik-compute false
```

实时节点需要 Linux ROS 2 环境中的 `rclpy`、消息包，以及 NumPy、SciPy、roslibpy、Pinocchio。默认使用当前 Python 解释器。需要其他环境时再设置 `runtime.python` 或 `runtime.conda_env`；实时运行还需按本机 ROS 2 安装路径加载 `setup.bash`：

```bash
python -m pip install -r app/src/requirements_mocap_ik.txt
python run.py check --teleop
```

默认后端不需要 CasADi、IPOPT 或旧官方 `.so`。使用 legacy 官方后端时，环境检测会额外检查 CasADi，且第三方官方包需完整保留。

双手 SDK 位于 `third_party/revo2_hand_ik_sdk`，其配置和 native 库路径在 `hand_ik` 段设置。SDK 在独立子进程运行；只测试双臂时可传 `--hand-ik-compute false`。计算开关和 `hand_control.publish` 发布开关分开。

启动器收到 `--hand-ik-compute false` 时，也会关闭本次会话的手部发布；不会修改配置文件中保存的手部偏好。

## 一次静止启动

1. 按厂家要求佩戴并完成动捕设备自身校准，确认肩、髋、肘、腕和手等节点有效。
2. 操作者双臂自然下垂，保持静止约 2 秒。机器人保持当前姿态并持续提供有效关节反馈，无需跟人摆同一姿态。
3. 等待启动零位就绪；程序以本次人体参考和机器人当前 FK 建立映射，再开始动作。

不需要胸前 Home、三组转腕、逐侧六方向运动或第二个 45 帧标定窗口。启动窗口参数沿用 `startup_hold_seconds`、`startup_human_vertical_cosine`、`startup_human_min_wrist_drop`、`startup_human_max_lateral_offset`、`startup_max_home_rms`。回放可单独设置同名参数，未设置则使用 `teleop` 的启动参数。

`position_scale=1.0` 是骨段方向映射的正常增益，`orientation_scale=1.0` 是手部相对旋转的正常增益。它们不会取消 URDF 限位，也不代表人体与机器人关节角逐一相等。

手部姿态增益为 `1` 时使用相对启动零位的身体局部旋转；小于 `1` 时逐帧缩放旋转增量并累积，重新对齐时清零。这样跨过 ±180°、转完一圈后改变转轴时，目标仍保持连续。

跟踪或手臂反馈失效后，实时桥接撤销自动对齐，双臂和双手停发，并等待新的稳定窗口。程序保留上一已提交目标的历史；若失效期间实测姿态已移动到与该目标不相容的位置，会锁定本次会话并提示检查后重新启动。发送异常同样需要重新启动。回放遇到失效直接中止，不在异常后自动移动回基线。

## 命令

从仓库根目录执行。

```bash
# 菜单：环境检测、录制、回放、实时控制、日志图表
python run.py

# 实时只计算，不发布；仍需机器人当前反馈
python run.py teleop --robot-command false --hand-ik-compute false

# 实时发布
python run.py teleop --robot-command true

# 录制完整 UDP 数据，不连接机器人
python run.py record --duration 10

# 离线预览：显式覆盖配置中的 execute/use_robot_status
python run.py replay examples/skeleton_demo.csv --dry-run --hand-ik-compute false

# 执行已检查的回放
python run.py replay app/data/vr_udp_capture/你的录制.csv --execute --yes

# 本机模拟 UDP 录制自测，以及离线回归
python run.py test
python -m pytest -q app/src/tests
```

新骨架回放需要含完整上半身节点、跟踪信息与有效静止启动片段的记录。旧版仅左右手位姿记录请使用 legacy 模式分析，不能补空节点伪装成完整骨架。

仓库配置可能保留原有 `robot_command`、`execute` 和手部发布偏好；预览请显式使用对应关闭选项。`--dry-run` 不连接机器人；`teleop --robot-command false` 仍订阅状态，但不发送动作命令。

## 配置与边界

- `rosbridge`：机器人地址、端口与状态超时。
- `record`：UDP 地址、端口和录制路径。
- `teleop` / `replay`：模式、控制侧、映射增益、启动窗口与命令边界。
- `hand_ik` / `hand_control`：SDK 计算、通道映射、反馈和发布。
- `workspace_calibration` / `wrist_calibration`：legacy 工具输出及输入路径，默认 `runtime_path=null`。

`skeleton` 拒绝非空人工标定路径，拒绝旧官方后端与 `position_only`。错误会在启动子进程前报出。旧配置里若显式使用旧模式，仍按旧模式运行；要切换新路径，请同时设置 `control_mode=skeleton`、`ik_backend=teleop_dls`、`ik_task=pose` 并清空两个 `runtime_path`。

URDF 限位和离线测试不能证明实际环境无碰撞，也不替代机器人硬件保护。失去跟踪、反馈过期或命令校验失败时，软件停发命令；停发不是硬件急停。新路径的默认增益不绕过这些检查。

## Legacy

`direct` 保留原有方向 DLS 与独立 45 帧稳定窗口；`ik` 保留旧末端位姿映射及可选标定输入；`raise_template` 保留姿态模板。旧官方后端的腰部行为只属于这些高级兼容流程。

人工工具有明确的高级命令名：

```bash
python run.py legacy-workspace-calibrate --stage guide
python run.py legacy-wrist-calibrate --help
python run.py legacy-calibrate --help
```

原名称继续兼容。详细操作见 [legacy 标定说明](legacy-calibration.md)。
