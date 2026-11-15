# URDF 双臂独立物理仿真

这个环境用于验证当前骨段方向与手部姿态 IK，以及一条合成 64 节点骨架的完整映射路径，不需要物理机器人、ROS、Revo2 SDK 或旧官方求解器 `.so`。Pinocchio 只生成 IK 命令，PyBullet DIRECT 独立执行电机控制与动力学，下一控制帧读取 PyBullet 的实测关节位置。运动过程中不会把解直接写入实测状态。

## 安装与运行

在仓库根目录使用独立虚拟环境，避免修改现有机器人 Python 环境：

```bash
python3.12 -m venv .venv-simulation
.venv-simulation/bin/python -m pip install -r requirements_simulation.txt
.venv-simulation/bin/python run.py simulate --output app/data/simulation --control-hz 100 --physics-substeps 4 --duration 4
```

依赖至少需要 Python 3.11；本次完整验证使用 Python 3.12。其他版本尚未完成同样的引擎与时序验证。

也可直接运行独立入口：

```bash
.venv-simulation/bin/python app/src/teleop_simulation.py --output app/data/simulation --control-hz 100 --physics-substeps 4 --duration 4 --realtime
.venv-simulation/bin/python -m pytest -q app/src/tests/test_simulation.py
```

默认以虚拟时间快速执行，适合回归；`--realtime` 用绝对单调时钟目标节拍进行定时。`--duration` 是每个场景的仿真秒数。退出码 `0` 表示全部验证通过，`2` 表示至少一项验证失败，`1` 表示依赖、模型或参数错误。报告分别给出 `accuracy_safety_passed`、`control_compute_passed`、`simulation_budget_passed`、`realtime_pacing_passed`；即使精度和控制计算通过，物理模拟器整步超预算时 overall 仍失败。失败时仍保存诊断报告，不把这些状态合并宣传为通过。

BLAS 线程限制为 1，避免小矩阵求解产生额外并行调度开销。`skeleton-current` 与 `skeleton-100hz` 使用相同参数：6 次迭代、阻尼 0.08、单步 0.06rad、平滑 0.65、方向权重 1/0.85、姿态权重 0.25。两个名称不代表独立算法版本；算法优化前后应使用固定轨迹和相同参数另做比较。

## 模型与控制

- 固定基座，重力 `[0,0,-9.81] m/s²`，默认控制 100Hz、每控制步 4 个物理子步（400Hz）。
- 活动 URDF 为 `app/src/models/walker_tienkung_ei_description/walker_tienkung_ei.urdf`。全部关节 origin、axis、限位，以及 link 惯量与质量保持原值，并用签名核对；加载时启用 `URDF_USE_INERTIA_FROM_FILE`。
- 只有缺失的 STL/mesh 替换为半径 35mm 的简化球体。输出 manifest 逐项记录替换。已有可访问 mesh 使用绝对路径，不移动原资产。
- 双臂使用 PyBullet `POSITION_CONTROL`，增益 0.18/1，力矩上限读取 URDF，最大速度取 URDF 与操作者速度限制的交集。`--max-joint-velocity` 默认 0.48rad/s，与当前发布配置相同。非手臂关节固定在初始零位目标，保持与 Pinocchio reduced model 一致。
- 物理 motor 使用该速度交集的 98%，默认 0.4704rad/s，留出明确的 2% 数值伺服余量；命令与违规检查仍使用 0.48rad/s 操作上限，URDF 不变。此前无余量的不可达场景出现 0.481001rad/s 峰值、1 个子步超过 0.481rad/s 检查阈值。没有重置实际速度或放宽验收阈值，该差异记录在原始首轮报告中；此余量也不构成硬件速度保证。
- 解算使用当前主路径的 `solve_segment_directions`，任务为上臂/前臂方向和手部姿态。当前测量值用于线性化，上一已提交命令只用于连续性约束。
- 正常候选先取 URDF、启动基线 ±1rad、当前实测及上一提交目标各自 ±min(0.06rad,速度×标称周期) 的交集并裁剪。额外记录原始求解结果是否非法，不用裁剪掩盖求解器超限。故意超限的 `limit_guard` 候选直接拒绝。初始化场景与独立 FK 核对允许 `resetJointState`，闭环运动阶段只调用 motor control 和 `stepSimulation`。

## 独立核对与场景

首先在 8 个确定性随机姿态下比较 Pinocchio 与 PyBullet 的末端 FK、骨段方向。末端位置阈值 2µm，方向/姿态阈值 20µrad。核对的是世界 link frame，使用 `getLinkState` 的 link frame 字段，不将惯性质心坐标误当关节坐标。

| 场景 | 目的 |
|---|---|
| reachable | 从可达关节轨迹 FK 生成方向与姿态目标，实际位置由物理引擎独立测量 |
| wrist_twist | 双腕转动时检查姿态跟随与末端误差 |
| left_only | 仅给左臂任务，右侧保留基线 |
| near_limits | 逐渐逼近肩滚与腕滚的 URDF/当前基线操作包络交集边界 |
| unreachable | 注入不可达方向与远端显示目标，要求不越硬限位和速度边界，不要求跟上 |
| tracking_loss | 中段注入跟踪失效，检查该段不提交新目标；物理电机仍保持上一目标 |
| limit_guard | 故意构造超限候选，要求外部命令检查拒绝送入引擎 |
| mocap_end_to_end | 合成规范 64 节点，经实际协议编码/解码、2秒静止门禁、自动对齐、完整骨架映射和物理闭环 |

最后一个场景使用固定人体肩髋与 0.33/0.34m 骨段，以机器人中立/目标骨段方向和 TCP 旋转生成可解释输入，再附加人体整体平移和 yaw。未用的手指、腿部节点明确标为未跟踪，不伪造真实手指观测；这是上半身映射验证，不运行手部 SDK。门禁与 `SkeletonRetargeter` 使用实际生产实现，不预生成映射后的目标。

可达场景在最后 20% 稳定段检查方向误差 <0.12rad、末端位置 <45mm、姿态 <0.22rad；每个物理子步检查实测硬限位和速度违规，另外检查故障期间发送新目标数、未控制右臂的保持情况，以及完整控制步计算耗时 p99 小于控制周期。这些是仿真回归阈值，不是机器人精度规格。

## 输出与查看

结果目录包含：

```text
metrics.json               # 参数、FK核对、场景结果、误差与时序分位数
frames.json                # 默认20Hz回放采样：实测q、目标、骨架与末端姿态
simulation.html            # 自包含交互运动回放，无CDN、无网络请求
preview.png                # PyBullet TinyRenderer静态图
simulation_geometry.urdf   # 仅缺失几何被替换的派生模型
```

用浏览器直接打开 `simulation.html`：可切换场景、播放/暂停、拖动时间与视角、缩放；蓝色为 URDF 物理骨架，黄色为目标末端，绿色为实际末端，彩色轴显示姿态。页面显示跟踪失效/候选拒绝状态以及误差和耗时。

`solver_ms` 记录任务求解，`target_generation_ms` 记录测试 fixture 生成。设备侧合成原始包及参考 FK 在计时前生成，不计入主机实时预算。`pipeline_ms` 保留每次测量、求解、校验、物理子步、独立误差计算及回放采样。第八场景仍在实时环路真实解码 64 节点包（`packet_decode_ms`），计算身体参考、映射目标和 IK，另外输出 `mapper_ms` 与 `pipeline_full_mapper_ms`，没有预生成映射后的结果。回放骨架采样约20Hz；控制和误差每100Hz、物理限位/速度每400Hz验证。模型装载、预热、启动对齐、报告落盘和渲染不计入运动控制步。

`control_compute_ms` 是实测反馈读取、协议解码、完整 mapper/IK、生产 guard 和命令提交的计算时段；`physics_ms` 是物理子步及每子步限位/速度测量，`validation_observer_ms` 是独立 FK 误差核对和回放采样。完整 `pipeline_ms` 包含三者，不把模拟器开销归咎于机器人控制算法，也不把它隐藏。

`compute_budget_misses` 是完整仿真步计算超过 10ms 的帧数；仅启用 `--realtime` 时才统计 `wall_period_overruns`（实际周期超过目标周期的 105%），同时记录周期和唤醒迟到分位数。定时模式还要求周期 p99 ≤11ms、唤醒迟到 p99 ≤1ms（100Hz时）。快速执行的吞吐量不能作为实际 100Hz 调度证明，也不声称硬实时保证。

## 验证范围

这是固定基座、简化几何和模拟执行器的物理闭环。自碰撞未启用，没有场地或真实负载模型，不能据此宣称真机碰撞安全。URDF 惯量和限位被保留，但电机增益、摩擦和执行器响应没有经过硬件辨识。100Hz 仿真通过不等于 ROS、网络、手部 SDK 和真机也能在同样周期内完成。

前七个场景是 solver-target level，第八个覆盖合成上半身的实际协议与完整映射，仍未覆盖真实设备噪声、ROS 接收/发布、手部 SDK 与真实跟踪失效恢复。`tracking_loss` 是仿真输入故障注入，`limit_guard` 检查独立安全边界，不把它们描述为完整机器人系统验收。

## 本次验证结果

专用 Linux / Python 3.12 环境中，10 项仿真专测通过。Pinocchio/PyBullet 的 8 个独立姿态核对最大末端位置差为 `4.80e-8 m`，姿态差为 `1.71e-7 rad`；33 个运动连杆质量误差为零、主惯量最大差约 `1.11e-16 kg·m²`。

最终 8×4 秒、100Hz 控制/400Hz 物理的实际定时运行中，全部功能、精度和安全判据通过；所有场景控制计算 p99 均低于 10ms。普通可达、腕扭转、单侧与跟踪恢复场景的末段最大末端误差为 `1.87–3.68mm`。接近操作包络的 `near_limits` 场景末段最大误差为 `43.92mm / 0.1805rad`，接近预设 `45mm / 0.22rad` 阈值，不能将其概括为毫米级精度。

完整 64 节点场景的控制计算 p99 为 `7.774ms`；物理子步阶段 p99 为 `3.666ms`，独立误差/回放观察阶段为 `0.585ms`。包含三者的完整仿真步 p99 为 `11.547ms`，实际周期 p99 为 `11.551ms`，400 帧中有 49 帧完整仿真步超过 10ms。功能/控制计算通过，物理模拟器整步预算与定时判据未通过，overall 保持失败；这不是硬实时或完整机器人系统 100Hz 证明。

最终记录中硬限位、速度、原始求解器非法候选均为零；跟踪失效期间没有新目标提交，故意超限候选被拒绝 120 次。机器可读结果见 [simulation-100hz.json](validation/simulation-100hz.json)，包含 CPU、Python/依赖版本与源码 SHA256。提交版本仅移除机器绝对路径（URDF 改仓库相对路径，BLAS 库保留文件名），数值不变；未脱敏原始报告留在用户输出目录。元数据在采样结束后补录，未重新采样或改动控制环路。

早期失败报告也被保留：第一轮存在重复单 link FK/旋转构造及 100Hz 全骨架观察开销；后续使用同一引擎状态的批量读取、等价四元数转换、20Hz 回放采样，并将设备侧合成输入预生成，真实解码和映射仍逐帧执行。另按实际生产配置收紧实测反馈速度包络，并给模拟 motor 增加明确 2% 数值余量。没有重置运动中的实际状态、放宽误差/时序阈值或调整求解参数来换取通过。

实现依据：[PyBullet 官方 Quickstart](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf)、[官方电机与仿真示例](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py)、[官方 link frame/FK 示例](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/jacobian.py)。
