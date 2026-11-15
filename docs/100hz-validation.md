# 100 Hz 计算与仿真验证

目标周期为 10 ms。这里分别验证动态 64 节点映射的计算预算、控制回调的调度与慢手部隔离、独立 URDF 物理引擎的闭环结果。不同范围的数字不混用；仿真与本机计算结果不等于实体机器人网络链路的时延。

## 同机算法对比

CPU 为 Xeon E5-2690 v4，14 核 28 线程；Python 3.12.16、Pinocchio 3.6.0、NumPy 2.5.4、SciPy 1.18.1，BLIS 实际使用 1 线程。优化前后使用同一解释器、同一 URDF、相同确定性动态输入，每场景 600 帧。

| 场景 | 优化前 p95 / p99 ms | 优化后 p95 / p99 ms |
|---|---:|---:|
| 双臂可达运动 | 10.27 / 10.90 | 6.26 / 7.07 |
| 转腕 | 9.54 / 11.01 | 4.82 / 5.24 |
| 单侧 | 7.38 / 8.80 | 4.25 / 4.57 |
| 不可达输入 | 8.78 / 9.88 | 6.68 / 7.37 |

2400 帧中超过 10 ms 的次数由 59 降至 0；原有 6 次迭代、增益和全部安全约束保留。最大逐帧关节命令差小于 `1e-11 rad`，所有硬限位、实测步长、上一命令步长和基线包络违反均为零。详见 [机器可读报告](validation/ik-benchmark.json)。

主要变化是每次迭代只更新一次运动学、双臂独立的批量 7 轴 DLS、复用最终 FK 与残差，并减少重复校验。原测量环境已经是单线程；进程级线程数配置不是上述提速的主要原因。

复测当前代码：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  python tools/benchmark_ik.py --frames 600 --output app/data/benchmark.json
```

比较旧代码时，在另一个 worktree 检出报告中的 baseline commit，再使用 `--source-dir /path/to/old/app/src`。工具记录源码与 URDF 哈希、环境、耗时、误差和约束违反；性能随处理器、负载和依赖版本变化。

## 控制链路

手臂默认计算与发布上限为 100 Hz；手部 SDK 在独立的 20 Hz latest-only 工作线程执行。旧结果保留原输入的接收时间，不会重新贴上当前时间；失效、异步故障与会话结束均有门禁。

运动范围按秒限制：`max_joint_velocity=0.48 rad/s` 与 URDF 速度、原有单步上限共同生效，丢失周期不会攒出更大的运动步长。手部默认归一化速度 `1/s`，例行帧日志默认 1 Hz，故障日志即时保留。动捕或反馈低于控制频率时，跟踪会保守下降；不能靠重复旧反馈获得未经确认的位移。

在同一 CPU 的专用仿真环境（Pinocchio 3.7.0、NumPy 2.3.5、SciPy 1.18.1、BLAS 1 线程）中，实际调用生产 `control_loop`，使用动态双臂/转腕输入、100 Hz 内存反馈和阻塞 50 ms 的假手部 SDK。预热经过真实 2 秒静止门禁；每场记录 600 个定时回调，并执行默认 1 Hz JSON 日志序列化。

| 反馈场景 | 实际手臂发布 | 回调 p95 / p99 / max ms | 超过 10ms |
|---|---:|---:|---:|
| 跟随上一目标的反馈 | 99.943 Hz（600/600） | 7.82 / 8.78 / 10.11 | 1/600 |
| 固定不动的反馈 | 99.982 Hz（600/600） | 7.90 / 8.93 / 10.23 | 1/600 |

每场 SDK 调用 119 次，118 次已完成结果全部新鲜，左右手各发布 118 次；结果输入年龄 p99 为 76.43 / 77.55 ms。最后一次尚在执行的任务在退出时关闭。两场安全/速度违反均为零，无会话故障；固定反馈场景确认命令不会持续累加未确认位移。完整数据见 [control-loop-100hz.json](validation/control-loop-100hz.json)。这验证了软件回调的平均 100 Hz 能力，仍有尾部抖动；输入已由内存提供，ROS 消息传输、UDP 接收线程和真实 SDK 不包含在这项测量中。

复现（使用专用仿真环境，基准会复用测试中的内存 ROS fixture）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_control_loop.py --frames 600 --output app/data/control-loop.json
```

录制回放以 100 Hz 控制 tick 采样原始时间轴，不会加速源记录。20 Hz、2.95 秒的 60 帧回归在 100 Hz 下执行 2.96 秒（含末帧一个 tick），保留 2 秒启动门禁；重复帧保留原接收时间，不能延长新鲜度。

## 独立 URDF 仿真

环境和场景见 [仿真说明](simulation.md)。固定基座、简化几何、模型惯量与模拟电机控制提供离线验证条件，不包含真实接触环境、实体执行器、ROS 调度、网络或原生双手 SDK 的验收。

8 个场景的功能、精度与安全判据通过，全部控制计算 p99 <10 ms。前七个 solver 目标场景完整仿真步 p99 为 8.55–9.22 ms；第八个真实解码 64 节点、经过自动对齐和完整映射的场景，控制计算 p99 为 7.77 ms，加入物理子步和独立诊断后为 11.55 ms（49/400 帧超预算）。完整模拟器的预算与定时判据未通过，机器报告的 overall 保持 `false`，不宣称整条仿真链或真机硬实时达标。

普通可达、转腕、单侧和跟踪恢复的末段最大末端误差为 1.87–3.68 mm；接近操作包络的场景为 43.92 mm / 0.1805 rad，接近预设 45 mm / 0.22 rad 阈值。硬限位、速度和原始求解器非法候选为零，跟踪丢失期间无新目标提交，故意超限的 120 次候选均被拒绝。完整参数与结果见 [simulation-100hz.json](validation/simulation-100hz.json)。

## 回归检查（骨架路径阶段）

专用 Linux / Python 3.12 环境全套回归：`236 passed, 7 skipped, 130 subtests passed`；包含 10 项独立物理仿真专测。7 项跳过是旧官方 native IK 集成用例，需另备 CasADi/Pinocchio CasADi 支持及官方 `.so`，并显式设置 `VR_RUN_VENDOR_IK_TESTS=1` 才运行；本次未加载这些 native 库或手部 SDK。默认测试收集无需旧版依赖。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
```

另外通过源码编译检查、启动器参数转发测试与 `git diff --check`。`python run.py test` 的本机 UDP 自测收到 30 个有效包、0 个非法包；合成示例 dry-run 保留 41 帧启动窗口，生成 20 个运动预览命令，未连接机器人。HTML 回放已人工检查播放/暂停、场景切换、末帧拖动和重置视角，浏览器控制台无错误。单元回归通过与完整仿真定时失败分别保留，不把两者混写为全部验收通过。

## VR 手柄空间 IK 扩展

独立 `controller` 入口使用 Unity/OpenXR JSON 位姿，直接映射三维位置与旋转，不需要人体骨架。新增 `solve_cartesian` 的纯 IK 基准使用生产参数（6 次迭代、阻尼 0.08、平滑 1.0、100Hz 对应单步 0.0048rad），5 个场景各 600 帧。p99 最大 2.303ms、单帧最大 3.195ms，3000 帧均低于 10ms；全部硬限位、实测步长、上一命令步长和未控制侧约束违反为零。报告见 [cartesian-ik-benchmark.json](validation/cartesian-ik-benchmark.json)。这是合成反馈下的纯 IK，不包含映射或物理执行。

八个实际协议解码、抓握锚定、完整 mapper、生产发布守卫和独立物理引擎的场景，全部功能与安全判据通过；主机控制计算 p99 最大 6.065ms。普通三轴平移、转腕、单侧、失效恢复和重握场景末段最大末端误差 0.308–0.717mm；接近包络场景为 22.06mm / 0.1057rad。不可达目标只要求安全，实际残差另行保留。重新锚定帧关节目标跳变为零，硬限位/速度/历史违反、非法输入提交和未控制电机目标变更均为零。

包含物理和观察开销的完整仿真中，near_limits p99=10.358ms、clutch_reanchor p99=10.180ms，后者唤醒迟到 p99=1.114ms，超过既定 10ms / 1ms 阈值。因此完整仿真预算和定时判据仍为失败，overall 保持 `false`。详见 [controller-ik.json](validation/controller-ik.json)。参考三轴输入已独立核对；旧 fixture 数值失稳的失败报告与诊断也保留，没有放宽验收阈值。

新增路径后，专用 Linux 环境最终全套回归为 **289 passed, 7 skipped, 178 subtests passed**。默认预览的真实 loopback UDP 测试接收 70 个有效帧、拒绝 1 个非法包、计算 80 次候选，机器人发布次数为零；覆盖松手、重新握把和跟踪失效。源码编译检查通过。Unity 发送器已依据官方 API 审查并检查语法，尚未在实际 Unity 项目、头显、手柄或真机上验证。使用和直接机器人基坐标 API 见 [手柄说明](vr-controller.md)。
