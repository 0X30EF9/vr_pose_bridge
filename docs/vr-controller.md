# VR 手柄空间位姿控制

这条路径把左右 VR 手柄的三维位置和旋转转换为机械臂末端位姿，再执行 Cartesian IK。它不读取人体骨架，不需要六方向或 hand→wrist 人工标定，也不解析未知的 `C8 FF` 二进制手柄包。每侧握把是独立的 deadman/clutch：按下时记录手柄和机器人实测 TCP 锚点，当帧关节目标等于实测位置；之后只跟随相对于该锚点的变化。松开后该侧停止发送新目标。

## 先运行预览

在仓库根目录、已安装 [README](../README.md) 所列依赖的 Python 环境中运行：

```bash
python run.py controller
```

默认监听 `0.0.0.0:15010`，以 100Hz 控制 tick 使用最新有效输入；`ideal_feedback_preview` 使用跟随已接受目标的虚拟反馈，不连接 ROS，也不发送机器人命令。需要物理仿真时运行：

```bash
python run.py controller-simulate
```

读取真实机器人反馈但不发布目标：

```bash
python run.py controller --preview-robot
```

明确启用机械臂发布：

```bash
python run.py controller --execute
```

实际发布按当前握把状态只包含左臂 7 轴、右臂 7 轴或双臂 14 轴；这条路径不发布腰部和机器人手部命令。没有启动预置动作或自动回零。默认限制为 `--max-joint-velocity 0.48` rad/s，并叠加 URDF 速度/机械限位、实测与上一已提交命令的步长限制及固定会话基线包络。重新握把不重置会话基线。

## Unity / OpenXR 发送器

发送器是 [VrPoseUdpSender.cs](../examples/unity/VrPoseUdpSender.cs)，使用标准 `UnityEngine.XR.InputDevices` Feature API。Unity 官方 OpenXR 文档明确支持 Feature API；需要启用与设备相符的 Interaction Profile，并通过项目的 OpenXR Project Validation。[Unity OpenXR 输入](https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.16/manual/input.html)、[项目配置](https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.16/manual/project-configuration.html)。

1. 在 Unity 6 项目中安装并启用 XR Plug-in Management / OpenXR，选择对应平台及手柄 Interaction Profile。
2. 使用 SteamVR 时，让 Unity 的 **Play Mode OpenXR Runtime** 使用 SteamVR（该选项可用时），或使用已正确设置的系统 OpenXR runtime。这里不要求安装专有 SteamVR Unity 插件。[官方 runtime 选择说明](https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.16/manual/project-configuration.html#openxr-runtime)。
3. 将 C# 文件放入 Unity 项目的 `Assets`，挂到一个启用的 GameObject。
4. 将 `receiverHost` 设为运行 Python 接收端的 IPv4 地址。同机可保留 `127.0.0.1`；端口默认 `15010`。
5. 先启动 Python 预览，再进入 Unity Play 模式。先松开握把，确认设备跟踪有效，再按住要控制的那一侧并移动、旋转。

代码使用 `GetDeviceAtXRNode(LeftHand/RightHand)`，同时读取 `isTracked`、包含 Position/Rotation 的 `trackingState`、`devicePosition`、`deviceRotation` 和 `gripButton`。不支持或读取失败的握把输入不会被猜测为已释放，也不会用其他按钮替代。跟踪失效时仍保留已知握把状态，避免恢复跟踪后在握把持续按住时自动恢复运动。[InputDevices API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/XR.InputDevices.GetDeviceAtXRNode.html)、[CommonUsages](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/XR.CommonUsages.html)。

`sendRateHz=100` 是发送上限。发送发生在 Unity `Update` 中，实际新样本频率受应用和设备帧率限制，例如 90Hz。发送器不追赶补发同一采样的 burst；主机 100Hz 可以保持最新输入继续求解，但这不等于设备以 100Hz 采样。接收端保留该输入最初的主机接收时间，不会因为重复使用而刷新过期时间。

组件禁用、应用失焦或暂停时会尽力发送 `enabled=false`、`tracked=false` 的停止输入；网络断开时依靠输入超时停发。发送器也监听 `XRInputSubsystem.trackingOriginUpdated`：原点改变后发送失效输入，并要求各侧在有效跟踪下松开再按。正常操作仍应先松开握把，再进行 recenter 或更改 tracking origin。[Unity 原点更新事件](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/XR.XRInputSubsystem-trackingOriginUpdated.html)。

发送器还保存每侧 `InputDevice` 身份，用公开的 `Equals` 比较（Unity 内部比较设备 ID），并监听 `deviceDisconnected`。同一个 LeftHand/RightHand 节点被另一台设备替换时，即使新设备立即有效且握把仍按住，也会锁存该侧失效，等待有效松手后重新按下。组件禁用时解除事件订阅。[Unity 输入设备官方实现](https://github.com/Unity-Technologies/UnityCsReference/blob/master/Modules/XR/ScriptBindings/XRInput.bindings.cs)。

停止发布不能保证机器人控制器取消已经接收的运动目标；代码没有虚构硬件急停协议。实际运行的网络、反馈和控制器停止能力需要在对应机器人上核验。

## JSON UDP 协议 v1

每个 UDP 数据报包含一个 UTF-8 JSON 对象，最大 16 KiB；字段名是 `sequence`：

```json
{
  "version": 1,
  "type": "vr_controller",
  "sequence": 1,
  "timestamp": 12.345,
  "coordinate_system": "unity",
  "left": {
    "position": [0.0, 1.1, 0.3],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "tracked": true,
    "enabled": false,
    "activation_id": 0
  },
  "right": {
    "position": [0.0, 1.1, 0.3],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "tracked": true,
    "enabled": false,
    "activation_id": 0
  }
}
```

- `sequence` 是非负整数，发送端会话内严格递增；`timestamp` 是严格递增的源端秒数，不要求与主机时钟同步。Unity 使用不受 `timeScale` 影响的双精度启动时间。[Unity 时钟 API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Time-realtimeSinceStartupAsDouble.html)。
- `position` 单位为米，`rotation_xyzw` 顺序固定为 x/y/z/w，所有值必须有限；四元数范数需在单位长度的 1% 内，接收时会归一化。失效设备仍发送合法的零位置和单位四元数，并明确标记 `tracked=false`。
- `tracked` 与 `enabled` 必须是真正的 JSON 布尔值。`enabled` 表示握把，不是“坐标可用”。
- `activation_id` 为可选非负整数，Unity 示例始终提供，每次真实握把上升沿递增。它可识别正常运行中因 UDP/latest 合并而漏掉的松手包，并将新握把周期重新锚定为零位移。它不能绕过跟踪失效后的松手要求。
- `received_at` 不属于发送协议。主机在接收时使用自己的 monotonic 时钟；重复包、逆序包及不前进的源时间不能刷新已接受输入。输入间隔超过默认 0.3 秒也会撤销锚点，新包仍按住握把时不能直接恢复。

接收端固定首个有效 UDP 来源的 IP 和端口。Unity 组件重启、发送 socket 端口改变或发送端序号/时钟重置后，需要重启 `controller` 接收会话；不会自动清除固定机器人基线和已提交命令历史。

## 坐标和相对映射

Unity 使用左手坐标，+X 向右、+Y 向上、+Z 向前。[Unity 坐标说明](https://docs.unity3d.com/2022.3/Documentation/Manual/QuaternionAndEulerRotationsInUnity.html)。原生 OpenXR 的 LOCAL 空间是右手系，+X 向右、+Y 向上、−Z 向前；原生 OpenVR/SteamVR 也采用该轴约定和米制。[OpenXR reference spaces](https://github.com/KhronosGroup/OpenXR-Docs/blob/main/specification/sources/chapters/spaces.adoc)、[Valve OpenVR 头文件](https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h)。

机器人目标坐标统一为 FLU：+X 前、+Y 左、+Z 上。

| `coordinate_system` | 输入位置 `(x,y,z)` 转机器人位置 | 用途 |
| --- | --- | --- |
| `unity` | `(z,-x,y)` | 本仓库 Unity 发送器，包括使用 SteamVR OpenXR runtime 时 |
| `openxr` | `(-z,-x,y)` | 原生 OpenXR / 同轴约定的原生 SteamVR 发送器 |
| `robot` | `(x,y,z)` | 已自行转换到机器人 FLU 的发送器 |

位置使用 `p_robot = S p_input`，旋转使用 `R_robot = S R_input Sᵀ`。Unity 的换基矩阵行列式为 −1，不能直接把它当四元数旋转，也不能只交换四元数分量。原生 OpenVR 的 w/x/y/z 四元数需要先转换为本协议的 x/y/z/w 顺序。

同一侧的位置和旋转必须来自同一个稳定 tracking space。抓握锚点只消除初始位置和姿态偏置，不自动识别房间的航向：默认把 tracking forward 映射为 robot forward。若现场 tracking 空间的 yaw 不同，应先正确设置输入坐标，或在发送端完成自定义变换后使用 `robot`；不要在抓握中改变空间原点。

按下握把时记录手柄 `C₀` 和机器人实测 TCP `T₀`，之后目标为：

```text
p_target = p_T₀ + position_scale × (p_C − p_C₀)
R_target = scaled_relative_controller_rotation × R_T₀
```

增益默认均为 1，可配置为 `[0,1]`。单位旋转增益直接使用 `R_C R_C₀ᵀ`；小于 1 的旋转增益按相邻新输入的短旋转增量缩放后累计，跨越 ±π 或连续转圈不会产生主值分支跳变。`--ik-task position_only` 明确只要求位置；默认 `pose` 同时要求位置和姿态。

## Python 集成契约与验证范围

`parse_controller_packet(data, received_at)` 返回不可变 `ControllerFrame`。`VrControllerRetargeter.ingest(frame, now)` 应对每个新包调用，以保留批量输入里的松手/失效边沿；它不求解。100Hz 控制循环再调用 `update(latest, current_q, last_q_cmd, now)`，传入完整实测 14 轴和调用方真正发送成功后的目标。mapper 不提前提交候选命令。`invalidate(reason)` 撤销锚点但保留会话基线；未启用侧保留当前反馈，输出 `control_side` 只包含本帧启用侧。

### 直接传机器人基坐标系目标

不通过 UDP，也可以直接调用数值 IK。以下是离线示例，没有发送接口：

```python
import sys
import time
import numpy as np
sys.path.insert(0, "app/src")  # 从仓库根目录运行
from teleop_ik_core import TeleopArmIK
from teleop_ik_types import TeleopIKConfig, TeleopIKInput
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS

solver = TeleopArmIK(TeleopIKConfig(
    joint_names=list(ARM_JOINT_NAMES), motor_ids=list(ARM_MOTOR_IDS),
    task_mode="pose", control_dt=0.01, max_iterations=6,
    max_joint_delta_per_step=0.0048, smoothing_alpha=1.0,
    enable_local_log=False, log_to_console=False,
    show_model_load_progress=False, log_ik_result=False,
))
current_q = np.zeros(14)  # 仅离线示例；真机必须换成完整、新鲜的实测反馈
solver.set_home_q(current_q)  # 会话参考只初始化一次
left_target, right_target = solver.end_effector_poses(current_q)
left_target = left_target.copy()       # 两个目标都是合法的 4×4 SE(3)
left_target[0, 3] += 0.005             # 模型基坐标系中向前 5 mm
last_q_cmd = None                      # 或调用方上一条真正发送成功的完整 14 轴目标
result = solver.solve_cartesian(TeleopIKInput(
    timestamp=time.monotonic(), control_side="left",
    target_pose_l=left_target, target_pose_r=right_target,
    current_q=current_q, last_q_cmd=last_q_cmd,
))
candidate = result.command             # 候选，尚未发送；不能提前当成 last_q_cmd
print(result.reason, result.converged, candidate.positions)
```

当前求解器使用固定基座、腰部锁在 0 的 **14 轴机械臂 reduced model**。目标 4×4 必须位于该机器人模型基坐标系；操作点为 `L_ee` / `R_ee`，包含模型原有固定工具 offset，不是裸腕转轴。世界坐标目标或实际转腰后的世界 TCP 不能直接当成同一基坐标输入；本次 API 没有扩展腰部运动学或发布腰部命令。

每次求解以 `current_q` 为实测起点，`last_q_cmd` 只约束连续性；未控制侧保持实测位置。`limited` / `iteration_budget` 是合法有界诊断，表示限幅或迭代预算内尚未收敛，应结合 `converged` 和残差判断效果。数值 API 自身不验证网络年龄、不发布命令；真实发布还须通过反馈/输入新鲜度、URDF 速度与硬界、每秒速度、已提交历史及固定基线门禁。

Python 协议/坐标/抓握/过期/限位逻辑有离线单测；Cartesian 求解和物理闭环由独立仿真验证。C# 通过了 API stub 的语法编译检查，但当前修改未在真实 Unity 项目、头显、手柄或机器人上运行，不能把代码和仿真验证视为设备联调完成。

本次最终全套回归为 `289 passed, 7 skipped, 178 subtests passed`。纯空间 IK 的 3000 帧基准 p99 最大 2.303ms；八个物理闭环场景的功能/安全通过，主机控制计算 p99 最大 6.065ms。完整物理仿真仍有两个场景 p99 略超 10ms，未通过的定时判据保留为失败。详细数据、精度范围和复测命令见 [验证说明](100hz-validation.md#vr-手柄空间-ik-扩展)、[纯 IK 报告](validation/cartesian-ik-benchmark.json) 与 [物理闭环报告](validation/controller-ik.json)。
