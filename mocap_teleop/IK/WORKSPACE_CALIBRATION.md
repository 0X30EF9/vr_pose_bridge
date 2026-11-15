# 工作空间标定操作说明

本说明用于标定 Formotion G2 动捕手部位移到天工 2 Pro 双臂工作空间的映射。

标定分为两部分：

1. 人按照 CLI 提示完成 Home 和六方向动作，程序只采集 UDP 动捕数据。
2. 程序离线调用官方 IK，计算机器人安全工作空间并生成标定 JSON。

标定过程不会发布 `/arm/cmd_pos`、`/waist/cmd_pos` 或灵巧手命令。

> **标定 Home 与遥操作启动 Home 是两个不同概念。** 本文的胸前 Home 只用于离线拟合坐标轴、方向比例和安全边界，不会直接成为机器人启动目标。每次实时遥操作启动时，操作者和机器人都必须双臂自然下垂并稳定保持约 `2 秒`；程序随后以人体当前姿态和机器人当前 `/arm/status` FK 分别建立本次运行零位。门禁完成前不会发布手臂或腰部命令。

## 1. 启动前准备

- 清空机器人双臂、腰部周围的人员和物体。
- 正确佩戴 Revo2 手套，确保双手和上半身都在动捕范围内。
- 操作者站在机器人正前方，身体朝向与机器人朝向一致。
- 双脚站稳；标定过程中不要移动脚、转身、侧身或弯腰。
- 关闭实时遥操作和动作回放，确保没有其他进程占用 UDP 端口 `15000`。
- 第一阶段只标定位置，手腕姿态跟随保持关闭。

## 2. 人体 Home 姿态

CLI 显示 `HOME` 后，按以下方式站立：

1. 身体正对机器人，双肩放松，不耸肩。
2. 双肘自然弯曲约 `30°～60°`，不要完全伸直。
3. 双手放在胸前偏前方，间距约与肩同宽。
4. 左手保持在身体左侧，右手保持在身体右侧。
5. 手掌和手腕保持自然中立，不主动旋转手腕。
6. 按回车后继续保持静止，直到 CLI 显示 `CAPTURE_ACCEPTED pose=home`。

程序使用整个稳定窗口的中位数作为 Home，不使用单帧数据。

## 3. 每个方向的操作方法

每个方向都执行同一套动作：

1. 按提示回到 Home，并保持另一只手不动。
2. 阅读 CLI 显示的方向说明。
3. 按回车后，在 `1～2 秒` 内平稳移动目标手。
4. 到达目标位置后保持静止，直到显示 `CAPTURE_ACCEPTED`。
5. 再回到 Home，等待下一条提示。

不要快速甩手，也不要为了达到指定厘米数而勉强伸到极限。距离只需要大致满足范围，程序以动捕实际位移为准。

### forward：向前

- 从胸前 Home 沿身体正前方移动约 `15～25cm`。
- 肘部仍保持轻微弯曲。
- 不要身体前倾，不要耸肩。

### backward：向后

- 从 Home 向胸口方向收回约 `10～15cm`。
- 不要身体后仰。
- 手不要贴到胸口。

### outward：向身体外侧

- 左手向身体左侧移动约 `15～20cm`。
- 右手向身体右侧移动约 `15～20cm`。
- 不要转动腰部或肩膀。

### inward：向身体内侧

- 左手向身体右侧移动约 `8～15cm`。
- 右手向身体左侧移动约 `8～15cm`。
- 不要跨过身体中线，不要让双手相碰。

### up：向上

- 从 Home 垂直向上移动约 `15～20cm`。
- 肩膀保持放松，不要耸肩。
- 不要同时向前伸手。

### down：向下

- 从 Home 垂直向下移动约 `10～15cm`。
- 肘部不要完全伸直。
- 不要弯腰或下蹲。

## 4. CLI 提示的含义

- `CALIBRATION_PREFLIGHT`：正在显示安全条件和采集参数。
- `RETURN_HOME`：目标手和另一只手都回到 Home。
- `MOVE_NOW`：按说明移动当前目标手，然后保持。
- `CAPTURE_ACCEPTED`：当前窗口有效，可以继续下一步。
- `CAPTURE_RETRY`：当前窗口不稳定、位移过小或方向混杂，需要重做当前动作。
- `ANALYZE_RESULT OK`：人体坐标系和方向数据通过检查。
- `SCAN_RESULT OK`：官方 IK 工作空间扫描完成。
- `CALIBRATION_OUTPUT`：最终标定 JSON 的路径。

出现 `CAPTURE_RETRY` 时，只重做当前动作，不需要从头开始。若连续失败，检查：

- 动捕 UDP 是否持续输出；
- 目标手是否移动了至少 `8cm`；
- 另一只手和身体是否保持静止；
- 到达位置后是否真正保持了约 `2 秒`；
- 当前方向是否做反。

## 5. 配置与产物路径

默认配置位于 `run_ik_config.json` 的 `workspace_calibration` 段：

- 原始采集保存到 `data/workspace_calibration/raw/session_<时间戳>/capture.json`；
- 动捕分析保存为 `data/workspace_calibration/analysis.json`，已存在时自动生成带时间戳的新文件；
- 最终标定保存为 `data/workspace_calibration/calibration.json`，已存在时默认不覆盖；
- 只有显式传入 `--overwrite` 才允许覆盖派生 JSON；原始 session 始终不覆盖。

最终 `calibration.json` 必须满足 `schema_version=1` 且 `valid=true`。官方 IK 的实时遥操作和回放会在启动前加载该文件；文件缺失、版本不兼容或质量未通过时直接拒绝启动。

标定接入后，`position_scale` 只表示标定安全边界内的总增益。左右手和六个方向仍使用标定文件中的独立比例，并在组合方向超界时投影回安全工作空间。

## 6. 推荐命令

查看说明，不连接设备：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python \
  /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
  workspace-calibrate --stage guide
```

执行完整标定：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python \
  /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
  workspace-calibrate --stage all --side both
```

只采集人体动作：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python \
  /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
  workspace-calibrate --stage capture --side both
```

分析已有采集并扫描官方 IK：

```bash
/home/lenovo/miniconda3/envs/mocap_ik/bin/python \
  /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
  workspace-calibrate --stage analyze --input <标定采集目录>

/home/lenovo/miniconda3/envs/mocap_ik/bin/python \
  /home/lenovo/dzy/ik_update/mocap_teleop/IK/run_ik.py \
  workspace-calibrate --stage scan --input <analysis.json>
```

## 7. 标定结果验收

最终报告必须满足：

- 坐标旋转矩阵为右手系，`det(R)` 接近 `+1`；
- 左右手每个方向的动捕位移均不小于配置阈值；
- 静止窗口抖动不超过配置阈值；
- 官方 IK 至少在每个方向找到一个有效安全边界；
- 标定 JSON 记录原始采集目录、URDF、solver 和质量指标。

未通过质量检查的结果不会标记为 `valid=true`，实时遥操作和回放不得加载。

## 8. 真机验证顺序

标定文件生成后，不要立即进行大范围双臂动作。应依次执行：

1. 左臂，安全边界的 `30%`；
2. 右臂，安全边界的 `30%`；
3. 左右单臂，安全边界的 `60%`；
4. 双臂低速；
5. 最后再单独标定手腕姿态和真实灵巧手 TCP。

任意一步出现方向错误、接近身体、关节突跳或明显末端误差时立即停止，不扩大范围。