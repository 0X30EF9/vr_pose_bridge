# Legacy 人工标定说明

这些工具仅供维护旧 `--control-mode ik` 末端映射。默认 `skeleton` 路径直接使用 64 节点 UDP 骨架、机器人 URDF 和一次自然下垂静止启动，不需要本文的人工操作；菜单不包含人工标定步骤。动捕设备自身的校准仍按厂家要求完成。

## 默认路径与历史工具

新路径不加载腕点或工作空间 JSON，并会拒绝显式 `--wrist-calibration`、`--workspace-calibration` 或配置中的非空 `runtime_path`。不要把历史 JSON 改标签后接入新路径。

旧末端模式可显式启用合法标定：

```bash
python run.py teleop --control-mode ik --ik-backend tiangong2pro_official --workspace-calibration /path/to/workspace.json --wrist-calibration /path/to/wrist.json
```

这条命令沿用当前发布开关；只预览时加 `--robot-command false`。旧后端需要完整的第三方官方 IK 包与相应依赖。

## 两种人工工具的目的

腕点工具采集胸前、体侧、抬高手腕三组动作，每组保持腕点不移动并转腕，用于估计或验证 hand→anatomical wrist 偏移。当前完整骨架已有腕节点，不需要通过这种流程重新估计腕点。

工作空间工具先采集胸前 Home 和逐侧六方向动作，估计坐标轴及个人动作幅度；再离线调用旧官方 IK，从给定机器人 Home 扫描各方向数学边界。该工具不发布机器人控制话题。扫描结果依赖模型、Home 和固定姿态，不是实际环境碰撞验证。

## 查看与运行

从仓库根目录执行：

```bash
# 只显示说明，无设备访问
python run.py legacy-workspace-calibrate --stage guide

# 旧人工腕点工具
python run.py legacy-wrist-calibrate --stage all --side both

# 旧工作空间采集与离线扫描
python run.py legacy-workspace-calibrate --stage all --side both

# 顺序运行两种旧工具
python run.py legacy-calibrate
```

原 `calibrate`、`wrist-calibrate`、`workspace-calibrate` 名称仍作为兼容入口。组合工具先验证或生成腕点结果，再生成候选工作空间文件；任何阶段失败会保留已有工作空间结果。

## 旧工作空间动作

Home：身体静止，双手在胸前，肘部弯曲，手腕中立。每个方向先回 Home，另一只手保持不动；按提示移动并保持，等待 `CAPTURE_ACCEPTED`。

| 方向 | 操作 |
|---|---|
| forward | 向身体正前方移动约 15～25cm |
| backward | 向身体方向收回约 10～15cm |
| outward | 左手向左、右手向右约 15～20cm |
| inward | 向身体中线约 8～15cm，不交叉 |
| up | 向上约 15～20cm，不耸肩 |
| down | 向下约 10～15cm，不靠弯腰补位 |

`CAPTURE_RETRY` 只需重做当前动作。`ANALYZE_RESULT` 检查采集质量，`SCAN_RESULT` 检查模型扫描，最终 JSON 的 `valid` 必须为 `true`。

这里的胸前 Home 仅用于旧离线拟合。实时旧模式仍以人体自然下垂稳定片段和机器人当前反馈建立本次运行零位，机器人无需摆成人体的同一姿态。

## 配置与文件

配置位于 `app/run_ik_config.json`。`calibration_output` 只定义新产物位置，默认写入 `app/data/wrist_calibration` 或 `app/data/workspace_calibration`。`runtime_path` 只供旧运行模式选择输入，默认 `null`。CLI 路径优先于配置；显式缺失、无效或不兼容文件会报错。

历史采集及旧 JSON 保留在 `archive/history_data`。旧工作空间示例使用 `control_frame=L_ee_R_ee`，原始记录只有手位姿和胸口位置，缺少解剖腕点观测，不能真实重生成当前 `wrist_roll` 模型。只改 JSON 的标签无效。若继续维护旧末端路径，应重新采集并扫描，生成满足 `schema_version=1`、`valid=true`、`control_frame=wrist_roll` 的文件。

标定边界是给定条件下的模型估计，不能替代真实机器人和场地的验证。日常使用新骨架路径时，无需生成、选择或维护这些文件。
