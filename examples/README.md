# 离线骨架样例

`skeleton_demo.csv` 是可重复生成的合成数据：60 帧、20 Hz，前 2.2 秒为自然下垂静止姿态，之后轻微弯肘和转动手部。它用于检查自动对齐、CSV 解析与双臂映射，不是设备实测或机器人执行轨迹。未使用的指节和腿部节点为占位数据，不能用来验证手部 SDK 或全身动作。

在仓库根目录执行：

```bash
python examples/generate_skeleton_demo.py
python run.py replay examples/skeleton_demo.csv --dry-run --hand-ik-compute false
```
