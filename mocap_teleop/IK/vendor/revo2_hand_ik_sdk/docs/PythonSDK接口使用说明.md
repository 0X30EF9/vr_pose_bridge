# Python SDK 接口使用说明

## 1. 正式接口边界

Python SDK 正式入口为：

```python
from revo2_hand_ik import Revo2DualHandIkSdk
```

正式集成时，外部业务层负责准备输入数据，然后调用：

```python
result = sdk.solve(
    left_points=left_points,
    right_points=right_points,
    left_quaternions_wxyz=left_quats,
    right_quaternions_wxyz=right_quats,
    sequence=frame_id,
    timestamp_ms=timestamp_ms,
)
```

SDK 内部不直接监听 UDP。UDP 只存在于 `python/examples/udp_to_sdk_test.py`，用于真实数据测试和交付例程。

## 2. 最小调用示例

```python
from revo2_hand_ik import Revo2DualHandIkSdk

config = "bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json"
library = "lib/librevo2_ik_sdk.so"

with Revo2DualHandIkSdk(config, library_path=library) as sdk:
    result = sdk.solve(
        left_points=left_points,
        right_points=right_points,
        left_quaternions_wxyz=left_quats,
        right_quaternions_wxyz=right_quats,
        sequence=sequence,
        timestamp_ms=timestamp_ms,
    )

joint_positions = result["joint_positions"]
```

## 3. 输出单位

Python SDK 公开输出为：

```text
joint_unit = rad
joint_positions = 12 路关节角，单位 rad
```

当前版本不输出 `0..100` 归一化值，也不会在 Python 层做 `clamp(0, 100)`。

底层 C ABI 可能仍保留历史字段 `position_norm_0_100`，但 Python SDK 对外结果已过滤该字段。

## 4. 双手输出顺序

`joint_positions` 固定为 12 路：

```text
0  left_thumb_flex
1  left_thumb_abduct
2  left_index
3  left_middle
4  left_ring
5  left_pinky
6  right_thumb_flex
7  right_thumb_abduct
8  right_index
9  right_middle
10 right_ring
11 right_pinky
```

`joint_positions_named` 提供同名字典，便于业务侧按名称取值。

## 5. SDK 与测试例程关系

正式接口：

```text
业务程序 -> Revo2DualHandIkSdk.solve(...) -> rad 输出
```

测试例程：

```text
Formotion G2 UDP -> udp_to_sdk_test.py -> Revo2DualHandIkSdk.solve(...) -> RViz
```

因此，交付给甲方时既可以只集成接口，也可以参考 UDP 例程理解输入字段如何转成 SDK 参数。


## 6. 启动标定与重置

创建 SDK 对象后，前若干次 `solve()` 会执行鲁棒开掌标定。调用方应持续输入真实帧，并保持双手完全开掌静止约 2 秒。输出中的：

```python
result["left"]["debug"]
result["right"]["debug"]
```

包含以下状态字段：

```text
calibrated
calibration_stage
calibration_status
calibration_samples
calibration_required
calibration_warmup_remaining
calibration_rejected
calibration_quality_failures
calibration_point_rms_m
calibration_length_cv
```

业务程序应以左右手 `calibrated == True` 作为标定完成条件。切换佩戴者、重新佩戴手套或设备端重新标定后，调用：

```python
sdk.reset()
```

然后重新保持完全开掌，直到左右手再次完成标定。异常帧会自动触发当前连续窗口重新计数，不需要因为单次拒绝就重建 SDK 对象。
