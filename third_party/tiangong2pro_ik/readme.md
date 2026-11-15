# 天工 2.0 Pro 逆运动学 (IK) 求解器

本项目提供了针对天工 2.0 Pro 机器人的双臂逆运动学求解方案，基于 Pinocchio 动力学库和 CasADi 优化框架实现。主要包含基础优化版本和高性能代码生成版本。

## 安装依赖
```bash
conda install pinocchio=3.6.0 -c conda-forge
```

## 项目结构

- [old.py](old.py): 原始的 IK 实现，使用 CasADi 的 `Opti` 接口。适用于快速测试和验证，但在实时性要求较高的场景下性能有限。
- [tiangong_ik.py](tiangong_ik.py): 高性能 IK 引擎。支持以下特性：
    - 手动锁定无关关节（如腿部、腰部），仅对双臂进行求解。
    - 支持 CasADi 代码生成（C code generation）以导出外部求解器。
- [gen_so.py](gen_so.py): 用于将 IK 逻辑编译为 `.so` 动态库的工具脚本。编译后的求解器具有极高的执行效率（通常在 1ms 左右）。
- [test_so_ik.py](test_so_ik.py): 演示如何调用生成的 `.so` 文件进行实时 IK 求解。
- [tienkung2_pro/](tienkung2_pro/): 存放机器人的 URDF 模型、Mesh 资源及相关配置文件。

## 依赖要求

- **Python 3.10+**
- **Pinocchio**: 用于机器人模型解析与运动学计算。
- **CasADi**: 提供自动微分与优化框架接口。
- **IPOPT**: 底层非线性优化求解器。
- **GCC**: 导出 `.so` 时需要系统环境具备 C 语言编译能力。

## 快速上手

### 1. 运行基础示例
```bash
python old.py
```

### 2. 构建高性能求解器
运行以下命令生成 `.so` 文件（默认保存在 `.casadi_cache/`）：
```bash
python gen_so.py
```

### 3. 使用生成的求解器
```bash
python test_so_ik.py --useso=1 
```
默认不加载so,使用实时编译

## 开发者笔记

- **模型同步**: 在 Pinocchio 中修改 `model`（如添加 Frame）后，必须重新调用 `model.createData()`，否则会发生内存越界导致段错误。
- **求解效率**: 若需极速体验，请务必使用 `gen_so.py` 方案。
- **关节映射**: `get_arm_q` 方法可方便地从完整关节向量中提取出指定的 7 自由度手臂关节。


## 执行对比

### old.py

```bash
(ik) soom@soom-thinkpad:/data/work/ik$ python old.py 
Loading robot model...
Robot model loaded.
Joint ID: 0, Name: universe
Joint ID: 1, Name: body_yaw_joint
Joint ID: 2, Name: shoulder_pitch_l_joint
Joint ID: 3, Name: shoulder_roll_l_joint
Joint ID: 4, Name: shoulder_yaw_l_joint
Joint ID: 5, Name: elbow_pitch_l_joint
Joint ID: 6, Name: elbow_yaw_l_joint
Joint ID: 7, Name: wrist_pitch_l_joint
Joint ID: 8, Name: wrist_roll_l_joint
Joint ID: 9, Name: shoulder_pitch_r_joint
Joint ID: 10, Name: shoulder_roll_r_joint
Joint ID: 11, Name: shoulder_yaw_r_joint
Joint ID: 12, Name: elbow_pitch_r_joint
Joint ID: 13, Name: elbow_yaw_r_joint
Joint ID: 14, Name: wrist_pitch_r_joint
Joint ID: 15, Name: wrist_roll_r_joint
Reduced robot model created.
Creating symbolic variables
Setting up optimization problem...
test

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

IK solve time: 66.51 ms dist L:3.591mm R:16.260mm 
IK solve time: 7.29 ms dist L:2.118mm R:4.700mm 
IK solve time: 7.93 ms dist L:1.756mm R:3.273mm 
IK solve time: 8.35 ms dist L:1.609mm R:3.038mm 
IK solve time: 8.21 ms dist L:1.521mm R:2.999mm 
IK solve time: 7.97 ms dist L:1.459mm R:2.997mm 
-----------------right hand-------------------
Target set: [0.38596, -0.36838, 0.13425]
求解得到的右手 7 个关节角度: [-1.23139  0.04345  0.20048  0.26175  0.28218  0.86154  0.47724]
Right Hand FK Position (Pinocchio): [ 0.38723 -0.36862  0.13155]
Right Hand FK Position (Manual):    [ 0.38723 -0.36862  0.13155]
------------------left hand-------------------
Target set: [0, 0.26419, -0.17]
求解得到的左手 7 个关节角度: [ 0.1306  -0.01497  0.15632  0.0438   0.15449 -0.14214 -0.11319]
Left Hand FK Position (Pinocchio):  [ 0.00002  0.26424 -0.17146]
----------------------------------------------
```
### 使用so
```bash
(ik) soom@soom-thinkpad:/data/work/ik$ python test_so_ik.py --useso=1
Initializing IK Engine from /data/work/ik/.casadi_cache/ik_solver.so...
Loading external solver from /data/work/ik/.casadi_cache/ik_solver.so...
Solver loaded successfully.

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

IK solve: 12.10 ms | iters: 1/5 | dist L:0.657mm R:1.291mm | rot err L:0.0009 R:0.0047
IK solve: 5.43 ms | iters: 2/5 | dist L:0.098mm R:0.025mm | rot err L:0.0000 R:0.0002
IK solve: 5.48 ms | iters: 3/5 | dist L:0.018mm R:0.017mm | rot err L:0.0000 R:0.0001
-----------------right hand-------------------
Target set: [0.38596, -0.36838, 0.13425]
求解得到的右手 7 个关节角度: [-1.22743786  0.05978102  0.21879012  0.23627918  0.30994498  0.89574756
  0.52252952]
Right Hand FK Position (Pinocchio): [ 0.38596865 -0.36838051  0.13423502]
Right Hand FK Position (Manual):    [ 0.38596865 -0.36838051  0.13423502]
------------------left hand-------------------
Target set: [0, 0.26419, -0.17]
求解得到的左手 7 个关节角度: [ 0.12618764 -0.01540439  0.16841887  0.073506    0.16445674 -0.16839378
 -0.12107895]
Left Hand FK Position (Pinocchio):  [-6.37691747e-07  2.64191763e-01 -1.70017799e-01]
----------------------------------------------
```

### 使用jit 动态编译

```bash
(ik) soom@soom-thinkpad:/data/work/ik$ python test_so_ik.py --useso=0
useso=0, skip loading .so, use Python solver instead.

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

IK solve: 6.23 ms | iters: 1/5 | dist L:0.657mm R:1.291mm | rot err L:0.0009 R:0.0047
IK solve: 2.79 ms | iters: 2/5 | dist L:0.098mm R:0.025mm | rot err L:0.0000 R:0.0002
IK solve: 2.67 ms | iters: 3/5 | dist L:0.018mm R:0.017mm | rot err L:0.0000 R:0.0001
-----------------right hand-------------------
Target set: [0.38596, -0.36838, 0.13425]
求解得到的右手 7 个关节角度: [-1.22743786  0.05978102  0.21879012  0.23627918  0.30994498  0.89574756
  0.52252952]
Right Hand FK Position (Pinocchio): [ 0.38596865 -0.36838051  0.13423502]
Right Hand FK Position (Manual):    [ 0.38596865 -0.36838051  0.13423502]
------------------left hand-------------------
Target set: [0, 0.26419, -0.17]
求解得到的左手 7 个关节角度: [ 0.12618764 -0.01540439  0.16841887  0.073506    0.16445674 -0.16839378
 -0.12107895]
Left Hand FK Position (Pinocchio):  [-6.37691747e-07  2.64191763e-01 -1.70017799e-01]
```