import os
import shlex
import subprocess
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
import time

class TiangongArmIK:
    def __init__(self,urdf_path='tienkung2_pro/urdf/tiangong2.0_pro.urdf', mesh_path='tienkung2_pro/urdf', jit=False, so_path=None):
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.useJit = jit
        
        # 1. 模型加载
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
        self.mixed_jointsToLockIDs = [
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",

            "head_yaw_joint", "head_pitch_joint", "head_roll_joint",

            "left_index_1_joint", "left_index_2_joint", "left_little_1_joint", "left_little_2_joint",
            "left_middle_1_joint", "left_middle_2_joint", "left_ring_1_joint", "left_ring_2_joint",
            "left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint",

            "right_index_1_joint", "right_index_2_joint", "right_little_1_joint", "right_little_2_joint",
            "right_middle_1_joint", "right_middle_2_joint", "right_ring_1_joint", "right_ring_2_joint",
            "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"
        ]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        
        self.L_hand_id_joint = self.reduced_robot.model.getJointId('wrist_roll_l_joint')
        self.R_hand_id_joint = self.reduced_robot.model.getJointId('wrist_roll_r_joint')
        self.reduced_robot.model.addFrame(pin.Frame('L_ee', self.L_hand_id_joint, pin.SE3(np.eye(3), np.zeros(3)), pin.FrameType.OP_FRAME))
        self.reduced_robot.model.addFrame(pin.Frame('R_ee', self.R_hand_id_joint, pin.SE3(np.eye(3), np.zeros(3)), pin.FrameType.OP_FRAME))
        
        self.reduced_robot.data = self.reduced_robot.model.createData()
        
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        
        self.joint_indices = self._get_joint_indices()
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.last_iteration_count = 0
        self.last_return_status = "not_run"
        self.last_position_error = float("nan")
        self.last_orientation_error = float("nan")
        self.last_solver_exception = None
        
        # 2. 核心逻辑
        if so_path and os.path.exists(so_path):
            print(f"Loading external solver from {so_path}...")
            self.ik_solver_func = casadi.external('ik_solver', so_path)
            print("Solver loaded successfully.")
        else:
            self._setup_solver()

    def _get_joint_indices(self):
        indices = {}
        for name in self.reduced_robot.model.names:
            if name == "universe": continue
            j_id = self.reduced_robot.model.getJointId(name)
            idx = self.reduced_robot.model.joints[j_id].idx_q
            indices[name] = idx
        return indices

    def _setup_solver(self):
        nq = self.reduced_robot.model.nq
        cmodel = cpin.Model(self.reduced_robot.model)
        cdata = cmodel.createData()
        
        q = casadi.SX.sym("q", nq)
        q_init = casadi.SX.sym("q_init", nq)
        target_l = casadi.SX.sym("target_l", 4, 4)
        target_r = casadi.SX.sym("target_r", 4, 4)
        
        cpin.framesForwardKinematics(cmodel, cdata, q)
        e_p_l = cdata.oMf[self.L_hand_id].translation - target_l[:3,3]
        e_p_r = cdata.oMf[self.R_hand_id].translation - target_r[:3,3]
        e_r_l = cpin.log3(cdata.oMf[self.L_hand_id].rotation @ target_l[:3,:3].T)
        e_r_r = cpin.log3(cdata.oMf[self.R_hand_id].rotation @ target_r[:3,:3].T)
        
        cost = 60 * (casadi.sumsqr(e_p_l) + casadi.sumsqr(e_p_r)) + \
               3 * (casadi.sumsqr(e_r_l) + casadi.sumsqr(e_r_r)) + \
               1e-4 * casadi.sumsqr(q) + \
               0.01 * casadi.sumsqr(q - q_init)
               
        low = self.reduced_robot.model.lowerPositionLimit.copy()
        high = self.reduced_robot.model.upperPositionLimit.copy()
        low[low < -1e10] = -100
        high[high > 1e10] = 100

        # 构建 NLP 问题字典
        # 使用 casadi.vec 确保矩阵展开为列向量，与后续输入对齐
        p_vec = casadi.vertcat(casadi.vec(target_l), casadi.vec(target_r), q_init)
        nlp = {'x': q, 'f': cost, 'p': p_vec}
        # 创建求解器实例 (使用 IPOPT 求解器)
        # 优化选项：启用 JIT 提高执行速度，调整 ipopt 内部容差
        jit_options = {"flags": ["-O3"], "verbose": False}
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 30,         # 减少最大迭代次数，IK 通常很快收敛
                'tol': 1e-4,            # 适当放开内部容差，因为外部有阈值检查
                'acceptable_tol': 1e-3,
                'linear_solver': 'mumps' # 确保使用高效的线性求解器
            },
            'print_time': False,
            'expand': True,             # 展开 SX 图，对代码生成更友好
            'jit': self.useJit,                # 开启 JIT
            'compiler': 'shell',        # 使用系统编译器（gcc/clang）
            'jit_options': jit_options,
        }
        

        solver = casadi.nlpsol('solver', 'ipopt', nlp, opts)

        # 这里的调用实际上只是为了构建计算图，不会真正执行求解（因为是在定义 Function）
        # 但这种写法生成了一个内部包含 solver 调用的 Function
        p_vec = casadi.vertcat(target_l.reshape((-1,1)), target_r.reshape((-1,1)), q_init)
        res = solver(x0=q_init, p=p_vec, lbx=low, ubx=high)
        self.ik_solver_func = casadi.Function('ik_solver', [target_l, target_r, q_init], [res['x']])

    def export_so(self, output_so="ik_solver.so", condaPath=None):
        print("Generating C code...")
        c_file = "ik_solver.c"
        # 使用 CodeGenerator 自动拉入依赖（包括内部 nlpsol 所需符号）
        cg = casadi.CodeGenerator(c_file)
        # 先展开图，避免运行期再触发插件调用，生成纯数值代码
        f_expanded = self.ik_solver_func.expand()
        cg.add(f_expanded)
        cg.generate()
        
        casadi_path = os.path.dirname(casadi.__file__)
        conda_prefix = condaPath or os.environ.get('CONDA_PREFIX')
        if not conda_prefix:
            raise RuntimeError("无法确定 Conda 环境路径")

        command = [
            "gcc",
            "-fPIC",
            "-shared",
            "-O3",
            c_file,
            "-Dcasadi_inf=INFINITY",
            "-o",
            output_so,
            f"-I{conda_prefix}/include",
            f"-I{casadi_path}/include",
            f"-L{casadi_path}",
            f"-L{conda_prefix}/lib",
            f"-Wl,-rpath,{casadi_path}:{conda_prefix}/lib",
            "-lcasadi",
            "-lipopt",
            "-lstdc++",
            "-lm",
        ]

        print(f"Executing: {shlex.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"IK 动态库编译失败，gcc exit code={result.returncode}")
        print(f"Successfully created {output_so}")

    def get_arm_q(self, q, side="r"):
        side_prefix = "_r" if side == "r" else "_l"
        arm_joint_names = [f"shoulder_pitch{side_prefix}_joint", f"shoulder_roll{side_prefix}_joint", f"shoulder_yaw{side_prefix}_joint",
                           f"elbow_pitch{side_prefix}_joint", f"elbow_yaw{side_prefix}_joint", f"wrist_pitch{side_prefix}_joint", f"wrist_roll{side_prefix}_joint"]
        return np.array([q[self.joint_indices[name]] for name in arm_joint_names])

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        q_init=None,
        max_iters=1,
        pos_threshold=1e-4,
        ori_threshold=1e-4,
        verbose=False,
    ):
        """
        进行逆运动学求解，支持多次迭代以逼近阈值。
        :param left_wrist: 左手目标位姿 (4x4 矩阵)
        :param right_wrist: 右手目标位姿 (4x4 矩阵)
        :param q_init: 可选，作为迭代起点的关节初始值
        :param max_iters: 最大外部迭代次数；实时控制默认只执行一次 NLP 求解
        :param pos_threshold: 位置误差阈值 (m)
        :param ori_threshold: 角度误差阈值 (rad)
        :param verbose: 是否打印每次外部迭代的耗时和残差
        :return: 求解后的完整关节角度序列
        """
        if q_init is not None:
            self.init_data = q_init
        
        iter_count = 0
        self.last_iteration_count = 0
        self.last_return_status = "not_run"
        self.last_position_error = float("nan")
        self.last_orientation_error = float("nan")
        self.last_solver_exception = None
        for i in range(max_iters):
            iter_count = i + 1
            self.last_iteration_count = iter_count
            try:
                start_time = time.time()
                    
                # 显式使用 DM 类型进行数据传递，这在 external call 中更安全
                arg_l = casadi.DM(left_wrist)
                arg_r = casadi.DM(right_wrist)
                arg_q = casadi.DM(self.init_data)
                
                sol_q_dm = self.ik_solver_func(arg_l, arg_r, arg_q)
                sol_q = np.array(sol_q_dm).flatten()
            
                # 计算当前误差
                placement_L = self.reduced_robot.framePlacement(sol_q, self.L_hand_id)
                placement_R = self.reduced_robot.framePlacement(sol_q, self.R_hand_id)
                
                dist_L = np.linalg.norm(placement_L.translation - left_wrist[:3, 3])
                dist_R = np.linalg.norm(placement_R.translation - right_wrist[:3, 3])
                
                # 角度误差计算 (SO3 空间偏差)
                rot_err_mat_L = placement_L.rotation.T @ left_wrist[:3, :3]
                rot_err_mat_R = placement_R.rotation.T @ right_wrist[:3, :3]
                rot_err_L = np.linalg.norm(pin.log3(rot_err_mat_L))
                rot_err_R = np.linalg.norm(pin.log3(rot_err_mat_R))
                self.last_position_error = float(max(dist_L, dist_R))
                self.last_orientation_error = float(max(rot_err_L, rot_err_R))
                
                # 更新当前解，供下次迭代（热启动）
                self.init_data = sol_q

                if verbose:
                    print(f"IK solve: {(time.time() - start_time)*1000:.2f} ms | iters: {iter_count}/{max_iters} | dist L:{dist_L*1000:.3f}mm R:{dist_R*1000:.3f}mm | rot err L:{rot_err_L:.4f} R:{rot_err_R:.4f}")

                # 外部 .so 只导出解向量，无法读取 Ipopt stats；这里记录可观测的末端残差状态。
                if self.last_position_error < pos_threshold and self.last_orientation_error < ori_threshold:
                    self.last_return_status = "residual_converged"
                    break
                self.last_return_status = "max_outer_iterations"
                    
            except Exception as e:
                self.last_return_status = "exception"
                self.last_solver_exception = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"IK solve iteration {i} failed: {e}")
                break
        
        return self.init_data
