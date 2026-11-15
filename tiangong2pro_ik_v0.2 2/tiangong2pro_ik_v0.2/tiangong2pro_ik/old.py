import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys
import logging_mp
logger_mp = logging_mp.get_logger(__name__)


######from teleop.utils.weighted_moving_filter import WeightedMovingFilter

class tiangongpro_42_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        print("Loading robot model...")
        self.robot = pin.RobotWrapper.BuildFromURDF('tienkung2_pro/urdf/tiangong2.0_pro.urdf', 'tienkung2_pro/urdf')

        print("Robot model loaded.")    
        # 只锁定腿部关节，保留手臂关节
        self.mixed_jointsToLockIDs = [
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",

            "waist_yaw_joint", "head_yaw_joint", "head_pitch_joint", "head_roll_joint",

            "left_index_1_joint", "left_index_2_joint", "left_little_1_joint", "left_little_2_joint",
            "left_middle_1_joint", "left_middle_2_joint", "left_ring_1_joint", "left_ring_2_joint",
            "left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint",

            "right_index_1_joint", "right_index_2_joint", "right_little_1_joint", "right_little_2_joint",
            "right_middle_1_joint", "right_middle_2_joint", "right_ring_1_joint", "right_ring_2_joint",
            "right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"
        ]


        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # 在构建简化机器人后，打印所有关节名称和ID

        for idx, joint_name in enumerate(self.reduced_robot.model.names):
            print(f"Joint ID: {idx}, Name: {joint_name}")


        print("Reduced robot model created.")   

        self.reduced_robot.model.addFrame(  #向机器人模型添加新的框架
            pin.Frame('L_ee',
                    self.reduced_robot.model.getJointId('wrist_roll_l_joint'),
		            #0，
                    pin.SE3(np.eye(3),
                        np.array([0.00,0,0]).T),
                    pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                    self.reduced_robot.model.getJointId('wrist_roll_r_joint'),
		            #0，
                    pin.SE3(np.eye(3),
                        np.array([0.00,0,0]).T),
                    pin.FrameType.OP_FRAME)
        )

        self.reduced_robot.data = self.reduced_robot.model.createData()

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        print(f"Creating symbolic variables")
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )




        print("Setting up optimization problem...")
        #设置一个非线性优化问题，用于求解机器人的逆运动学问题
        self.opti = casadi.Opti()   #创建一个CasADi优化问题对象
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)    #创建一个优化变量，表示机器人的关节角度
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # 创建一个参数，表示上一时刻的关节角度for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))#计算平移误差的平方和
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))#计算旋转误差的平方和
        self.regularization_cost = casadi.sumsqr(self.var_q)    #计算关节角度的平方和
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)  #当前关节角度与上一时刻关节角度差的平方和 for smooth

        # Setting optimization constraints and goals
        #设置优化问题的约束条件和目标函数

        self.opti.subject_to(self.opti.bounded(     #向优化问题添加约束条件
            self.reduced_robot.model.lowerPositionLimit,    #各关节的下限位置
            self.var_q,                                     #定义的优化变量
            self.reduced_robot.model.upperPositionLimit)    #关节的上限位置
        )
        #优化问题的目标函数目标是找到使这个表达式值最小的关节角度
            # 50 * self.translational_cost: 平移误差成本，权重为50
            # self.rotation_cost: 旋转误差成本，权重为1
            # 0.02 * self.regularization_cost: 正则化成本，权重为0.02
            # 0.1 * self.smooth_cost: 平滑性成本，权重为0.1
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)



        opts = {        #创建一个字典，包含IPOPT求解器的各种配置选项
            'ipopt':{
                'print_level':0,        #0表示最小输出，只显示关键信息
                'max_iter':200,          #最大迭代次数   
                'tol':1e-6              #目标函数的变化小于这个值时，认为已经收敛
            },
            'print_time':False,#表示不打印求解时间
            'calc_lam_p':False # False表示不计算拉格朗日乘子    https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        #self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None
        self.joint_indices = self._get_joint_indices()


    def _get_joint_indices(self):
        """ 获取简化模型中各关节的索引映射 """
        indices = {}
        for name in self.reduced_robot.model.names:
            if name == "universe": continue
            j_id = self.reduced_robot.model.getJointId(name)
            idx = self.reduced_robot.model.joints[j_id].idx_q
            indices[name] = idx
        return indices


    def get_arm_q(self, q, side="r"):
        """ 从全关节状态 q 中提取指定侧手臂的关节角度 """
        side_prefix = "_r" if side == "r" else "_l"
        arm_joint_names = [f"shoulder_pitch{side_prefix}_joint", f"shoulder_roll{side_prefix}_joint", f"shoulder_yaw{side_prefix}_joint",
                           f"elbow_pitch{side_prefix}_joint", f"elbow_yaw{side_prefix}_joint", f"wrist_pitch{side_prefix}_joint", f"wrist_roll{side_prefix}_joint"]
        return np.array([q[self.joint_indices[name]] for name in arm_joint_names])

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        t = time.time()

        # 如果提供了当前电机角度，将其作为优化的初始值
        # 设置优化变量的初始值为self.init_data
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)



        # 设置左右手腕目标位姿参数
        # 设置上一时刻的关节角度参数（用于平滑性约束）          
        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # 尝试求解优化问题
            # 获取解（关节角度）
            # 将解添加到平滑滤波器中
            # 获取滤波后的关节角度

            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            #self.smooth_filter.add_data(sol_q)
            #sol_q = self.smooth_filter.filtered_data


            # 计算关节速度（这里都乘以0.0，所以实际上是零速度）
            # 更新self.init_data为当前解
            # 使用递归牛顿欧拉算法（RNEA）计算关节力矩            

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            placement_L = self.reduced_robot.framePlacement(sol_q, self.L_hand_id)
            placement_R = self.reduced_robot.framePlacement(sol_q, self.R_hand_id)
            
            dist_L = np.linalg.norm(placement_L.translation - left_wrist[:3, 3])
            dist_R = np.linalg.norm(placement_R.translation - right_wrist[:3, 3])

            print(f"IK solve time: {(time.time() - t)*1000:.2f} ms dist L:{dist_L*1000:.3f}mm R:{dist_R*1000:.3f}mm ")

            return sol_q, sol_tauff
        
        except Exception as e:
            print("IK求解失败:", e)
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            #self.smooth_filter.add_data(sol_q)
            #sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
        


def compute_wrist_roll_r_pose(q_body, q_arm):

    import numpy as np

    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])

    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

    def rpy_to_matrix(r, p, y):
        # Roll-Pitch-Yaw to rotation matrix (ZYX order)
        return rot_z(y) @ rot_y(p) @ rot_x(r)

    def trans(x, y, z):
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        return T

    # q_body: body_yaw_joint
    # q_arm: [shoulder_pitch_r, shoulder_roll_r, shoulder_yaw_r, elbow_pitch_r, elbow_yaw_r, wrist_pitch_r, wrist_roll_r]
    
    # 0. body_yaw_joint (parent: pelvis/universe, child: body_yaw_link)
    # Based on URDF: origin rpy="0 0 0" xyz="0 0 0"
    T = rot_z(q_body)

    # 1. shoulder_pitch_r_joint (parent: body_yaw_link)
    # URDF: origin rpy="-0.087266 0 0" xyz="0 -0.1444 0.37712"
    T = T @ trans(0, -0.1444, 0.37712) @ rpy_to_matrix(-0.087266, 0, 0) @ rot_y(q_arm[0])
    
    # 2. shoulder_roll_r_joint
    # URDF: origin rpy="0 0 0" xyz="0 -0.068 0"
    T = T @ trans(0, -0.068, 0) @ rot_x(q_arm[1])
    
    # 3. shoulder_yaw_r_joint
    # URDF: origin rpy="0 0 0" xyz="0 0 -0.1025"
    T = T @ trans(0, 0, -0.1025) @ rot_z(q_arm[2])
    
    # 4. elbow_pitch_r_joint
    # URDF: origin rpy="0 0 0" xyz="0.02 0 -0.1975"
    T = T @ trans(0.02, 0, -0.1975) @ rot_y(q_arm[3])
    
    # 5. elbow_yaw_r_joint
    # URDF: origin rpy="0 0 0" xyz="-0.02 0 -0.0543"
    T = T @ trans(-0.02, 0, -0.0543) @ rot_z(q_arm[4])
    
    # 6. wrist_pitch_r_joint
    # URDF: origin rpy="0 0 0" xyz="0 0 -0.1887"
    T = T @ trans(0, 0, -0.1887) @ rot_y(q_arm[5])
    
    # 7. wrist_roll_r_joint
    # URDF: origin rpy="0 0 0" xyz="0 0 -0.02"
    T = T @ trans(0, 0, -0.02) @ rot_x(q_arm[6])
    
    return T



if __name__ == "__main__":

    arm_ik = tiangongpro_42_ArmIK(Unit_Test = True, Visualization = False)


    '''
    0点:   [0,-26.419,-18]
    '''
    '''
    [-0.60, 0.00, 1.20, -1.00, 0.00, 0.00, 0.00]
    [0.329,-0.028,0.035]


    右手预备抓取位置
    [0.20, -0.80, 0.00, -2.00, 0.80, 0.00, 0.00]
    [23.366,-38.045,20.312]
    '''



    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0,0.26419,-0.17]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array( [0.38596,-0.36838,0.13425]  ),
    )

    print("test")
    for _ in range(5):
        arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
    
    sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

    sol_q_body = sol_q[arm_ik.joint_indices["body_yaw_joint"]]
    sol_q_left = arm_ik.get_arm_q(sol_q, side="l")
    sol_q_right = arm_ik.get_arm_q(sol_q, side="r")

    
    # 动态寻找关节索引
    sol_q_body = sol_q[arm_ik.joint_indices["body_yaw_joint"]]
    sol_q_left = arm_ik.get_arm_q(sol_q, side="l")
    sol_q_right = arm_ik.get_arm_q(sol_q, side="r")

    print("-----------------right hand-------------------")
    print(f"Target set: [0.38596, -0.36838, 0.13425]")
    print("求解得到的右手 7 个关节角度:", sol_q_right)
    # 使用 Pinocchio 的正向运动学验证 (建议以此为准)
    R_pos_pin = arm_ik.reduced_robot.framePlacement(sol_q, arm_ik.R_hand_id).translation
    print(f"Right Hand FK Position (Pinocchio): {R_pos_pin}")
    # Manual FK 现在包含 body_yaw 并修正了参数
    print(f"Right Hand FK Position (Manual):    {compute_wrist_roll_r_pose(sol_q_body, sol_q_right)[:3, 3]}")

    print("------------------left hand-------------------")
    print(f"Target set: [0, 0.26419, -0.17]")
    print("求解得到的左手 7 个关节角度:", sol_q_left)
    L_pos_pin = arm_ik.reduced_robot.framePlacement(sol_q, arm_ik.L_hand_id).translation
    print(f"Left Hand FK Position (Pinocchio):  {L_pos_pin}")
    print("----------------------------------------------")
