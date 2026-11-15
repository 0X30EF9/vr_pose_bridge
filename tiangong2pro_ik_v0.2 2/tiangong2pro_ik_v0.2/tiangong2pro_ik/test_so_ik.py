from tiangong_ik import TiangongArmIK
import numpy as np
import pinocchio as pin
import os,argparse

CASADI_CACHE = os.path.join(os.getcwd(), ".casadi_cache")
os.makedirs(CASADI_CACHE, exist_ok=True)

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

def main(useso=True):
    so_path = None
    if useso:
        so_path = os.path.join(CASADI_CACHE, "ik_solver.so")
        if not os.path.exists(so_path):
            print(f"Error: {so_path} not found. Run generate_ik_so.py first.")
            so_path = None
            print(f"Initializing IK Engine :{so_path}...")
        else:
            print(f"Initializing IK Engine from {so_path}...")
    else:
        print("useso=0, skip loading .so, use Python solver instead.")
    
    arm_ik = TiangongArmIK(so_path=so_path,jit=True)

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0,0.26419,-0.17]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array( [0.38596,-0.36838,0.13425]  ),
    )
    
    sol_q = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

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

    # 强制退出，避免析构函数执行冲突导致的 munmap_chunk() 报错
    # 比如: CasADi/IPOPT 对象析构时可能引发内存错误


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--useso", type=int, default=0, choices=[0, 1], help="1 to load cached ik_solver.so, 0 to use Python solver")
    args = parser.parse_args()
    main(useso=bool(args.useso))
