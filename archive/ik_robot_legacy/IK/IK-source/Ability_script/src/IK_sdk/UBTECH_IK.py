import os, sys, time, logging
from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin

try:
    import meshcat.geometry as mg
except ModuleNotFoundError:
    mg = None

from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

def _find_default_model_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "models" / "walker_tienkung_ei_description"
        if (candidate / "walker_tienkung_ei.urdf").is_file():
            return candidate
    return Path(__file__).resolve().parent / "walker_tienkung_ei_description"


_DEFAULT_MODEL_DIR = _find_default_model_dir()
_DEFAULT_URDF_PATH = _DEFAULT_MODEL_DIR / "walker_tienkung_ei.urdf"

casadi_path = os.environ.get("CASADI_PATH")
if casadi_path:
    casadi.GlobalOptions.setCasadiPath(casadi_path)

class UBTECH_TainGong_ArmIK:

    def __init__(self, urdf_path: str | os.PathLike | None = None, package_dir: str | os.PathLike | None = None):

        resolved_urdf_path = Path(urdf_path) if urdf_path is not None else _DEFAULT_URDF_PATH
        resolved_package_dir = Path(package_dir) if package_dir is not None else _DEFAULT_MODEL_DIR
        if not resolved_urdf_path.is_file():
            raise FileNotFoundError(f"URDF 文件不存在: {resolved_urdf_path}")

        # 加载相关的urdf模型，需要给定两个参数，第一是urdf_path，第二是package_dirs
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(resolved_urdf_path),
            str(resolved_package_dir))

        # 需要锁住的相关关节，按顺序排列，需要根据当前的urdf进行相关的排列，已经验证
        self.mixed_jointsToLockIDs = [
            "head_roll_joint", "head_pitch_joint", "head_yaw_joint", "waist_yaw_joint",
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",]

        self.reduced_robot = self.robot.buildReducedRobot(list_of_joints_to_lock=self.mixed_jointsToLockIDs, reference_configuration=np.array([0.0] * self.robot.model.nq), )

        self.reduced_robot.model.addFrame( pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('wrist_roll_l_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.025, -0.07, -0.21])),
                      pin.FrameType.OP_FRAME) )

        self.reduced_robot.model.addFrame( pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('wrist_roll_r_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.025, 0.07, -0.21])),
                      pin.FrameType.OP_FRAME) )

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l, self.cTf_r  = casadi.SX.sym("tf_l", 4, 4), casadi.SX.sym("tf_r", 4, 4)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        self.L_hand_id, self.R_hand_id  = self.reduced_robot.model.getFrameId("L_ee"), self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3]
                )
            ],
        )

        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T)
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)

        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l,self.param_tf_r  = self.opti.parameter(4, 4), self.opti.parameter(4, 4)

        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)  # TODO 平滑处理，可能会出现问题

        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )

        self.opti.minimize( 20 * self.translational_cost + 0.05 * self.rotation_cost + 0.0001 * self.regularization_cost + 0.005 * self.smooth_cost)

        opts = {'ipopt': {'print_level': 0, 'max_iter': 300, 'tol': 1e-6}, 'print_time': False, 'calc_lam_p': False }

        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        logging.warning("UBTECH_TainGong_ArmIK initialized")

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None, print_error: bool = True):

        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q

        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()

            sol_q = self.opti.value(self.var_q)

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if print_error:

                self.print_errors(sol_q, left_wrist, right_wrist)

            return np.around(sol_q, 4).tolist(), sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(
                f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")

            return current_lr_arm_motor_q.tolist(), np.zeros(self.reduced_robot.model.nv)

    def print_errors(self, q, left_target, right_target):
        """计算并打印位置和姿态误差"""

        data = self.reduced_robot.model.createData()
        pin.framesForwardKinematics(self.reduced_robot.model, data, q)
        pin.updateFramePlacements(self.reduced_robot.model, data)

        left_frame_id = self.reduced_robot.model.getFrameId("L_ee")
        right_frame_id = self.reduced_robot.model.getFrameId("R_ee")

        left_pose = data.oMf[left_frame_id]
        right_pose = data.oMf[right_frame_id]

        # 位置误差
        left_pos_error = np.linalg.norm(left_pose.translation - left_target[:3, 3])
        right_pos_error = np.linalg.norm(right_pose.translation - right_target[:3, 3])

        # 姿态误差（使用四元数）
        left_rot_actual = pin.Quaternion(left_pose.rotation)
        left_rot_target = pin.Quaternion(left_target[:3, :3])
        left_rot_error = np.arccos(2 * np.square(np.dot(left_rot_actual.coeffs(), left_rot_target.coeffs())) - 1)

        right_rot_actual = pin.Quaternion(right_pose.rotation)
        right_rot_target = pin.Quaternion(right_target[:3, :3])
        right_rot_error = np.arccos(2 * np.square(np.dot(right_rot_actual.coeffs(), right_rot_target.coeffs())) - 1)

        print(f"Position Errors - Left: {left_pos_error:.6f}, Right: {right_pos_error:.6f}")
        print(f"Orientation Errors - Left: {left_rot_error:.6f}, Right: {right_rot_error:.6f}")

if __name__ == "__main__":

    FK: bool = False

    joint_angles = [-0.0, 0.07, -0.0, -0.0, 0.0, 0.0, 0.0,  0.400, -0.2600, -0.1000, -1.9999, 1.1999, 0.0002, -0.0007]

    if FK:

        arm_fk = UBTECH_TainGong_ArmFK(Unit_Test=False, Visualization=False)

        eef = arm_fk.solve_fk(joint_angles=joint_angles)

        print("the arm current positiois {}, {}".format(eef["left_arm"]["position"], eef["right_arm"]["position"], ))


        left_position = np.asarray([0.00005, 0.26419, -0.17755])
        right_position = np.asarray([0.536, 0.022, 0.012])
        left_rotation = eef["left_arm"]["rotation_matrix"]
        right_rotation = eef["right_arm"]["rotation_matrix"]

    else:

        left_position = np.asarray([0.02505, 0.2670, -0.3909])
        left_rotation = np.asarray([[1.0, 0.0, -0.0], [-0.0, 0.9876, -0.1566], [0.0, 0.1566, 0.9876]])

        right_position = np.asarray([0.4379, -0.0883, -0.2174])
        right_position[2] = right_position[2] + 0.02
        right_rotation = np.asarray([[-0.1189, -0.62844, -0.7687], [0.99, -0.070, -0.096], [0.006, -0.77, 0.6323]])

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(left_rotation),
        left_position,
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(right_rotation),
        right_position,
    )

    arm_ik = UBTECH_TainGong_ArmIK()

    target_pose, _ = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_motor_q=np.asarray(joint_angles))

    print("IK test finished.",target_pose)
