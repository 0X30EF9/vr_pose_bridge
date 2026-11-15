"""基于IK的能力封装@LiuXin--2025.10.24，借鉴宇树的Ik能力，使用pinocchio和casadi进行反解，需要传入的参数如下：
   @ x，y, z 相对与机器人末端的平移矩阵
   @ rx, ry, rz 相对于机器人末端的旋转矩阵
   """
import pinocchio as pin

import logging, sys, threading, uuid, json, math, argparse, time, flask, roslibpy, copy

import numpy as np

from flask import request
from pathlib import Path
from collections import deque

from IKAbility_utils import (find_free_port, detect_ability_name, load_robot_host_port,
                             left_init_position, left_init_rotation, right_init_position, right_init_rotation, motors)

from ability_sdk.src.AbilitySDK import ability_stub, ipc_server_handle, log, task

from IK_sdk.UBTECH_IK import UBTECH_TainGong_ArmIK


class IK(ability_stub.AbilityInterface):

    def __init__(self):

        base = Path(sys.argv[0]).name

        if base.endswith('.ability'):
            base = base[:-len('.ability')]
        if '.' in base:
            base = base.split('.', 1)[0]

        ABILITY_NAME = base or 'Head'

        logging.warning(f"[CFG] Detected RAW primary='{ABILITY_NAME}', fallback='{detect_ability_name('HeadAbility')}'")

        cfg_host, cfg_port, ABILITY_NAME, LAST_CFG_PATH_ = load_robot_host_port(ABILITY_NAME=ABILITY_NAME)

        self.app = flask.Flask(__name__)

        self.server_thread, self.port, self.server = None, None, None
        self.running_flag, self.head_joint_control_flag, self.target_tracking_flag = False, False, False
        self.required_keys = {'x', 'y', 'z', 'roll', 'yaw', 'pitch'}

        self.client = roslibpy.Ros(host=cfg_host, port=cfg_port)
        self.joint_state = deque()

        self.left_motors = copy.deepcopy(motors[:7])
        self.right_motors = copy.deepcopy(motors[7:]) 
        self.motors = copy.deepcopy(motors)

        self.client.run()
        logging.warning("IKAbility initialized")

        # 需要做反解的手臂状态，0是只对左手做，1是只对右手做，2是全部做，是否需要用到夹爪，0是不需要，1是需要
        self.joint_id, self.claw_id = 0, 0
        self.max_timeout = 10

        # Step1、初始化能力和flask框架的相关内容

        logging.warning(
            f"IKAbility abilityName='{ABILITY_NAME}' from executable; using websocket {cfg_host}:{cfg_port}; cfg={LAST_CFG_PATH_}")

        logging.warning(f"IKAbility abilityName='{ABILITY_NAME}' from executable")

        @self.app.route('/api/task/start_task', methods=['POST'])
        def start_task_route():
            """存在三个预设状态，第一是单独调用左手，第二是单独调用右手， 第三个全部做，第三是否启用夹爪"""

            route_start = time.time()

            try:
                data = json.loads(request.data)

                if data is None:
                    return {"error": "Invalid or missing JSON data"}, 400

                elif data.get("task_type", -1) not in [0, 1, 2]:
                    return {"error": "Unsupported task_type or no task_type provided"}, 401

                if data.get("task_type", -1) == 0:
                    self.joint_id = 0

                    result = task.execute_task_simple(
                        self.joint_control_func,
                        executor_id=uuid.uuid4(),
                        input=data.get("payload")
                    )

                    logging.info(f"[Head] Route total duration: {time.time() - route_start:.3f}s")
                    return result, 200

                elif data.get("task_type", -1) == 1:
                    self.joint_id = 1
                    exec_start = time.time()
                    result = task.execute_task_simple(
                        self.joint_control_func,
                        executor_id=uuid.uuid4(),
                        input=data.get("payload")
                    )

                    logging.info(f"[Head] task_type=exec duration: {time.time() - exec_start:.3f}s")
                    return result, 200

                elif data.get("task_type", -1) == 2:

                    return {"error": f"The current version does not support IK for both left and right Joints: "}, 400

            except Exception as e:

                logging.info(f"[Head] Route total duration: {time.time() - route_start:.3f}s")
                return {"error": f"Failed to process request: {str(e)}"}, 400

        # 状态路由
        @self.app.route('/api/status', methods=['GET'])
        def status_route():

            start_time = time.time()

            self._joint_subscribe()

            while not self.joint_state:

                if time.time() - start_time > self.max_timeout:
                    logging.error("无法获取数据！超时")

                    return {"message": "task execute timeout"}

                time.sleep(0.2)

            current_position = copy.deepcopy( self.joint_state.pop())

            self.joint_state.clear()

            return {"message": f"Current joint state is {current_position}"}, 200

        # 取消任务路由
        @self.app.route('/api/task/cancel_task', methods=['POST'])
        def cancel_task_route():
            try:
                data = request.get_json()

                if data is None:
                    return {"error": "Invalid or missing JSON data"}, 400
                task_type = data.get("task_type", -1)
                if task_type != 1:
                    return {"error": "Unsupported task_type or no task_type provided"}, 401
                
                logging.warning(f"Cancel Task received data: {data}")

                # 取消并且回到初始位置
                if task_type == 0 or task_type == 1:
                    result = task.execute_task_simple(
                        self.cancel_joint_control_func,
                        executor_id=uuid.uuid4(),
                        input=data.get("payload")
                    )

                    return result, 200

            except Exception as e:
                return {"error": f"Failed to process request: {str(e)}"}, 400

    # TODO 下方代码为特有代码，进行标识，当前IK主要分为两个能力，第一是IK计算的能力，第二是IK运动的能力，暴露在外部的只有一个，因为IK解算将其封装为了对象
    @staticmethod
    def ryp_from_quaternion(x, y, z, roll_rad, pitch_rad, yaw_rad):

        """将当前传入的3维旋转矩阵变成3*3的矩阵，并转换成Ik逆解所需要的格式"""

        translation = np.asarray([x, y, z])

        logging.warning(f"Target translation is {translation}")

        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

        rotation_matrix = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        tf_target = pin.SE3(pin.Quaternion(rotation_matrix), translation)
        logging.warning(f"Target SE3 is {tf_target.homogeneous}")

        return tf_target

    def _joint_subscribe(self):
        """通过rosbrige获取当前的消息"""

        def joint_state_call_back(msg):
            try:
                if self.client.is_connected:
                    self.joint_state.append([round((msg["status"][index]["pos"]), 4) for index in range(len(msg["status"]))])
                    if len(self.joint_state) > 20:
                        self.joint_state.popleft()

            except Exception as e:
                print(f"Error processing message: {e}")
                joint_sub.unsubscribe()

        joint_sub = roslibpy.Topic(self.client, '/arm/status', 'bodyctrl_msgs/msg/MotorStatusMsg')
        joint_sub.subscribe(joint_state_call_back)

    def _joint_move(self, motors, position: list):
        """驱动机器人进行移动"""

        publisher = roslibpy.Topic(self.client, "/arm/cmd_pos", "bodyctrl_msgs/msg/CmdSetMotorPosition")
        publisher.advertise()

        try:
            header = {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''}

            for motor, value in zip(motors, position):
                motor["pos"] = value

            cmd_msg = {'header': header, 'cmds': motors}

            logging.warning(f"Publishing joint command")

            publisher.publish(cmd_msg)

        except Exception as e:

            return False
        
        logging.warning(f"Joint command published successfully")

        publisher.unadvertise()

        return True

    @staticmethod
    def claw_move(client, position):

        """驱动机器人进行夹爪操作"""
        pass

    def joint_control_func(self, task_status: task.TaskStatus, input: dict = None) -> dict:

        self._joint_subscribe()

        target_joint = None

        message = "Arrived"    

        if self.running_flag:
            logging.error("Another IK_control Task is running")
            return {"message": "another task is running"}

        if self.required_keys - input.keys():

            logging.error(f"Missing required parameters: {self.required_keys - input.keys()}")

            return {"message": f"Missing required parameters{self.required_keys - input.keys()}"}

        self.running_flag = True
        self.head_joint_control_flag = True

        # Step1、首先获取当前的手臂状态值
        start_time = time.time()

        while not self.joint_state:

            if time.time() - start_time > self.max_timeout:
                logging.error("无法获取数据！超时")

                return {"message": "task execute timeout"}

            time.sleep(0.2)

        self.cancel_joint_control_position = copy.deepcopy(self.joint_state.pop())

        # Step2、根据当前的id进行赋值和解算操作， 将当前右手臂的值赋0操作，并且进行解算

        start_time_IK = time.time()
        if self.joint_id == 0:

            logging.warning(f"Current left target input is {input}")
            logging.warning(f"Current left initial position is {input.get('x', 0.0)}")
            logging.warning(f"Current left initial rotation is {input.get('y', 0.0)}")

            target_left_position = self.ryp_from_quaternion(x= input.get('x', 0.0) +  input.get('dx', 0.0), y=input.get('y', 0.0) +  input.get('dy', 0.0),
                                                            z= input.get('z', 0.0) +  input.get('dz', 0.0),

                                                            roll_rad=input.get('roll', 0.0),
                                                            pitch_rad=input.get('pitch', 0.0),
                                                            yaw_rad=input.get('yaw', 0.0))

            target_right_position = pin.SE3(pin.Quaternion(right_init_rotation), right_init_position)

            target_joint, _ = self.IK.solve_ik(left_wrist=target_left_position.homogeneous,
                                               right_wrist=target_right_position.homogeneous,
                                               current_lr_arm_motor_q=self.joint_state.pop())
            
            # Step3、根据当前的夹爪状态值进行运行操作

            if not self._joint_move(motors=self.left_motors, position=target_joint[:7]):

                message = "Control failed, please check the parameters"

        elif self.joint_id == 1:
            
            logging.warning(f"Current left target input is {input}")
            logging.warning(f"Current left initial position is {input.get('x', 0.0)}")
            logging.warning(f"Current left initial rotation is {input.get('y', 0.0)}")

            target_right_position = self.ryp_from_quaternion(x=input.get('x', 0.0) +  input.get('dx', 0.0), y=input.get('y', 0.0) +  input.get('dy', 0.0),
                                                             z=input.get('z', 0.0) +  input.get('dz', 0.0),
                                                             roll_rad=input.get('roll', 0.0),
                                                             pitch_rad=input.get('pitch', 0.0),
                                                             yaw_rad=input.get('yaw', 0.0))

            target_left_position = pin.SE3(pin.Quaternion(left_init_rotation), left_init_position)

            target_joint, _ = self.IK.solve_ik(left_wrist=target_left_position.homogeneous,
                                               right_wrist=target_right_position.homogeneous,
                                               current_lr_arm_motor_q=self.joint_state.pop())
            
                    
            if not self._joint_move(motors=self.right_motors, position=target_joint[7:]):

                message = "Control failed, please check the parameters"

        elif self.joint_id == 2:
            pass

        logging.warning(f"Current joint state is {self.joint_state.pop()}")

        logging.warning(f"Target joint angles are {target_joint}")
        logging.warning(f"IK solve duration: {time.time() - start_time_IK:.3f}s")

        ret = {"message": message}

        self.running_flag = False
        self.head_joint_control_flag = False

        self.joint_state.clear()

        # 增加一个timeout时间，保证当前的IK执行结束
        while time.time() - start_time_IK < input.get("timeout", 5):

            time.sleep(0.2)

        return ret

    def cancel_joint_control_func(self, task_status: task.TaskStatus, input: dict = None) -> dict:

        logging.warning(f"Cancel Task received data")

        start_time_cancel = time.time()

        if not self.running_flag or not self.head_joint_control_flag:

            return {"message": f"IK_control Task is not running {self.running_flag} {self.head_joint_control_flag}"}

        self.running_flag = False
        self.target_tracking_flag = False

        logging.warning(f"Cancel Task")
                    
        message = "Task Cancel success"

        if not self._joint_move(position=self.cancel_joint_control_position, motors=self.motors):

            message = "Task is canceled, but failed to stay current pose"

        while time.time() - start_time_cancel < input.get("timeout", 4):

            time.sleep(0.2)

        return {"message": message}

    # 下方代码为通用代码，可以直接复用
    def on_start(self):
        logging.warning("on_start called")

    def on_connect(self):

        self.IK = UBTECH_TainGong_ArmIK()

        self.port = find_free_port()
        self.ability_info.ability_port = self.port
        self.stop_event = threading.Event()

        def run_flask():
            try:
                from werkzeug.serving import make_server
                self.server = make_server('0.0.0.0', self.port, self.app, threaded=True)
                self.server.serve_forever()
            except Exception as e:
                logging.error(f"Flask server error: {e}")

        self.server_thread = threading.Thread(target=run_flask)
        self.server_thread.daemon = True
        self.server_thread.start()
        logging.warning(f"on_connect called, Flask server started on port {self.port}")

    def on_disconnect(self):
        logging.warning("on_disconnect called")
        if self.server_thread and self.server:
            try:
                self.stop_event.set()
                self.server.shutdown()
                self.server.server_close()
                # Send shutdown request to the server
                self.server_thread.join(timeout=3.0)
                if self.server_thread.is_alive():
                    logging.warning("Server thread did not terminate within timeout")
                else:
                    logging.warning("Flask server thread terminated")
            except Exception as e:
                logging.error(f"Error during server shutdown: {e}")
            finally:
                self.server_thread = None
                self.stop_event = None
                self.server = None
                logging.warning("Flask server stopped")
        else:
            logging.warning("No active server thread to stop")

    def on_terminate(self):
        logging.warning("on_terminate called")


if __name__ == "__main__":
    log.configure_logging(sys.argv[0])
    ability = IK()
    handle = ipc_server_handle.HttpIpcServerHandle(sys.argv, ability)
    handle.run()
    logging.warning("HeadAbility process ended")
