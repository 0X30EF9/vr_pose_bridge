"""基于IK的能力封装@LiuXin--2025.10.24，借鉴宇树的Ik能力，使用pinocchio和casadi进行反解，需要传入的参数如下：
   @ x，y, z 相对与机器人末端的平移矩阵
   @ rx, ry, rz 相对于机器人末端的旋转矩阵
   """

import logging, time, flask, roslibpy

from collections import deque
# import ipopt
import casadi as ca
print(ca.file)

joint_state = deque()

class IK():

    def __init__(self):
        cfg_host = "192.168.1.110"
        cfg_port = 9091

        self.client = roslibpy.Ros(host=cfg_host, port=cfg_port)
        self.client.run()
        logging.warning("IKAbility initialized")
        self.app = flask.Flask(__name__)
        self.max_timeout = 20
        self.joint_state = deque()

    def status_route(self):
        """获取当前的关节状态"""

        start_time = time.time()

        self.joint_subscribe(client=self.client)

        while not self.joint_state:

            if time.time() - start_time > self.max_timeout:
                logging.error("无法获取数据！超时")

                return {"message": "task execute timeout"}

            time.sleep(0.2)
        
        return {"message": f"Current joint state is {self.joint_state.pop()}"}
    
    def joint_subscribe(self, client):
        """通过rosbrige获取当前的消息"""
        logging.warning("Subscribing to /arm/status topic")

        def joint_state_call_back(msg):
            try:
                if client.is_connected:
                    self.joint_state.append([round((msg["status"][index]["pos"]), 4) for index in range(len(msg["status"]))])
                    # logging.warning(f"Current joint state is {joint_state.pop()}")

                    if len(self.joint_state) > 20:

                        self.joint_state.popleft()

            except Exception as e:
                print(f"Error processing message: {e}")
                joint_sub.unsubscribe()

        joint_sub = roslibpy.Topic(client, '/arm/status', 'bodyctrl_msgs/msg/MotorStatusMsg')
        joint_sub.subscribe(joint_state_call_back)
 
if __name__ == "__main__":

    ability = IK()
    ability.status_route()  
