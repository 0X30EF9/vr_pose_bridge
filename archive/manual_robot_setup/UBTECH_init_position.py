import sys
import time

import roslibpy
import copy
import logging
import argparse

from urllib.parse import urlparse

from constant import left_up_position, left_down_position, right_up_position, right_down_position


control_dict = {"name": 0.0, "pos": 0.0, "spd": 0.7, "cur": 2.0}
joint_motors = [copy.deepcopy(control_dict) for _ in range(14)]
head_motors = [copy.deepcopy(control_dict) for _ in range(3)]

for i, motor in enumerate(joint_motors, 11):
    motor["name"] = (i if i < 18 else i + 3)

for i, motor in enumerate(head_motors, 0):
    motor["name"] = i

logging.basicConfig(level=logging.INFO)


def joint_move(joint_publisher, position: list):
    """驱动机器人进行移动"""

    try:
        header = {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''}

        for motor, value in zip(joint_motors, position):
            motor["pos"] = value

        cmd_msg = {'header': header, 'cmds': joint_motors}

        joint_publisher.publish(cmd_msg)

    except Exception as e:

        return False

    return True


def head_move(client, position: list):
    """驱动机器人进行头部移动"""

    publisher = roslibpy.Topic(client, "/head/cmd_pos", "bodyctrl_msgs/msg/CmdSetMotorPosition")
    publisher.advertise()

    try:
        header = {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''}

        for motor, value in zip(head_motors, position):
            motor["pos"] = value

        cmd_msg = {'header': header, 'cmds': head_motors}

        publisher.publish(cmd_msg)

    except Exception as e:

        return False

    publisher.unadvertise()

    return True


def main(client, joint_position: list, head_position: list = None, head_control: bool = False, ):

    joint_publisher = roslibpy.Topic(client, "/arm/cmd_pos", "bodyctrl_msgs/msg/CmdSetMotorPosition")
    joint_publisher.advertise()

    if head_control:

        assert len(head_position) > 0, "Head position not zero, please check the input"

        head_move(client=client, position=head_position)

    try:

        for current_position in joint_position:

            joint_move(joint_publisher=joint_publisher, position=current_position)

            time.sleep(2)
            logging.info(current_position)

        joint_publisher.unadvertise()

        return True

    except Exception as e:

        logging.error(f"Error {e}")

        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='控制机器人移动到初始位置')

    parser.add_argument('--joint_control', type=str, help='输入up或者down，前者是胳膊抬起来，后者是胳膊放下，但是注意，放下是直接放下', default="up")
    parser.add_argument('--joint_control_id', type=int, help='输入0或者1，前者是左手，后者是右手', default=0)
    parser.add_argument('--action', type=int, choices=[1, 2, 3, 4], help='数字选项：1=左手抬起 2=右手抬起 3=左手放下 4=右手放下（会覆盖 joint_control 与 joint_control_id）', default=None)

    parser.add_argument('--head_control', type=bool, help='输入是否使用头部控制', default=False)
    parser.add_argument('--max_timeout', type=int, help='最大timeout时间', default=100)

    parser.add_argument('--url', type=str, help='当前的网桥')

    args = parser.parse_args()

    # 如果使用 --action 参数，则映射到现有的 joint_control 与 joint_control_id
    if args.action is not None:
        assert args.action in [1, 2, 3, 4], "args.action 必须是 1-4 之间的整数"
        if args.action == 1:
            args.joint_control = "up"
            args.joint_control_id = 0
        elif args.action == 2:
            args.joint_control = "up"
            args.joint_control_id = 1
        elif args.action == 3:
            args.joint_control = "down"
            args.joint_control_id = 0
        elif args.action == 4:
            args.joint_control = "down"
            args.joint_control_id = 1

    client_ = urlparse(f"ws://{args.url}" if args.url else "ws://192.168.1.110:9091")

    client = roslibpy.Ros(host=client_.hostname, port=client_.port)

    client.run()

    start_time = time.time()

    while not client.is_connected:

        if time.time() - start_time > args.max_timeout:

            logging.error("无法获取数据！超时")

            sys.exit(500)

        time.sleep(0.2)
        

    logging.info(f"Current rosbrige ip and post is {client}")

    if args.joint_control and "up" in args.joint_control:

        assert args.joint_control_id in [0, 1], "args.joint_control_id is Error"

        position = left_up_position if args.joint_control_id == 0 else right_up_position

        if not main(client=client, joint_position=position, head_control=args.head_control):

            logging.error("Move False")

        logging.info("Move Success")

    elif args.joint_control and "down" in args.joint_control:

        assert args.joint_control_id in [0, 1], "args.joint_control_id is Error"

        position = left_down_position if args.joint_control_id == 0 else right_down_position

        if not main(client=client, joint_position=position, head_control=args.head_control):

            logging.error("Move False")

        logging.info("Move Success")

    else:

        logging.error(f"The current input for `args.control` is {args.joint_control}")

        raise ImportError("The current input for `args.joint_control` is incorrect. Please check your input.")
