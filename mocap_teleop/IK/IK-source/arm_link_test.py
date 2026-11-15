from __future__ import annotations

import argparse
import copy
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import roslibpy

SOURCE_DIR = Path(__file__).resolve().parent
INIT_POSITION_DIR = SOURCE_DIR / "init_position"
if str(INIT_POSITION_DIR) not in sys.path:
    sys.path.insert(0, str(INIT_POSITION_DIR))

from teleop_hand_control import (
    HandControlError,
    build_hand_joint_state_message,
    parse_hand_state_message,
)
from teleop_robot_constants import (  # noqa: E402
    ARM_MOTOR_TEMPLATES,
    HAND_CMD_TOPICS,
    HAND_JOINT_STATE_TYPE,
    HAND_STATE_TOPICS,
    NATURAL_DOWN_Q,
)

ARM_STATUS_TOPIC = "/arm/status"
ARM_STATUS_TYPE = "bodyctrl_msgs/msg/MotorStatusMsg"
ARM_CMD_TOPIC = "/arm/cmd_pos"
ARM_CMD_TYPE = "bodyctrl_msgs/msg/CmdSetMotorPosition"
ARM_MOTOR_IDS = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]

DEFAULT_HOST = "192.168.1.110"
DEFAULT_PORT = 9091
DEFAULT_CONNECT_TIMEOUT = 8.0
DEFAULT_STATUS_TIMEOUT = 5.0
DEFAULT_HOLD_SECONDS = 3.0
DEFAULT_RESTORE_SECONDS = 3.0

TEST_PROFILES = {
    "left": {
        "ids": [12, 14],
        "description": "左臂安全小幅测试：肩 roll 外展 + 肘 pitch 弯曲",
    },
    "right": {
        "ids": [22, 24],
        "description": "右臂安全小幅测试：肩 roll 外展 + 肘 pitch 弯曲",
    },
    "both": {
        "ids": [12, 14, 22, 24],
        "description": "左右臂同时安全小幅测试：肩 roll 外展 + 肘 pitch 弯曲",
    },
}

SAFE_TEST_DELTAS = {
    12: +1.0,  # 左肩 roll 外展
    14: -1.0,  # 左肘 pitch 弯曲
    22: -1.0,  # 右肩 roll 外展
    24: -1.0,  # 右肘 pitch 弯曲
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手臂与灵巧手 rosbridge 链路测试：读取反馈、顺序执行小幅动作并自动恢复",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="rosbridge 地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="rosbridge 端口")
    parser.add_argument("--side", choices=("left", "right", "both"), default="left", help="测试手臂：left/right/both")
    parser.add_argument("--scale", type=float, default=0.35, help="参考 init_position 目标的插值比例，默认 0.35")
    parser.add_argument("--min-delta", type=float, default=0.08, help="至少制造的单关节目标变化，默认 0.08 rad")
    parser.add_argument("--max-delta", type=float, default=0.35, help="单关节最大目标变化限幅，默认 0.35 rad")
    parser.add_argument("--spd", type=float, default=0.7, help="电机速度参数，默认对齐 init_position")
    parser.add_argument("--cur", type=float, default=2.0, help="电机电流参数，默认对齐 init_position")
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS, help="动作目标保持时间")
    parser.add_argument("--restore-seconds", type=float, default=DEFAULT_RESTORE_SECONDS, help="发布结束姿态后的观察时间")
    parser.add_argument(
        "--finish-pose",
        choices=("initial", "baseline", "natural_down"),
        default="initial",
        help="测试结束后固定到的姿态：initial/baseline=测试开始时 /arm/status 读到的位置；natural_down=手写自然下垂近似姿态",
    )
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--status-timeout", type=float, default=DEFAULT_STATUS_TIMEOUT)
    parser.add_argument("--feedback-threshold", type=float, default=0.03, help="判定手臂有动作的最小反馈变化")
    parser.add_argument("--hand-delta", type=float, default=0.20, help="双手 1..5 号通道的握紧变化，默认 0.20")
    parser.add_argument("--hand-feedback-threshold", type=float, default=0.02, help="判定灵巧手有动作的最小反馈变化")
    parser.add_argument("--arm-only", action="store_true", help="仅运行旧版手臂测试，不测试灵巧手")
    parser.add_argument("--execute", action="store_true", help="真正发布控制命令；不加则只做 dry-run 预览")
    parser.add_argument("--yes", action="store_true", help="跳过二次确认，配合 --execute 使用")
    return parser.parse_args()


def first_present(item: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        if key in item:
            return item[key]
    return None


def normalize_status(msg: dict[str, Any]) -> dict[int, dict[str, float]]:
    raw_items = msg.get("status", [])
    if not isinstance(raw_items, list):
        return {}

    by_id = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        motor_id = first_present(item, ("name", "id", "motor_id", "motorId"))
        pos = first_present(item, ("pos", "position", "angle", "q"))
        try:
            parsed_id = int(motor_id)
            by_id[parsed_id] = {
                "pos": float(pos),
                "speed": float(item.get("speed", 0.0)),
                "current": float(item.get("current", 0.0)),
                "temperature": float(item.get("temperature", 0.0)),
                "error": float(item.get("error", 0.0)),
            }
        except Exception:
            continue
    return by_id


class ArmStatusBuffer:
    def __init__(self):
        self.samples: list[tuple[float, dict[int, dict[str, float]]]] = []
        self.lock = threading.Lock()

    def callback(self, msg):
        by_id = normalize_status(msg)
        if by_id:
            with self.lock:
                self.samples.append((time.time(), by_id))
                if len(self.samples) > 200:
                    self.samples = self.samples[-200:]

    def latest_complete(self) -> tuple[float, dict[int, dict[str, float]]] | None:
        with self.lock:
            for sample in reversed(self.samples):
                if all(motor_id in sample[1] for motor_id in ARM_MOTOR_IDS):
                    return sample[0], copy.deepcopy(sample[1])
        return None

    def wait_complete(self, timeout: float):
        start = time.time()
        while time.time() - start < timeout:
            sample = self.latest_complete()
            if sample is not None:
                return sample
            time.sleep(0.05)
        return None

    def latest_ids(self) -> list[int]:
        with self.lock:
            if not self.samples:
                return []
            return sorted(self.samples[-1][1].keys())


class HandStatusBuffer:
    def __init__(self, side: str):
        self.side = side
        self.sample: tuple[float, tuple[float, ...]] | None = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def callback(self, msg: dict[str, Any]) -> None:
        try:
            positions = parse_hand_state_message(msg)
        except HandControlError as exc:
            with self.lock:
                self.sample = None
                self.error = str(exc)
            return
        with self.lock:
            self.sample = (time.time(), positions)
            self.error = None

    def latest_valid(self) -> tuple[float, tuple[float, ...]] | None:
        with self.lock:
            return self.sample

    def wait_valid(self, timeout: float) -> tuple[float, tuple[float, ...]] | None:
        start = time.time()
        while time.time() - start < timeout:
            sample = self.latest_valid()
            if sample is not None:
                return sample
            time.sleep(0.05)
        return None


def build_hand_test_target(baseline: tuple[float, ...], delta: float) -> tuple[float, ...]:
    return tuple(max(0.0, value - delta) if index < 5 else value for index, value in enumerate(baseline))


def hand_feedback_delta(
    baseline: tuple[float, ...],
    observed: tuple[float, ...],
) -> float:
    return max(abs(observed[index] - baseline[index]) for index in range(5))


def connect_rosbridge(host: str, port: int, timeout: float) -> roslibpy.Ros:
    client = roslibpy.Ros(host=host, port=port)
    client.run()
    start = time.time()
    while not client.is_connected:
        if time.time() - start > timeout:
            raise TimeoutError(f"连接 rosbridge 超时: ws://{host}:{port}")
        time.sleep(0.05)
    return client


def positions_from_status(status: dict[int, dict[str, float]]) -> dict[int, float]:
    return {motor_id: status[motor_id]["pos"] for motor_id in ARM_MOTOR_IDS}


def natural_down_positions() -> dict[int, float]:
    return {motor_id: float(NATURAL_DOWN_Q[index]) for index, motor_id in enumerate(ARM_MOTOR_IDS)}


def build_test_target(args: argparse.Namespace, baseline: dict[int, float]) -> dict[int, float]:
    target = dict(baseline)
    step = min(max(args.min_delta, 0.0), args.max_delta)
    for motor_id in TEST_PROFILES[args.side]["ids"]:
        direction = SAFE_TEST_DELTAS[motor_id]
        target[motor_id] = baseline[motor_id] + direction * step
    return target


def build_finish_target(args: argparse.Namespace, baseline: dict[int, float]) -> dict[int, float]:
    if args.finish_pose in ("initial", "baseline"):
        return dict(baseline)
    return natural_down_positions()


def build_cmd_msg(positions: dict[int, float], spd: float, cur: float) -> dict[str, Any]:
    motors = copy.deepcopy(ARM_MOTOR_TEMPLATES)
    for motor in motors:
        motor_id = int(motor["name"])
        motor["pos"] = float(positions[motor_id])
        motor["spd"] = float(spd)
        motor["cur"] = float(cur)
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
        "cmds": motors,
    }


def print_status_summary(label: str, positions: dict[int, float]) -> None:
    print(label)
    for motor_id in ARM_MOTOR_IDS:
        print(f"  id={motor_id} pos={positions[motor_id]:+.6f}")


def print_delta_summary(label: str, base: dict[int, float], target: dict[int, float]) -> None:
    print(label)
    for motor_id in ARM_MOTOR_IDS:
        delta = target[motor_id] - base[motor_id]
        if abs(delta) > 1e-9:
            print(f"  id={motor_id} base={base[motor_id]:+.6f} target={target[motor_id]:+.6f} delta={delta:+.6f}")


def confirm_execution(args: argparse.Namespace) -> None:
    if not args.execute:
        return
    if args.yes:
        return
    print("即将顺序发布手臂与双灵巧手控制命令，并在每阶段结束后自动恢复。")
    print(f"手臂: {ARM_CMD_TOPIC}; 灵巧手: {HAND_CMD_TOPICS['left']}, {HAND_CMD_TOPICS['right']}")
    print("灵巧手仅改变 1..5 号弯曲通道，6 号拇指旋转保持反馈基线。")
    print("请确认人员在机器人旁、急停可用、手臂和双手周围无干涉。")
    answer = input("输入 EXECUTE_ROBOT_TEST 继续: ").strip()
    if answer != "EXECUTE_ROBOT_TEST":
        raise SystemExit("用户未确认，取消发布")


def publish(topic: roslibpy.Topic, positions: dict[int, float], spd: float, cur: float, label: str) -> None:
    topic.publish(build_cmd_msg(positions, spd, cur))
    print(label)


def max_abs_delta(base: dict[int, float], observed: dict[int, float], ids: list[int]) -> float:
    if not ids:
        return 0.0
    return max(abs(observed[motor_id] - base[motor_id]) for motor_id in ids)


def any_error(status: dict[int, dict[str, float]]) -> list[int]:
    return [motor_id for motor_id in ARM_MOTOR_IDS if abs(status[motor_id].get("error", 0.0)) > 1e-9]


def main() -> int:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale 必须 > 0")
    if args.min_delta < 0 or args.max_delta <= 0 or args.min_delta > args.max_delta:
        raise ValueError("--min-delta/--max-delta 参数不合法")
    if not 0 < args.hand_delta <= 1:
        raise ValueError("--hand-delta 必须在 (0, 1] 内")

    print(f"ROBOT_LINK_TEST rosbridge=ws://{args.host}:{args.port} side={args.side} execute={args.execute}")
    print(f"PROFILE {TEST_PROFILES[args.side]['description']}")

    try:
        client = connect_rosbridge(args.host, args.port, args.connect_timeout)
    except Exception as exc:
        if args.execute:
            print(f"FAIL_CONNECT rosbridge=ws://{args.host}:{args.port} error={exc}")
            return 10
        print(f"WARN_CONNECT_DRY_RUN rosbridge=ws://{args.host}:{args.port} error={exc}; use offline preview baselines")
        baseline = natural_down_positions()
        target = build_test_target(args, baseline)
        finish_target = build_finish_target(args, baseline)
        print_status_summary("BASELINE_ARM_POSITIONS_OFFLINE", baseline)
        print_delta_summary("TARGET_PREVIEW", baseline, target)
        print_delta_summary(f"FINISH_PREVIEW_{args.finish_pose.upper()}", baseline, finish_target)
        if not args.arm_only:
            for side in ("left", "right"):
                hand_baseline = (1.0,) * 6
                print(f"HAND_TARGET_PREVIEW side={side} baseline={hand_baseline} target={build_hand_test_target(hand_baseline, args.hand_delta)}")
        print("DRY_RUN_ONLY no command published; rosbridge is not connected")
        return 0
    print("ROSBRIDGE_CONNECTED")

    status_buffer = ArmStatusBuffer()
    status_topic = roslibpy.Topic(client, ARM_STATUS_TOPIC, ARM_STATUS_TYPE)
    status_topic.subscribe(status_buffer.callback)
    hand_status_buffers: dict[str, HandStatusBuffer] = {}
    hand_status_topics: dict[str, roslibpy.Topic] = {}
    if not args.arm_only:
        for side in ("left", "right"):
            hand_buffer = HandStatusBuffer(side)
            hand_topic = roslibpy.Topic(client, HAND_STATE_TOPICS[side], HAND_JOINT_STATE_TYPE)
            hand_topic.subscribe(hand_buffer.callback)
            hand_status_buffers[side] = hand_buffer
            hand_status_topics[side] = hand_topic

    cmd_topic = None
    hand_cmd_topics: dict[str, roslibpy.Topic] = {}
    hand_baselines: dict[str, tuple[float, ...]] = {}
    try:
        sample = status_buffer.wait_complete(args.status_timeout)
        if sample is None:
            received_ids = status_buffer.latest_ids()
            missing_ids = [motor_id for motor_id in ARM_MOTOR_IDS if motor_id not in received_ids]
            message = (
                f"topic={ARM_STATUS_TOPIC} timeout={args.status_timeout}s "
                f"expected_ids={ARM_MOTOR_IDS} received_ids={received_ids} missing_ids={missing_ids}"
            )
            if args.execute:
                print(f"FAIL_STATUS_TIMEOUT {message}")
                return 11
            print(f"WARN_STATUS_TIMEOUT_DRY_RUN {message}; use natural_down as preview baseline")
            baseline = natural_down_positions()
        else:
            _, baseline_status = sample
            error_ids = any_error(baseline_status)
            if error_ids:
                print(f"WARN baseline motor error ids={error_ids}")
            baseline = positions_from_status(baseline_status)

        if not args.arm_only:
            for side, hand_buffer in hand_status_buffers.items():
                hand_sample = hand_buffer.wait_valid(args.status_timeout)
                if hand_sample is None:
                    message = (
                        f"topic={HAND_STATE_TOPICS[side]} timeout={args.status_timeout}s "
                        f"error={hand_buffer.error or 'missing'}"
                    )
                    if args.execute:
                        print(f"FAIL_HAND_STATUS_TIMEOUT {message}")
                        return 12
                    print(f"WARN_HAND_STATUS_TIMEOUT_DRY_RUN {message}; use open-hand preview baseline")
                    hand_baselines[side] = (1.0,) * 6
                else:
                    hand_baselines[side] = hand_sample[1]

        target = build_test_target(args, baseline)
        finish_target = build_finish_target(args, baseline)
        controlled_ids = TEST_PROFILES[args.side]["ids"]
        hand_targets = {
            side: build_hand_test_target(hand_baseline, args.hand_delta)
            for side, hand_baseline in hand_baselines.items()
        }

        print_status_summary("BASELINE_ARM_POSITIONS", baseline)
        print_delta_summary("TARGET_PREVIEW", baseline, target)
        print_delta_summary(f"FINISH_PREVIEW_{args.finish_pose.upper()}", baseline, finish_target)
        for side in hand_targets:
            print(f"HAND_TARGET_PREVIEW side={side} baseline={hand_baselines[side]} target={hand_targets[side]}")
        print(
            f"COMMAND_PARAMS spd={args.spd} cur={args.cur} hold={args.hold_seconds}s "
            f"finish_pose={args.finish_pose} observe={args.restore_seconds}s hand_delta={args.hand_delta}"
        )

        confirm_execution(args)
        if not args.execute:
            print("DRY_RUN_ONLY no command published; add --execute to publish after confirming safety")
            return 0

        cmd_topic = roslibpy.Topic(client, ARM_CMD_TOPIC, ARM_CMD_TYPE)
        cmd_topic.advertise()
        for side in hand_targets:
            hand_topic = roslibpy.Topic(client, HAND_CMD_TOPICS[side], HAND_JOINT_STATE_TYPE)
            hand_topic.advertise()
            hand_cmd_topics[side] = hand_topic
        time.sleep(0.2)

        publish(cmd_topic, target, args.spd, args.cur, "PUBLISHED_ARM_TEST_TARGET")
        time.sleep(args.hold_seconds)
        move_sample = status_buffer.latest_complete()
        if move_sample is None:
            raise TimeoutError("手臂动作后没有完整反馈")
        _, moved_status = move_sample
        moved = positions_from_status(moved_status)
        move_delta = max_abs_delta(baseline, moved, controlled_ids)
        other_delta = max_abs_delta(baseline, moved, [mid for mid in ARM_MOTOR_IDS if mid not in controlled_ids])
        print_delta_summary("OBSERVED_ARM_AFTER_MOVE", baseline, moved)
        print(f"ARM_MOVE_CONTROLLED_MAX_DELTA {move_delta:.6f}")
        print(f"ARM_MOVE_OTHER_MAX_DELTA {other_delta:.6f}")

        publish(cmd_topic, finish_target, args.spd, args.cur, f"PUBLISHED_ARM_FINISH_{args.finish_pose.upper()}")
        time.sleep(args.restore_seconds)
        finish_sample = status_buffer.latest_complete()
        if finish_sample is None:
            raise TimeoutError("手臂结束姿态发布后没有完整反馈")
        _, finished_status = finish_sample
        finished = positions_from_status(finished_status)
        finish_error = max_abs_delta(finish_target, finished, ARM_MOTOR_IDS)
        arm_pass = move_delta >= args.feedback_threshold and finish_error <= max(args.feedback_threshold, 0.08)
        print(f"ARM_FINISH_MAX_ERROR {finish_error:.6f}")
        print("PASS_ARM" if arm_pass else "FAIL_ARM")

        hand_pass = True
        if hand_cmd_topics:
            for side, topic in hand_cmd_topics.items():
                topic.publish(build_hand_joint_state_message(hand_targets[side]))
                print(f"PUBLISHED_HAND_TEST_TARGET side={side}")
            time.sleep(args.hold_seconds)
            hand_move_deltas: dict[str, float] = {}
            for side, hand_buffer in hand_status_buffers.items():
                observed = hand_buffer.latest_valid()
                if observed is None:
                    hand_move_deltas[side] = 0.0
                else:
                    hand_move_deltas[side] = hand_feedback_delta(hand_baselines[side], observed[1])
                print(f"HAND_MOVE_MAX_DELTA side={side} value={hand_move_deltas[side]:.6f}")
            for side, topic in hand_cmd_topics.items():
                topic.publish(build_hand_joint_state_message(hand_baselines[side]))
                print(f"PUBLISHED_HAND_RESTORE_BASELINE side={side}")
            time.sleep(args.restore_seconds)
            for side, hand_buffer in hand_status_buffers.items():
                restored = hand_buffer.latest_valid()
                restore_error = 1.0 if restored is None else max(
                    abs(value - base) for value, base in zip(restored[1], hand_baselines[side])
                )
                side_pass = (
                    hand_move_deltas[side] >= args.hand_feedback_threshold
                    and restore_error <= max(args.hand_feedback_threshold, 0.08)
                )
                hand_pass = hand_pass and side_pass
                print(f"HAND_RESTORE_MAX_ERROR side={side} value={restore_error:.6f}")
                print(f"{'PASS' if side_pass else 'FAIL'}_{side.upper()}_HAND")

        if arm_pass and hand_pass:
            print("PASS_ROBOT_LINK_TEST arm and inspire hands completed")
            return 0
        return 2 if not arm_pass else 3
    finally:
        if cmd_topic is not None:
            try:
                cmd_topic.unadvertise()
            except Exception:
                pass
        for side, hand_topic in hand_cmd_topics.items():
            try:
                if side in hand_baselines:
                    hand_topic.publish(build_hand_joint_state_message(hand_baselines[side]))
                hand_topic.unadvertise()
            except Exception:
                pass
        try:
            status_topic.unsubscribe()
        except Exception:
            pass
        for hand_status_topic in hand_status_topics.values():
            try:
                hand_status_topic.unsubscribe()
            except Exception:
                pass
        client.terminate()


if __name__ == "__main__":
    raise SystemExit(main())