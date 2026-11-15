#!/usr/bin/env python3
"""Render direct-control output and four-stage retarget diagnostics in RViz.

Layer 0 is the raw FSS landmark skeleton in unmodified FSS world coordinates.
Layer A is the calibrated palm-mapped skeleton used directly by the
position-palm solvers. Layer B repeats the exact green task points; they must
overlap the corresponding Layer A landmarks. Layer C is final post-safety FK.
Keeping these layers visible separates source mapping from IK/manifold and
safety-filter residuals.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 16124
DEBUG_LISTEN_IP = "127.0.0.1"
DEBUG_LISTEN_PORT = 16125
PUBLISH_HZ = 100.0
STALE_WARN_S = 1.0
LEGACY_CONTROL_PROTOCOL = "revo2_finger_data_set_v1"
SDK_RAD_CONTROL_PROTOCOL = "revo2_dual_hand_ik_sdk_v2"
EXPECTED_DEBUG_PROTOCOL = "revo2_ik_debug_overlay_v1"

# The URDF distal joints are mimic joints; only the 12 active parent joints are
# published and robot_state_publisher computes the remaining visual joints from
# URDF mimic tags. New Python SDK test packets carry radians directly. Legacy
# native/Python bridge packets may still carry finger_data_set[0..100]; those are
# converted to radians only inside this RViz helper.
CONTROL_CHANNELS = (
    ("left_thumb_proximal_joint", 0),
    ("left_thumb_metacarpal_joint", 1),
    ("left_index_proximal_joint", 2),
    ("left_middle_proximal_joint", 3),
    ("left_ring_proximal_joint", 4),
    ("left_pinky_proximal_joint", 5),
    ("right_thumb_proximal_joint", 6),
    ("right_thumb_metacarpal_joint", 7),
    ("right_index_proximal_joint", 8),
    ("right_middle_proximal_joint", 9),
    ("right_ring_proximal_joint", 10),
    ("right_pinky_proximal_joint", 11),
)

# Publish mimic joints explicitly as well. This avoids depending on the mimic
# implementation details of the robot_state_publisher version installed with
# ROS2 Humble and guarantees that every movable visual joint receives a value.
VISUAL_JOINT_SOURCES = (
    ("left_thumb_metacarpal_joint", 1, 1.0),
    ("left_thumb_proximal_joint", 0, 1.0),
    ("left_thumb_distal_joint", 0, 1.0),
    ("left_index_proximal_joint", 2, 1.0),
    ("left_index_distal_joint", 2, 1.155),
    ("left_middle_proximal_joint", 3, 1.0),
    ("left_middle_distal_joint", 3, 1.155),
    ("left_ring_proximal_joint", 4, 1.0),
    ("left_ring_distal_joint", 4, 1.155),
    ("left_pinky_proximal_joint", 5, 1.0),
    ("left_pinky_distal_joint", 5, 1.155),
    ("right_thumb_metacarpal_joint", 7, 1.0),
    ("right_thumb_proximal_joint", 6, 1.0),
    ("right_thumb_distal_joint", 6, 1.0),
    ("right_index_proximal_joint", 8, 1.0),
    ("right_index_distal_joint", 8, 1.155),
    ("right_middle_proximal_joint", 9, 1.0),
    ("right_middle_distal_joint", 9, 1.155),
    ("right_ring_proximal_joint", 10, 1.0),
    ("right_ring_distal_joint", 10, 1.155),
    ("right_pinky_proximal_joint", 11, 1.0),
    ("right_pinky_distal_joint", 11, 1.155),
)

RAW_POINT_ORDER = (
    "hand",
    "thumb_near", "thumb_middle", "thumb_far",
    "index_near", "index_middle", "index_far",
    "middle_near", "middle_middle", "middle_far",
    "ring_near", "ring_middle", "ring_far",
    "little_near", "little_middle", "little_far",
)

POINT_ORDER = (
    "thumb_near", "thumb_middle", "thumb_far",
    "index_near", "index_middle", "index_far",
    "middle_near", "middle_middle", "middle_far",
    "ring_near", "ring_middle", "ring_far",
    "little_near", "little_middle", "little_far",
)
SKELETON_SEGMENTS = (
    ("thumb_near", "thumb_middle"), ("thumb_middle", "thumb_far"),
    ("index_near", "index_middle"), ("index_middle", "index_far"),
    ("middle_near", "middle_middle"), ("middle_middle", "middle_far"),
    ("ring_near", "ring_middle"), ("ring_middle", "ring_far"),
    ("little_near", "little_middle"), ("little_middle", "little_far"),
)


@dataclass(frozen=True)
class OverlayTask:
    name: str
    target_middle: Point | None
    target_far: Point | None
    robot_middle: Point | None
    robot_far: Point | None


@dataclass(frozen=True)
class OverlayHand:
    side: str
    raw_frame_id: str
    raw_points: dict[str, Point]
    frame_id: str
    calibrated: bool
    points: dict[str, Point]
    tasks: tuple[OverlayTask, ...]


def clamp_0_100(value: object) -> float:
    return max(0.0, min(100.0, float(value)))


def _point_from_json(raw: object) -> Point | None:
    if not isinstance(raw, Mapping) or not bool(raw.get("valid", False)):
        return None
    try:
        return Point(x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


class ControlUdpToRviz(Node):
    def __init__(self, listen_ip: str, listen_port: int, debug_listen_ip: str, debug_listen_port: int, joint_topic: str) -> None:
        super().__init__("revo2_control_udp_to_rviz")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._joint_topic = joint_topic
        self._publisher = self.create_publisher(JointState, joint_topic, qos)
        self._overlay_publisher = self.create_publisher(MarkerArray, "/revo2/local_debug/mocap_overlay", qos)
        self._socket = self._make_socket(listen_ip, listen_port)
        self._debug_socket = self._make_socket(debug_listen_ip, debug_listen_port)
        self._joint_positions_rad = [0.0] * 12
        self._previous_rx_positions = [0.0] * 12
        self._last_packet_joint_delta = 0.0
        self._moving_packets = 0
        self._last_rx = 0.0
        self._last_debug_rx = 0.0
        self._received = 0
        self._debug_received = 0
        self._invalid = 0
        self._debug_invalid = 0
        self._last_health_log = 0.0
        self._overlay_hands: dict[str, OverlayHand] = {}
        self.create_timer(0.002, self._drain_udp)
        self.create_timer(0.002, self._drain_debug_udp)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_overlay)
        self.create_timer(1.0, self._health)
        self.get_logger().info(
            "control UDP RViz mirror ready: "
            f"control_udp={listen_ip}:{listen_port} debug_overlay_udp={debug_listen_ip}:{debug_listen_port} "
            f"topics={joint_topic},/revo2/local_debug/mocap_overlay "
            f"rate={PUBLISH_HZ:.1f}Hz"
        )

    @staticmethod
    def _make_socket(ip: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ip, port))
        sock.setblocking(False)
        return sock

    def destroy_node(self) -> bool:
        try:
            self._socket.close()
            self._debug_socket.close()
        finally:
            return super().destroy_node()

    def _drain_udp(self) -> None:
        # Keep only the newest packet. RViz should represent the latest actual
        # control output, not a delayed queue of historical hand states.
        while True:
            try:
                raw, _source = self._socket.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as exc:
                self.get_logger().error(f"UDP receive error: {exc}")
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
                protocol = payload.get("protocol")
                if protocol == SDK_RAD_CONTROL_PROTOCOL:
                    values = payload.get("joint_positions")
                    if not isinstance(values, list) or len(values) != 12:
                        raise ValueError("joint_positions must contain exactly 12 radian values")
                    if payload.get("joint_unit") != "rad":
                        raise ValueError("joint_unit must be rad for SDK output")
                    self._joint_positions_rad = [float(value) for value in values]
                elif protocol == LEGACY_CONTROL_PROTOCOL:
                    values = payload.get("finger_data_set")
                    if not isinstance(values, list) or len(values) != 12:
                        raise ValueError("finger_data_set must contain exactly 12 values")
                    legacy_upper = (1.03, 1.57, 1.41, 1.41, 1.41, 1.41,
                                    1.03, 1.57, 1.41, 1.41, 1.41, 1.41)
                    self._joint_positions_rad = [
                        clamp_0_100(value) * upper / 100.0
                        for value, upper in zip(values, legacy_upper)
                    ]
                else:
                    raise ValueError(f"unexpected protocol: {protocol!r}")
                if not all(float("-inf") < value < float("inf") for value in self._joint_positions_rad):
                    raise ValueError("joint_positions contains NaN or Inf")
                self._last_packet_joint_delta = max(
                    abs(current - previous)
                    for current, previous in zip(self._joint_positions_rad, self._previous_rx_positions)
                )
                if self._last_packet_joint_delta > 1e-6:
                    self._moving_packets += 1
                self._previous_rx_positions = list(self._joint_positions_rad)
                self._last_rx = time.monotonic()
                self._received += 1
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                self._invalid += 1
                if self._invalid <= 5 or self._invalid % 100 == 0:
                    self.get_logger().warn(f"ignore invalid control mirror UDP packet: {exc}")

    def _parse_overlay_hand(self, raw: object) -> OverlayHand | None:
        if not isinstance(raw, Mapping):
            return None
        side = str(raw.get("side", ""))
        if side not in ("left", "right"):
            return None

        raw_frame_id = str(raw.get("raw_frame_id", "world"))
        raw_points: dict[str, Point] = {}
        raw_raw_points = raw.get("raw_points", [])
        if isinstance(raw_raw_points, list):
            for item in raw_raw_points:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name", ""))
                point = _point_from_json(item.get("point"))
                if name in RAW_POINT_ORDER and point is not None:
                    raw_points[name] = point

        frame_id = str(raw.get("frame_id", ""))
        calibrated = bool(raw.get("calibrated", False))
        points: dict[str, Point] = {}
        raw_points_mapped = raw.get("points", [])
        if calibrated and frame_id and isinstance(raw_points_mapped, list):
            for item in raw_points_mapped:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name", ""))
                point = _point_from_json(item.get("point"))
                if name in POINT_ORDER and point is not None:
                    points[name] = point

        # Raw input is valid before the open-pose calibration completes. Keep
        # it visible in that phase even though mapped/task/FK layers are absent.
        if not raw_points and not points:
            return None

        raw_thumb_task = raw.get("thumb_task", {})
        raw_thumb_task = raw_thumb_task if isinstance(raw_thumb_task, Mapping) else {}
        tasks: list[OverlayTask] = [OverlayTask(
            name="thumb",
            target_middle=_point_from_json(raw_thumb_task.get("target_middle")),
            target_far=_point_from_json(raw_thumb_task.get("target_far")),
            robot_middle=_point_from_json(raw_thumb_task.get("robot_middle")),
            robot_far=_point_from_json(raw_thumb_task.get("robot_far")),
        )]
        raw_long_tasks = raw.get("long_finger_tasks", [])
        if isinstance(raw_long_tasks, list):
            for raw_task in raw_long_tasks:
                if not isinstance(raw_task, Mapping) or not bool(raw_task.get("valid", False)):
                    continue
                name = str(raw_task.get("name", ""))
                if name not in {"index", "middle", "ring", "little"}:
                    continue
                tasks.append(OverlayTask(
                    name=name,
                    target_middle=_point_from_json(raw_task.get("target_middle")),
                    target_far=_point_from_json(raw_task.get("target_far")),
                    robot_middle=_point_from_json(raw_task.get("robot_middle")),
                    robot_far=_point_from_json(raw_task.get("robot_far")),
                ))
        return OverlayHand(
            side=side,
            raw_frame_id=raw_frame_id,
            raw_points=raw_points,
            frame_id=frame_id,
            calibrated=calibrated,
            points=points,
            tasks=tuple(tasks),
        )

    def _drain_debug_udp(self) -> None:
        while True:
            try:
                raw, _source = self._debug_socket.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as exc:
                self.get_logger().error(f"debug overlay UDP receive error: {exc}")
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("protocol") != EXPECTED_DEBUG_PROTOCOL:
                    raise ValueError(f"unexpected protocol: {payload.get('protocol')!r}")
                hands = payload.get("hands")
                if not isinstance(hands, list):
                    raise ValueError("hands must be a list")
                latest: dict[str, OverlayHand] = {}
                for raw_hand in hands:
                    hand = self._parse_overlay_hand(raw_hand)
                    if hand is not None:
                        latest[hand.side] = hand
                self._overlay_hands = latest
                self._last_debug_rx = time.monotonic()
                self._debug_received += 1
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                self._debug_invalid += 1
                if self._debug_invalid <= 5 or self._debug_invalid % 100 == 0:
                    self.get_logger().warn(f"ignore invalid debug overlay UDP packet: {exc}")

    def _publish(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint for joint, _source_index, _multiplier in VISUAL_JOINT_SOURCES]
        msg.position = [
            self._joint_positions_rad[source_index] * multiplier
            for _joint, source_index, multiplier in VISUAL_JOINT_SOURCES
        ]
        self._publisher.publish(msg)

    def _base_marker(self, frame_id: str, namespace: str, marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=0.25).to_msg()
        return marker

    def _publish_overlay(self) -> None:
        markers = MarkerArray()
        for side, hand in self._overlay_hands.items():
            is_left = side == "left"
            raw_color = _color(1.0, 0.25, 0.10, 0.88) if is_left else _color(0.35, 0.45, 1.0, 0.88)
            source_color = _color(0.10, 0.95, 0.95, 0.78) if is_left else _color(0.95, 0.20, 0.90, 0.78)
            target_color = _color(0.20, 1.0, 0.25, 1.0)
            robot_color = _color(1.0, 0.85, 0.15, 1.0)
            base_id = 0 if is_left else 100

            # Layer 0: exact raw FSS positions in the original FSS/world frame.
            # No scale, palm frame, coordinate retargeting, or zero-pose offset.
            if hand.raw_points:
                raw_lines = self._base_marker(hand.raw_frame_id, f"revo2_fss_raw_{side}", base_id + 50, Marker.LINE_LIST)
                raw_lines.scale.x = 0.0025
                raw_lines.color = raw_color
                for first, second in SKELETON_SEGMENTS:
                    if first in hand.raw_points and second in hand.raw_points:
                        raw_lines.points.extend((hand.raw_points[first], hand.raw_points[second]))
                raw_points = self._base_marker(hand.raw_frame_id, f"revo2_fss_raw_{side}", base_id + 51, Marker.SPHERE_LIST)
                raw_points.scale.x = raw_points.scale.y = raw_points.scale.z = 0.0085
                raw_points.color = raw_color
                raw_points.points = [hand.raw_points[name] for name in RAW_POINT_ORDER if name in hand.raw_points]
                markers.markers.extend((raw_lines, raw_points))

            if not hand.calibrated or not hand.frame_id or not hand.points:
                continue

            # Layer A: uniform-scale palm-mapped FSS geometry. This is a
            # visualization source layer, not necessarily the exact task point
            # because task points use per-landmark calibrated scales.
            lines = self._base_marker(hand.frame_id, f"revo2_mocap_source_{side}", base_id, Marker.LINE_LIST)
            lines.scale.x = 0.0025
            lines.color = source_color
            for first, second in SKELETON_SEGMENTS:
                if first in hand.points and second in hand.points:
                    lines.points.extend((hand.points[first], hand.points[second]))
            markers.markers.append(lines)

            points = self._base_marker(hand.frame_id, f"revo2_mocap_source_{side}", base_id + 1, Marker.SPHERE_LIST)
            points.scale.x = points.scale.y = points.scale.z = 0.0075
            points.color = source_color
            points.points = [hand.points[name] for name in POINT_ORDER if name in hand.points]
            markers.markers.append(points)

            # Layer B: exact current objective points. Green target segments
            # expose the positions actually passed to thumb DLS / long-finger
            # one-dimensional curve projection.
            target_lines = self._base_marker(hand.frame_id, f"revo2_ik_targets_{side}", base_id + 2, Marker.LINE_LIST)
            target_lines.scale.x = 0.0035
            target_lines.color = target_color
            target_points = self._base_marker(hand.frame_id, f"revo2_ik_targets_{side}", base_id + 3, Marker.SPHERE_LIST)
            target_points.scale.x = target_points.scale.y = target_points.scale.z = 0.010
            target_points.color = target_color

            # Layer C: actual final FK after SafetyFilter. Yellow points are
            # directly comparable with the green target points.
            robot_points = self._base_marker(hand.frame_id, f"revo2_ik_fk_{side}", base_id + 4, Marker.SPHERE_LIST)
            robot_points.scale.x = robot_points.scale.y = robot_points.scale.z = 0.011
            robot_points.color = robot_color

            for task in hand.tasks:
                if task.target_middle is not None:
                    target_points.points.append(task.target_middle)
                if task.target_far is not None:
                    target_points.points.append(task.target_far)
                if task.target_middle is not None and task.target_far is not None:
                    target_lines.points.extend((task.target_middle, task.target_far))
                if task.robot_middle is not None:
                    robot_points.points.append(task.robot_middle)
                if task.robot_far is not None:
                    robot_points.points.append(task.robot_far)

            markers.markers.extend((target_lines, target_points, robot_points))

        if markers.markers:
            self._overlay_publisher.publish(markers)

    def _health(self) -> None:
        now = time.monotonic()
        age = float("inf") if self._last_rx <= 0.0 else now - self._last_rx
        debug_age = float("inf") if self._last_debug_rx <= 0.0 else now - self._last_debug_rx
        if age > STALE_WARN_S:
            self.get_logger().warn(
                f"waiting for/stale direct-control mirror UDP: age={age:.3f}s "
                f"received={self._received} invalid={self._invalid}"
            )
        if debug_age > STALE_WARN_S:
            self.get_logger().warn(
                f"waiting for/stale raw/mapped-input overlay UDP: age={debug_age:.3f}s "
                f"received={self._debug_received} invalid={self._debug_invalid}"
            )
        elif now - self._last_health_log >= 5.0:
            self._last_health_log = now
            self.get_logger().info(
                "control/debug mirror healthy: "
                f"control_received={self._received} control_age={age:.3f}s "
                f"moving_packets={self._moving_packets} last_joint_delta={self._last_packet_joint_delta:.6f}rad "
                f"joint_abs_max={max(abs(v) for v in self._joint_positions_rad):.6f}rad "
                f"joint_subscribers={self._publisher.get_subscription_count()} "
                f"overlay_received={self._debug_received} overlay_age={debug_age:.3f}s"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render direct control plus raw and mapped FSS landmarks in RViz")
    parser.add_argument("--listen-ip", default=LISTEN_IP, help=f"control UDP bind IP (default: {LISTEN_IP})")
    parser.add_argument("--listen-port", type=int, default=LISTEN_PORT, help=f"control UDP port (default: {LISTEN_PORT})")
    parser.add_argument("--debug-listen-ip", default=DEBUG_LISTEN_IP, help=f"overlay UDP bind IP (default: {DEBUG_LISTEN_IP})")
    parser.add_argument("--debug-listen-port", type=int, default=DEBUG_LISTEN_PORT, help=f"overlay UDP port (default: {DEBUG_LISTEN_PORT})")
    parser.add_argument("--joint-topic", default="/joint_states", help="JointState output topic (default: /joint_states)")
    args = parser.parse_args()
    for field in ("listen_port", "debug_listen_port"):
        value = int(getattr(args, field))
        if not 1 <= value <= 65535:
            parser.error(f"--{field.replace('_', '-')} must be in 1..65535")
    return args


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = ControlUdpToRviz(args.listen_ip, args.listen_port, args.debug_listen_ip, args.debug_listen_port, args.joint_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        # ROS2 Humble may invalidate the context before spin() returns when the
        # launcher forwards SIGINT/SIGTERM.  Suppress only that shutdown path.
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
