from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

QUAT_ORDER = "xyzw"
DEVICE_MODE_GLOVE = "glove"
DEVICE_MODE_CONTROLLER = "controller"
DEVICE_MODE_AUTO = "auto"
DEVICE_MODES = (DEVICE_MODE_GLOVE, DEVICE_MODE_CONTROLLER, DEVICE_MODE_AUTO)
GLOVE_SYNC_HEADERS = (bytes((0xAA, 0x55)), bytes((0x55, 0xAA)))
OBSERVED_CONTROLLER_SYNC_HEADERS = (bytes((0xC8, 0xFF)),)
MOCAP_G2_FRAME_ID = 0x0000
MOCAP_G2_JOINT_COUNT = 64
MOCAP_G2_FLOATS_PER_JOINT = 7
MOCAP_G2_PACKET_TOTAL_BYTES = 1860
MOCAP_G2_LIVE_LENGTH_FIELD = 1850
MOCAP_G2_JOINT_POSE_BYTES = MOCAP_G2_JOINT_COUNT * MOCAP_G2_FLOATS_PER_JOINT * 4
MOCAP_G2_MIN_PACKET_BYTES = 6 + 8 + MOCAP_G2_JOINT_POSE_BYTES
EXPECTED_UDP_PACKET_SIZE = MOCAP_G2_PACKET_TOTAL_BYTES

LEGACY_FRAME_ID = 0x0001
LEGACY_JOINT_COUNT = 47
LEGACY_PACKET_SIZES = (1366, 1370)
LEGACY_PAYLOAD_LENGTHS = (1356, 1360)
LEGACY_ROTATION_OFFSET = 8
LEGACY_POSITION_OFFSET = 208
LEGACY_ROTATION_STRIDE = 4
LEGACY_POSITION_STRIDE = 3

MOCAP_JOINT_NAMES: tuple[str, ...] = (
    "head", "neck", "chest", "spine1", "spine2", "waist",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "left_thumb_near", "left_thumb_mid", "left_thumb_far", "left_thumb_end",
    "left_index_near", "left_index_mid", "left_index_far", "left_index_end",
    "left_middle_near", "left_middle_mid", "left_middle_far", "left_middle_end",
    "left_ring_near", "left_ring_mid", "left_ring_far", "left_ring_end",
    "left_pinky_near", "left_pinky_mid", "left_pinky_far", "left_pinky_end",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "right_thumb_near", "right_thumb_mid", "right_thumb_far", "right_thumb_end",
    "right_index_near", "right_index_mid", "right_index_far", "right_index_end",
    "right_middle_near", "right_middle_mid", "right_middle_far", "right_middle_end",
    "right_ring_near", "right_ring_mid", "right_ring_far", "right_ring_end",
    "right_pinky_near", "right_pinky_mid", "right_pinky_far", "right_pinky_end",
)
MOCAP_JOINT_INDEX = MappingProxyType({name: index for index, name in enumerate(MOCAP_JOINT_NAMES)})

if len(MOCAP_JOINT_NAMES) != MOCAP_G2_JOINT_COUNT:
    raise AssertionError("Formotion G2 关节表必须包含 64 个节点")


class MocapPacketError(ValueError):
    pass


class UnsupportedDevicePacketError(MocapPacketError):
    """数据报属于已识别设备模式，但该模式的帧布局尚未验证。"""


@dataclass(frozen=True)
class JointPose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    tracked: bool = True

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.quaternion_xyzw) != 4:
            raise ValueError("JointPose 维度必须为 position[3] + quaternion_xyzw[4]")
        if not all(math.isfinite(value) for value in self.position + self.quaternion_xyzw):
            raise ValueError("JointPose 包含 NaN/Inf")
        if self.tracked:
            norm = math.sqrt(sum(value * value for value in self.quaternion_xyzw))
            if norm <= 1e-8:
                raise ValueError("已跟踪节点的四元数长度为 0")
            object.__setattr__(
                self,
                "quaternion_xyzw",
                tuple(float(value / norm) for value in self.quaternion_xyzw),
            )


@dataclass(frozen=True)
class MocapFrame:
    timestamp_s: float
    timestamp_us: int
    receive_time_s: float
    joints: tuple[JointPose, ...]
    source: str
    wire: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.joints) != MOCAP_G2_JOINT_COUNT:
            raise ValueError(f"MocapFrame 必须包含 {MOCAP_G2_JOINT_COUNT} 个节点")
        if not math.isfinite(self.timestamp_s) or not math.isfinite(self.receive_time_s):
            raise ValueError("MocapFrame 时间戳无效")
        object.__setattr__(self, "wire", MappingProxyType(dict(self.wire)))

    def joint(self, name: str) -> JointPose:
        try:
            return self.joints[MOCAP_JOINT_INDEX[name]]
        except KeyError as exc:
            raise KeyError(f"未知动捕节点: {name}") from exc

    def to_replay_dict(self, frame_index: int | None = None) -> dict[str, Any]:
        left_hand = self.joint("left_hand")
        right_hand = self.joint("right_hand")
        left_wrist = self.joint("left_wrist")
        right_wrist = self.joint("right_wrist")
        result: dict[str, Any] = {
            "receive_time": float(self.timestamp_s),
            "packet_timestamp_ms": int(self.timestamp_us // 1000),
            "packet_timestamp_us": int(self.timestamp_us),
            "packet_size": self.wire.get("packet_size"),
            # 旧字段保留给已有回放与手部链路；其物理含义始终是 hand 节点。
            "left_pos": list(left_hand.position),
            "left_quat": list(left_hand.quaternion_xyzw),
            "right_pos": list(right_hand.position),
            "right_quat": list(right_hand.quaternion_xyzw),
            "left_hand_pos": list(left_hand.position),
            "left_hand_quat": list(left_hand.quaternion_xyzw),
            "right_hand_pos": list(right_hand.position),
            "right_hand_quat": list(right_hand.quaternion_xyzw),
            "left_wrist_pos": list(left_wrist.position),
            "left_wrist_quat": list(left_wrist.quaternion_xyzw),
            "right_wrist_pos": list(right_wrist.position),
            "right_wrist_quat": list(right_wrist.quaternion_xyzw),
            "hand_wrist_tracking": {
                "left": {"hand": left_hand.tracked, "wrist": left_wrist.tracked},
                "right": {"hand": right_hand.tracked, "wrist": right_wrist.tracked},
            },
            "skeleton": upper_body_skeleton(self),
            "mocap_frame": self,
        }
        if frame_index is not None:
            result["frame_index"] = int(frame_index)
        return result


def snapshot_mocap_data(data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """复制控制帧外层容器；内部 MocapFrame 和骨架数据按只读值共享。"""
    return None if data is None else dict(data)


@dataclass(frozen=True)
class _WireLayout:
    header_byte_order: str
    pose_byte_order: str
    timestamp_offset: int
    pose_offset: int
    sync_reversed: bool
    length_field: int


def normalize_quaternion_xyzw(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise MocapPacketError("四元数必须包含 4 个值")
    quat = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in quat):
        raise MocapPacketError("四元数包含 NaN/Inf")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 1e-8:
        raise MocapPacketError("四元数长度为 0")
    return tuple(value / norm for value in quat)


def _g2_length_score(length: int, packet_size: int) -> int:
    if packet_size == MOCAP_G2_PACKET_TOTAL_BYTES and length == MOCAP_G2_LIVE_LENGTH_FIELD:
        return 8
    if length == packet_size:
        return 6
    if length + 6 == packet_size:
        return 5
    if length in (MOCAP_G2_PACKET_TOTAL_BYTES, MOCAP_G2_PACKET_TOTAL_BYTES - 6) and packet_size >= MOCAP_G2_PACKET_TOTAL_BYTES:
        return 4
    if length in (MOCAP_G2_MIN_PACKET_BYTES, MOCAP_G2_MIN_PACKET_BYTES - 6) and packet_size >= MOCAP_G2_MIN_PACKET_BYTES:
        return 2
    return -1


def _header_candidates(data: bytes) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for byte_order, prefix in (("little", "<"), ("big", ">")):
        packet_id, length = struct.unpack_from(prefix + "HH", data, 2)
        if packet_id == MOCAP_G2_FRAME_ID and _g2_length_score(length, len(data)) >= 0:
            candidates.append((length, byte_order))
    return candidates


def _pose_block_score(data: bytes, pose_offset: int, byte_order: str) -> float:
    if pose_offset < 0 or pose_offset + MOCAP_G2_JOINT_POSE_BYTES > len(data):
        return float("-inf")
    prefix = "<" if byte_order == "little" else ">"
    score = 0.0
    normalized_quaternion_count = 0
    for index in (0, 5, 7, 10, 20, 36, 39, 49, 60, 63):
        values = struct.unpack_from(prefix + "7f", data, pose_offset + index * 28)
        if not all(math.isfinite(value) for value in values):
            return float("-inf")
        max_position = max(abs(value) for value in values[:3])
        score += 1.0 if max_position < 100.0 else -10.0
        norm = math.sqrt(sum(value * value for value in values[3:]))
        if norm <= 1e-8:
            score += 0.05
        elif 0.5 <= norm <= 1.5:
            normalized_quaternion_count += 1
            score += 4.0 - min(3.0, abs(norm - 1.0) * 8.0)
        else:
            score -= min(10.0, abs(norm - 1.0))
    if normalized_quaternion_count < 6:
        return float("-inf")
    return score


def _select_g2_layout(data: bytes) -> _WireLayout:
    headers = _header_candidates(data)
    if not headers:
        little_id, little_length = struct.unpack_from("<HH", data, 2)
        big_id, big_length = struct.unpack_from(">HH", data, 2)
        raise MocapPacketError(
            "不支持的 G2 包头: "
            f"size={len(data)} little(id=0x{little_id:04X},length={little_length}) "
            f"big(id=0x{big_id:04X},length={big_length})"
        )
    length, header_order = max(
        headers,
        key=lambda item: (_g2_length_score(item[0], len(data)), item[1] == "little"),
    )
    candidates: list[tuple[float, int, int, str]] = []
    for timestamp_offset, pose_offset in ((6, 14), (8, 16)):
        for pose_order in ("little", "big"):
            candidates.append((_pose_block_score(data, pose_offset, pose_order), timestamp_offset, pose_offset, pose_order))
    score, timestamp_offset, pose_offset, pose_order = max(candidates, key=lambda item: item[0])
    if not math.isfinite(score) or score < -1.0:
        raise MocapPacketError("无法识别 G2 timestamp/pose 对齐或浮点字节序")
    return _WireLayout(
        header_byte_order=header_order,
        pose_byte_order=pose_order,
        timestamp_offset=timestamp_offset,
        pose_offset=pose_offset,
        sync_reversed=data[:2] == bytes((0x55, 0xAA)),
        length_field=int(length),
    )


def _decode_g2_packet(data: bytes, receive_time_s: float) -> MocapFrame:
    if len(data) < MOCAP_G2_MIN_PACKET_BYTES:
        raise MocapPacketError(f"G2 UDP 包过短: got={len(data)} minimum={MOCAP_G2_MIN_PACKET_BYTES}")
    layout = _select_g2_layout(data)
    prefix = "<" if layout.pose_byte_order == "little" else ">"
    timestamp_us = int(struct.unpack_from(prefix + "Q", data, layout.timestamp_offset)[0])
    joints: list[JointPose] = []
    for index in range(MOCAP_G2_JOINT_COUNT):
        x, y, z, qw, qx, qy, qz = struct.unpack_from(prefix + "7f", data, layout.pose_offset + index * 28)
        values = (x, y, z, qw, qx, qy, qz)
        if not all(math.isfinite(value) for value in values):
            raise MocapPacketError(f"G2 节点 {index} 包含 NaN/Inf")
        quat_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        tracked = quat_norm > 1e-8
        joints.append(
            JointPose(
                position=(float(x), float(y), float(z)),
                quaternion_xyzw=(float(qx), float(qy), float(qz), float(qw)) if tracked else (0.0, 0.0, 0.0, 1.0),
                tracked=tracked,
            )
        )
    timestamp_s = timestamp_us / 1_000_000.0 if timestamp_us > 0 else float(receive_time_s)
    return MocapFrame(
        timestamp_s=timestamp_s,
        timestamp_us=timestamp_us,
        receive_time_s=float(receive_time_s),
        joints=tuple(joints),
        source="formotion_g2_udp",
        wire={
            "layout": "formotion_g2_mocap_v1",
            "packet_size": len(data),
            "length_field": layout.length_field,
            "header_byte_order": layout.header_byte_order,
            "pose_byte_order": layout.pose_byte_order,
            "timestamp_offset": layout.timestamp_offset,
            "pose_offset": layout.pose_offset,
            "sync_reversed": layout.sync_reversed,
        },
    )


def _decode_legacy_packet(data: bytes, receive_time_s: float) -> MocapFrame:
    if len(data) not in LEGACY_PACKET_SIZES:
        raise MocapPacketError(f"旧 FSS UDP 包长度不匹配: got={len(data)} expected={LEGACY_PACKET_SIZES}")
    header1, header2, packet_id, payload_length = struct.unpack_from("<BBHH", data, 0)
    if (header1, header2) != (0xAA, 0x55) or packet_id != LEGACY_FRAME_ID:
        raise MocapPacketError("旧 FSS 包头或 ID 无效")
    if payload_length not in LEGACY_PAYLOAD_LENGTHS or len(data) != 6 + payload_length + 4:
        raise MocapPacketError("旧 FSS payload 长度无效")
    unpacked = struct.unpack("<BBHHI 3f 188f 6H 6H 141f I" if payload_length == 1356 else "<BBHHI 3f 188f 6H 6H 141f I I", data)
    timestamp_ms = int(unpacked[4])
    joints: list[JointPose] = []
    for index in range(MOCAP_G2_JOINT_COUNT):
        if index < LEGACY_JOINT_COUNT:
            quat_offset = LEGACY_ROTATION_OFFSET + index * LEGACY_ROTATION_STRIDE
            pos_offset = LEGACY_POSITION_OFFSET + index * LEGACY_POSITION_STRIDE
            qx, qy, qz, qw = (float(value) for value in unpacked[quat_offset:quat_offset + 4])
            position = tuple(float(value) for value in unpacked[pos_offset:pos_offset + 3])
            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            tracked = norm > 1e-8
            quat = (qx, qy, qz, qw) if tracked else (0.0, 0.0, 0.0, 1.0)
        else:
            position = (0.0, 0.0, 0.0)
            quat = (0.0, 0.0, 0.0, 1.0)
            tracked = False
        joints.append(JointPose(position=position, quaternion_xyzw=quat, tracked=tracked))
    timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 0 else float(receive_time_s)
    return MocapFrame(
        timestamp_s=timestamp_s,
        timestamp_us=timestamp_ms * 1000,
        receive_time_s=float(receive_time_s),
        joints=tuple(joints),
        source="legacy_fss_udp",
        wire={"layout": "legacy_fss_motion_v1", "packet_size": len(data), "length_field": payload_length},
    )


def _packet_mode(data: bytes) -> str | None:
    sync = data[:2]
    if sync in GLOVE_SYNC_HEADERS:
        return DEVICE_MODE_GLOVE
    if sync in OBSERVED_CONTROLLER_SYNC_HEADERS:
        return DEVICE_MODE_CONTROLLER
    return None


def parse_mocap_packet(
    data: bytes,
    receive_time: float,
    *,
    device_mode: str = DEVICE_MODE_GLOVE,
) -> MocapFrame:
    if device_mode not in DEVICE_MODES:
        raise ValueError(f"不支持的设备模式: {device_mode!r}; expected={DEVICE_MODES}")
    if len(data) < 6:
        raise MocapPacketError(f"UDP 包过短: {len(data)}")

    packet_mode = _packet_mode(data)
    if packet_mode is None:
        raise MocapPacketError(
            f"未知同步头: {data[0]:02X} {data[1]:02X}; "
            f"selected_mode={device_mode} size={len(data)}"
        )
    if device_mode != DEVICE_MODE_AUTO and packet_mode != device_mode:
        raise MocapPacketError(
            f"设备模式不匹配: selected={device_mode} packet={packet_mode} "
            f"sync={data[0]:02X} {data[1]:02X} size={len(data)}"
        )
    if packet_mode == DEVICE_MODE_CONTROLLER:
        raise UnsupportedDevicePacketError(
            "检测到手柄模式 C8 FF 数据报，但当前缺少厂商帧布局；"
            f"size={len(data)}，已拒绝生成遥操作骨架"
        )

    candidates = _header_candidates(data)
    if candidates:
        return _decode_g2_packet(data, receive_time)
    return _decode_legacy_packet(data, receive_time)


def frame_from_joint_rows(
    rows_xyzw: Sequence[Sequence[float]],
    *,
    timestamp_s: float,
    timestamp_us: int,
    receive_time_s: float | None = None,
    source: str = "canonical",
    wire: Mapping[str, Any] | None = None,
) -> MocapFrame:
    if len(rows_xyzw) != MOCAP_G2_JOINT_COUNT:
        raise ValueError(f"节点行数必须为 {MOCAP_G2_JOINT_COUNT}")
    joints: list[JointPose] = []
    for index, row in enumerate(rows_xyzw):
        if len(row) != 7:
            raise ValueError(f"节点 {index} 必须为 x,y,z,qx,qy,qz,qw")
        values = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"节点 {index} 包含 NaN/Inf")
        norm = math.sqrt(sum(value * value for value in values[3:]))
        joints.append(
            JointPose(
                position=values[:3],
                quaternion_xyzw=values[3:] if norm > 1e-8 else (0.0, 0.0, 0.0, 1.0),
                tracked=norm > 1e-8,
            )
        )
    return MocapFrame(
        timestamp_s=float(timestamp_s),
        timestamp_us=int(timestamp_us),
        receive_time_s=float(timestamp_s if receive_time_s is None else receive_time_s),
        joints=tuple(joints),
        source=source,
        wire={} if wire is None else wire,
    )


def encode_mocap_g2_frame(
    frame: MocapFrame,
    *,
    total_length: int = MOCAP_G2_PACKET_TOTAL_BYTES,
    length_field: int | None = None,
) -> bytes:
    if total_length < MOCAP_G2_MIN_PACKET_BYTES:
        raise ValueError(f"G2 包总长不能小于 {MOCAP_G2_MIN_PACKET_BYTES}")
    if length_field is None:
        length_field = MOCAP_G2_LIVE_LENGTH_FIELD if total_length == MOCAP_G2_PACKET_TOTAL_BYTES else total_length
    output = bytearray(total_length)
    struct.pack_into("<BBHHQ", output, 0, 0xAA, 0x55, MOCAP_G2_FRAME_ID, int(length_field), int(frame.timestamp_us))
    offset = 14
    for joint in frame.joints:
        qx, qy, qz, qw = joint.quaternion_xyzw
        struct.pack_into("<7f", output, offset, *joint.position, qw, qx, qy, qz)
        offset += 28
    return bytes(output)


def upper_body_skeleton(frame: MocapFrame) -> dict[str, dict[str, dict[str, Any]]]:
    skeleton: dict[str, dict[str, dict[str, Any]]] = {"left": {}, "right": {}, "torso": {}}
    for side in ("left", "right"):
        for joint_name in ("shoulder", "elbow", "wrist", "hand"):
            pose = frame.joint(f"{side}_{joint_name}")
            skeleton[side][joint_name] = {
                "pos": list(pose.position),
                "quat": list(pose.quaternion_xyzw),
                "tracked": pose.tracked,
            }
    for name in ("waist", "spine1", "spine2", "chest", "neck", "head", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        pose = frame.joint(name)
        skeleton["torso"][name] = {
            "pos": list(pose.position),
            "quat": list(pose.quaternion_xyzw),
            "tracked": pose.tracked,
        }
    return skeleton


def to_jsonable(value: Any) -> Any:
    if isinstance(value, MocapFrame):
        return {
            "timestamp_s": value.timestamp_s,
            "timestamp_us": value.timestamp_us,
            "receive_time_s": value.receive_time_s,
            "source": value.source,
            "wire": dict(value.wire),
        }
    if isinstance(value, JointPose):
        return {
            "position": value.position,
            "quaternion_xyzw": value.quaternion_xyzw,
            "tracked": value.tracked,
        }
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value