"""Formotion/FSS mocap UDP decoder/encoder used by the Python SDK examples.

The current test chain consumes the Formotion Glove G2 ``Mocap Frame`` UDP
layout documented by the supplied quick-start manual:

``AA 55 | uint16 id=0x0000 | uint16 length=1850 | uint64 timestamp_us | float[64][7] joints_pose | reserved``

Each ``joints_pose`` item stores ``x, y, z, w, qx, qy, qz`` in world
coordinates.  The complete UDP datagram is 1860 bytes.  The verified live device writes
``1850`` into the Length field.  The parser also accepts earlier example
variants that wrote total or payload length and derives offsets from the actual
packet size.

The legacy FSS motion frame layout used by previous project versions is still
accepted for local regression and old CSV recordings:

``AA 55 | uint16 id=0x0001 | uint16 payload_len | legacy payload | uint32 crc``
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Mapping, Sequence

HEADER1 = 0xAA
HEADER2 = 0x55
HEADER_STRUCT = struct.Struct("<BBHH")
HEADER_STRUCT_BE = struct.Struct(">BBHH")
CRC_STRUCT = struct.Struct("<I")

# New Formotion Glove G2 mocap frame protocol.
MOCAP_G2_FRAME_ID = 0x0000
MOCAP_G2_JOINT_COUNT = 64
MOCAP_G2_FLOATS_PER_JOINT = 7
MOCAP_G2_JOINT_POSE_BYTES = MOCAP_G2_JOINT_COUNT * MOCAP_G2_FLOATS_PER_JOINT * 4
MOCAP_G2_TIMESTAMP_BYTES = 8
MOCAP_G2_MIN_PACKET_BYTES = HEADER_STRUCT.size + MOCAP_G2_TIMESTAMP_BYTES + MOCAP_G2_JOINT_POSE_BYTES
MOCAP_G2_PACKET_TOTAL_BYTES = 1860
# Real Formotion G2 devices observed on 2026-07-10 send a 1860-byte
# datagram while writing 1850 into the uint16 Length field.  Keep this
# convention explicit instead of treating the field as the UDP datagram size.
MOCAP_G2_LIVE_LENGTH_FIELD = 1850
MOCAP_G2_LIVE_LENGTH_EXCLUDED_BYTES = MOCAP_G2_PACKET_TOTAL_BYTES - MOCAP_G2_LIVE_LENGTH_FIELD
MOCAP_G2_RESERVED_BYTES = MOCAP_G2_PACKET_TOTAL_BYTES - MOCAP_G2_MIN_PACKET_BYTES
MOCAP_G2_PAYLOAD_BYTES = MOCAP_G2_PACKET_TOTAL_BYTES - HEADER_STRUCT.size
MOCAP_G2_MIN_PAYLOAD_BYTES = MOCAP_G2_MIN_PACKET_BYTES - HEADER_STRUCT.size

# Legacy project protocol retained for compatibility with old recordings/tests.
MOTION_FRAME_ID = 0x0001
JOINT_COUNT = 47
FINGER_VALUES_PER_HAND = 6
BASE_PAYLOAD_BYTES = 1352
PAYLOAD_WITH_TIMESTAMP_BYTES = 1356
PAYLOAD_WITH_TIMESTAMP_AND_INNER_CRC_BYTES = 1360
VALID_PAYLOAD_LENGTHS = {
    PAYLOAD_WITH_TIMESTAMP_BYTES,
    PAYLOAD_WITH_TIMESTAMP_AND_INNER_CRC_BYTES,
}

FSS_CRC32_INITIAL = 1
FSS_CRC32_POLYNOMIAL = 0xEDB88320


def _make_crc32_table() -> tuple[int, ...]:
    table: list[int] = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = ((value >> 1) ^ FSS_CRC32_POLYNOMIAL) if (value & 1) else (value >> 1)
        table.append(value & 0xFFFFFFFF)
    return tuple(table)


FSS_CRC32_TABLE = _make_crc32_table()

# Legacy 47-joint CSV names.  ``csv_udp_replay.py`` can still read old CSV
# files and remap their hand landmarks into the new 64-joint UDP frame.
JOINT_NAMES: tuple[str, ...] = (
    "head", "chest",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "hip",
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_upper_leg", "right_lower_leg", "right_foot",
    "left_thumb_near", "left_thumb_middle", "left_thumb_far",
    "left_index_near", "left_index_middle", "left_index_far",
    "left_middle_near", "left_middle_middle", "left_middle_far",
    "left_ring_near", "left_ring_middle", "left_ring_far",
    "left_little_near", "left_little_middle", "left_little_far",
    "right_thumb_near", "right_thumb_middle", "right_thumb_far",
    "right_index_near", "right_index_middle", "right_index_far",
    "right_middle_near", "right_middle_middle", "right_middle_far",
    "right_ring_near", "right_ring_middle", "right_ring_far",
    "right_little_near", "right_little_middle", "right_little_far",
)
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# Formotion G2 64-joint order verified against the new vendor CSV export.
# The CSV stores one complete pose for every protocol slot:
#   pos_<name>_X/Y/Z + quat_<name>_W/X/Y/Z
#
# Important naming differences:
# - the protocol CSV uses ``mid`` while the SDK uses ``middle``;
# - the protocol CSV uses ``pinky`` while the SDK uses ``little``.
MOCAP_G2_JOINT_NAMES: tuple[str, ...] = (
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
if len(MOCAP_G2_JOINT_NAMES) != MOCAP_G2_JOINT_COUNT:
    raise AssertionError("Formotion G2 joint table must contain exactly 64 entries")

MOCAP_G2_JOINT_INDEX = {name: index for index, name in enumerate(MOCAP_G2_JOINT_NAMES)}

# Internal SDK hand names -> verified protocol indices.
# Tips/end points are present in the protocol but the IK SDK only consumes
# hand + near/middle/far.
MOCAP_G2_HAND_INDEX: dict[str, dict[str, int]] = {
    "left": {
        "hand": MOCAP_G2_JOINT_INDEX["left_hand"],
        "thumb_near": MOCAP_G2_JOINT_INDEX["left_thumb_near"],
        "thumb_middle": MOCAP_G2_JOINT_INDEX["left_thumb_mid"],
        "thumb_far": MOCAP_G2_JOINT_INDEX["left_thumb_far"],
        "index_near": MOCAP_G2_JOINT_INDEX["left_index_near"],
        "index_middle": MOCAP_G2_JOINT_INDEX["left_index_mid"],
        "index_far": MOCAP_G2_JOINT_INDEX["left_index_far"],
        "middle_near": MOCAP_G2_JOINT_INDEX["left_middle_near"],
        "middle_middle": MOCAP_G2_JOINT_INDEX["left_middle_mid"],
        "middle_far": MOCAP_G2_JOINT_INDEX["left_middle_far"],
        "ring_near": MOCAP_G2_JOINT_INDEX["left_ring_near"],
        "ring_middle": MOCAP_G2_JOINT_INDEX["left_ring_mid"],
        "ring_far": MOCAP_G2_JOINT_INDEX["left_ring_far"],
        "little_near": MOCAP_G2_JOINT_INDEX["left_pinky_near"],
        "little_middle": MOCAP_G2_JOINT_INDEX["left_pinky_mid"],
        "little_far": MOCAP_G2_JOINT_INDEX["left_pinky_far"],
    },
    "right": {
        "hand": MOCAP_G2_JOINT_INDEX["right_hand"],
        "thumb_near": MOCAP_G2_JOINT_INDEX["right_thumb_near"],
        "thumb_middle": MOCAP_G2_JOINT_INDEX["right_thumb_mid"],
        "thumb_far": MOCAP_G2_JOINT_INDEX["right_thumb_far"],
        "index_near": MOCAP_G2_JOINT_INDEX["right_index_near"],
        "index_middle": MOCAP_G2_JOINT_INDEX["right_index_mid"],
        "index_far": MOCAP_G2_JOINT_INDEX["right_index_far"],
        "middle_near": MOCAP_G2_JOINT_INDEX["right_middle_near"],
        "middle_middle": MOCAP_G2_JOINT_INDEX["right_middle_mid"],
        "middle_far": MOCAP_G2_JOINT_INDEX["right_middle_far"],
        "ring_near": MOCAP_G2_JOINT_INDEX["right_ring_near"],
        "ring_middle": MOCAP_G2_JOINT_INDEX["right_ring_mid"],
        "ring_far": MOCAP_G2_JOINT_INDEX["right_ring_far"],
        "little_near": MOCAP_G2_JOINT_INDEX["right_pinky_near"],
        "little_middle": MOCAP_G2_JOINT_INDEX["right_pinky_mid"],
        "little_far": MOCAP_G2_JOINT_INDEX["right_pinky_far"],
    },
}

# v0.10.31: the right-hand ring/little IMUs are back in their normal anatomical
# installation.  Keep the explicit empty map so a source-specific remap can be
# reintroduced only with a documented hardware change.
RIGHT_HAND_SOURCE_NAME_REMAP: dict[str, str] = {}


class FssPacketError(ValueError):
    pass


def fss_crc32(data: bytes | bytearray | memoryview) -> int:
    """Return the legacy raw FSS outer CRC.

    This is kept for legacy ``id=0x0001`` frames.  The new Formotion G2 mocap
    frame does not document an outer CRC in the 15000 data packet.
    """
    crc = FSS_CRC32_INITIAL
    for byte in memoryview(data).cast("B"):
        crc = FSS_CRC32_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFF


@dataclass(frozen=True)
class FssPacket:
    raw: bytes
    packet_id: int
    payload_length: int
    timestamp_ms: int
    payload_offset: int
    finger_offset_in_payload: int
    crc_ok: bool
    layout: str = "legacy_fss_motion_v1"
    timestamp_us: int | None = None
    joint_pose_offset: int | None = None
    joint_count: int = JOINT_COUNT
    header_byte_order: str = "little"
    pose_byte_order: str = "little"
    timestamp_offset: int | None = None
    sync_reversed: bool = False

    @property
    def has_timestamp(self) -> bool:
        return True

    def hand_values(self, side: str) -> tuple[int, int, int, int, int, int]:
        if self.layout != "legacy_fss_motion_v1":
            return (0, 0, 0, 0, 0, 0)
        side_offset = 0 if side == "left" else 12 if side == "right" else None
        if side_offset is None:
            raise ValueError("side must be 'left' or 'right'")
        offset = self.payload_offset + self.finger_offset_in_payload + side_offset
        return tuple(struct.unpack_from("<6H", self.raw, offset))

    @staticmethod
    def _checked_hand_values(values: Sequence[int], *, side: str) -> tuple[int, int, int, int, int, int]:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if len(values) != FINGER_VALUES_PER_HAND:
            raise ValueError(f"{side} hand requires exactly six uint16 values")
        return tuple(max(0, min(65535, int(v))) for v in values)

    def replace_both_hand_values(
        self,
        left_values: Sequence[int],
        right_values: Sequence[int],
    ) -> bytes:
        """Clone a legacy frame and update both six-value hand fields.

        The new G2 mocap frame does not contain these legacy normalized hand
        fields.  Use SDK rad output instead of trying to write back into the
        input mocap packet.
        """
        if self.layout != "legacy_fss_motion_v1":
            raise ValueError("replace_both_hand_values only supports legacy id=0x0001 frames")
        left = self._checked_hand_values(left_values, side="left")
        right = self._checked_hand_values(right_values, side="right")
        output = bytearray(self.raw)
        start = self.payload_offset + self.finger_offset_in_payload
        struct.pack_into("<6H", output, start, *left)
        struct.pack_into("<6H", output, start + FINGER_VALUES_PER_HAND * 2, *right)
        crc = fss_crc32(output[:-CRC_STRUCT.size])
        CRC_STRUCT.pack_into(output, len(output) - CRC_STRUCT.size, crc)
        return bytes(output)

    def replace_hand_values(self, side: str, values: Sequence[int]) -> bytes:
        if side == "left":
            return self.replace_both_hand_values(values, self.hand_values("right"))
        if side == "right":
            return self.replace_both_hand_values(self.hand_values("left"), values)
        raise ValueError("side must be 'left' or 'right'")


def _g2_length_score(length: int, packet_size: int) -> int:
    """Score a G2 Length field against the received UDP datagram size.

    The verified live device sends packet_size=1860 with Length=1850.
    Earlier local examples used total or payload length.  Accept the verified
    device convention first, then retain those older variants while still
    requiring a complete 64x7 pose block.
    """
    # Verified live-device convention: datagram=1860, Length=1850.
    # The field excludes 10 bytes from the complete UDP datagram.
    if (
        packet_size == MOCAP_G2_PACKET_TOTAL_BYTES
        and length == MOCAP_G2_LIVE_LENGTH_FIELD
        and length + MOCAP_G2_LIVE_LENGTH_EXCLUDED_BYTES == packet_size
    ):
        return 8
    if length == packet_size:
        return 6
    if HEADER_STRUCT.size + length == packet_size:
        return 5
    if length == MOCAP_G2_PACKET_TOTAL_BYTES and packet_size in (1860, 1862, 1864, 1868):
        return 4
    if length == MOCAP_G2_PAYLOAD_BYTES and packet_size in (1860, 1862, 1864, 1868):
        return 3
    if length in (MOCAP_G2_MIN_PACKET_BYTES, MOCAP_G2_MIN_PAYLOAD_BYTES) and packet_size >= MOCAP_G2_MIN_PACKET_BYTES:
        return 2
    return -1


def _pose_block_score(datagram: bytes, joint_pose_offset: int, byte_order: str) -> float:
    """Return a plausibility score for one float[64][7] layout candidate."""
    end = joint_pose_offset + MOCAP_G2_JOINT_POSE_BYTES
    if joint_pose_offset < 0 or end > len(datagram):
        return float("-inf")

    prefix = "<" if byte_order == "little" else ">"
    # Include body, hand roots, and finger joints.  Correct decoding normally
    # produces finite positions and unit-length quaternions.
    sample_indices = (0, 5, 10, 15, 20, 31, 39, 44, 49, 60, 63)
    score = 0.0
    for joint_index in sample_indices:
        offset = joint_pose_offset + joint_index * MOCAP_G2_FLOATS_PER_JOINT * 4
        try:
            px, py, pz, qw, qx, qy, qz = struct.unpack_from(prefix + "7f", datagram, offset)
        except struct.error:
            return float("-inf")
        values = (px, py, pz, qw, qx, qy, qz)
        if not all(math.isfinite(value) for value in values):
            return float("-inf")

        max_position = max(abs(px), abs(py), abs(pz))
        if max_position < 10000.0:
            score += 1.0
        else:
            score -= 10.0

        quat_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        # Permit all-zero untracked joints, but strongly prefer normalized data.
        if quat_norm < 1e-8:
            score += 0.1
        elif 0.5 <= quat_norm <= 1.5:
            score += 4.0 - min(3.0, abs(quat_norm - 1.0) * 8.0)
        else:
            score -= min(10.0, abs(quat_norm - 1.0))
    return score


def _select_g2_pose_layout(datagram: bytes) -> tuple[int, int, str]:
    """Select packed/aligned timestamp offsets and float byte order.

    Packed serialization places uint64 timestamp at offset 6 and joint poses at
    offset 14.  A native C/C++ struct may insert two alignment bytes, producing
    offsets 8 and 16.  Both little- and big-endian float layouts are diagnosed.
    """
    candidates: list[tuple[float, int, int, str]] = []
    for timestamp_offset, joint_pose_offset in ((6, 14), (8, 16)):
        for byte_order in ("little", "big"):
            candidates.append(
                (_pose_block_score(datagram, joint_pose_offset, byte_order), timestamp_offset, joint_pose_offset, byte_order)
            )
    score, timestamp_offset, joint_pose_offset, byte_order = max(candidates, key=lambda item: item[0])
    if not math.isfinite(score) or score < -1.0:
        details = ", ".join(
            f"ts={ts}/pose={pose}/{order}:score={candidate_score:.2f}"
            for candidate_score, ts, pose, order in candidates
        )
        raise FssPacketError(f"unable to locate a plausible G2 float[64][7] block ({details})")
    return timestamp_offset, joint_pose_offset, byte_order


def _parse_mocap_g2_packet(
    datagram: bytes,
    *,
    packet_id: int,
    length: int,
    header_byte_order: str,
    sync_reversed: bool,
) -> FssPacket:
    if packet_id != MOCAP_G2_FRAME_ID:
        raise FssPacketError(f"unsupported Formotion G2 packet id: 0x{packet_id:04X}")
    if len(datagram) < MOCAP_G2_MIN_PACKET_BYTES:
        raise FssPacketError(
            f"Formotion G2 packet too short: got {len(datagram)}, minimum {MOCAP_G2_MIN_PACKET_BYTES}"
        )
    if _g2_length_score(length, len(datagram)) < 0:
        raise FssPacketError(
            f"unsupported Formotion G2 length={length}, packet_size={len(datagram)}; "
            f"expected live length {MOCAP_G2_LIVE_LENGTH_FIELD}, total {MOCAP_G2_PACKET_TOTAL_BYTES}, "
            f"or payload {MOCAP_G2_PAYLOAD_BYTES}"
        )

    timestamp_offset, joint_pose_offset, pose_byte_order = _select_g2_pose_layout(datagram)
    prefix = "<" if pose_byte_order == "little" else ">"
    timestamp_us = struct.unpack_from(prefix + "Q", datagram, timestamp_offset)[0]
    payload_length = int(length)
    return FssPacket(
        raw=datagram,
        packet_id=packet_id,
        payload_length=payload_length,
        timestamp_ms=int(timestamp_us // 1000),
        payload_offset=HEADER_STRUCT.size,
        finger_offset_in_payload=-1,
        crc_ok=True,
        layout="formotion_g2_mocap_v1",
        timestamp_us=int(timestamp_us),
        joint_pose_offset=joint_pose_offset,
        joint_count=MOCAP_G2_JOINT_COUNT,
        header_byte_order=header_byte_order,
        pose_byte_order=pose_byte_order,
        timestamp_offset=timestamp_offset,
        sync_reversed=sync_reversed,
    )


def _parse_legacy_packet(datagram: bytes, *, packet_id: int, length: int, check_crc: bool) -> FssPacket:
    if packet_id != MOTION_FRAME_ID:
        raise FssPacketError(f"unsupported FSS packet id: 0x{packet_id:04X}")
    if length not in VALID_PAYLOAD_LENGTHS:
        raise FssPacketError(f"unsupported FSS payload length: {length}")
    expected_size = HEADER_STRUCT.size + length + CRC_STRUCT.size
    if len(datagram) != expected_size:
        raise FssPacketError(f"packet size mismatch: got {len(datagram)}, expected {expected_size}")
    packet_crc = CRC_STRUCT.unpack_from(datagram, len(datagram) - 4)[0]
    computed_crc = fss_crc32(datagram[:-4])
    if check_crc and packet_crc != computed_crc:
        raise FssPacketError(
            "FSS outer CRC mismatch (raw seed=1/no final XOR): "
            f"packet=0x{packet_crc:08X}, computed=0x{computed_crc:08X}"
        )

    payload_offset = HEADER_STRUCT.size
    timestamp_ms = struct.unpack_from("<I", datagram, payload_offset)[0]
    offset = 4 + 3 * 4 + JOINT_COUNT * 4 * 4
    return FssPacket(
        raw=datagram,
        packet_id=packet_id,
        payload_length=length,
        timestamp_ms=timestamp_ms,
        payload_offset=payload_offset,
        finger_offset_in_payload=offset,
        crc_ok=(packet_crc == computed_crc),
    )


def _decode_header_candidates(datagram: bytes) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for byte_order, header_struct in (("little", HEADER_STRUCT), ("big", HEADER_STRUCT_BE)):
        _h1, _h2, packet_id, length = header_struct.unpack_from(datagram, 0)
        if packet_id == MOCAP_G2_FRAME_ID and _g2_length_score(length, len(datagram)) >= 0:
            candidates.append((packet_id, length, byte_order))
        elif packet_id == MOTION_FRAME_ID and length in VALID_PAYLOAD_LENGTHS:
            candidates.append((packet_id, length, byte_order))
    return candidates


def parse_fss_packet(datagram: bytes, *, check_crc: bool = True) -> FssPacket:
    """Parse new G2 mocap frames and legacy FSS frames.

    Live device streams are accepted with little/big-endian uint16 headers,
    packed/aligned uint64 timestamps, and either ``AA 55`` or ``55 AA`` sync
    byte order.  The selected layout is exposed in :class:`FssPacket` for logs.
    """
    if len(datagram) < HEADER_STRUCT.size:
        raise FssPacketError(f"packet too short: {len(datagram)} bytes")
    h1, h2 = datagram[0], datagram[1]
    sync_reversed = (h1, h2) == (HEADER2, HEADER1)
    if (h1, h2) not in ((HEADER1, HEADER2), (HEADER2, HEADER1)):
        raise FssPacketError(f"bad sync bytes: 0x{h1:02X} 0x{h2:02X}")

    candidates = _decode_header_candidates(datagram)
    if not candidates:
        little_id, little_length = struct.unpack_from("<HH", datagram, 2)
        big_id, big_length = struct.unpack_from(">HH", datagram, 2)
        raise FssPacketError(
            "unsupported header: "
            f"size={len(datagram)} little(id=0x{little_id:04X},length={little_length}) "
            f"big(id=0x{big_id:04X},length={big_length})"
        )

    # Prefer exact length matches, then little-endian for backward compatibility.
    def candidate_score(item: tuple[int, int, str]) -> tuple[int, int]:
        packet_id, length, byte_order = item
        length_score = _g2_length_score(length, len(datagram)) if packet_id == MOCAP_G2_FRAME_ID else 10
        return length_score, 1 if byte_order == "little" else 0

    packet_id, length, header_byte_order = max(candidates, key=candidate_score)
    if packet_id == MOCAP_G2_FRAME_ID:
        return _parse_mocap_g2_packet(
            datagram,
            packet_id=packet_id,
            length=length,
            header_byte_order=header_byte_order,
            sync_reversed=sync_reversed,
        )
    if header_byte_order != "little" or sync_reversed:
        raise FssPacketError("legacy id=0x0001 packets require AA 55 and little-endian header fields")
    return _parse_legacy_packet(datagram, packet_id=packet_id, length=length, check_crc=check_crc)

def _payload_joint_positions_offset(packet: FssPacket) -> int:
    return 4 + 3 * 4 + JOINT_COUNT * 4 * 4 + 2 * FINGER_VALUES_PER_HAND * 2


def _payload_joint_rotations_offset(packet: FssPacket) -> int:
    return 4 + 3 * 4


def _hand_names() -> tuple[str, ...]:
    return (
        "hand",
        "thumb_near", "thumb_middle", "thumb_far",
        "index_near", "index_middle", "index_far",
        "middle_near", "middle_middle", "middle_far",
        "ring_near", "ring_middle", "ring_far",
        "little_near", "little_middle", "little_far",
    )


def _extract_hand_mocap_g2(packet: FssPacket, side: str) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float, float]]]:
    assert packet.joint_pose_offset is not None
    index_map = MOCAP_G2_HAND_INDEX[side]
    points: dict[str, tuple[float, float, float]] = {}
    quats: dict[str, tuple[float, float, float, float]] = {}
    for short_name in _hand_names():
        source_short_name = RIGHT_HAND_SOURCE_NAME_REMAP.get(short_name, short_name) if side == "right" else short_name
        joint_index = index_map[source_short_name]
        offset = packet.joint_pose_offset + joint_index * MOCAP_G2_FLOATS_PER_JOINT * 4
        prefix = "<" if packet.pose_byte_order == "little" else ">"
        px, py, pz, qw, qx, qy, qz = struct.unpack_from(prefix + "7f", packet.raw, offset)
        points[short_name] = (float(px), float(py), float(pz))
        quats[short_name] = (float(qw), float(qx), float(qy), float(qz))
    return points, quats


def _extract_hand_mocap_legacy(packet: FssPacket, side: str) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float, float]]]:
    prefix = f"{side}_"
    position_base = packet.payload_offset + _payload_joint_positions_offset(packet)
    rotation_base = packet.payload_offset + _payload_joint_rotations_offset(packet)
    points: dict[str, tuple[float, float, float]] = {}
    quats: dict[str, tuple[float, float, float, float]] = {}
    for short_name in _hand_names():
        source_short_name = RIGHT_HAND_SOURCE_NAME_REMAP.get(short_name, short_name) if side == "right" else short_name
        source_name = prefix + source_short_name
        idx = JOINT_INDEX[source_name]
        px, py, pz = struct.unpack_from("<3f", packet.raw, position_base + idx * 12)
        qx, qy, qz, qw = struct.unpack_from("<4f", packet.raw, rotation_base + idx * 16)
        points[short_name] = (float(px), float(py), float(pz))
        quats[short_name] = (float(qw), float(qx), float(qy), float(qz))
    return points, quats


def extract_hand_mocap(packet: FssPacket, side: str) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float, float]]]:
    """Extract the hand-only points/rotations consumed by ``Revo2HandIkSdk``.

    Position tuple order is ``x, y, z``.  Quaternion tuple order returned to the
    SDK is always ``w, x, y, z``.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if packet.layout == "formotion_g2_mocap_v1":
        return _extract_hand_mocap_g2(packet, side)
    return _extract_hand_mocap_legacy(packet, side)


def mocap_g2_csv_required_columns() -> set[str]:
    """Return required pose columns for a native Formotion G2 CSV export."""
    required: set[str] = set()
    for joint_name in MOCAP_G2_JOINT_NAMES:
        required.update(
            {
                f"pos_{joint_name}_X",
                f"pos_{joint_name}_Y",
                f"pos_{joint_name}_Z",
                f"quat_{joint_name}_W",
                f"quat_{joint_name}_X",
                f"quat_{joint_name}_Y",
                f"quat_{joint_name}_Z",
            }
        )
    return required


def mocap_g2_csv_row_to_joint_poses(
    row: Mapping[str, object],
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Convert one native G2 CSV row into protocol-order ``float[64][7]``.

    Input columns are the names emitted by the new recorder, for example
    ``pos_left_index_mid_X`` and ``quat_right_pinky_far_W``.  Output row order
    is exactly :data:`MOCAP_G2_JOINT_NAMES`, and each pose is
    ``x, y, z, w, qx, qy, qz``.
    """

    def value(column: str) -> float:
        raw = row.get(column)
        if raw is None or raw == "":
            raise ValueError(f"missing required G2 CSV value: {column}")
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid float in G2 CSV column {column}: {raw!r}") from exc

    poses: list[tuple[float, float, float, float, float, float, float]] = []
    for joint_name in MOCAP_G2_JOINT_NAMES:
        poses.append(
            (
                value(f"pos_{joint_name}_X"),
                value(f"pos_{joint_name}_Y"),
                value(f"pos_{joint_name}_Z"),
                value(f"quat_{joint_name}_W"),
                value(f"quat_{joint_name}_X"),
                value(f"quat_{joint_name}_Y"),
                value(f"quat_{joint_name}_Z"),
            )
        )
    return poses


def build_mocap_g2_frame_packet(
    *,
    timestamp_us: int,
    joint_poses: Sequence[Sequence[float]],
    total_length: int = MOCAP_G2_PACKET_TOTAL_BYTES,
    length_field: int | None = None,
) -> bytes:
    """Build one Formotion G2 ``Mocap Frame`` UDP datagram.

    ``joint_poses`` must contain 64 rows.  Each row is ``x, y, z, w, qx, qy,
    qz``.  The UDP datagram is 1860 bytes.  By default the uint16 ``Length``
    field is 1850, matching the verified live Formotion G2 stream.  Reserved
    tail bytes are zero-filled.
    """
    if len(joint_poses) != MOCAP_G2_JOINT_COUNT:
        raise ValueError(f"joint_poses must contain {MOCAP_G2_JOINT_COUNT} rows")
    if total_length < MOCAP_G2_MIN_PACKET_BYTES:
        raise ValueError(f"total_length must be at least {MOCAP_G2_MIN_PACKET_BYTES}")
    if length_field is None:
        length_field = (
            MOCAP_G2_LIVE_LENGTH_FIELD
            if total_length == MOCAP_G2_PACKET_TOTAL_BYTES
            else total_length
        )
    if not 0 <= int(length_field) <= 0xFFFF:
        raise ValueError("length_field must fit uint16")
    payload_length = total_length - HEADER_STRUCT.size
    payload = bytearray(payload_length)
    struct.pack_into("<Q", payload, 0, int(timestamp_us) & 0xFFFFFFFFFFFFFFFF)
    offset = MOCAP_G2_TIMESTAMP_BYTES
    for index, pose in enumerate(joint_poses):
        if len(pose) != MOCAP_G2_FLOATS_PER_JOINT:
            raise ValueError(f"joint pose {index} must contain x,y,z,w,qx,qy,qz")
        struct.pack_into("<7f", payload, offset, *(float(v) for v in pose))
        offset += MOCAP_G2_FLOATS_PER_JOINT * 4
    return HEADER_STRUCT.pack(HEADER1, HEADER2, MOCAP_G2_FRAME_ID, int(length_field)) + bytes(payload)


def build_fss_motion_packet(
    *,
    timestamp_ms: int,
    root_position: Sequence[float],
    joint_rotations_xyzw: Sequence[Sequence[float]],
    joint_positions: Sequence[Sequence[float]],
    left_hand_values: Sequence[int] | None = None,
    right_hand_values: Sequence[int] | None = None,
    payload_length: int = PAYLOAD_WITH_TIMESTAMP_BYTES,
) -> bytes:
    """Build one legacy FSS whole-body motion UDP datagram.

    Kept for backward-compatible tests.  New SDK validation should use
    :func:`build_mocap_g2_frame_packet` instead.
    """
    if payload_length not in VALID_PAYLOAD_LENGTHS:
        raise ValueError(
            "legacy CSV replay supports FSS timestamped payload lengths "
            f"{sorted(VALID_PAYLOAD_LENGTHS)}, got {payload_length}"
        )
    if len(root_position) != 3:
        raise ValueError("root_position must contain exactly three values")
    if len(joint_rotations_xyzw) != JOINT_COUNT:
        raise ValueError(f"joint_rotations_xyzw must contain {JOINT_COUNT} quaternions")
    if len(joint_positions) != JOINT_COUNT:
        raise ValueError(f"joint_positions must contain {JOINT_COUNT} positions")

    left = FssPacket._checked_hand_values(left_hand_values or (0,) * FINGER_VALUES_PER_HAND, side="left")
    right = FssPacket._checked_hand_values(right_hand_values or (0,) * FINGER_VALUES_PER_HAND, side="right")

    payload = bytearray(payload_length)
    offset = 0
    struct.pack_into("<I", payload, offset, int(timestamp_ms) & 0xFFFFFFFF)
    offset += 4
    struct.pack_into("<3f", payload, offset, *(float(v) for v in root_position))
    offset += 12

    for index, quat in enumerate(joint_rotations_xyzw):
        if len(quat) != 4:
            raise ValueError(f"joint rotation {index} must contain x,y,z,w")
        struct.pack_into("<4f", payload, offset, *(float(v) for v in quat))
        offset += 16

    struct.pack_into("<6H", payload, offset, *left)
    offset += 12
    struct.pack_into("<6H", payload, offset, *right)
    offset += 12

    for index, position in enumerate(joint_positions):
        if len(position) != 3:
            raise ValueError(f"joint position {index} must contain x,y,z")
        struct.pack_into("<3f", payload, offset, *(float(v) for v in position))
        offset += 12

    if offset != PAYLOAD_WITH_TIMESTAMP_BYTES:
        raise AssertionError(
            f"legacy FSS motion fields size mismatch: wrote {offset}, expected {PAYLOAD_WITH_TIMESTAMP_BYTES}"
        )

    header = HEADER_STRUCT.pack(HEADER1, HEADER2, MOTION_FRAME_ID, payload_length)
    raw_without_crc = header + bytes(payload)
    return raw_without_crc + CRC_STRUCT.pack(fss_crc32(raw_without_crc))


def legacy_csv_rows_to_g2_joint_poses(
    *,
    positions_xyz: Mapping[str, Sequence[float]],
    rotations_xyzw: Mapping[str, Sequence[float]],
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Map old 47-column CSV hand landmarks into a 64-joint G2 frame.

    This helper exists only for CSV replay.  It lets old recordings feed the new
    UDP input example without changing the SDK API.  Missing body joints remain
    zero with identity orientation; hand joints are copied into the documented
    64-joint slots used by :data:`MOCAP_G2_HAND_INDEX`.
    """
    identity = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    poses: list[tuple[float, float, float, float, float, float, float]] = [identity for _ in range(MOCAP_G2_JOINT_COUNT)]

    # Body hand anchors, useful for overlay/raw debugging.
    anchor_map = {
        "left_hand": MOCAP_G2_HAND_INDEX["left"]["hand"],
        "right_hand": MOCAP_G2_HAND_INDEX["right"]["hand"],
    }
    for legacy_name, index in anchor_map.items():
        pos = positions_xyz.get(legacy_name)
        quat = rotations_xyzw.get(legacy_name)
        if pos is not None and quat is not None:
            x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
            qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            poses[index] = (x, y, z, qw, qx, qy, qz)

    for side in ("left", "right"):
        index_map = MOCAP_G2_HAND_INDEX[side]
        for short_name, joint_index in index_map.items():
            if short_name == "hand":
                continue
            legacy_name = f"{side}_{short_name}"
            pos = positions_xyz.get(legacy_name)
            quat = rotations_xyzw.get(legacy_name)
            if pos is None or quat is None:
                continue
            x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
            qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            poses[joint_index] = (x, y, z, qw, qx, qy, qz)
    return poses
