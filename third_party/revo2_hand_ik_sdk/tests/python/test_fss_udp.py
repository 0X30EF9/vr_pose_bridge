from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from revo2_hand_ik.fss_udp import (  # noqa: E402
    FssPacketError,
    HEADER_STRUCT,
    MOCAP_G2_HAND_INDEX,
    MOCAP_G2_JOINT_NAMES,
    MOCAP_G2_LIVE_LENGTH_FIELD,
    MOCAP_G2_PACKET_TOTAL_BYTES,
    MOCAP_G2_FRAME_ID,
    build_mocap_g2_frame_packet,
    MOTION_FRAME_ID,
    PAYLOAD_WITH_TIMESTAMP_BYTES,
    PAYLOAD_WITH_TIMESTAMP_AND_INNER_CRC_BYTES,
    CRC_STRUCT,
    extract_hand_mocap,
    fss_crc32,
    mocap_g2_csv_required_columns,
    mocap_g2_csv_row_to_joint_poses,
    parse_fss_packet,
)


def make_packet() -> bytes:
    payload = bytearray(PAYLOAD_WITH_TIMESTAMP_BYTES)
    struct.pack_into("<I", payload, 0, 12345)
    # root, rotations, hand fields, positions remain zero except selected data.
    rotation_base = 4 + 3 * 4
    position_base = 4 + 3 * 4 + 47 * 4 * 4 + 12 * 2
    # left hand index is 5; position and quat demonstrate extractor order.
    struct.pack_into("<4f", payload, rotation_base + 5 * 16, 0.1, 0.2, 0.3, 0.4)
    struct.pack_into("<3f", payload, position_base + 5 * 12, 1.0, 2.0, 3.0)
    # Right raw FSS outer-finger source triplets are intentionally distinct.
    # 41..43 is raw `right_ring_*`; 44..46 is raw `right_little_*`.
    struct.pack_into("<3f", payload, position_base + 41 * 12, 41.0, 42.0, 43.0)
    struct.pack_into("<3f", payload, position_base + 44 * 12, 44.0, 45.0, 46.0)
    fingers_offset = 4 + 3 * 4 + 47 * 4 * 4
    struct.pack_into("<6H", payload, fingers_offset, 1, 2, 3, 4, 5, 6)
    struct.pack_into("<6H", payload, fingers_offset + 12, 11, 12, 13, 14, 15, 16)
    packet = bytearray(HEADER_STRUCT.pack(0xAA, 0x55, MOTION_FRAME_ID, len(payload)) + payload)
    packet += CRC_STRUCT.pack(fss_crc32(packet))
    return bytes(packet)


def make_g2_packet() -> bytes:
    poses = [(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) for _ in range(64)]
    left_hand_index = MOCAP_G2_HAND_INDEX["left"]["hand"]
    left_thumb_index = MOCAP_G2_HAND_INDEX["left"]["thumb_near"]
    right_ring_index = MOCAP_G2_HAND_INDEX["right"]["ring_near"]
    poses[left_hand_index] = (1.0, 2.0, 3.0, 0.4, 0.1, 0.2, 0.3)
    poses[left_thumb_index] = (4.0, 5.0, 6.0, 0.8, 0.5, 0.6, 0.7)
    poses[right_ring_index] = (56.0, 57.0, 58.0, 1.0, 0.0, 0.0, 0.0)
    return build_mocap_g2_frame_packet(timestamp_us=12_345_678, joint_poses=poses)


def make_g2_packet_big_endian() -> bytes:
    raw = bytearray(MOCAP_G2_PACKET_TOTAL_BYTES)
    struct.pack_into(">BBHH", raw, 0, 0xAA, 0x55, MOCAP_G2_FRAME_ID, MOCAP_G2_PACKET_TOTAL_BYTES)
    struct.pack_into(">Q", raw, 6, 98_765_432)
    offset = 14
    for index in range(64):
        pose = (float(index), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        struct.pack_into(">7f", raw, offset, *pose)
        offset += 28
    return bytes(raw)


def make_g2_packet_aligned() -> bytes:
    raw = bytearray(MOCAP_G2_PACKET_TOTAL_BYTES)
    struct.pack_into("<BBHH", raw, 0, 0xAA, 0x55, MOCAP_G2_FRAME_ID, MOCAP_G2_PACKET_TOTAL_BYTES)
    # Two native-struct alignment bytes at offsets 6..7.
    struct.pack_into("<Q", raw, 8, 22_222_000)
    offset = 16
    for index in range(64):
        pose = (float(index), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        struct.pack_into("<7f", raw, offset, *pose)
        offset += 28
    return bytes(raw)


class FssUdpTest(unittest.TestCase):

    def test_crc_matches_supplied_cpp_decoder_reference(self) -> None:
        # Reference value from udp_receiever/src/fss_decoder/FssDecoder.cpp:
        # initial accumulator = 1, crc32_tab update, no final XOR.
        self.assertEqual(fss_crc32(b"123456789"), 0x3A8639CB)
        self.assertEqual(fss_crc32(bytes((0xAA, 0x55, 0x01, 0x00, 0x4C, 0x05))), 0x17323275)

    def test_timestamp_is_required_by_receiver_layout(self) -> None:
        payload = bytearray(PAYLOAD_WITH_TIMESTAMP_BYTES - 4)
        packet = bytearray(HEADER_STRUCT.pack(0xAA, 0x55, MOTION_FRAME_ID, len(payload)) + payload)
        packet += CRC_STRUCT.pack(fss_crc32(packet))
        with self.assertRaises(FssPacketError):
            parse_fss_packet(bytes(packet), check_crc=True)

    def test_accepts_optional_trailing_frame_crc_field(self) -> None:
        payload = bytearray(PAYLOAD_WITH_TIMESTAMP_AND_INNER_CRC_BYTES)
        struct.pack_into("<I", payload, 0, 12345)
        # The final 4 bytes are producer-owned/opaque to the supplied decoder.
        struct.pack_into("<I", payload, PAYLOAD_WITH_TIMESTAMP_BYTES, 0x13572468)
        packet = bytearray(HEADER_STRUCT.pack(0xAA, 0x55, MOTION_FRAME_ID, len(payload)) + payload)
        packet += CRC_STRUCT.pack(fss_crc32(packet))
        parsed = parse_fss_packet(bytes(packet), check_crc=True)
        self.assertEqual(parsed.timestamp_ms, 12345)


    def test_extract_hand_points_and_quaternion_order(self) -> None:
        packet = parse_fss_packet(make_packet())
        points, quats = extract_hand_mocap(packet, "left")
        self.assertEqual(points["hand"], (1.0, 2.0, 3.0))
        for actual, expected in zip(quats["hand"], (0.4, 0.1, 0.2, 0.3)):
            self.assertAlmostEqual(actual, expected, places=6)


    def test_right_ring_little_source_mapping_follows_normal_hardware_order(self) -> None:
        packet = parse_fss_packet(make_packet())
        points, _quats = extract_hand_mocap(packet, "right")
        # v0.10.31: restored IMU installation means FSS labels are already
        # anatomically correct and must reach the SDK unchanged.
        self.assertEqual(points["ring_near"], (41.0, 42.0, 43.0))
        self.assertEqual(points["little_near"], (44.0, 45.0, 46.0))


    def test_crc_failure(self) -> None:
        raw = bytearray(make_packet())
        raw[20] ^= 0xFF
        with self.assertRaises(FssPacketError):
            parse_fss_packet(bytes(raw), check_crc=True)


    def test_formotion_g2_verified_joint_indices(self) -> None:
        self.assertEqual(len(MOCAP_G2_JOINT_NAMES), 64)
        self.assertEqual(MOCAP_G2_HAND_INDEX["left"]["hand"], 10)
        self.assertEqual(MOCAP_G2_HAND_INDEX["left"]["thumb_near"], 15)
        self.assertEqual(MOCAP_G2_HAND_INDEX["left"]["index_middle"], 20)
        self.assertEqual(MOCAP_G2_HAND_INDEX["left"]["little_far"], 33)
        self.assertEqual(MOCAP_G2_HAND_INDEX["right"]["hand"], 39)
        self.assertEqual(MOCAP_G2_HAND_INDEX["right"]["thumb_near"], 44)
        self.assertEqual(MOCAP_G2_HAND_INDEX["right"]["little_far"], 62)

    def test_native_g2_csv_row_mapping(self) -> None:
        row: dict[str, str] = {}
        for joint_index, joint_name in enumerate(MOCAP_G2_JOINT_NAMES):
            row[f"pos_{joint_name}_X"] = str(joint_index + 0.1)
            row[f"pos_{joint_name}_Y"] = str(joint_index + 0.2)
            row[f"pos_{joint_name}_Z"] = str(joint_index + 0.3)
            row[f"quat_{joint_name}_W"] = "1.0"
            row[f"quat_{joint_name}_X"] = "0.0"
            row[f"quat_{joint_name}_Y"] = "0.0"
            row[f"quat_{joint_name}_Z"] = "0.0"

        self.assertTrue(mocap_g2_csv_required_columns().issubset(row.keys()))
        poses = mocap_g2_csv_row_to_joint_poses(row)
        self.assertEqual(len(poses), 64)
        self.assertEqual(poses[15][:3], (15.1, 15.2, 15.3))
        self.assertEqual(poses[39][:3], (39.1, 39.2, 39.3))

        packet = parse_fss_packet(
            build_mocap_g2_frame_packet(timestamp_us=1000, joint_poses=poses)
        )
        left_points, _ = extract_hand_mocap(packet, "left")
        right_points, _ = extract_hand_mocap(packet, "right")
        for actual, expected in zip(left_points["thumb_near"], (15.1, 15.2, 15.3)):
            self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(right_points["hand"], (39.1, 39.2, 39.3)):
            self.assertAlmostEqual(actual, expected, places=5)


    def test_parse_verified_live_g2_length_field_1850(self) -> None:
        raw = make_g2_packet()
        self.assertEqual(len(raw), 1860)
        self.assertEqual(struct.unpack_from("<H", raw, 4)[0], MOCAP_G2_LIVE_LENGTH_FIELD)
        packet = parse_fss_packet(raw)
        self.assertEqual(packet.layout, "formotion_g2_mocap_v1")
        self.assertEqual(packet.payload_length, 1850)
        self.assertEqual(packet.timestamp_offset, 6)
        self.assertEqual(packet.joint_pose_offset, 14)

    def test_still_accepts_documented_total_length_field_1860(self) -> None:
        raw = bytearray(make_g2_packet())
        struct.pack_into("<H", raw, 4, MOCAP_G2_PACKET_TOTAL_BYTES)
        packet = parse_fss_packet(bytes(raw))
        self.assertEqual(packet.layout, "formotion_g2_mocap_v1")
        self.assertEqual(packet.payload_length, MOCAP_G2_PACKET_TOTAL_BYTES)

    def test_parse_formotion_g2_mocap_frame(self) -> None:
        packet = parse_fss_packet(make_g2_packet(), check_crc=True)
        self.assertEqual(packet.layout, "formotion_g2_mocap_v1")
        self.assertEqual(packet.packet_id, MOCAP_G2_FRAME_ID)
        self.assertEqual(len(packet.raw), MOCAP_G2_PACKET_TOTAL_BYTES)
        self.assertEqual(packet.timestamp_us, 12_345_678)
        self.assertEqual(packet.timestamp_ms, 12_345)

    def test_extract_formotion_g2_points_and_quaternion_order(self) -> None:
        packet = parse_fss_packet(make_g2_packet())
        points, quats = extract_hand_mocap(packet, "left")
        self.assertEqual(points["hand"], (1.0, 2.0, 3.0))
        self.assertEqual(points["thumb_near"], (4.0, 5.0, 6.0))
        for actual, expected in zip(quats["hand"], (0.4, 0.1, 0.2, 0.3)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_formotion_g2_has_no_outer_crc_requirement(self) -> None:
        raw = bytearray(make_g2_packet())
        raw[-1] ^= 0x55
        packet = parse_fss_packet(bytes(raw), check_crc=True)
        self.assertEqual(packet.layout, "formotion_g2_mocap_v1")

    def test_parse_live_g2_big_endian_header_and_pose(self) -> None:
        packet = parse_fss_packet(make_g2_packet_big_endian())
        self.assertEqual(packet.header_byte_order, "big")
        self.assertEqual(packet.pose_byte_order, "big")
        self.assertEqual(packet.joint_pose_offset, 14)
        self.assertEqual(packet.timestamp_us, 98_765_432)
        points, _ = extract_hand_mocap(packet, "left")
        self.assertAlmostEqual(points["hand"][0], 10.0)

    def test_parse_live_g2_aligned_native_struct(self) -> None:
        packet = parse_fss_packet(make_g2_packet_aligned())
        self.assertEqual(packet.header_byte_order, "little")
        self.assertEqual(packet.pose_byte_order, "little")
        self.assertEqual(packet.timestamp_offset, 8)
        self.assertEqual(packet.joint_pose_offset, 16)
        self.assertEqual(packet.timestamp_us, 22_222_000)
        points, _ = extract_hand_mocap(packet, "right")
        self.assertAlmostEqual(points["hand"][0], 39.0)

    def test_parse_live_g2_reversed_sync_bytes(self) -> None:
        raw = bytearray(make_g2_packet())
        raw[0], raw[1] = raw[1], raw[0]
        packet = parse_fss_packet(bytes(raw))
        self.assertTrue(packet.sync_reversed)
        self.assertEqual(packet.layout, "formotion_g2_mocap_v1")


if __name__ == "__main__":
    unittest.main()
