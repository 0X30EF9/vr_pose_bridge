from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mocap_csv_format import timestamp_us_from_csv
from mocap_udp_recorder import RAW_FRAME_HEADER, capture_udp
from teleop_mocap_protocol import (
    MOCAP_G2_FRAME_ID,
    MOCAP_G2_LIVE_LENGTH_FIELD,
    MOCAP_G2_PACKET_TOTAL_BYTES,
    MocapPacketError,
    UnsupportedDevicePacketError,
    encode_mocap_g2_frame,
    frame_from_joint_rows,
    parse_mocap_packet,
    snapshot_mocap_data,
)


def joint_rows() -> list[tuple[float, ...]]:
    rows = [(float(index), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) for index in range(64)]
    rows[10] = (1.0, 2.0, 3.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    return rows


def make_wire_packet(
    *,
    aligned: bool = False,
    byte_order: str = "little",
    length_field: int = MOCAP_G2_LIVE_LENGTH_FIELD,
    sync_reversed: bool = False,
) -> bytes:
    prefix = "<" if byte_order == "little" else ">"
    timestamp_offset, pose_offset = (8, 16) if aligned else (6, 14)
    raw = bytearray(MOCAP_G2_PACKET_TOTAL_BYTES)
    sync = (0x55, 0xAA) if sync_reversed else (0xAA, 0x55)
    struct.pack_into(prefix + "BBHH", raw, 0, *sync, MOCAP_G2_FRAME_ID, length_field)
    struct.pack_into(prefix + "Q", raw, timestamp_offset, 12_345_678)
    for index, row in enumerate(joint_rows()):
        x, y, z, qx, qy, qz, qw = row
        struct.pack_into(prefix + "7f", raw, pose_offset + index * 28, x, y, z, qw, qx, qy, qz)
    return bytes(raw)


class _FakeUdpSocket:
    def __init__(self, datagrams: list[bytes]) -> None:
        self.datagrams = list(datagrams)

    def bind(self, _address) -> None:
        pass

    def setsockopt(self, *_args) -> None:
        pass

    def settimeout(self, _timeout: float) -> None:
        pass

    def recvfrom(self, _size: int):
        if self.datagrams:
            return self.datagrams.pop(0), ("127.0.0.1", 12345)
        time.sleep(0.002)
        raise socket.timeout

    def close(self) -> None:
        pass


class MocapProtocolTest(unittest.TestCase):
    def test_compact_live_packet_and_wxyz_boundary(self) -> None:
        frame = parse_mocap_packet(make_wire_packet(), receive_time=99.0)
        self.assertEqual(frame.timestamp_us, 12_345_678)
        self.assertEqual(frame.wire["length_field"], 1850)
        self.assertEqual(frame.wire["timestamp_offset"], 6)
        self.assertEqual(frame.wire["pose_offset"], 14)
        self.assertEqual(frame.joint("left_hand").position, (1.0, 2.0, 3.0))
        expected = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        for actual, target in zip(frame.joint("left_hand").quaternion_xyzw, expected):
            self.assertAlmostEqual(actual, target, places=6)

    def test_replay_dict_keeps_hand_compatibility_and_exposes_wrist(self) -> None:
        frame = frame_from_joint_rows(
            joint_rows(),
            timestamp_s=12.345678,
            timestamp_us=12_345_678,
        )
        replay = frame.to_replay_dict()

        self.assertEqual(replay["left_pos"], replay["left_hand_pos"])
        self.assertEqual(replay["left_quat"], replay["left_hand_quat"])
        self.assertEqual(replay["left_wrist_pos"], [9.0, 0.0, 1.0])
        self.assertEqual(replay["right_wrist_pos"], [38.0, 0.0, 1.0])
        self.assertTrue(replay["hand_wrist_tracking"]["left"]["hand"])
        self.assertTrue(replay["hand_wrist_tracking"]["left"]["wrist"])

    def test_accepts_documented_total_length(self) -> None:
        frame = parse_mocap_packet(
            make_wire_packet(length_field=MOCAP_G2_PACKET_TOTAL_BYTES),
            receive_time=99.0,
        )
        self.assertEqual(frame.wire["length_field"], 1860)

    def test_aligned_native_struct(self) -> None:
        frame = parse_mocap_packet(make_wire_packet(aligned=True), receive_time=99.0)
        self.assertEqual(frame.wire["timestamp_offset"], 8)
        self.assertEqual(frame.wire["pose_offset"], 16)
        self.assertEqual(frame.joint("right_hand").position[0], 39.0)

    def test_big_endian_header_timestamp_and_pose(self) -> None:
        frame = parse_mocap_packet(
            make_wire_packet(byte_order="big", length_field=MOCAP_G2_PACKET_TOTAL_BYTES),
            receive_time=99.0,
        )
        self.assertEqual(frame.wire["header_byte_order"], "big")
        self.assertEqual(frame.wire["pose_byte_order"], "big")
        self.assertEqual(frame.joint("right_hand").position[0], 39.0)

    def test_reversed_sync_header(self) -> None:
        frame = parse_mocap_packet(make_wire_packet(sync_reversed=True), receive_time=99.0)
        self.assertTrue(frame.wire["sync_reversed"])

    def test_encoder_round_trip_is_64_joint_g2(self) -> None:
        source = frame_from_joint_rows(
            joint_rows(),
            timestamp_s=12.345678,
            timestamp_us=12_345_678,
        )
        raw = encode_mocap_g2_frame(source)
        decoded = parse_mocap_packet(raw, receive_time=99.0)
        self.assertEqual(len(raw), 1860)
        self.assertEqual(len(decoded.joints), 64)
        self.assertEqual(decoded.joint("left_hand").position, (1.0, 2.0, 3.0))

    def test_rejects_invalid_sync_and_non_finite_pose(self) -> None:
        bad_sync = bytearray(make_wire_packet())
        bad_sync[:2] = b"\x00\x00"
        with self.assertRaises(MocapPacketError):
            parse_mocap_packet(bytes(bad_sync), receive_time=99.0)

        bad_pose = bytearray(make_wire_packet())
        struct.pack_into("<f", bad_pose, 14, float("nan"))
        with self.assertRaises(MocapPacketError):
            parse_mocap_packet(bytes(bad_pose), receive_time=99.0)

    def test_device_mode_dispatch_rejects_mismatch_and_unverified_controller(self) -> None:
        glove_packet = make_wire_packet()
        with self.assertRaisesRegex(MocapPacketError, "设备模式不匹配"):
            parse_mocap_packet(
                glove_packet,
                receive_time=99.0,
                device_mode="controller",
            )

        controller_packet = b"\xC8\xFF" + bytes(range(32))
        with self.assertRaisesRegex(UnsupportedDevicePacketError, "缺少厂商帧布局"):
            parse_mocap_packet(
                controller_packet,
                receive_time=99.0,
                device_mode="controller",
            )
        with self.assertRaises(UnsupportedDevicePacketError):
            parse_mocap_packet(
                controller_packet,
                receive_time=99.0,
                device_mode="auto",
            )

    def test_auto_mode_accepts_verified_glove_packet(self) -> None:
        frame = parse_mocap_packet(
            make_wire_packet(),
            receive_time=99.0,
            device_mode="auto",
        )
        self.assertEqual(frame.source, "formotion_g2_udp")

    def test_control_snapshot_supports_immutable_mocap_frame(self) -> None:
        frame = parse_mocap_packet(make_wire_packet(), receive_time=99.0)
        source = frame.to_replay_dict()

        snapshot = snapshot_mocap_data(source)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNot(snapshot, source)
        self.assertIs(snapshot["mocap_frame"], frame)
        self.assertIs(snapshot["mocap_frame"].wire, frame.wire)
        snapshot["current_q"] = [0.0] * 14
        self.assertNotIn("current_q", source)
        self.assertIsNone(snapshot_mocap_data(None))

    def test_controller_recording_saves_bounded_raw_diagnostics(self) -> None:
        packets = [b"\xC8\xFF" + bytes([index]) * 30 for index in range(3)]
        fake_socket = _FakeUdpSocket(packets)
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                output=Path(temp_dir),
                no_raw=True,
                device_mode="controller",
                diagnostic_samples=2,
                diagnostic_bytes=8,
                ip="127.0.0.1",
                port=15000,
                duration=0.02,
                flush_every=1,
            )
            with patch("mocap_udp_recorder.socket.socket", return_value=fake_socket):
                self.assertEqual(capture_udp(args), 0)

            meta_path = next(Path(temp_dir).glob("*.meta.json"))
            with meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
            self.assertEqual(meta["device_mode"], "controller")
            self.assertEqual(meta["received_packets"], 3)
            self.assertEqual(meta["valid_packets"], 0)
            self.assertEqual(meta["invalid_packets"], 3)
            self.assertEqual(meta["unsupported_samples_saved"], 2)

            diagnostic_path = Path(meta["unsupported_samples_path"])
            raw = diagnostic_path.read_bytes()
            header_size = struct.calcsize(RAW_FRAME_HEADER)
            _timestamp_ns, first_size = struct.unpack_from(RAW_FRAME_HEADER, raw, 0)
            second_offset = header_size + first_size
            _timestamp_ns, second_size = struct.unpack_from(RAW_FRAME_HEADER, raw, second_offset)
            self.assertEqual((first_size, second_size), (32, 32))
            self.assertEqual(len(raw), header_size * 2 + 64)

    def test_csv_timestamp_unit_detection(self) -> None:
        timestamp_us, source = timestamp_us_from_csv(1_783_763_930_244.0)
        self.assertEqual(timestamp_us, 1_783_763_930_244_000)
        self.assertEqual(source, "milliseconds_in_systemTimestampUs")
        timestamp_us, source = timestamp_us_from_csv(1_783_763_930_244_000)
        self.assertEqual(timestamp_us, 1_783_763_930_244_000)
        self.assertEqual(source, "microseconds")


if __name__ == "__main__":
    unittest.main()