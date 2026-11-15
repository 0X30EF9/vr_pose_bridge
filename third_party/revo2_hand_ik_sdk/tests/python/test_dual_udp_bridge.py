from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from revo2_hand_ik.fss_udp import CRC_STRUCT, HEADER_STRUCT, MOTION_FRAME_ID, PAYLOAD_WITH_TIMESTAMP_BYTES, fss_crc32  # noqa: E402
from revo2_hand_ik.udp_bridge import FssUdpDualHandIkBridge  # noqa: E402


def make_packet() -> bytes:
    payload = bytearray(PAYLOAD_WITH_TIMESTAMP_BYTES)
    struct.pack_into("<I", payload, 0, 12345)
    packet = bytearray(HEADER_STRUCT.pack(0xAA, 0x55, MOTION_FRAME_ID, len(payload)) + payload)
    packet += CRC_STRUCT.pack(fss_crc32(packet))
    return bytes(packet)


class FakeSdk:
    def __init__(self, side: str, values: tuple[float, float, float, float, float, float]):
        self.side = side
        self.values = values
        self.calls: list[tuple[int, int]] = []

    def solve(self, points: dict, quats: dict, *, sequence: int, timestamp_ms: int) -> dict:
        del points, quats
        self.calls.append((sequence, timestamp_ms))
        names = ("thumb_flex_joint", "thumb_abduct_joint", "index_joint", "middle_joint", "ring_joint", "pinky_joint")
        return {"joints": [{"name": f"{self.side}_{name}", "rad": value} for name, value in zip(names, self.values)], "debug": {}}


class DirectBridgeTest(unittest.TestCase):
    def test_one_input_frame_generates_exact_12_channel_output(self) -> None:
        bridge = FssUdpDualHandIkBridge.__new__(FssUdpDualHandIkBridge)
        bridge.udp = SimpleNamespace(check_crc=True, role_name="Revo2HandIK")
        bridge.received = 42
        bridge.left_sdk = FakeSdk("left", (10, 60, 20, 30, 40, 50))
        bridge.right_sdk = FakeSdk("right", (11, 61, 21, 31, 41, 51))
        bridge.recorder = None

        payload = json.loads(bridge.process_datagram(make_packet()).decode("utf-8"))
        self.assertEqual(payload["joint_positions"], [10.0, 60.0, 20.0, 30.0, 40.0, 50.0, 11.0, 61.0, 21.0, 31.0, 41.0, 51.0])
        self.assertEqual(payload["sequence"], 42)
        self.assertEqual(payload["timestamp_ms"], 12345)
        self.assertEqual(bridge.left_sdk.calls, [(42, 12345)])
        self.assertEqual(bridge.right_sdk.calls, [(42, 12345)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
