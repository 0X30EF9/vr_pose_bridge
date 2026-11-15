from __future__ import annotations

import socket
import unittest

from teleop_hand_ik import hand_ik_request
from teleop_mocap_protocol import MOCAP_G2_JOINT_COUNT, encode_mocap_g2_frame, frame_from_joint_rows, parse_mocap_packet


class UdpLoopbackTest(unittest.TestCase):
    def test_local_g2_udp_uses_canonical_frame_without_publishers(self) -> None:
        rows = [
            (index * 0.001, index * 0.002, 1.0 + index * 0.001, 0.0, 0.0, 0.0, 1.0)
            for index in range(MOCAP_G2_JOINT_COUNT)
        ]
        source = frame_from_joint_rows(rows, timestamp_s=1.25, timestamp_us=1_250_000)
        datagram = encode_mocap_g2_frame(source)

        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1.0)
            sender.sendto(datagram, receiver.getsockname())
            received, address = receiver.recvfrom(4096)
        finally:
            sender.close()
            receiver.close()

        self.assertEqual(address[0], "127.0.0.1")
        frame = parse_mocap_packet(received, receive_time=2.0)
        replay = frame.to_replay_dict()
        hand_request = hand_ik_request(frame, sequence=0)
        self.assertEqual(replay["mocap_frame"], frame)
        self.assertEqual(hand_request["left_points"]["hand"], frame.joint("left_hand").position)
        self.assertEqual(hand_request["right_points"]["hand"], frame.joint("right_hand").position)
        self.assertNotIn("publish", hand_request)
        self.assertNotIn("host", hand_request)
        self.assertNotIn("port", hand_request)


if __name__ == "__main__":
    unittest.main()