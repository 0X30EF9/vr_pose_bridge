from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from teleop_joint_schema import ARM_JOINT_NAMES, HAND_JOINT_ORDER
from visualize_joint_commands import build_report, latest_session, load_robot_model, load_samples, render_html


def log_line(record: dict) -> str:
    return "2026-07-14 10:00:00,000 INFO " + json.dumps(record) + "\n"


class JointCommandVisualizationTest(unittest.TestCase):
    def test_loads_sent_preview_and_local_hand_records(self) -> None:
        records = [
            {
                "section": "robot_command",
                "frame": 1,
                "time": 100.0,
                "data": {
                    "enabled": True,
                    "control_side": "both",
                    "topic": "/arm/cmd_pos",
                    "positions": [index / 10 for index in range(14)],
                    "message": {"cmds": [{"name": motor_id} for motor_id in (*range(11, 18), *range(21, 28))]},
                },
            },
            {
                "section": "robot_command",
                "frame": 2,
                "time": 100.1,
                "data": {
                    "enabled": False,
                    "control_side": "right",
                    "positions": [index / 10 for index in range(7)],
                },
            },
            {
                "section": "hand_ik_local_result",
                "frame": 2,
                "time": 100.2,
                "data": {
                    "joint_order": list(HAND_JOINT_ORDER),
                    "joint_positions": [index / 20 for index in range(12)],
                    "published": False,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.log"
            path.write_text("ignored line\n" + "".join(log_line(record) for record in records), encoding="utf-8")
            samples, invalid = load_samples(path)

        self.assertEqual(invalid, 0)
        self.assertEqual([sample.channel for sample in samples], ["arm_sent", "arm_preview", "hand_local"])
        self.assertEqual(samples[0].names, ARM_JOINT_NAMES)
        self.assertEqual(samples[1].names, ARM_JOINT_NAMES[7:])
        self.assertEqual(samples[2].names, HAND_JOINT_ORDER)
        self.assertTrue(samples[0].published)
        self.assertFalse(samples[1].published)

    def test_latest_session_and_html_are_self_contained(self) -> None:
        records = [
            {
                "section": "robot_command",
                "frame": frame,
                "time": timestamp,
                "data": {"enabled": True, "control_side": "left", "positions": [float(frame)] * 7},
            }
            for frame, timestamp in ((1, 10.0), (2, 10.1), (1, 30.0), (2, 30.1))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            samples, invalid = load_samples(path)
            selected = latest_session(samples, 10.0)
            report = build_report(selected, source=path, invalid_relevant=invalid, max_points=100)
            html = render_html(report)

        self.assertEqual([sample.frame for sample in selected], [1, 2])
        self.assertEqual(report["meta"]["arm_sent_count"], 2)
        self.assertIn("const REPORT=", html)
        self.assertIn("shoulder_pitch_l_joint", html)
        self.assertIn('id="robotCanvas"', html)
        self.assertIn('data-view="front"', html)
        self.assertIn('data-view="side"', html)
        self.assertIn('data-view="top"', html)
        self.assertIn("X 前 / Y 左 / Z 上", html)
        self.assertIn("function classifyPose", html)
        self.assertIn("目标→最终 FK 误差", html)
        self.assertIn("function forwardChain", html)
        self.assertNotIn("__REPORT_DATA__", html)
        self.assertNotIn("https://", html)

    def test_robot_model_is_extracted_from_urdf_as_two_continuous_chains(self) -> None:
        model = load_robot_model()

        self.assertEqual(model["base_link"], "waist_yaw_link")
        self.assertEqual([joint["name"] for joint in model["chains"]["left"]], list(ARM_JOINT_NAMES[:7]))
        self.assertEqual([joint["name"] for joint in model["chains"]["right"]], list(ARM_JOINT_NAMES[7:]))
        self.assertEqual(model["chains"]["left"][0]["origin"], [0.0, 0.14539, 0.37721])
        self.assertEqual(model["chains"]["right"][0]["origin"], [0.0, -0.14539, 0.37721])
        self.assertEqual(model["chains"]["left"][0]["axis"], [0.0, 1.0, 0.0])
        self.assertLess(
            model["chains"]["left"][0]["lower"],
            model["chains"]["left"][0]["upper"],
        )

    def test_direct_retarget_diagnostics_are_attached_to_command_frame(self) -> None:
        residual = {"left": {"upper_angle_rad": 0.1, "forearm_angle_rad": 0.2}}
        records = [
            {
                "section": "direct_retarget_result",
                "frame": 7,
                "time": 99.9,
                "data": {
                    "meta": {
                        "limited": True,
                        "solver_meta": {"iterations": 6, "residual_by_side": residual},
                    }
                },
            },
            {
                "section": "robot_command",
                "frame": 7,
                "time": 100.0,
                "data": {"enabled": True, "control_side": "both", "positions": [0.0] * 14},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.log"
            path.write_text("".join(log_line(record) for record in records), encoding="utf-8")
            samples, invalid = load_samples(path)
            report = build_report(samples, source=path, invalid_relevant=invalid, max_points=100)

        self.assertEqual(invalid, 0)
        self.assertEqual(samples[0].diagnostics["solver_meta"]["iterations"], 6)
        self.assertEqual(report["tracks"][0]["samples"][0][3]["solver_meta"]["residual_by_side"], residual)

    def test_invalid_relevant_record_is_counted(self) -> None:
        record = {
            "section": "robot_command",
            "frame": 1,
            "time": 1.0,
            "data": {"enabled": True, "control_side": "both", "positions": [float("nan")] * 14},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.log"
            path.write_text(log_line(record), encoding="utf-8")
            samples, invalid = load_samples(path)
        self.assertEqual(samples, [])
        self.assertEqual(invalid, 1)


if __name__ == "__main__":
    unittest.main()