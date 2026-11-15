from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


LAUNCHER = Path(__file__).resolve().parents[2] / "run_ik.py"
SPEC = importlib.util.spec_from_file_location("skeleton_launcher_tests", LAUNCHER)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
REPOSITORY = next(parent for parent in LAUNCHER.parents if (parent / "run.py").is_file())


class SkeletonLauncherFlowTest(unittest.TestCase):
    def config(self, root: Path, override: dict | None = None) -> Path:
        recorded = root / "record.csv"
        recorded.touch()
        value = {
            "runtime": {"python": sys.executable, "conda_env": None, "pause_after_action": False},
            "teleop": {"skip_check": True, "robot_command": False},
            "replay": {"input": str(recorded), "execute": False},
            "hand_ik": {"compute": False},
            "hand_control": {"publish": False},
        }
        if override:
            value = launcher.deep_merge(value, override)
        config = root / "config.json"
        config.write_text(json.dumps(value), encoding="utf-8")
        return config

    def invoke(self, config: Path, arguments: list[str]):
        with (
            patch.object(sys, "argv", [str(LAUNCHER), "--config", str(config), *arguments]),
            patch.object(launcher.subprocess, "call", return_value=0) as child,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = launcher.main()
        self.assertEqual(code, 0)
        child.assert_called_once()
        return child.call_args.args[0]

    def assert_option(self, command: list[str], flag: str, expected: str) -> None:
        self.assertEqual(command[command.index(flag) + 1], expected)

    def test_real_cli_dispatches_default_live_and_preview_to_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            for action in ("teleop", "replay"):
                with self.subTest(action=action):
                    command = self.invoke(config, [action])
                    for flag, expected in (
                        ("--control-mode", "skeleton"), ("--ik-backend", "teleop_dls"),
                        ("--ik-task", "pose"), ("--position-scale", "1.0"),
                        ("--orientation-scale", "1.0"), ("--startup-hold-seconds", "2.0"),
                    ):
                        self.assert_option(command, flag, expected)
                    self.assertNotIn("--workspace-calibration", command)
                    self.assertNotIn("--wrist-calibration", command)
            self.assertNotIn("--execute", command)
            self.assert_option(command, "--prepose", "none")

    def test_explicit_dry_run_overrides_an_executing_user_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {"replay": {"execute": True, "use_robot_status": True}})
            command = self.invoke(config, ["replay", "--dry-run"])
            self.assertNotIn("--execute", command)
            self.assertNotIn("--use-robot-status", command)

    def test_relative_replay_input_is_resolved_before_child_changes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            relative = Path("examples") / "skeleton_demo.csv"
            command = self.invoke(config, ["replay", str(relative), "--dry-run"])
            entry_index = command.index(str(launcher.REPLAY_ENTRY))
            self.assertEqual(command[entry_index + 1], str(relative.resolve()))

    def test_disabling_hand_compute_disables_session_hand_publication_without_editing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {"hand_ik": {"compute": True}, "hand_control": {"publish": True}})
            for action in ("teleop", "replay"):
                command = self.invoke(config, [action, "--hand-ik-compute", "false"])
                settings = json.loads(command[command.index("--hand-control-json") + 1])
                self.assertFalse(settings["publish"])
            self.assertTrue(json.loads(config.read_text(encoding="utf-8"))["hand_control"]["publish"])

    def test_menu_self_test_uses_a_valid_full_skeleton_packet(self) -> None:
        from teleop_mocap_protocol import parse_mocap_packet
        from teleop_startup_pose import NaturalDownStartupConfig, check_human_natural_down

        frame = parse_mocap_packet(launcher.build_fake_mocap_packet(), 1.0)
        self.assertEqual(len(frame.joints), 64)
        valid, reasons, _ = check_human_natural_down(frame, NaturalDownStartupConfig())
        self.assertTrue(valid, reasons)

    def test_null_runtime_in_real_cli_uses_the_current_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {"runtime": {"python": None, "conda_env": None}})
            command = self.invoke(config, ["teleop"])
            self.assertEqual(command[0], sys.executable)
            self.assertNotIn("conda", command)

    def test_cli_and_config_calibration_paths_fail_before_starting_any_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for action in ("teleop", "replay"):
                for section in ("workspace_calibration", "wrist_calibration"):
                    for supplied_by in ("cli", "config"):
                        with self.subTest(action=action, section=section, supplied_by=supplied_by):
                            override = {section: {"runtime_path": str(root / "old.json")}} if supplied_by == "config" else None
                            config = self.config(root, override)
                            arguments = [action]
                            if supplied_by == "cli":
                                arguments.extend(("--" + section.replace("_", "-"), str(root / "old.json")))
                            with (
                                patch.object(sys, "argv", [str(LAUNCHER), "--config", str(config), *arguments]),
                                patch.object(launcher.subprocess, "call") as child,
                                patch.object(launcher, "check_environment") as check,
                                contextlib.redirect_stderr(io.StringIO()) as error,
                            ):
                                with self.assertRaises(SystemExit) as raised:
                                    launcher.main()
                            self.assertEqual(raised.exception.code, 2)
                            self.assertIn("不接受人工标定路径", error.getvalue())
                            child.assert_not_called()
                            check.assert_not_called()

    def test_skeleton_rejects_legacy_backend_and_position_only_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            for action in ("teleop", "replay"):
                for extra in (("--ik-backend", "tiangong2pro_official"), ("--ik-task", "position_only")):
                    with (
                        self.subTest(action=action, extra=extra),
                        patch.object(sys, "argv", [str(LAUNCHER), "--config", str(config), action, *extra]),
                        patch.object(launcher.subprocess, "call") as child,
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        with self.assertRaises(SystemExit) as raised:
                            launcher.main()
                        self.assertEqual(raised.exception.code, 2)
                        child.assert_not_called()

    def test_legacy_ik_still_forwards_explicit_calibration_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)
            calibration = root / "legacy.json"
            calibration.write_text("{}", encoding="utf-8")
            for action in ("teleop", "replay"):
                command = self.invoke(config, [action, "--control-mode", "ik", "--ik-backend", "tiangong2pro_official",
                                               "--workspace-calibration", str(calibration)])
                self.assert_option(command, "--control-mode", "ik")
                self.assert_option(command, "--workspace-calibration", str(calibration))

    def test_single_startup_window_parameters_are_forwarded_in_both_flows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), {"teleop": {"startup_hold_seconds": 3.0}})
            for action in ("teleop", "replay"):
                command = self.invoke(config, [action, "--startup-hold-seconds", "1.5"])
                self.assert_option(command, "--startup-hold-seconds", "1.5")
            command = self.invoke(config, ["replay"])
            self.assert_option(command, "--startup-hold-seconds", "3.0")

    def test_menu_starts_skeleton_without_inserting_a_calibration_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            with (
                patch.object(sys, "argv", [str(LAUNCHER), "--config", str(config), "--no-pause", "menu"]),
                patch("builtins.input", side_effect=["4", "1", "1", "0"]),
                patch.object(launcher.subprocess, "call", return_value=0) as child,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(launcher.main(), 0)
            child.assert_called_once()
            self.assert_option(child.call_args.args[0], "--control-mode", "skeleton")

    def test_default_environment_check_does_not_require_casadi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            with (
                patch.object(sys, "argv", [str(LAUNCHER), "--config", str(config), "check", "--teleop"]),
                patch.object(launcher, "check_import", return_value=True) as check,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(launcher.main(), 0)
            modules = [call.args[0] for call in check.call_args_list]
            self.assertIn("pinocchio", modules)
            self.assertNotIn("casadi", modules)

    def test_repository_entry_point_help_runs_without_robot_dependencies(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, str(REPOSITORY / "run.py"), "teleop", "--help"],
                                capture_output=True, text=True, encoding="utf-8", timeout=10, env=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skeleton", result.stdout)


if __name__ == "__main__":
    unittest.main()
