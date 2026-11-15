from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


RUN_IK_PATH = Path(__file__).resolve().parents[2] / "run_ik.py"
RUN_IK_SPEC = importlib.util.spec_from_file_location("run_ik", RUN_IK_PATH)
if RUN_IK_SPEC is None or RUN_IK_SPEC.loader is None:
    raise RuntimeError(f"无法加载统一入口: {RUN_IK_PATH}")
run_ik = importlib.util.module_from_spec(RUN_IK_SPEC)
RUN_IK_SPEC.loader.exec_module(run_ik)


class CombinedCalibrationWorkflowTest(unittest.TestCase):
    def make_args(self, root: Path) -> Namespace:
        config = copy.deepcopy(run_ik.DEFAULT_CONFIG)
        config["runtime"]["python"] = sys.executable
        config["runtime"]["conda_env"] = None
        config["wrist_calibration"]["capture_output"] = str(root / "wrist" / "raw")
        config["wrist_calibration"]["calibration_output"] = str(root / "wrist" / "calibration.json")
        config["workspace_calibration"]["capture_output"] = str(root / "workspace" / "raw")
        config["workspace_calibration"]["calibration_output"] = str(root / "workspace" / "calibration.json")
        return Namespace(
            command="calibrate",
            launcher_config=config,
            python=sys.executable,
            conda_env=None,
            device_mode=None,
            ip=None,
            port=None,
        )

    def test_success_runs_wrist_before_workspace_and_publishes_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.make_args(root)
            calls: list[str] = []

            def fake_wrist(stage_args: Namespace) -> int:
                calls.append("wrist")
                stage_args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
                stage_args.calibration_output.write_text('{"valid": true}', encoding="utf-8")
                return 0

            def fake_workspace(stage_args: Namespace) -> int:
                calls.append("workspace")
                self.assertEqual(stage_args.wrist_calibration, root / "wrist" / "calibration.json")
                self.assertNotEqual(stage_args.calibration_output, root / "workspace" / "calibration.json")
                stage_args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
                stage_args.calibration_output.write_text('{"valid": true}', encoding="utf-8")
                return 0

            with (
                patch.object(run_ik, "run_wrist_calibration", side_effect=fake_wrist),
                patch.object(run_ik, "run_workspace_calibration", side_effect=fake_workspace),
            ):
                code = run_ik.run_combined_calibration(args)

            final_path = root / "workspace" / "calibration.json"
            self.assertEqual(code, 0)
            self.assertEqual(calls, ["wrist", "workspace"])
            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8")), {"valid": True})
            self.assertEqual(list(final_path.parent.glob(".*.candidate.json")), [])

    def test_reuses_valid_direct_wrist_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.make_args(root)
            wrist_path = root / "wrist" / "calibration.json"
            wrist_path.parent.mkdir(parents=True, exist_ok=True)
            wrist_path.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "control_point": "anatomical_wrist",
                        "sides": {
                            side: {"valid": True, "position_source": "wrist_joint"}
                            for side in ("left", "right")
                        },
                    }
                ),
                encoding="utf-8",
            )
            stages: list[str] = []

            def fake_wrist(stage_args: Namespace) -> int:
                stages.append(stage_args.stage)
                self.assertEqual(stage_args.input, wrist_path)
                return 0

            def fake_workspace(stage_args: Namespace) -> int:
                stage_args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
                stage_args.calibration_output.write_text('{"valid": true}', encoding="utf-8")
                return 0

            with (
                patch.object(run_ik, "run_wrist_calibration", side_effect=fake_wrist),
                patch.object(run_ik, "run_workspace_calibration", side_effect=fake_workspace),
            ):
                code = run_ik.run_combined_calibration(args)

            self.assertEqual(code, 0)
            self.assertEqual(stages, ["validate"])

    def test_wrist_failure_blocks_workspace_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(Path(tmp))
            with (
                patch.object(run_ik, "run_wrist_calibration", return_value=2),
                patch.object(run_ik, "run_workspace_calibration") as workspace,
            ):
                code = run_ik.run_combined_calibration(args)

            self.assertEqual(code, 2)
            workspace.assert_not_called()

    def test_workspace_failure_preserves_existing_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.make_args(root)
            wrist_path = root / "wrist" / "calibration.json"
            final_path = root / "workspace" / "calibration.json"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text('{"version": "trusted"}', encoding="utf-8")

            def fake_wrist(stage_args: Namespace) -> int:
                wrist_path.parent.mkdir(parents=True, exist_ok=True)
                wrist_path.write_text('{"valid": true}', encoding="utf-8")
                return 0

            def fake_workspace(stage_args: Namespace) -> int:
                stage_args.calibration_output.write_text('{"valid": false}', encoding="utf-8")
                return 2

            with (
                patch.object(run_ik, "run_wrist_calibration", side_effect=fake_wrist),
                patch.object(run_ik, "run_workspace_calibration", side_effect=fake_workspace),
            ):
                code = run_ik.run_combined_calibration(args)

            self.assertEqual(code, 2)
            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8")), {"version": "trusted"})
            self.assertEqual(list(final_path.parent.glob(".*.candidate.json")), [])

    def test_invalid_candidate_preserves_existing_final_file_even_after_child_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.make_args(root)
            final_path = root / "workspace" / "calibration.json"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text('{"version": "trusted"}', encoding="utf-8")

            def invalid_workspace(stage_args: Namespace) -> int:
                stage_args.calibration_output.write_text('{"valid": false}', encoding="utf-8")
                return 0

            with (
                patch.object(run_ik, "run_wrist_calibration", return_value=0),
                patch.object(run_ik, "run_workspace_calibration", side_effect=invalid_workspace),
            ):
                self.assertEqual(run_ik.run_combined_calibration(args), 2)
            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8")), {"version": "trusted"})
            self.assertEqual(list(final_path.parent.glob(".*.candidate.json")), [])


class RuntimeCalibrationConfigurationTest(unittest.TestCase):
    def launcher_args(
        self, root: Path, command: str, extra: tuple[str, ...] = (), runtime_paths=None,
    ) -> Namespace:
        config = {
            "runtime": {"python": sys.executable, "conda_env": None},
            "workspace_calibration": {"calibration_output": "old-workspace.json"},
            "wrist_calibration": {"calibration_output": "old-wrist.json"},
        }
        if runtime_paths is not None:
            for section, path in runtime_paths.items():
                config[section]["runtime_path"] = str(path)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        argv = [str(RUN_IK_PATH), "--config", str(config_path), command, *extra]
        if command == "teleop":
            argv.append("--skip-check")
        with patch.object(sys, "argv", argv):
            return run_ik.apply_config(run_ik.parse_args())

    def test_legacy_configs_do_not_implicitly_load_calibration_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for command in ("teleop", "replay"):
                with self.subTest(command=command):
                    args = self.launcher_args(Path(tmp), command)
                    self.assertIsNone(args.workspace_calibration)
                    self.assertIsNone(args.wrist_calibration)
                    self.assertEqual(
                        args.launcher_config["workspace_calibration"]["calibration_output"], "old-workspace.json",
                    )

    def test_runtime_paths_are_forwarded_and_cli_overrides_the_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configured = root / "configured.json"
            explicit = root / "explicit.json"
            configured.write_text("{}", encoding="utf-8")
            explicit.write_text("{}", encoding="utf-8")
            paths = {"workspace_calibration": configured, "wrist_calibration": configured}
            for command in ("teleop", "replay"):
                overrides = ("--workspace-calibration", str(explicit), "--wrist-calibration", str(explicit))
                for extra, expected in (((), configured), (overrides, explicit)):
                    with self.subTest(command=command, explicit=bool(extra)):
                        args = self.launcher_args(root, command, extra, paths)
                        with (
                            patch.object(run_ik, "run_child", return_value=0) as child,
                            patch.object(run_ik, "resolve_replay_input", return_value=root / "record.csv"),
                        ):
                            self.assertEqual(getattr(run_ik, "run_" + command)(args), 0)
                        child_command = child.call_args.args[0]
                        for flag in ("--workspace-calibration", "--wrist-calibration"):
                            self.assertEqual(child_command[child_command.index(flag) + 1], str(expected))

    def test_explicit_missing_calibration_is_rejected_before_starting_a_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for command in ("teleop", "replay"):
                for flag in ("--workspace-calibration", "--wrist-calibration"):
                    with self.subTest(command=command, flag=flag):
                        args = self.launcher_args(root, command, (flag, str(root / "missing.json")))
                        with (
                            patch.object(run_ik, "run_child") as child,
                            patch.object(run_ik, "resolve_replay_input", return_value=root / "record.csv"),
                        ):
                            with self.assertRaises(FileNotFoundError):
                                getattr(run_ik, "run_" + command)(args)
                            child.assert_not_called()

    def test_workspace_guide_uses_the_normal_launcher_without_device_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self.launcher_args(Path(tmp), "workspace-calibrate", ("--stage", "guide"))
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, str(RUN_IK_PATH), "--config", str(args.config),
                 "workspace-calibrate", "--stage", "guide"],
                capture_output=True, text=True, encoding="utf-8", timeout=10, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Home", result.stdout)

    def test_combined_cli_uses_output_paths_without_implicitly_enabling_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.launcher_args(root, "calibrate")
            config = json.loads(args.config.read_text(encoding="utf-8"))
            config["wrist_calibration"]["calibration_output"] = str(root / "wrist.json")
            config["workspace_calibration"]["calibration_output"] = str(root / "workspace.json")
            args.config.write_text(json.dumps(config), encoding="utf-8")
            stages = []

            def finish_stage(stage_args: Namespace) -> int:
                stages.append(stage_args.command)
                stage_args.calibration_output.write_text('{"valid": true}', encoding="utf-8")
                return 0

            with (
                patch.object(sys, "argv", [str(RUN_IK_PATH), "--config", str(args.config), "calibrate"]),
                patch.object(run_ik, "run_wrist_calibration", side_effect=finish_stage),
                patch.object(run_ik, "run_workspace_calibration", side_effect=finish_stage),
            ):
                self.assertEqual(run_ik.main(), 0)
            self.assertEqual(stages, ["wrist-calibrate", "workspace-calibrate"])
            self.assertTrue((root / "workspace.json").is_file())
            configured = run_ik.load_config(args.config)
            self.assertIsNone(configured["workspace_calibration"]["runtime_path"])
            self.assertIsNone(configured["wrist_calibration"]["runtime_path"])

    def test_wrist_validation_uses_the_normal_launcher_without_device_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "wrist.json"
            quality = {
                "sample_count": 60, "group_count": 3, "rotation_excitation_rad": 0.5,
                "rotation_rank": 2, "outlier_fraction": 0.0,
                "position_observation": "explicit_wrist_joint", "hand_wrist_shared_origin": True,
                "hand_wrist_origin_rms_m": 0.0, "hand_wrist_origin_p95_m": 0.0,
            }
            payload = {
                "schema_version": 1, "valid": True, "control_point": "anatomical_wrist",
                "sides": {
                    side: {"valid": True, "position_source": "wrist_joint",
                           "hand_to_wrist_offset": [0.0, 0.0, 0.0], "quality": quality}
                    for side in ("left", "right")
                },
            }
            calibration_path.write_text(json.dumps(payload), encoding="utf-8")
            args = self.launcher_args(root, "wrist-calibrate", ("--stage", "validate", "--input", str(calibration_path)))
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, str(RUN_IK_PATH), "--config", str(args.config),
                 "wrist-calibrate", "--stage", "validate", "--input", str(calibration_path)],
                capture_output=True, text=True, encoding="utf-8", timeout=10, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("WRIST_CALIBRATION_VALID", result.stdout)


if __name__ == "__main__":
    unittest.main()
