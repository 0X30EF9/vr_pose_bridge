from __future__ import annotations

import copy
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()