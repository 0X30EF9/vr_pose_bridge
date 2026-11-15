import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_launcher_skeleton as flows


class ControllerLauncherTest(unittest.TestCase):
    def test_controller_defaults_to_preview_even_with_existing_robot_execution_preferences(self):
        fixture = flows.SkeletonLauncherFlowTest()
        with tempfile.TemporaryDirectory() as temporary:
            config = fixture.config(Path(temporary), {"teleop": {"robot_command": True}, "replay": {"execute": True}})
            command = fixture.invoke(config, ["controller", "--duration", "1"])
            self.assertIn(str(flows.launcher.SOURCE_DIR / "vr_controller_bridge.py"), command)
            fixture.assert_option(command, "--control-rate", "100.0")
            fixture.assert_option(command, "--udp-port", "15010")
            self.assertNotIn("--execute", command)
            self.assertNotIn("--preview-robot", command)
            command = fixture.invoke(config, ["controller", "--execute", "--ik-task", "position_only", "--host", "test-host"])
            self.assertIn("--execute", command)
            fixture.assert_option(command, "--ik-task", "position_only")
            fixture.assert_option(command, "--host", "test-host")

    def test_controller_simulation_forwards_coordinate_and_scene_without_robot_args(self):
        fixture = flows.SkeletonLauncherFlowTest()
        with tempfile.TemporaryDirectory() as temporary:
            config = fixture.config(Path(temporary))
            command = fixture.invoke(config, ["controller-simulate", "--coordinate-system", "unity", "--scenario", "translation_xyz", "--realtime"])
            fixture.assert_option(command, "--coordinate-system", "unity")
            fixture.assert_option(command, "--scenario", "translation_xyz")
            self.assertIn("--realtime", command)
            self.assertNotIn("--host", command)
            self.assertNotIn("--execute", command)

    def test_menu_supplies_configuration_for_each_simulation_and_controller_entry(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ"):
            fixture = flows.SkeletonLauncherFlowTest()
            config = fixture.config(Path(temporary))
            with patch.object(sys, "argv", ["run.py", "--config", str(config), "menu"]):
                base = flows.launcher.parse_args()
            for name in ("simulate", "controller", "controller-simulate"):
                args = flows.launcher.make_menu_args(base, name)
                with patch.object(flows.launcher.subprocess, "call", return_value=0) as child, contextlib.redirect_stdout(io.StringIO()):
                    runner = {"simulate": flows.launcher.run_simulation,
                              "controller": flows.launcher.run_controller,
                              "controller-simulate": flows.launcher.run_controller_simulation}[name]
                    self.assertEqual(runner(args), 0)
                child.assert_called_once()


if __name__ == "__main__":
    unittest.main()
