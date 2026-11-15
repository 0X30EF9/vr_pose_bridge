from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_launcher_skeleton as flows


class HighRateLauncherTest(unittest.TestCase):
    def setUp(self):
        self.fixture = flows.SkeletonLauncherFlowTest()
        self.launcher = flows.launcher

    def test_live_and_replay_receive_period_and_velocity_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture.config(Path(directory))
            live = self.fixture.invoke(config, ["teleop"])
            for flag, value in (("--control-rate", "100.0"), ("--safe-rate", "100.0"),
                                ("--max-joint-velocity", "0.48"), ("--hand-control-rate", "20.0"),
                                ("--max-hand-velocity", "1.0"), ("--log-rate", "1.0")):
                self.fixture.assert_option(live, flag, value)
            replay = self.fixture.invoke(config, ["replay", "--control-rate", "80", "--max-joint-velocity", "0.3"])
            self.fixture.assert_option(replay, "--rate", "80.0")
            self.fixture.assert_option(replay, "--max-joint-velocity", "0.3")

    def test_small_matrix_thread_caps_are_process_local_and_configurable(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            config = self.fixture.config(Path(directory))
            with patch.object(sys, "argv", ["run.py", "--config", str(config), "--blas-threads", "2", "teleop"]):
                self.launcher.parse_args()
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")

    def test_simulation_dispatch_never_needs_a_robot_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture.config(root)
            command = self.fixture.invoke(config, ["simulate", "--duration", "0.3", "--realtime", "--scenario", "mocap_end_to_end", "--output", str(root / "sim")])
            self.assertIn(str(self.launcher.SOURCE_DIR / "teleop_simulation.py"), command)
            self.fixture.assert_option(command, "--control-hz", "100.0")
            self.fixture.assert_option(command, "--duration", "0.3")
            self.assertIn("--realtime", command)
            self.fixture.assert_option(command, "--scenario", "mocap_end_to_end")
            self.assertNotIn("--host", command)
            self.assertNotIn("--execute", command)

    def test_invalid_timing_is_rejected_before_starting_a_child(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture.config(Path(directory))
            for flag, value in (("--control-rate", "0"), ("--hand-control-rate", "nan"),
                                ("--max-joint-velocity", "-1"), ("--max-hand-velocity", "0"), ("--log-rate", "inf")):
                with (self.subTest(flag=flag), patch.object(sys, "argv", ["run.py", "--config", str(config), "teleop", flag, value]),
                      patch.object(self.launcher.subprocess, "call") as child, contextlib.redirect_stderr(io.StringIO())):
                    with self.assertRaises(SystemExit):
                        self.launcher.main()
                    child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
