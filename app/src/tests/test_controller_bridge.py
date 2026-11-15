from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np

from teleop_ik_types import IKCommand
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
from vr_controller_bridge import ControllerCommandGate, parse_args, publish_controller_command


def command(q, side="both"):
    return IKCommand(1., side, list(ARM_JOINT_NAMES), np.asarray(q, dtype=float), list(ARM_MOTOR_IDS))


class ControllerPublicationGateTest(unittest.TestCase):
    def setUp(self):
        self.q = np.zeros(14)
        self.gate = ControllerCommandGate(self.q)

    def test_rate_is_independent_of_physical_motion_and_does_not_bank_time(self):
        first = self.gate.prepare(command(self.q+.1), self.q, 1., 1., 1.)
        np.testing.assert_allclose(first.positions, .0048)
        self.gate.commit(first, 1.)
        early = self.gate.prepare(command(self.q+.1), first.positions, 1., 1., 1.001)
        np.testing.assert_allclose(early.positions, .0048+.00048)
        late = self.gate.prepare(command(self.q+.1), first.positions, 10., 10., 10.)
        np.testing.assert_allclose(late.positions, .0096)

    def test_inactive_committed_target_survives_other_arm_publications(self):
        right_q = self.q.copy()
        for step in range(1, 6):
            target = right_q.copy()
            target[7:] += .004
            checked = self.gate.prepare(command(target, "right"), right_q, step, step, step)
            self.gate.commit(checked, step)
            right_q = checked.positions
        left = self.gate.prepare(command(self.q+.1, "left"), self.q, 6., 6., 6.)
        self.gate.commit(left, 6.)
        np.testing.assert_allclose(self.gate.reference(self.q)[7:], .02)
        with self.assertRaises(RuntimeError):
            self.gate.prepare(command(self.q, "right"), self.q, 7., 7., 7.)

    def test_stagnant_feedback_prevents_unconfirmed_motion_accumulation(self):
        for step in range(10):
            now = 1.+step*.01
            checked = self.gate.prepare(command(self.q+.1), self.q, now, now, now)
            self.gate.commit(checked, now)
            self.assertLessEqual(float(np.max(checked.positions)), .0048+1e-12)

    def test_baseline_and_inputs_are_copied_and_limits_remain_fixed(self):
        initial = self.q.copy()
        gate = ControllerCommandGate(initial, max_from_baseline=.01)
        initial[:] = .5
        candidate = command(self.q+.05)
        before = candidate.positions.copy()
        checked = gate.prepare(candidate, self.q+.009, 1., 1., 1.)
        np.testing.assert_allclose(checked.positions, .01)
        np.testing.assert_array_equal(candidate.positions, before)
        np.testing.assert_array_equal(gate.baseline, self.q)

    def test_invalid_schema_feedback_limits_and_stale_input_cannot_publish(self):
        for received, source, now in ((0., 1., 1.), (1., 0., 1.), (2., 1., 1.), (1., 2., 1.)):
            with self.subTest(received=received, source=source), self.assertRaises(RuntimeError):
                self.gate.prepare(command(self.q), self.q, received, source, now)
        for invalid in (replace(command(self.q), waist_position=0.),
                        replace(command(self.q), motor_ids=list(reversed(ARM_MOTOR_IDS))),
                        command(np.full(14, np.nan)), command(np.full(14, 20.))):
            with self.assertRaises(ValueError):
                self.gate.prepare(invalid, self.q, 1., 1., 1.)
        left = ControllerCommandGate(self.q, control_side="left")
        with self.assertRaises(ValueError):
            left.prepare(command(self.q, "right"), self.q, 1., 1., 1.)

    def test_failed_send_does_not_commit_any_history(self):
        publish = Mock(side_effect=RuntimeError("disconnected during send"))
        with self.assertRaises(RuntimeError):
            publish_controller_command(self.gate, command(self.q+.1, "left"), 1.,
                                       lambda: (1., self.q), publish, clock=lambda: 1.)
        publish.assert_called_once()
        np.testing.assert_array_equal(self.gate.reference(self.q), self.q)
        self.assertFalse(self.gate._sent["left"])

    def test_feedback_expiring_at_actual_send_boundary_stops_publication(self):
        publish = Mock()
        clock = iter((1., 1., 1.6))
        with self.assertRaises(RuntimeError):
            publish_controller_command(self.gate, command(self.q), 1., lambda: (1., self.q),
                                       publish, clock=lambda: next(clock))
        publish.assert_not_called()
        self.assertFalse(self.gate._sent["left"])

    def test_dynamic_side_sends_only_seven_motors_and_commits_after_success(self):
        publish = Mock()
        checked = publish_controller_command(self.gate, command(self.q+.1, "left"), 1.,
                                            lambda: (1., self.q), publish, clock=lambda: 1.)
        self.assertEqual([motor["name"] for motor in publish.call_args.args[0]["cmds"]], list(ARM_MOTOR_IDS[:7]))
        np.testing.assert_array_equal(checked.positions[7:], self.q[7:])
        self.assertTrue(self.gate._sent["left"])
        self.assertFalse(self.gate._sent["right"])

    def test_preview_is_default_and_robot_modes_cannot_use_synthetic_feedback(self):
        args = parse_args([])
        self.assertFalse(args.execute)
        self.assertFalse(args.preview_robot)
        with self.assertRaises(SystemExit):
            parse_args(["--execute", "--initial-q", *["0"]*14])


if __name__ == "__main__":
    unittest.main()
