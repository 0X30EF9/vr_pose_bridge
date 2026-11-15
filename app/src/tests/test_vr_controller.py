from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop_ik_types import IKCommand
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS
from teleop_vr_controller import (
    VrControllerConfig, VrControllerRetargeter, controller_basis,
    parse_controller_packet, pose_in_robot,
)


def packet(sequence=1, *, left=None, right=None, coordinate_system="robot", timestamp=None):
    def pose(changes):
        result = {"position": [0., 0., 0.], "rotation_xyzw": [0., 0., 0., 1.],
                  "tracked": True, "enabled": False, "activation_id": 0}
        result.update(changes or {})
        return result
    return {"version": 1, "type": "vr_controller", "sequence": sequence,
            "timestamp": sequence * .01 if timestamp is None else timestamp,
            "coordinate_system": coordinate_system, "left": pose(left), "right": pose(right)}


class FakeCartesianSolver:
    def __init__(self):
        self.joint_limits = [(-2., 2.)] * 14
        self.home_calls, self.requests = [], []
        self.reason = "ok"

    def set_home_q(self, q):
        self.home_calls.append(q.copy())

    def end_effector_poses(self, q):
        result = []
        for offset, lateral in ((0, .2), (7, -.2)):
            pose = np.eye(4)
            pose[:3, 3] = np.array([.4, lateral, .8]) + q[offset:offset+3]
            pose[:3, :3] = Rotation.from_euler("xyz", q[offset+3:offset+6]).as_matrix()
            result.append(pose)
        return result

    def solve_cartesian(self, request):
        self.requests.append(request)
        q = request.current_q.copy()
        indices = slice(None) if request.control_side == "both" else slice(0, 7) if request.control_side == "left" else slice(7, 14)
        q[indices] += .1
        return SimpleNamespace(reason=self.reason, converged=False, command=IKCommand(
            request.timestamp, request.control_side, list(ARM_JOINT_NAMES), q, list(ARM_MOTOR_IDS)))


class ControllerProtocolTest(unittest.TestCase):
    def test_host_receipt_is_not_taken_from_packet(self):
        data = packet()
        data["received_at"] = 999999.
        frame = parse_controller_packet(json.dumps(data).encode(), received_at=100.)
        self.assertEqual(frame.received_at, 100.)
        self.assertEqual(frame.sequence, 1)

    def test_invalid_schema_numeric_flags_or_pose_is_rejected(self):
        mutations = (
            lambda p: p.update(sequence=True), lambda p: p.update(sequence=1.0),
            lambda p: p.update(version=True), lambda p: p.update(timestamp=float("nan")),
            lambda p: p.update(coordinate_system=[]), lambda p: p.update(type="c8ff"),
            lambda p: p["left"].update(tracked=1), lambda p: p["left"].update(enabled="true"),
            lambda p: p["left"].update(position=[True, 0, 0]),
            lambda p: p["left"].update(rotation_xyzw=[0, 0, 0, 0]),
            lambda p: p["left"].update(rotation_xyzw=[0, 0, 0, 2]),
            lambda p: p["left"].update(position=[float("inf"), 0, 0]),
            lambda p: p["left"].update(activation_id=-1), lambda p: p.pop("right"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = packet()
                mutation(data)
                with self.assertRaises(ValueError):
                    parse_controller_packet(data, 100.)
        for data in (b"\xc8\xff", "[]", '{"version":1,"version":1}', " " * 16385):
            with self.assertRaises(ValueError):
                parse_controller_packet(data, 100.)

    def test_basis_translation_and_rotation_include_unity_reflection(self):
        for system, expected in (("unity", [3, -1, 2]), ("openxr", [-3, -1, 2]), ("robot", [1, 2, 3])):
            with self.subTest(system=system):
                raw_rotation = Rotation.from_euler("xyz", [.3, -.2, .4]).as_matrix()
                frame = parse_controller_packet(packet(coordinate_system=system, left={
                    "position": [1, 2, 3], "rotation_xyzw": Rotation.from_matrix(raw_rotation).as_quat().tolist(),
                }), 100.)
                transformed = pose_in_robot(frame.left, system)
                basis = controller_basis(system)
                np.testing.assert_allclose(transformed[:3, 3], expected)
                np.testing.assert_allclose(transformed[:3, :3] @ (basis @ np.array([.1, .2, .3])),
                                           basis @ (raw_rotation @ np.array([.1, .2, .3])), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(transformed[:3, :3]), 1.)
        self.assertAlmostEqual(np.linalg.det(controller_basis("unity")), -1.)


class ControllerRetargetTest(unittest.TestCase):
    def setUp(self):
        self.solver = FakeCartesianSolver()
        self.mapper = VrControllerRetargeter(solver=self.solver)
        self.q = np.zeros(14)

    def frame(self, sequence, **kwargs):
        received_at = kwargs.pop("received_at", 100. + sequence * .01)
        return parse_controller_packet(packet(sequence, **kwargs), received_at)

    def update(self, frame, q=None, previous=None, now=None):
        return self.mapper.update(frame, self.q if q is None else q, previous,
                                  frame.received_at if now is None else now)

    def test_first_clutch_frame_is_exactly_measured_tcp_and_q_without_solving(self):
        self.q[:] = .1
        command = self.update(self.frame(1, left={"enabled": True}, right={"enabled": True}))
        np.testing.assert_array_equal(command.positions, self.q)
        self.assertEqual(self.solver.requests, [])
        self.assertEqual(command.control_side, "both")
        for side, measured in zip(("left", "right"), self.solver.end_effector_poses(self.q)):
            np.testing.assert_array_equal(command.meta["target_pose_by_side"][side], measured)
            self.assertTrue(command.meta["reanchored_by_side"][side])

    def test_cartesian_delta_and_dynamic_inactive_side_are_preserved(self):
        self.update(self.frame(1, left={"enabled": True, "position": [2., 3., 4.]}))
        command = self.update(self.frame(2, left={"enabled": True, "position": [2.1, 3.2, 4.3],
                                               "rotation_xyzw": Rotation.from_euler("z", .2).as_quat().tolist()}))
        self.assertEqual(command.control_side, "left")
        self.assertEqual(command.meta["active_sides"], ("left",))
        target = command.meta["target_pose_by_side"]["left"]
        np.testing.assert_allclose(target[:3, 3], [.5, .4, 1.1])
        np.testing.assert_allclose(target[:3, :3], Rotation.from_euler("z", .2).as_matrix(), atol=1e-12)
        np.testing.assert_allclose(command.positions[:7], .0048)
        np.testing.assert_array_equal(command.positions[7:], self.q[7:])

    def test_release_and_reanchor_do_not_move_or_reset_session_baseline(self):
        self.update(self.frame(1, left={"enabled": True}))
        self.assertIsNone(self.update(self.frame(2)))
        moved_q = np.full(14, .2)
        command = self.update(self.frame(3, left={"enabled": True, "position": [20., 0., 0.]}), q=moved_q)
        np.testing.assert_array_equal(command.positions, moved_q)
        np.testing.assert_array_equal(self.mapper.session_baseline_q, np.zeros(14))
        self.assertEqual(len(self.solver.home_calls), 1)

    def test_tracking_loss_is_per_side_and_activation_change_cannot_unlock_it(self):
        self.update(self.frame(1, left={"enabled": True, "activation_id": 1}, right={"enabled": True}))
        for index, changes in ((2, {"tracked": False, "enabled": True, "activation_id": 1}),
                               (3, {"enabled": True, "activation_id": 2})):
            command = self.update(self.frame(index, left=changes, right={"enabled": True}))
            self.assertEqual(command.control_side, "right")
        self.update(self.frame(4, left={"enabled": False, "activation_id": 2}, right={"enabled": True}))
        command = self.update(self.frame(5, left={"enabled": True, "activation_id": 3}, right={"enabled": True}))
        self.assertEqual(command.control_side, "both")
        self.assertTrue(command.meta["reanchored_by_side"]["left"])
        np.testing.assert_array_equal(command.positions[:7], self.q[:7])
        self.assertEqual(self.solver.requests[-1].control_side, "right")

    def test_activation_counter_recovers_a_dropped_release_with_zero_motion(self):
        self.update(self.frame(1, left={"enabled": True, "activation_id": 1}))
        command = self.update(self.frame(2, left={"enabled": True, "activation_id": 2, "position": [5., 0., 0.]}))
        np.testing.assert_array_equal(command.positions, self.q)
        self.assertTrue(command.meta["reanchored_by_side"]["left"])
        self.assertEqual(self.solver.requests, [])

    def test_ingest_batch_keeps_release_edges_without_solving_intermediate_packets(self):
        self.update(self.frame(1, left={"enabled": True}))
        self.mapper.ingest(self.frame(2), now=100.02)
        final = self.frame(3, left={"enabled": True, "position": [10., 0., 0.]})
        self.mapper.ingest(final, now=final.received_at)
        self.assertEqual(self.solver.requests, [])
        command = self.update(final)
        np.testing.assert_array_equal(command.positions, self.q)
        self.assertTrue(command.meta["reanchored_by_side"]["left"])

    def test_duplicate_reverse_and_stopped_source_time_cannot_refresh_receipt(self):
        first = self.frame(1, received_at=100., left={"enabled": True})
        self.update(first)
        for data in (self.frame(1, received_at=100.29, left={"enabled": True}),
                     self.frame(0, received_at=100.29),
                     self.frame(2, received_at=100.29, timestamp=first.timestamp)):
            self.assertFalse(self.mapper.ingest(data, now=100.29))
            self.assertEqual(self.mapper.latest_frame.received_at, 100.)
        self.assertIsNone(self.update(first, now=100.31))
        self.assertIsNone(self.mapper.latest_frame)

    def test_long_receipt_gap_requires_a_release_then_zero_motion_press(self):
        self.update(self.frame(1, received_at=1., left={"enabled": True, "activation_id": 1}))
        self.assertIsNone(self.update(self.frame(2, received_at=1.5, left={"enabled": True, "activation_id": 2})))
        self.assertIsNone(self.update(self.frame(3, received_at=1.51)))
        command = self.update(self.frame(4, received_at=1.52, left={"enabled": True, "activation_id": 3}))
        np.testing.assert_array_equal(command.positions, self.q)
        self.assertTrue(command.meta["reanchored_by_side"]["left"])

    def test_manual_invalid_packet_invalidation_cannot_be_bypassed_by_a_new_press_id(self):
        self.update(self.frame(1, left={"enabled": True, "activation_id": 1}))
        self.mapper.invalidate("invalid_packet")
        self.assertIsNone(self.update(self.frame(2, left={"enabled": True, "activation_id": 99})))
        self.update(self.frame(3))
        command = self.update(self.frame(4, left={"enabled": True, "activation_id": 100}))
        np.testing.assert_array_equal(command.positions, self.q)

    def test_repeated_hold_does_not_commit_generated_q_or_rescale_rotation_twice(self):
        self.mapper = VrControllerRetargeter(replace(VrControllerConfig(), orientation_scale=.5), solver=self.solver)
        self.update(self.frame(1, left={"enabled": True}))
        held = self.frame(2, left={"enabled": True, "rotation_xyzw": Rotation.from_euler("z", .4).as_quat().tolist()})
        first, second = self.update(held), self.update(held)
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_allclose(second.meta["target_pose_by_side"]["left"][:3, :3], Rotation.from_euler("z", .2).as_matrix(), atol=1e-12)
        np.testing.assert_array_equal(self.solver.requests[-1].last_q_cmd, self.q)

    def test_fractional_orientation_gain_is_continuous_through_pi_and_full_turn(self):
        self.mapper = VrControllerRetargeter(replace(VrControllerConfig(), orientation_scale=.5), solver=self.solver)
        self.update(self.frame(1, left={"enabled": True}))
        for index, angle in enumerate((45, 90, 135, 179, 181, 225, 270, 315, 360), 2):
            quaternion = Rotation.from_euler("z", angle, degrees=True).as_quat()
            if index % 2:
                quaternion = -quaternion
            command = self.update(self.frame(index, left={"enabled": True, "rotation_xyzw": quaternion.tolist()}))
            np.testing.assert_allclose(command.meta["target_pose_by_side"]["left"][:3, :3],
                                       Rotation.from_euler("z", angle * .5, degrees=True).as_matrix(), atol=1e-10)

    def test_fractional_rotation_gain_handles_changing_axes_in_input_order(self):
        self.mapper = VrControllerRetargeter(replace(VrControllerConfig(), orientation_scale=.5), solver=self.solver)
        self.update(self.frame(1, left={"enabled": True}))
        physical = expected = np.eye(3)
        for index, axis in enumerate(("x", "y", "z", "x", "y"), 2):
            increment = Rotation.from_euler(axis, .3).as_matrix()
            physical = increment @ physical
            expected = Rotation.from_euler(axis, .15).as_matrix() @ expected
            command = self.update(self.frame(index, left={"enabled": True,
                "rotation_xyzw": Rotation.from_matrix(physical).as_quat().tolist()}))
            np.testing.assert_allclose(command.meta["target_pose_by_side"]["left"][:3, :3], expected, atol=1e-10)

    def test_fixed_baseline_envelope_and_invalid_feedback_are_not_reset_by_clutch(self):
        self.mapper = VrControllerRetargeter(replace(VrControllerConfig(), max_command_delta_from_baseline=.001), solver=self.solver)
        self.update(self.frame(1, left={"enabled": True}))
        command = self.update(self.frame(2, left={"enabled": True, "position": [.1, 0., 0.]}))
        self.assertLessEqual(np.max(command.positions), .001 + 1e-12)
        with self.assertRaises(ValueError):
            self.update(self.frame(3, left={"enabled": True}), q=np.full(14, float("nan")))
        self.assertIsNone(self.update(self.frame(4, left={"enabled": True})))

    def test_bounded_solver_diagnostics_are_preserved_and_unknown_failures_stop(self):
        self.update(self.frame(1, left={"enabled": True}))
        for index, reason in enumerate(("limited", "iteration_budget"), 2):
            self.solver.reason = reason
            command = self.update(self.frame(index, left={"enabled": True, "position": [.1, 0., 0.]}))
            self.assertEqual(command.meta["solver_meta"]["reason"], reason)
            self.assertFalse(command.meta["solver_meta"]["converged"])
        self.solver.reason = "invalid_target"
        with self.assertRaises(ValueError):
            self.update(self.frame(4, left={"enabled": True}))


if __name__ == "__main__":
    unittest.main()
