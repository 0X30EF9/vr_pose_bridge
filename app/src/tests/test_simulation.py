from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from teleop_simulation import (BulletWorld, build_solver, canonical_mocap_frame, clamp_operator_command, command_is_valid, distribution,
                               initial_configuration, model_signature, prepare_urdf,
                               run_scene, transform, validate_dynamics, validate_fk, write_viewer)


class SimulationModelTest(unittest.TestCase):
    def test_missing_geometry_replacement_preserves_joint_and_inertia_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "original.urdf"
            source.write_text('''<robot name="test"><link name="base"><inertial><origin xyz=".1 0 0"/>
                <mass value="2"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial>
                <visual><geometry><mesh filename="absent.stl"/></geometry></visual></link><link name="tip"/>
                <joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/>
                <origin xyz="0 .2 0" rpy="0 .1 0"/><axis xyz="0 1 0"/>
                <limit lower="-.5" upper=".8" velocity="2" effort="3"/></joint></robot>''', encoding="utf-8")
            destination = root / "derived.urdf"
            manifest = prepare_urdf(source, destination)
            before, after = ET.parse(source).getroot(), ET.parse(destination).getroot()
            self.assertEqual(model_signature(before), model_signature(after))
            self.assertEqual(len(manifest["missing_mesh_replacements"]), 1)
            self.assertIsNotNone(after.find("link/visual/geometry/sphere"))
            self.assertFalse(manifest["collision_safety_validated"])

    def test_candidate_guard_rejects_nan_hard_limits_and_step_velocity(self):
        q = np.zeros(14)
        limits = np.tile([-1., 1.], (14, 1))
        velocities = np.ones(14)
        valid = q + .005
        self.assertTrue(command_is_valid(valid, q, q, limits, velocities, .01))
        for invalid in (q + np.nan, q + 1.1, q + .02, np.zeros(13)):
            self.assertFalse(command_is_valid(invalid, q, q, limits, velocities, .01))

    def test_percentiles_are_computed_from_samples(self):
        result = distribution(range(101))
        self.assertEqual(result, {"p50": 50., "p95": 95., "p99": 99., "max": 100.})

    def test_physics_xyzw_transform_uses_the_link_rotation(self):
        pose = transform([1, 2, 3], [0, 0, np.sqrt(.5), np.sqrt(.5)])
        np.testing.assert_allclose(pose @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)

    def test_full_mapper_fixture_marks_unused_nodes_untracked(self):
        directions = {side: (np.array([0., 0., -1.]), np.array([0., 0., -1.])) for side in ("left", "right")}
        frame = canonical_mocap_frame(directions, [np.eye(4), np.eye(4)], 1.)
        self.assertEqual(len(frame.joints), 64)
        self.assertTrue(frame.joint("left_wrist").tracked)
        self.assertFalse(frame.joint("left_thumb_near").tracked)
        self.assertFalse(frame.joint("left_knee").tracked)

    def test_normal_candidate_is_clamped_to_operator_velocity_and_baseline(self):
        q = np.zeros(14)
        limits = np.tile([-2., 2.], (14, 1))
        result = clamp_operator_command(q+.5, q, q, q, limits, np.full(14, .48), .01)
        np.testing.assert_allclose(result, np.full(14, .0048))
        previous = np.full(14, .999)
        result = clamp_operator_command(q+1.5, previous, previous, q, limits, np.full(14, .48), .01)
        np.testing.assert_allclose(result, np.ones(14))

    def test_viewer_embeds_data_and_has_no_remote_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "viewer.html"
            write_viewer(path, [], [], {"description": "</script><script>bad</script>"})
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("__SIMULATION_DATA__", content)
            self.assertNotIn('src="https:', content)
            self.assertIn('<\\/script>', content)


HAS_ENGINE = importlib.util.find_spec("pybullet") is not None and importlib.util.find_spec("pinocchio") is not None


@unittest.skipUnless(HAS_ENGINE, "dedicated simulation environment with pin and pybullet required")
class IndependentPhysicsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.solver = build_solver("skeleton-100hz", .01)
        self.world = BulletWorld(Path(self.temporary.name), dt=.01, substeps=4)
        self.addCleanup(self.world.close)

    def test_independent_bullet_and_pinocchio_fk_agree(self):
        result = validate_fk(self.world, self.solver, samples=4)
        self.assertTrue(result["passed"], result)
        dynamics = validate_dynamics(self.world)
        self.assertTrue(dynamics["passed"], dynamics)

    def test_position_control_evolves_measured_state_instead_of_teleporting(self):
        start = initial_configuration(self.world.limits)
        self.world.reset(start)
        target = start.copy()
        target[3] -= .1
        self.world.command(target)
        np.testing.assert_allclose(self.world.measured()[0], start, atol=1e-10)
        self.world.step()
        measured = self.world.measured()[0]
        self.assertGreater(np.linalg.norm(measured-start), 1e-5)
        self.assertGreater(np.linalg.norm(measured-target), 1e-5)
        self.assertLess(abs(measured[3]-target[3]), abs(start[3]-target[3]))

    def test_tracking_loss_and_invalid_limit_candidate_are_observable(self):
        for name in ("tracking_loss", "limit_guard"):
            result, frames = run_scene(self.world, self.solver, name, .4, False)
            if name == "tracking_loss":
                self.assertGreater(result["fault_frames"], 0)
                self.assertEqual(result["commands_during_fault"], 0)
                self.assertTrue(all(not frame["command_applied"] for frame in frames if frame["fault"]))
            else:
                self.assertGreater(result["command_rejections"], 0)
            self.assertIsNone(result["wall_period_overruns"])


if __name__ == "__main__":
    unittest.main()
