from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_mocap_calibration import (
    RelativeMocapTargetMapper,
    build_relative_target,
    load_workspace_calibration,
    map_workspace_delta,
)
from teleop_workspace_calibration import analyze_capture, artifact_path, capture_quality_reasons, load_json


DIRECTIONS = ("forward", "backward", "outward", "inward", "up", "down")


def calibration_payload() -> dict[str, object]:
    scales = {
        "left": {
            "forward": 2.0,
            "backward": 1.0,
            "outward": 3.0,
            "inward": 4.0,
            "up": 2.0,
            "down": 1.0,
        },
        "right": {
            "forward": 2.0,
            "backward": 1.0,
            "outward": 3.0,
            "inward": 4.0,
            "up": 2.0,
            "down": 1.0,
        },
    }
    bounds = {
        side: {direction: {"safe_boundary_m": 0.2} for direction in DIRECTIONS}
        for side in ("left", "right")
    }
    return {
        "schema_version": 1,
        "valid": True,
        "control_frame": "wrist_roll",
        "base_from_mocap_rotation": np.eye(3).tolist(),
        "mocap_home": {"left": [1.0, 2.0, 3.0], "right": [1.0, -2.0, 3.0]},
        "direction_scales": scales,
        "workspace_bounds": bounds,
    }


class WorkspaceCalibrationTest(unittest.TestCase):
    def write_calibration(self, root: Path, payload: dict[str, object] | None = None) -> Path:
        path = root / "calibration.json"
        path.write_text(json.dumps(payload or calibration_payload()), encoding="utf-8")
        return path

    def test_optional_calibration_only_allows_an_omitted_path(self) -> None:
        self.assertIsNone(load_workspace_calibration(None, optional=True))
        self.assertIsNone(load_workspace_calibration("", optional=True))
        with self.assertRaisesRegex(ValueError, "未提供"):
            load_workspace_calibration(None)

    def test_explicit_missing_path_is_rejected_even_when_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_workspace_calibration(Path(tmp) / "missing.json", optional=True)

    def test_explicit_invalid_files_are_rejected_even_when_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field, value in (("valid", False), ("schema_version", 0), ("control_frame", "L_ee_R_ee")):
                with self.subTest(field=field):
                    payload = calibration_payload()
                    payload[field] = value
                    with self.assertRaises(ValueError):
                        load_workspace_calibration(self.write_calibration(root, payload), optional=True)
            invalid_json = root / "invalid.json"
            invalid_json.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_workspace_calibration(invalid_json, optional=True)

    def test_explicit_valid_optional_calibration_keeps_its_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = load_workspace_calibration(self.write_calibration(Path(tmp)), optional=True)
            self.assertIsNotNone(calibration)
            mapped, projected = map_workspace_delta(np.array([0.05, 0.0, 0.0]), "left", calibration, 1.0)
            np.testing.assert_allclose(mapped, [0.1, 0.0, 0.0])
            self.assertFalse(projected)

    def test_invalid_calibration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = calibration_payload()
            payload["valid"] = False
            path = self.write_calibration(Path(tmp), payload)
            with self.assertRaisesRegex(ValueError, "valid 必须为 true"):
                load_workspace_calibration(path)

    def test_directional_scales_distinguish_positive_and_negative_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = load_workspace_calibration(self.write_calibration(Path(tmp)))
            forward, projected = map_workspace_delta(np.array([0.05, 0.0, 0.0]), "left", calibration, 1.0)
            backward, _ = map_workspace_delta(np.array([-0.05, 0.0, 0.0]), "left", calibration, 1.0)
            right_inward, _ = map_workspace_delta(np.array([0.0, 0.05, 0.0]), "right", calibration, 1.0)
            np.testing.assert_allclose(forward, [0.10, 0.0, 0.0])
            np.testing.assert_allclose(backward, [-0.05, 0.0, 0.0])
            np.testing.assert_allclose(right_inward, [0.0, 0.20, 0.0])
            self.assertFalse(projected)

    def test_combined_motion_is_projected_into_safe_ellipsoid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = load_workspace_calibration(self.write_calibration(Path(tmp)))
            mapped, projected = map_workspace_delta(np.array([0.10, 0.10, 0.0]), "left", calibration, 1.0)
            self.assertTrue(projected)
            normalized = np.array([mapped[0] / 0.2, mapped[1] / 0.2, mapped[2] / 0.2])
            self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=9)

    def test_capture_quality_retries_only_the_failed_window(self) -> None:
        args = SimpleNamespace(
            min_valid_rate=0.9,
            max_direction_rms=0.055,
            min_direction_displacement=0.08,
        )
        home = {"left": np.zeros(3), "right": np.array([0.0, -0.4, 0.0])}
        frames = [
            {
                "left_pos": [0.02, 0.0, 0.0],
                "right_pos": [0.0, -0.2, 0.0],
            }
            for _ in range(8)
        ]
        _, reasons = capture_quality_reasons(
            frames,
            {"valid_rate": 0.5},
            "forward",
            "left",
            args,
            home,
        )
        self.assertTrue(any(reason.startswith("valid_rate=") for reason in reasons))
        self.assertTrue(any("left_displacement=" in reason for reason in reasons))
        self.assertTrue(any("right_body_relative_motion=" in reason for reason in reasons))

    def test_capture_quality_uses_body_relative_motion(self) -> None:
        args = SimpleNamespace(
            min_valid_rate=0.9,
            max_direction_rms=0.055,
            min_direction_displacement=0.08,
        )
        home = {"left": np.array([0.0, 0.4, 0.0]), "right": np.array([0.0, -0.4, 0.0])}
        frames = [
            {
                "body_pos": [0.25, 0.0, 0.0],
                "left_pos": [0.45, 0.4, 0.0],
                "right_pos": [0.25, -0.4, 0.0],
            }
            for _ in range(8)
        ]
        centers, reasons = capture_quality_reasons(
            frames,
            {"valid_rate": 1.0},
            "forward",
            "left",
            args,
            home,
        )
        np.testing.assert_allclose(centers["left"] - home["left"], [0.2, 0.0, 0.0])
        np.testing.assert_allclose(centers["right"] - home["right"], [0.0, 0.0, 0.0])
        self.assertEqual(reasons, [])

    def test_explicit_calibration_output_filename_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "operator_a_workspace.json"
            self.assertEqual(artifact_path(requested, "calibration"), requested)
            self.assertEqual(artifact_path(requested, "analysis"), Path(tmp) / "analysis.json")

    def test_analyze_capture_accepts_consistent_six_direction_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            homes = {
                "left": np.array([1.0, 0.3, 1.2]),
                "right": np.array([1.0, -0.3, 1.2]),
            }
            deltas = {
                "left": {
                    "forward": [0.16, 0.0, 0.0],
                    "backward": [-0.10, 0.0, 0.0],
                    "outward": [0.0, 0.14, 0.0],
                    "inward": [0.0, -0.09, 0.0],
                    "up": [0.0, 0.0, 0.15],
                    "down": [0.0, 0.0, -0.10],
                },
                "right": {
                    "forward": [0.16, 0.0, 0.0],
                    "backward": [-0.10, 0.0, 0.0],
                    "outward": [0.0, -0.14, 0.0],
                    "inward": [0.0, 0.09, 0.0],
                    "up": [0.0, 0.0, 0.15],
                    "down": [0.0, 0.0, -0.10],
                },
            }

            def frames(left: np.ndarray, right: np.ndarray) -> list[dict[str, list[float]]]:
                return [
                    {"left_pos": left.tolist(), "right_pos": right.tolist()}
                    for _ in range(10)
                ]

            windows: dict[str, object] = {
                "home": {
                    "frames": frames(homes["left"], homes["right"]),
                    "stats": {"valid_rate": 1.0},
                    "retries": [],
                }
            }
            for side in ("left", "right"):
                other = "right" if side == "left" else "left"
                for direction in DIRECTIONS:
                    moved = homes[side] + np.asarray(deltas[side][direction], dtype=float)
                    left = moved if side == "left" else homes[other]
                    right = moved if side == "right" else homes[other]
                    windows[f"{side}_{direction}"] = {
                        "frames": frames(left, right),
                        "stats": {"valid_rate": 1.0},
                        "retries": [],
                    }
            capture_path = root / "capture.json"
            capture_path.write_text(
                json.dumps(
                    {
                        "control_point": "anatomical_wrist",
                        "position_reference": "wrist_minus_chest",
                        "windows": windows,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input=capture_path,
                side="both",
                calibration_output=root / "calibration.json",
                overwrite=False,
                min_direction_displacement=0.08,
                max_direction_rms=0.055,
                min_valid_rate=0.9,
                min_direction_cosine=0.7,
                max_opposite_direction_cosine=-0.5,
            )

            output = analyze_capture(args)
            analysis = load_json(output)
            self.assertTrue(analysis["valid"])
            np.testing.assert_allclose(analysis["base_from_mocap_rotation"], np.eye(3), atol=1e-9)
            self.assertEqual(Path(analysis["source_capture"]), capture_path)

    def test_runtime_home_overrides_persisted_home_without_losing_workspace_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = calibration_payload()
            payload["position_reference"] = "hand_minus_chest"
            calibration = load_workspace_calibration(self.write_calibration(Path(tmp), payload))
            left_robot_home = np.eye(4)
            left_robot_home[:3, 3] = [0.2, 0.3, 0.4]
            right_robot_home = np.eye(4)
            right_robot_home[:3, 3] = [0.2, -0.3, 0.4]
            mapper = RelativeMocapTargetMapper(
                left_robot_home,
                right_robot_home,
                position_scale=1.0,
                orientation_scale=0.0,
                workspace_calibration=calibration,
            )
            left = np.eye(4)
            left[:3, 3] = [4.0, 5.0, 6.0]
            right = np.eye(4)
            right[:3, 3] = [4.0, -5.0, 6.0]
            chest = np.array([1.0, 1.0, 1.0])
            mapper.set_runtime_mocap_home(left, right, mocap_reference_position=chest)

            initial = mapper.map_targets(left, right, mocap_reference_position=chest)
            np.testing.assert_allclose(initial.left_target, left_robot_home, atol=1e-12)
            np.testing.assert_allclose(initial.right_target, right_robot_home, atol=1e-12)
            self.assertEqual(initial.meta["mocap_home_source"], "runtime_natural_down")
            self.assertEqual(initial.meta["workspace_calibration"], str(calibration.source_path))

            moved_left = left.copy()
            moved_left[0, 3] += 0.05
            moved = mapper.map_targets(moved_left, right, mocap_reference_position=chest)
            np.testing.assert_allclose(
                moved.left_target[:3, 3],
                left_robot_home[:3, 3] + [0.10, 0.0, 0.0],
                atol=1e-12,
            )

    def test_mapper_uses_persisted_home_instead_of_first_frame_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = load_workspace_calibration(self.write_calibration(Path(tmp)))
            left_robot_home = np.eye(4)
            right_robot_home = np.eye(4)
            mapper = RelativeMocapTargetMapper(
                left_robot_home,
                right_robot_home,
                position_scale=1.0,
                orientation_scale=0.0,
                workspace_calibration=calibration,
                control_side="left",
            )
            left = np.eye(4)
            left[:3, 3] = [1.05, 2.0, 3.0]
            right = np.eye(4)
            right[:3, 3] = [1.0, -2.0, 3.0]
            result = mapper.map_targets(left, right)
            np.testing.assert_allclose(result.left_target[:3, 3], [0.10, 0.0, 0.0])
            np.testing.assert_allclose(result.right_target, right_robot_home)
            self.assertEqual(result.meta["workspace_calibration"], str(calibration.source_path))

    def test_world_rotation_delta_maps_into_robot_base_before_tcp_home(self) -> None:
        base_from_mocap = R.from_euler("xyz", [0.25, -0.35, 0.15]).as_matrix()
        marker_mounting = R.from_euler("xyz", [0.8, -0.4, 1.1]).as_matrix()
        robot_home = np.eye(4)
        robot_home[:3, :3] = R.from_euler("xyz", [-0.2, 0.3, -0.5]).as_matrix()
        robot_home[:3, 3] = [0.2, 0.3, -0.4]
        mocap_home = np.eye(4)
        mocap_home[:3, :3] = marker_mounting
        mocap_home[:3, 3] = [1.0, 2.0, 3.0]

        expected_base_delta = R.from_euler("z", 0.4).as_matrix()
        mocap_world_delta = base_from_mocap.T @ expected_base_delta @ base_from_mocap
        mocap_current = mocap_home.copy()
        mocap_current[:3, :3] = mocap_world_delta @ mocap_home[:3, :3]

        target = build_relative_target(
            mocap_current,
            mocap_home,
            robot_home,
            position_scale=1.0,
            orientation_scale=1.0,
            base_from_mocap_rotation=base_from_mocap,
        )

        np.testing.assert_allclose(target[:3, 3], robot_home[:3, 3], atol=1e-12)
        np.testing.assert_allclose(
            target[:3, :3],
            expected_base_delta @ robot_home[:3, :3],
            atol=1e-12,
        )

    def test_fixed_left_right_marker_mounting_differences_cancel(self) -> None:
        base_delta = R.from_rotvec(np.array([0.15, -0.25, 0.35])).as_matrix()
        for mounting in (
            R.from_euler("xyz", [0.7, 0.2, -1.0]).as_matrix(),
            R.from_euler("xyz", [-0.6, 0.5, 0.9]).as_matrix(),
        ):
            mocap_home = np.eye(4)
            mocap_home[:3, :3] = mounting
            mocap_current = mocap_home.copy()
            mocap_current[:3, :3] = base_delta @ mounting
            robot_home = np.eye(4)

            target = build_relative_target(
                mocap_current,
                mocap_home,
                robot_home,
                position_scale=0.0,
                orientation_scale=1.0,
            )
            np.testing.assert_allclose(target[:3, :3], base_delta, atol=1e-12)

    def test_workspace_supports_independent_left_right_orientation_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = calibration_payload()
            left_base_from_mocap = R.from_euler("z", 0.3).as_matrix()
            right_base_from_mocap = R.from_euler("x", -0.4).as_matrix()
            payload["orientation_base_from_mocap_rotation"] = {
                "left": left_base_from_mocap.tolist(),
                "right": right_base_from_mocap.tolist(),
            }
            calibration = load_workspace_calibration(self.write_calibration(Path(tmp), payload))
            mapper = RelativeMocapTargetMapper(
                np.eye(4),
                np.eye(4),
                position_scale=0.0,
                orientation_scale=1.0,
                workspace_calibration=calibration,
            )
            left_home = np.eye(4)
            left_home[:3, :3] = R.from_euler("xyz", [0.8, -0.2, 0.4]).as_matrix()
            right_home = np.eye(4)
            right_home[:3, :3] = R.from_euler("xyz", [-0.7, 0.5, -0.3]).as_matrix()
            mapper.set_runtime_mocap_home(left_home, right_home)

            expected_delta = R.from_euler("y", 0.35).as_matrix()
            left_current = left_home.copy()
            left_current[:3, :3] = (
                left_base_from_mocap.T @ expected_delta @ left_base_from_mocap @ left_home[:3, :3]
            )
            right_current = right_home.copy()
            right_current[:3, :3] = (
                right_base_from_mocap.T @ expected_delta @ right_base_from_mocap @ right_home[:3, :3]
            )
            mapped = mapper.map_targets(left_current, right_current)

            np.testing.assert_allclose(mapped.left_target[:3, :3], expected_delta, atol=1e-12)
            np.testing.assert_allclose(mapped.right_target[:3, :3], expected_delta, atol=1e-12)
            self.assertEqual(mapped.meta["orientation_reference"], "world_delta_in_robot_base")

    def test_wrist_control_point_generates_tcp_arc_from_urdf_offset(self) -> None:
        wrist_to_tcp = np.eye(4)
        wrist_to_tcp[:3, 3] = [0.0, 0.0, -0.0848]
        mapper = RelativeMocapTargetMapper(
            np.eye(4),
            np.eye(4),
            position_scale=1.0,
            orientation_scale=1.0,
            left_control_to_target=wrist_to_tcp,
            right_control_to_target=wrist_to_tcp,
            control_frame="wrist_roll",
        )
        home = np.eye(4)
        mapper.set_runtime_mocap_home(home, home)

        initial = mapper.map_targets(home, home)
        np.testing.assert_allclose(initial.meta["left_control_target"], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(initial.left_target, wrist_to_tcp, atol=1e-12)

        current = np.eye(4)
        current[:3, :3] = R.from_euler("y", 0.4).as_matrix()
        rotated = mapper.map_targets(current, current)
        expected_tcp = current @ wrist_to_tcp
        np.testing.assert_allclose(rotated.meta["left_control_target"][:3, 3], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(rotated.left_target, expected_tcp, atol=1e-12)
        self.assertAlmostEqual(
            float(np.linalg.norm(rotated.left_target[:3, 3] - initial.left_target[:3, 3])),
            2.0 * 0.0848 * np.sin(0.2),
            places=12,
        )

    def test_orientation_scale_blends_from_locked_tcp_to_mocap_rotation(self) -> None:
        mocap_home = np.eye(4)
        mocap_current = np.eye(4)
        mocap_current[:3, :3] = R.from_euler("y", np.pi / 2.0).as_matrix()
        robot_home = np.eye(4)

        locked = build_relative_target(mocap_current, mocap_home, robot_home, 0.0, 0.0)
        blended = build_relative_target(mocap_current, mocap_home, robot_home, 0.0, 0.5)
        followed = build_relative_target(mocap_current, mocap_home, robot_home, 0.0, 1.0)

        np.testing.assert_allclose(locked[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(R.from_matrix(blended[:3, :3]).as_rotvec())), np.pi / 4.0)
        np.testing.assert_allclose(followed[:3, :3], mocap_current[:3, :3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
