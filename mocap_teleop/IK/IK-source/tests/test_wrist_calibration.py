from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as R

import calibrate_wrist

from teleop_wrist_calibration import (
    WristCalibrationThresholds,
    WristSideCalibration,
    fit_direct_wrist_joint,
    fit_hand_to_wrist_pivot,
    load_wrist_calibration,
    reconstruct_wrist_pose,
    reconstruct_wrist_pose_from_mapping,
    select_runtime_position_source,
)


class WristCalibrationTest(unittest.TestCase):
    @staticmethod
    def synthetic_samples(
        *,
        seed: int = 7,
        noise_m: float = 0.001,
        outlier_fraction: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        offset = np.array([-0.012, 0.004, -0.058])
        centers = {
            "front": np.array([0.35, 0.20, 1.20]),
            "side": np.array([0.25, 0.45, 1.10]),
            "raised": np.array([0.30, 0.25, 1.40]),
        }
        positions: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        groups: list[str] = []
        wrists: list[np.ndarray] = []
        for group, center in centers.items():
            for index in range(80):
                progress = index / 79.0
                rotvec = np.array(
                    [
                        0.55 * np.sin(progress * 2.0 * np.pi),
                        0.45 * np.cos(progress * 2.0 * np.pi),
                        0.50 * np.sin(progress * 4.0 * np.pi),
                    ]
                )
                rotation = R.from_rotvec(rotvec).as_matrix()
                positions.append(center - rotation @ offset + rng.normal(0.0, noise_m, 3))
                wrists.append(center + rng.normal(0.0, noise_m, 3))
                rotations.append(rotation)
                groups.append(group)
        positions_array = np.asarray(positions)
        if outlier_fraction > 0.0:
            outlier_count = int(round(len(positions_array) * outlier_fraction))
            indices = rng.choice(len(positions_array), outlier_count, replace=False)
            positions_array[indices] += rng.normal(0.0, 0.05, (outlier_count, 3))
        return positions_array, np.asarray(rotations), groups, np.asarray(wrists), offset

    def test_multigroup_pivot_fit_recovers_known_offset(self) -> None:
        positions, rotations, groups, wrists, expected = self.synthetic_samples()
        fit = fit_hand_to_wrist_pivot(positions, rotations, groups, wrist_positions=wrists)

        self.assertTrue(fit.valid, fit.reasons)
        np.testing.assert_allclose(fit.hand_to_wrist_offset, expected, atol=0.001)
        self.assertLess(fit.quality["pivot_rms_m"], 0.003)
        self.assertGreaterEqual(fit.quality["rotation_rank"], 2)
        self.assertEqual(select_runtime_position_source(fit), "wrist_joint")

    def test_direct_wrist_joint_accepts_shared_position_origin(self) -> None:
        positions, rotations, groups, _, _ = self.synthetic_samples()
        fit = fit_direct_wrist_joint(positions, rotations, groups, wrist_positions=positions.copy())

        self.assertTrue(fit.valid, fit.reasons)
        self.assertEqual(select_runtime_position_source(fit), "wrist_joint")
        np.testing.assert_allclose(fit.hand_to_wrist_offset, np.zeros(3), atol=0.0)
        self.assertEqual(fit.quality["hand_wrist_origin_rms_m"], 0.0)

    def test_direct_wrist_joint_rejects_distinct_position_origins(self) -> None:
        positions, rotations, groups, _, _ = self.synthetic_samples()
        wrists = positions + np.array([0.010, 0.0, 0.0])
        fit = fit_direct_wrist_joint(positions, rotations, groups, wrist_positions=wrists)

        self.assertFalse(fit.valid)
        self.assertTrue(any(reason.startswith("hand_wrist_origin_rms=") for reason in fit.reasons))
        self.assertEqual(select_runtime_position_source(fit), "hand_pose_plus_pivot_offset")

    def test_robust_fit_rejects_sparse_position_outliers(self) -> None:
        positions, rotations, groups, wrists, expected = self.synthetic_samples(outlier_fraction=0.08)
        fit = fit_hand_to_wrist_pivot(positions, rotations, groups, wrist_positions=wrists)

        self.assertTrue(fit.valid, fit.reasons)
        np.testing.assert_allclose(fit.hand_to_wrist_offset, expected, atol=0.003)
        self.assertGreater(fit.quality["outlier_fraction"], 0.03)
        self.assertLess(fit.quality["outlier_fraction"], 0.15)

    def test_insufficient_rotation_excitation_is_rejected(self) -> None:
        groups = ["front"] * 80
        rotations = np.repeat(np.eye(3)[None, :, :], 80, axis=0)
        positions = np.repeat(np.array([[0.2, 0.3, 1.0]]), 80, axis=0)
        fit = fit_hand_to_wrist_pivot(
            positions,
            rotations,
            groups,
            thresholds=WristCalibrationThresholds(min_groups=1),
        )

        self.assertFalse(fit.valid)
        self.assertTrue(any(reason.startswith("rotation_excitation=") for reason in fit.reasons))
        self.assertTrue(any(reason.startswith("rotation_rank=") for reason in fit.reasons))

    def test_reconstruct_wrist_pose_uses_explicit_position_source(self) -> None:
        hand_position = np.array([0.4, 0.2, 1.1])
        hand_rotation = R.from_euler("z", np.pi / 2.0)
        offset = np.array([0.1, 0.0, 0.0])
        pivot = WristSideCalibration(
            position_source="hand_pose_plus_pivot_offset",
            hand_to_wrist_offset=offset,
            quality={},
        )
        reconstructed = reconstruct_wrist_pose(
            hand_position,
            hand_rotation.as_quat(),
            None,
            pivot,
        )
        np.testing.assert_allclose(reconstructed[:3, 3], [0.4, 0.3, 1.1], atol=1e-12)

        direct = WristSideCalibration(
            position_source="wrist_joint",
            hand_to_wrist_offset=offset,
            quality={},
        )
        direct_pose = reconstruct_wrist_pose(
            hand_position,
            hand_rotation.as_quat(),
            [0.3, 0.25, 1.05],
            direct,
        )
        np.testing.assert_allclose(direct_pose[:3, 3], [0.3, 0.25, 1.05], atol=1e-12)
        np.testing.assert_allclose(direct_pose[:3, :3], hand_rotation.as_matrix(), atol=1e-12)

    def test_mapping_rejects_untracked_required_observation(self) -> None:
        frame = {
            "left_hand_pos": [0.4, 0.2, 1.1],
            "left_hand_quat": [0.0, 0.0, 0.0, 1.0],
            "left_wrist_pos": [0.4, 0.2, 1.05],
            "hand_wrist_tracking": {"left": {"hand": False, "wrist": True}},
        }
        pivot = WristSideCalibration(
            position_source="hand_pose_plus_pivot_offset",
            hand_to_wrist_offset=np.array([0.0, 0.0, -0.05]),
            quality={},
        )
        with self.assertRaisesRegex(ValueError, "left hand 未跟踪"):
            reconstruct_wrist_pose_from_mapping(frame, "left", pivot)

        frame["hand_wrist_tracking"]["left"] = {"hand": True, "wrist": False}
        reconstruct_wrist_pose_from_mapping(frame, "left", pivot)
        direct = WristSideCalibration(
            position_source="wrist_joint",
            hand_to_wrist_offset=np.array([0.0, 0.0, -0.05]),
            quality={},
        )
        with self.assertRaisesRegex(ValueError, "left wrist 未跟踪"):
            reconstruct_wrist_pose_from_mapping(frame, "left", direct)

    def test_loader_rejects_invalid_or_incomplete_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            payload = {
                "schema_version": 1,
                "valid": True,
                "control_point": "anatomical_wrist",
                "sides": {
                    side: {
                        "position_source": "wrist_joint",
                        "hand_to_wrist_offset": [0.0, 0.0, -0.05],
                        "valid": True,
                        "quality": {
                            "sample_count": 240,
                            "group_count": 3,
                            "rotation_excitation_rad": 0.5,
                            "rotation_rank": 3,
                            "outlier_fraction": 0.02,
                            "pivot_rms_m": 0.002,
                            "pivot_p95_m": 0.004,
                            "wrist_validation_rms_m": 0.003,
                        },
                    }
                    for side in ("left", "right")
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_wrist_calibration(path)
            self.assertEqual(set(calibration.sides), {"left", "right"})

            payload["sides"]["left"]["quality"]["pivot_rms_m"] = 0.050
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pivot_rms_m"):
                load_wrist_calibration(path)

            payload["valid"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid 必须为 true"):
                load_wrist_calibration(path)
    def test_one_step_flow_publishes_only_validated_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "raw" / "capture.json"
            raw_path.parent.mkdir()
            raw_path.write_text('{"samples": []}', encoding="utf-8")
            final_path = root / "calibration.json"
            final_path.write_text('{"version": "old"}', encoding="utf-8")
            args = Namespace(calibration_output=final_path, input=None, side="both")

            def fake_analyze(current: Namespace) -> Path:
                candidate = Path(current.calibration_output)
                candidate.write_text('{"valid": true}', encoding="utf-8")
                return candidate

            def fake_validate(current: Namespace) -> None:
                self.assertEqual(Path(current.input), Path(current.calibration_output))
                self.assertEqual(json.loads(Path(current.input).read_text(encoding="utf-8")), {"valid": True})

            with (
                patch.object(calibrate_wrist, "capture", return_value=raw_path),
                patch.object(calibrate_wrist, "analyze", side_effect=fake_analyze),
                patch.object(calibrate_wrist, "validate", side_effect=fake_validate),
            ):
                result = calibrate_wrist.run_all(args)

            self.assertEqual(result, final_path)
            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8")), {"valid": True})
            self.assertTrue(raw_path.is_file())
            self.assertEqual(list(root.glob("*.candidate.json")), [])

    def test_one_step_flow_preserves_existing_file_when_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "raw" / "capture.json"
            raw_path.parent.mkdir()
            raw_path.write_text('{"samples": []}', encoding="utf-8")
            final_path = root / "calibration.json"
            final_path.write_text('{"version": "trusted"}', encoding="utf-8")
            args = Namespace(calibration_output=final_path, input=None, side="both")

            def fake_analyze(current: Namespace) -> Path:
                candidate = Path(current.calibration_output)
                candidate.write_text('{"valid": false}', encoding="utf-8")
                return candidate

            with (
                patch.object(calibrate_wrist, "capture", return_value=raw_path),
                patch.object(calibrate_wrist, "analyze", side_effect=fake_analyze),
                patch.object(calibrate_wrist, "validate", side_effect=ValueError("质量门禁未通过")),
            ):
                with self.assertRaisesRegex(ValueError, "质量门禁未通过"):
                    calibrate_wrist.run_all(args)

            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8")), {"version": "trusted"})
            self.assertTrue(raw_path.is_file())
            self.assertEqual(list(root.glob("*.candidate.json")), [])


if __name__ == "__main__":
    unittest.main()