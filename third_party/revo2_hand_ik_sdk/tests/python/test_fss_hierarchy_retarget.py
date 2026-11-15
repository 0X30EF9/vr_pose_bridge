"""Configuration and source contracts for position-palm Revo2 retargeting."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_configs_use_position_palm_geometry_for_all_fingers() -> None:
    expected_sign = {"left": -1.0, "right": 1.0}
    for side in ("left", "right"):
        path = ROOT / "bin/conf/mocap_ik" / f"mocap_ik_revo2_{side}_hand_udp.json"
        config = json.loads(path.read_text(encoding="utf-8"))

        long_cfg = config["long_finger_retarget"]
        assert long_cfg["enabled"] is True
        fingers = long_cfg["fingers"]
        assert [entry["name"] for entry in fingers] == ["index", "middle", "ring", "little"]
        assert all(entry["retarget_mode"] == "position_palm_curve_projection_1d" for entry in fingers)
        assert all(entry["palm_root_joint"] == f"{side}_thumb_near" for entry in fingers)
        assert all(entry["palm_index_joint"] == f"{side}_index_near" for entry in fingers)
        expected_palm_outer_anchor = f"{side}_little_near"
        # v0.10.33 pairs the anatomical FSS outer anchor with its matching
        # robot pinky root on both hands. Mixing right_ring with robot_pinky
        # distorts the palm basis and causes target stretch.
        assert all(entry["palm_little_joint"] == expected_palm_outer_anchor for entry in fingers)
        assert all(entry["flex_sign"] == expected_sign[side] for entry in fingers)
        assert all(entry["calibration_frames"] >= 60 for entry in fingers)
        assert all(entry["calibration_warmup_frames"] >= 20 for entry in fingers)
        assert all(entry["calibration_max_point_step_m"] > 0.0 for entry in fingers)
        assert all(entry["calibration_max_segment_length_ratio"] > 1.0 for entry in fingers)
        assert all(entry["calibration_max_point_rms_m"] > 0.0 for entry in fingers)
        assert all(entry["calibration_max_length_cv"] > 0.0 for entry in fingers)
        assert all(entry["calibration_max_hinge_angle_rad"] == 0.65 for entry in fingers)
        assert all(0.0 < entry["output_lowpass_alpha"] < 1.0 for entry in fingers)
        for entry in fingers:
            # v0.10.35 applies the same input and projection jump fuses to
            # every finger on both sides; the policy is no longer a right-middle
            # special case.
            assert entry["max_segment_length_ratio"] > 1.0
            assert entry["max_target_step_m"] > 0.0
            assert entry["max_command_step_rad"] > 0.0
            robot_name = "pinky" if entry["name"] == "little" else entry["name"]
            assert entry["robot_palm_root_frame"] == f"{side}_thumb_metacarpal_link"
            assert entry["robot_palm_index_frame"] == f"{side}_index_proximal_link"
            assert entry["robot_palm_little_frame"] == f"{side}_pinky_proximal_link"
            assert entry["robot_middle_frame"] == f"{side}_{robot_name}_distal_link"
            expected_far = f"{side}_{robot_name}_tip_link"
            assert entry["robot_far_frame"] == expected_far
            assert entry["middle_position_weight"] > 0.0
            assert entry["far_position_weight"] > 0.0
            assert entry["projection_coarse_samples"] >= 7
            assert entry["projection_refine_iterations"] >= 0
            # v0.10.32 deliberately disables v0.10.31's nonlinear gain during
            # physical regression recovery; zero restores pure curve projection.
            assert entry["progressive_flex_gain"] == 0.0

        thumb = config["thumb_retarget"]["thumbs"][0]
        assert thumb["flex_feature_mode"] == "joint_geometry_ik_2dof"
        assert thumb["palm_frame_mode"] == "position_near_index_little"
        assert thumb["palm_index_joint"] == f"{side}_index_near"
        assert thumb["palm_little_joint"] == expected_palm_outer_anchor
        assert thumb["robot_index_frame"] == f"{side}_index_proximal_link"
        assert thumb["robot_little_frame"] == f"{side}_pinky_proximal_link"
        assert thumb["robot_root_frame"] == f"{side}_thumb_metacarpal_link"
        assert thumb["robot_middle_frame"] == f"{side}_thumb_proximal_link"
        assert thumb["robot_far_frame"] == f"{side}_thumb_distal_link"
        assert thumb["auto_position_scale"] is True
        assert thumb["calibration_frames"] >= 50
        assert thumb["calibration_warmup_frames"] >= 5
        assert thumb["calibration_max_point_step_m"] > 0.0
        assert thumb["calibration_max_segment_length_ratio"] > 1.0
        assert thumb["calibration_max_point_rms_m"] > 0.0
        assert thumb["calibration_max_length_cv"] > 0.0
        assert thumb["ik_max_iterations"] >= 4
        assert thumb["middle_position_weight"] > 0.0
        assert thumb["far_position_weight"] > 0.0
        assert thumb["joint_regularization_weight"] >= 0.0
        assert 0.0 < thumb["output_lowpass_alpha"] < 1.0


def test_native_bridge_keeps_normal_right_ring_pinky_source_indices() -> None:
    source = (ROOT / "src/apps/revo2_ik_dual_udp_bridge.cpp").read_text(encoding="utf-8")
    assert "constexpr HandIndices kRightHand{9, {32, 33, 34}, {35, 36, 37}, {38, 39, 40}, {41, 42, 43}, {44, 45, 46}};" in source
    assert "finger_data_set[10]=right_ring, [11]=right_pinky" in source




def test_right_palm_frame_keeps_anatomical_little_anchor_and_uniform_guards() -> None:
    config = json.loads((
        ROOT / "bin/conf/mocap_ik/mocap_ik_revo2_right_hand_udp.json"
    ).read_text(encoding="utf-8"))
    thumb = config["thumb_retarget"]["thumbs"][0]
    assert thumb["palm_little_joint"] == "right_little_near"
    fingers = config["long_finger_retarget"]["fingers"]
    assert all(entry["palm_little_joint"] == "right_little_near" for entry in fingers)
    assert [entry["near_joint"] for entry in fingers] == [
        "right_index_near", "right_middle_near", "right_ring_near", "right_little_near"
    ]
    assert all(entry["max_segment_length_ratio"] > 1.0 for entry in fingers)
    assert all(entry["max_target_step_m"] > 0.0 for entry in fingers)
    assert all(entry["max_command_step_rad"] > 0.0 for entry in fingers)

def test_thumb_and_long_finger_solvers_do_not_depend_on_fss_hand_pose() -> None:
    thumb = (ROOT / "src/algorithm/ik/thumb_2dof_ik.cpp").read_text(encoding="utf-8")
    long_finger = (ROOT / "src/algorithm/ik/long_finger_retarget.cpp").read_text(encoding="utf-8")

    assert "buildPalmPositionFrame" in thumb
    assert "palmLocalPosition" in thumb
    assert "R_base_from_palm" in thumb
    assert "handLocalPosition" not in thumb
    assert "target_middle" in thumb
    assert "target_far" in thumb
    assert "Eigen::Matrix<double, 6, 2> J" in thumb
    assert "cfg.distal_mimic_multiplier * J_full.col(distal.v)" in thumb
    assert "robot_tip_frame" not in thumb
    assert "target_tip" not in thumb

    assert "buildPalmPositionFrame" in long_finger
    assert "palm_root_joint" in long_finger
    assert "position_palm_curve_projection_1d" in long_finger
    assert "robot_middle_frame" in long_finger
    assert "projection_coarse_samples" in long_finger
    assert "target_middle" in long_finger
    assert "target_far" in long_finger
    assert "applyProgressiveFlexCompensation" in long_finger
    assert "cfg.progressive_flex_gain" in long_finger
    assert "held_invalid_long_finger_segment_geometry" in long_finger
    assert "held_invalid_long_finger_target_step" in long_finger
    assert "thumb_2dof_ik" not in long_finger
    assert "localOrientation" not in long_finger
    assert "signedTwistAngle" not in long_finger


def test_startup_calibration_rejects_unstable_or_closed_samples() -> None:
    long_finger = (ROOT / "src/algorithm/ik/long_finger_retarget.cpp").read_text(encoding="utf-8")
    thumb = (ROOT / "src/algorithm/ik/thumb_2dof_ik.cpp").read_text(encoding="utf-8")
    sdk = (ROOT / "src/sdk/revo2_ik_sdk.cpp").read_text(encoding="utf-8")

    assert "warming_up_open_hand_calibration" in long_finger
    assert "waiting_for_fully_open_long_finger" in long_finger
    assert "restarted_open_hand_calibration_after_unstable_frame" in long_finger
    assert "finalizeLongCalibrationStatistics" in long_finger
    assert "medianVector3" in long_finger
    assert "waiting_for_long_finger_open_hand_calibration" in thumb
    assert "restarted_thumb_calibration_after_unstable_frame" in thumb
    assert "finalizeThumbCalibrationStatistics" in thumb
    assert "allLongFingerCalibrated()" in sdk
    assert "populateCalibrationDiagnostics" in sdk
