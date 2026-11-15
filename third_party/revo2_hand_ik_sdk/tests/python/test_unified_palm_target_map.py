"""Regression contract for the v0.10.35 single palm-map retarget path."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_profiles_use_one_symmetric_guard_policy() -> None:
    for side in ("left", "right"):
        config = json.loads((
            ROOT / "bin/conf/mocap_ik" / f"mocap_ik_revo2_{side}_hand_udp.json"
        ).read_text(encoding="utf-8"))

        assert config["safety_filter"]["max_joint_velocity_rad_s"] == 5.0
        thumb = config["thumb_retarget"]["thumbs"][0]
        assert thumb["max_target_step_m"] > 0.0
        assert thumb["middle_position_scale"] == 1.0
        assert thumb["far_position_scale"] == 1.0

        for finger in config["long_finger_retarget"]["fingers"]:
            assert finger["middle_position_scale"] == 1.0
            assert finger["far_position_scale"] == 1.0
            assert finger["max_segment_length_ratio"] > 1.0
            assert finger["max_target_step_m"] > 0.0
            assert finger["max_command_step_rad"] > 0.0


def test_mapped_skeleton_and_solver_targets_share_the_same_transform() -> None:
    thumb = (ROOT / "src/algorithm/ik/thumb_2dof_ik.cpp").read_text(encoding="utf-8")
    long_finger = (ROOT / "src/algorithm/ik/long_finger_retarget.cpp").read_text(encoding="utf-8")
    sdk = (ROOT / "src/sdk/revo2_ik_sdk.cpp").read_text(encoding="utf-8")
    map_header = (ROOT / "src/mocap/calibrated_palm_map.h").read_text(encoding="utf-8")

    assert "struct CalibratedPalmMap" in map_header
    assert "calibratedPalmUniformScale" in map_header
    assert "runtime.palm_map.mapLocal(human_middle_from_near)" in thumb
    assert "runtime.palm_map.mapLocal(human_far_from_near)" in thumb
    assert "runtime.mapMiddleLocal(human_middle_palm)" in long_finger
    assert "runtime.mapFarLocal(human_far_palm)" in long_finger
    assert "thumb_state.palm_map.mapLocal(local)" in sdk
    assert "mapNearLocal(local)" in sdk
    assert "mapMiddleLocal(local)" in sdk
    assert "mapFarLocal(local)" in sdk

    # The old three-scale split is parse-compatible only; it must not be used
    # by either position-palm solver to form a target.
    assert "runtime.middle_position_scale" not in thumb
    assert "runtime.far_position_scale" not in thumb
    assert "runtime.middle_position_scale" not in long_finger
    assert "runtime.far_position_scale" not in long_finger


def test_curve_projection_has_a_nonzero_step_fuse() -> None:
    source = (ROOT / "src/algorithm/ik/long_finger_retarget.cpp").read_text(encoding="utf-8")
    assert "max_command_step_rad" in source
    assert "bounded_q = runtime.filtered_q_rad + std::clamp" in source
    assert "held_non_finite_long_finger_projection" in source


def test_thumb_endpoint_guards_do_not_apply_a_global_open_hand_bias() -> None:
    source = (ROOT / "src/algorithm/ik/thumb_2dof_ik.cpp").read_text(encoding="utf-8")

    # v0.10.39 retains the v0.10.35 DLS and only changes the two endpoint
    # zones. A calibrated constant subtraction would collapse all flex output,
    # as observed in v0.10.38; chord shortening is also invalid for this FSS
    # skeleton because both thumb segments have rigid lengths.
    assert "thumbRootCloseU" in source
    assert "applyEndpointOnlyFlexGuard" in source
    assert "human_middle0_from_near.x() - human_middle_from_near.x()" in source
    assert "open_flex_bias" not in source
    assert "strongGripFlexFloor" not in source
    assert "human_far_from_near.norm()" not in source

    for side in ("left", "right"):
        config = json.loads((
            ROOT / "bin/conf/mocap_ik" / f"mocap_ik_revo2_{side}_hand_udp.json"
        ).read_text(encoding="utf-8"))
        thumb = config["thumb_retarget"]["thumbs"][0]
        assert thumb["enable_extreme_flex_guard"] is True
        assert thumb["open_root_close_u_end"] > 0.0
        assert 0.0 < thumb["strong_grip_root_close_u_start"] < thumb["strong_grip_root_close_u_full"] <= 1.0
        assert 0.0 < thumb["strong_grip_min_flex_norm_at_full"] <= 1.0


def test_open_hand_neutral_alignment_removes_source_skeleton_tilt() -> None:
    # A source glove can report a nonzero palm-normal offset even at physical
    # full extension. A constant neutral correction must map the calibrated
    # source landmark exactly onto the robot q_open landmark while preserving
    # subsequent palm-local motion deltas.
    source_open = (0.020, 0.090, 0.018)
    robot_open = (0.024, 0.083, 0.000)
    scale = 0.8

    mapped_open = tuple(scale * value for value in source_open)
    correction = tuple(robot - mapped for robot, mapped in zip(robot_open, mapped_open))
    aligned_open = tuple(mapped + corr for mapped, corr in zip(mapped_open, correction))
    assert aligned_open == robot_open

    source_flexed = (0.020, 0.070, -0.012)
    aligned_flexed = tuple(scale * value + corr for value, corr in zip(source_flexed, correction))
    expected_delta = tuple(scale * (current - baseline)
                           for current, baseline in zip(source_flexed, source_open))
    actual_delta = tuple(current - baseline
                         for current, baseline in zip(aligned_flexed, robot_open))
    for actual, expected in zip(actual_delta, expected_delta):
        assert abs(actual - expected) < 1e-12


def test_curve_projection_never_extends_below_calibrated_open_command() -> None:
    source = (ROOT / "src/algorithm/ik/long_finger_retarget.cpp").read_text(encoding="utf-8")
    assert "projection_lower = std::clamp(runtime.robot_q0_rad, lower, upper)" in source
    assert "double best_q = projection_lower" in source
