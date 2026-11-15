"""Automatic body-relative retargeting of the parsed 64-node UDP skeleton.

Only a stable natural-down startup window is needed. Human world translation
and rotation are removed with the shoulder/hip body frame; hand rotations add
the twist information that segment directions alone cannot observe. This module
does not publish commands or assume a proposed command reached the robot.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from teleop_direct_retarget import (
    DirectRetargetConfig,
    SkeletonObservation,
    calibration_window_is_stable,
    controlled_indices,
    direction_angle,
    ensure_arm_q,
    median_home_arms,
    observation_is_safe,
    relative_direction_target,
    selected_sides,
    skeleton_observation,
)
from teleop_ik_core import TeleopArmIK, clip_to_limits, ensure_rotation, joint_limit_arrays
from teleop_ik_types import IKCommand, TeleopIKConfig

SKELETON_RETARGET_SOURCE = "body_relative_skeleton_dls"
# Quaternion sign changes and crossings of +/-180 degrees are measured on SO(3).
MAX_HAND_ROTATION_STEP_RAD = np.pi / 2.0


def validate_skeleton_frame(
    frame: dict[str, Any], config: DirectRetargetConfig
) -> tuple[SkeletonObservation, dict[str, np.ndarray]]:
    """Return segment observations and body-local hand rotations, or reject.

    Both anatomical wrists and hands must be tracked, even though the hand
    quaternion supplies the orientation task. No hand-to-wrist position offset
    or absolute hand position is needed for this retargeting path.
    """
    observation, reason = skeleton_observation(frame, config)
    if observation is None:
        raise ValueError(f"invalid_skeleton:{reason}")
    body_rotation = ensure_rotation(observation.body_frame.world_to_body, "body frame")
    rotations: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        for joint in ("wrist", "hand"):
            entry = frame.get("skeleton", {}).get(side, {}).get(joint)
            if not isinstance(entry, dict) or not entry.get("tracked", False):
                raise ValueError(f"untracked_{side}_{joint}")
            position = np.asarray(entry.get("pos"), dtype=float)
            quaternion = np.asarray(entry.get("quat"), dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"invalid_{side}_{joint}_position")
            norm = float(np.linalg.norm(quaternion))
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or not np.isfinite(norm) or norm <= 1e-8:
                raise ValueError(f"invalid_{side}_{joint}_quaternion")
            if joint == "hand":
                rotations[side] = body_rotation @ Rotation.from_quat(quaternion).as_matrix()
    return observation, rotations


def _body_local_observation(observation: SkeletonObservation) -> SkeletonObservation:
    """Reuse the existing length/jump gates in body coordinates, not world."""
    body = observation.body_frame
    arms = {
        side: replace(
            arm,
            shoulder_world=body.vector_to_local(arm.shoulder_world - body.origin),
            elbow_world=body.vector_to_local(arm.elbow_world - body.origin),
            wrist_world=body.vector_to_local(arm.wrist_world - body.origin),
        )
        for side, arm in observation.arms.items()
    }
    return replace(observation, arms=arms)


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(first @ second.T).as_rotvec()))


class SkeletonRetargeter:
    def __init__(
        self,
        config: DirectRetargetConfig,
        solver: TeleopArmIK | None = None,
        orientation_scale: float = 1.0,
    ):
        if len(config.joint_names) != 14 or len(config.motor_ids) != 14:
            raise ValueError("skeleton retarget requires 14 joint names and motor IDs")
        if len(set(config.joint_names)) != 14 or len(set(config.motor_ids)) != 14:
            raise ValueError("joint names and motor IDs must be unique")
        selected_sides(config.control_side)
        if not np.isfinite(orientation_scale) or not 0.0 <= orientation_scale <= 1.0:
            raise ValueError("orientation_scale must be finite and within [0, 1]")
        self.config = config
        self.orientation_scale = float(orientation_scale)
        self.solver = solver if solver is not None else TeleopArmIK(TeleopIKConfig(
            joint_names=config.joint_names,
            motor_ids=config.motor_ids,
            max_iterations=config.direction_iterations,
            damping=config.direction_damping,
            max_joint_delta_per_step=config.max_joint_delta_per_step,
            smoothing_alpha=config.smoothing_alpha,
            enable_local_log=False,
            log_to_console=False,
            show_model_load_progress=False,
            log_ik_result=False,
        ))
        self.aligned = False

    def _bounded_q(self, values: np.ndarray, name: str) -> np.ndarray:
        q = ensure_arm_q(values, name).copy()
        bounded = clip_to_limits(q, self.solver.joint_limits)
        if not np.allclose(q, bounded, atol=1e-8, rtol=0.0):
            raise ValueError(f"{name} exceeds URDF joint limits")
        return bounded

    def align(self, frames: Sequence[dict[str, Any]], baseline_q: np.ndarray) -> None:
        """Align one externally collected stable startup window, at least 3 frames.

        There is no second calibration-frame countdown after the startup gate.
        An unsuccessful realignment invalidates the previous alignment.
        """
        self.aligned = False
        baseline = self._bounded_q(baseline_q, "baseline_q")
        if len(frames) < 3:
            raise ValueError("alignment requires at least 3 stable frames")
        samples = [validate_skeleton_frame(frame, self.config) for frame in frames]
        observations = [_body_local_observation(sample[0]) for sample in samples]
        stable, metrics = calibration_window_is_stable(observations, self.config)
        if not stable:
            raise ValueError(f"unstable_natural_down_window:{metrics}")
        home_rotations: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            rotations = [sample[1][side] for sample in samples]
            center = Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()
            angles = np.asarray([_rotation_angle(rotation, center) for rotation in rotations])
            if np.sqrt(np.mean(angles ** 2)) > self.config.calibration_max_direction_rms_rad or np.max(angles) > 3.0 * self.config.calibration_max_direction_rms_rad:
                raise ValueError(f"unstable_{side}_hand_orientation")
            home_rotations[side] = center
        home_arms = median_home_arms(observations)
        for sample in observations:
            valid, reason = observation_is_safe(sample, home_arms, None, self.config)
            if not valid:
                raise ValueError(f"invalid_alignment:{reason}")
        self.solver.set_home_q(baseline)
        robot_poses = self.solver.end_effector_poses(baseline)
        self._robot_home_rotations = {
            side: ensure_rotation(pose[:3, :3], f"{side} robot home")
            for side, pose in zip(("left", "right"), robot_poses)
        }
        self._robot_home_directions = {
            side: self.solver.segment_directions(baseline, side) for side in ("left", "right")
        }
        self._baseline_q = baseline
        self._home_arms = home_arms
        self._home_hand_rotations = home_rotations
        self._previous_observation = observations[-1]
        self._previous_hand_rotations = {side: samples[-1][1][side].copy() for side in ("left", "right")}
        # Integration starts at the window mean, not the last noisy sample.
        # The separate previous sample above is only the tracking jump gate.
        self._integration_hand_rotations = {
            side: home_rotations[side].copy() for side in selected_sides(self.config.control_side)
        }
        self._scaled_hand_deltas = {side: np.eye(3) for side in selected_sides(self.config.control_side)}
        self.aligned = True

    def command_from_frame(
        self,
        frame: dict[str, Any],
        current_q: np.ndarray,
        last_q_cmd: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> IKCommand:
        if not self.aligned:
            raise ValueError("skeleton retargeter is not aligned")
        current = self._bounded_q(current_q, "current_q")
        reference = current if last_q_cmd is None else self._bounded_q(last_q_cmd, "last_q_cmd")
        stamp = float(frame.get("receive_time", 0.0) if timestamp is None else timestamp)
        if not np.isfinite(stamp):
            raise ValueError("timestamp must be finite")
        observation, hand_rotations = validate_skeleton_frame(frame, self.config)
        local_observation = _body_local_observation(observation)
        valid, reason = observation_is_safe(
            local_observation, self._home_arms, self._previous_observation, self.config
        )
        if not valid:
            raise ValueError(f"invalid_skeleton:{reason}")
        for side in ("left", "right"):
            if _rotation_angle(hand_rotations[side], self._previous_hand_rotations[side]) > MAX_HAND_ROTATION_STEP_RAD + 1e-8:
                raise ValueError(f"{side}_hand_orientation_jump")
        targets = {
            side: (
                relative_direction_target(self._home_arms[side].upper_direction_local, observation.arms[side].upper_direction_local, self._robot_home_directions[side][0], self.config.position_scale),
                relative_direction_target(self._home_arms[side].forearm_direction_local, observation.arms[side].forearm_direction_local, self._robot_home_directions[side][1], self.config.position_scale),
            )
            for side in selected_sides(self.config.control_side)
        }
        orientations = {}
        scaled_hand_deltas = {}
        for side in targets:
            if self.orientation_scale == 1.0:
                # Equivalent to telescoping all unscaled increments; computing
                # the absolute relative rotation avoids accumulated roundoff.
                scaled_hand_deltas[side] = hand_rotations[side] @ self._home_hand_rotations[side].T
            else:
                # Fractional gains act on each body-local SO(3) increment.
                # Scaling a global rotation logarithm has unavoidable branch
                # jumps when a full turn is followed by a change of axis.
                increment = Rotation.from_matrix(hand_rotations[side] @ self._integration_hand_rotations[side].T)
                scaled_increment = Rotation.from_rotvec(increment.as_rotvec() * self.orientation_scale).as_matrix()
                scaled_hand_deltas[side] = scaled_increment @ self._scaled_hand_deltas[side]
            orientations[side] = scaled_hand_deltas[side] @ self._robot_home_rotations[side]
        solved_q, solver_meta = self.solver.solve_segment_directions(
            timestamp=stamp,
            control_side=self.config.control_side,
            target_directions=targets,
            target_orientations=orientations,
            orientation_weight=self.solver.config.orientation_weight,
            current_q=current,
            last_q_cmd=reference,
            upper_weight=self.config.upper_direction_weight,
            forearm_weight=self.config.forearm_direction_weight,
            max_iterations=self.config.direction_iterations,
            damping=self.config.direction_damping,
        )
        proposed = ensure_arm_q(solved_q, "solved_q")
        safe_q = self._safe_command(proposed, current, reference)
        actual_poses = dict(zip(("left", "right"), self.solver.end_effector_poses(safe_q)))
        final_residuals = {}
        for side in targets:
            upper, forearm = self.solver.segment_directions(safe_q, side)
            final_residuals[side] = {
                "upper_angle_rad": direction_angle(upper, targets[side][0]),
                "forearm_angle_rad": direction_angle(forearm, targets[side][1]),
                "orientation_angle_rad": _rotation_angle(orientations[side], actual_poses[side][:3, :3]),
            }
        # Observation history is tracking quality state, not a submitted command.
        self._previous_observation = local_observation
        self._previous_hand_rotations = {side: rotation.copy() for side, rotation in hand_rotations.items()}
        self._integration_hand_rotations = {side: hand_rotations[side].copy() for side in targets}
        self._scaled_hand_deltas = scaled_hand_deltas
        return IKCommand(
            timestamp=stamp,
            control_side=self.config.control_side,
            joint_names=self.config.joint_names,
            positions=safe_q,
            motor_ids=self.config.motor_ids,
            meta={
                "solver": SKELETON_RETARGET_SOURCE,
                "source_by_side": {side: SKELETON_RETARGET_SOURCE for side in targets},
                "calibrated": True,
                "quality": "ok",
                "limited": bool(solver_meta.get("limited", False) or not np.allclose(safe_q, proposed, atol=1e-9)),
                "solver_meta": solver_meta,
                "target_directions": {side: {"upper": value[0], "forearm": value[1]} for side, value in targets.items()},
                "target_orientations": orientations,
                "final_residual_by_side": final_residuals,
                "safety": "urdf_limits+feedback_step+command_step+baseline_envelope",
            },
        )

    def _safe_command(self, proposed: np.ndarray, current: np.ndarray, reference: np.ndarray) -> np.ndarray:
        lower, upper = joint_limit_arrays(self.solver.joint_limits, 14)
        # The legacy direct envelope freezes wrists near +/-0.12 rad. Automatic
        # orientation mapping instead uses the common baseline bound on all axes.
        envelope = float(self.config.max_command_delta_from_baseline)
        step = self.config.max_joint_delta_per_step
        indices = controlled_indices(self.config.control_side)
        lower = np.maximum.reduce((lower, self._baseline_q - envelope, current - step, reference - step))
        upper = np.minimum.reduce((upper, self._baseline_q + envelope, current + step, reference + step))
        if np.any(lower[indices] > upper[indices]):
            raise ValueError("feedback, last command and baseline safety bounds do not overlap")
        safe = current.copy()
        safe[indices] = np.clip(proposed[indices], lower[indices], upper[indices])
        return safe
