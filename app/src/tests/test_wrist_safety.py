from __future__ import annotations

import unittest

import numpy as np

from teleop_wrist_safety import (
    WristBehaviorLimits,
    validate_wrist_behavior_candidate,
    wrist_position_errors,
)


class WristSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = WristBehaviorLimits(
            max_wrist_step_rad=0.06,
            max_elbow_step_rad=0.06,
            soft_limit_margin_rad=0.01,
            max_wrist_position_error_m=0.010,
        )
        self.joint_limits = [(-1.0, 1.0)] * 14

    def validate(
        self,
        candidate: np.ndarray,
        reference: np.ndarray | None = None,
        control_side: str = "both",
        errors: dict[str, float] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return validate_wrist_behavior_candidate(
            candidate,
            np.zeros(14) if reference is None else reference,
            self.joint_limits,
            control_side,
            {"left": 0.001, "right": 0.001} if errors is None else errors,
            self.limits,
        )

    def test_accepts_continuous_candidate(self) -> None:
        candidate = np.zeros(14)
        candidate[[3, 4, 5, 6, 10, 11, 12, 13]] = 0.04
        self.assertEqual(self.validate(candidate), {})

    def test_rejects_only_the_side_with_wrist_or_elbow_jump(self) -> None:
        candidate = np.zeros(14)
        candidate[5] = 0.08
        candidate[10] = -0.09
        reasons = self.validate(candidate)
        self.assertIn("left", reasons)
        self.assertIn("right", reasons)
        self.assertTrue(any(reason.startswith("wrist_step=") for reason in reasons["left"]))
        self.assertTrue(any(reason.startswith("elbow_step=") for reason in reasons["right"]))

        left_only = self.validate(candidate, control_side="left")
        self.assertEqual(set(left_only), {"left"})

    def test_soft_limit_allows_recovery_but_rejects_deeper_motion(self) -> None:
        reference = np.zeros(14)
        reference[5] = 0.995
        recovering = reference.copy()
        recovering[5] = 0.985
        self.assertEqual(self.validate(recovering, reference, control_side="left"), {})

        deeper = reference.copy()
        deeper[5] = 0.999
        reasons = self.validate(deeper, reference, control_side="left")
        self.assertTrue(any("outside_soft_limit" in reason for reason in reasons["left"]))

    def test_rejects_wrist_position_residual(self) -> None:
        reasons = self.validate(
            np.zeros(14),
            control_side="right",
            errors={"right": 0.015},
        )
        self.assertTrue(any(reason.startswith("wrist_position_error=") for reason in reasons["right"]))

    def test_wrist_position_errors_uses_control_point_not_tcp(self) -> None:
        actual_left = np.eye(4)
        actual_right = np.eye(4)
        actual_left[:3, 3] = [0.1, 0.2, 0.3]
        left_target = actual_left.copy()
        left_target[0, 3] += 0.004
        right_target = actual_right.copy()
        errors = wrist_position_errors(
            (actual_left, actual_right),
            {"left": left_target, "right": right_target},
            "left",
        )
        self.assertAlmostEqual(errors["left"], 0.004)
        self.assertNotIn("right", errors)


if __name__ == "__main__":
    unittest.main()