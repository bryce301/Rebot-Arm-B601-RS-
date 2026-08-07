import math
import unittest

import numpy as np

from scripts.rebot_motion_common import (
    JOINT_NAMES,
    limit_gripper_target_step,
    limit_gripper_trajectory_steps,
)


class GripperStepLimitTest(unittest.TestCase):
    def test_limits_only_gripper_target(self) -> None:
        previous = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        target = np.arange(len(JOINT_NAMES), dtype=np.float64)

        limited = limit_gripper_target_step(
            target,
            previous,
            JOINT_NAMES,
            math.radians(3.0),
        )

        np.testing.assert_array_equal(limited[:-1], target[:-1])
        self.assertAlmostEqual(limited[-1], math.radians(3.0))

    def test_limits_consecutive_trajectory_steps(self) -> None:
        trajectory = np.zeros((5, len(JOINT_NAMES)), dtype=np.float64)
        trajectory[:, -1] = np.radians([0.0, 30.0, -30.0, 100.0, 100.0])

        limited, clipped_count = limit_gripper_trajectory_steps(
            trajectory,
            JOINT_NAMES,
            math.radians(3.0),
        )

        steps_deg = np.degrees(np.diff(limited[:, -1]))
        self.assertTrue(np.all(np.abs(steps_deg) <= 3.0 + 1e-12))
        np.testing.assert_allclose(
            np.degrees(limited[:, -1]),
            [0.0, 3.0, 0.0, 3.0, 6.0],
            atol=1e-12,
        )
        self.assertEqual(clipped_count, 4)


if __name__ == "__main__":
    unittest.main()
