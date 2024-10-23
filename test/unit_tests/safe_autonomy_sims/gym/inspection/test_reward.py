"""Unit tests for rewards for the Inspection Gymnasium environment"""

import pytest
from safe_autonomy_simulation.sims.inspection.target import Target

from safe_autonomy_sims.gym.inspection.reward import observed_points_reward, weighted_observed_points_reward


def testObservedPointsRewardEnforcesInspectedPointsCannotBeUninspected():
    """Given a target being inspected, when determining the reward for inspected points and the target
    has fewer points inspected than previously, then an assertion error is raised"""
    target = Target("test_target", 2, 1)

    try:
        observed_points_reward(target, 1)
        pytest.fail("no assertion error raised when observed point is no longer observed")
    except AssertionError:
        pass


def testWeightedObservedPointsRewardEnforcesInspectedPointsCannotBeUninspected():
    """Given a target being inspected, when determining the reward for weighted inspected points and the target
    has fewer points inspected than previously, then an assertion error is raised"""
    target = Target("test_target", 2, 1)

    try:
        weighted_observed_points_reward(target, 0.5)
        pytest.fail("no assertion error raised when observed point is no longer observed")
    except AssertionError:
        pass
