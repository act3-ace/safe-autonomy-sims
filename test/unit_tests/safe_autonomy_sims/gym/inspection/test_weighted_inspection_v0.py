"""Unit tests for the weighted inspection environment"""

from test.unit_tests.safe_autonomy_sims.mock import MockInspectionPointSet, StaticCWHDynamics

from safe_autonomy_sims.gym.inspection.weighted_inspection_v0 import WeightedInspectionEnv


def test_weighted_inspection_env_terminates_on_inspection_threshold_met(mocker):
    """Given a weighted inspection environment and all other variables kept static, when the environment
    is stepped and the threshold for inspection success is met, the environment will report a termination
    event
    """
    # This is a little dangerous as other portions of the environment might be using CWHDynamics, but
    # it probably doesn't matter
    mocker.patch('safe_autonomy_simulation.sims.inspection.inspection_points.InspectionPointSet', MockInspectionPointSet)
    mocker.patch('safe_autonomy_simulation.sims.spacecraft.point_model.CWHDynamics', StaticCWHDynamics)

    env = WeightedInspectionEnv()
    env.reset()

    # First step, the environment should not be terminated so we can tell what's causing
    # the environment to terminate on the next step
    _, _, terminated, _, _ = env.step(action=env.action_space.sample())
    assert not terminated, "environment should not be terminated"

    _, _, terminated, _, _ = env.step(action=env.action_space.sample())

    assert terminated, "environment should be terminated after points inspected"
