"""Unit tests for the weighted inspection environment"""

from test.unit_tests.safe_autonomy_sims.mock import MockInspectionPointSet, StaticCWHDynamics

from safe_autonomy_sims.pettingzoo.inspection.weighted_multi_inspection_v0 import WeightedMultiInspectionEnv


def test_weighted_multi_inspection_env_terminates_on_inspection_threshold_met(mocker):
    """Given a weighted multiagent inspection environment and all other variables kept static, when the
    environment is stepped and the threshold for inspection success is met, the environment will report
    a termination event
    """
    # This is a little dangerous as other portions of the environment might be using CWHDynamics, but
    # it probably doesn't matter
    mocker.patch('safe_autonomy_simulation.sims.inspection.inspection_points.InspectionPointSet', MockInspectionPointSet)
    mocker.patch('safe_autonomy_simulation.sims.spacecraft.point_model.CWHDynamics', StaticCWHDynamics)

    env = WeightedMultiInspectionEnv()
    env.reset()

    actions = {name: None for name, _ in env.deputies.items()}

    # First step, the environment should not be terminated so we can tell what's causing
    # the environment to terminate on the next step
    _, _, terminations, _, _ = env.step(actions=actions)

    assert not any(terminations.values()), "environment should not be terminated"

    _, _, terminations, _, _ = env.step(actions=actions)

    assert any(terminations.values()), "environment should be terminated after points inspected"
