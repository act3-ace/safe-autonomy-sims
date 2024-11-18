"""Unit tests for the weighted inspection environment"""

import pytest
from test.unit_tests.safe_autonomy_sims.mock import (
    MockInspectionPointSet,
    StaticSixDOFDynamics,
)
from safe_autonomy_sims.pettingzoo.inspection.sixdof_multi_inspection_v0 import (
    WeightedSixDofMultiInspectionEnv,
)


@pytest.mark.unit_test
def test_weighted_6DOF_multi_inspection_env_terminates_on_inspection_threshold_met(
    mocker,
):
    """Given a weighted 6DOF multiagent inspection environment and all other variables kept static, when
    the environment is stepped and the threshold for inspection success is met, the environment will
    report a termination event
    """
    mocker.patch(
        "safe_autonomy_simulation.sims.inspection.inspection_points.InspectionPointSet",
        MockInspectionPointSet,
    )
    mocker.patch(
        "safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics",
        StaticSixDOFDynamics,
    )

    env = WeightedSixDofMultiInspectionEnv()
    env.reset()

    actions = {
        name: env.action_space(name).sample() for name, _ in env.deputies.items()
    }

    # First step, the environment should not be terminated so we can tell what's causing
    # the environment to terminate on the next step
    _, _, terminations, _, _ = env.step(actions=actions)

    assert not any(terminations.values()), "environment should not be terminated"

    _, _, terminations, _, _ = env.step(actions=actions)

    assert any(
        terminations.values()
    ), "environment should be terminated after points inspected"
