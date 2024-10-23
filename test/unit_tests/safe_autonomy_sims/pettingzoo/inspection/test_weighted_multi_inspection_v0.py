"""Unit tests for the weighted inspection environment"""

from test.unit_tests.safe_autonomy_sims.mock import MockInspectionPointSet, StaticDynamics

from safe_autonomy_sims.pettingzoo.inspection.weighted_multi_inspection_v0 import WeightedMultiInspectionEnv


def testWeightedMultiInspectionEnvTerminatesOnInspectionThresholdMet():
    """Given a weighted multiagent inspection environment and all other variables kept static, when the
    environment is stepped and the threshold for inspection success is met, the environment will report
    a termination event
    """
    env = WeightedMultiInspectionEnv()
    env.reset()

    # This isn't great because we're modifying the internal via a non-public interface, but Python allows it
    # To do this via a public interface, we'd either have to expose things we don't actually want to expose
    # just for testing or rearchitect to allow more modularity, but increases complexity.  Neither option
    # is particularly great right now, so we can live with swapping out internals.  However, this will make
    # this test brittle if the internal workings of the environment changes, such as lazy initialization
    # being done by some other func that is called after reset.
    env.chief._inspection_points = MockInspectionPointSet(parent=env.chief)  # pylint: disable=W0212
    actions = {}

    for name, deputy in env.deputies.items():
        deputy._dynamics = StaticDynamics()  # pylint: disable=W0212
        actions[name] = None

    # First step, the environment should not be terminated so we can tell what's causing
    # the environment to terminate on the next step
    _, _, terminations, _, _ = env.step(actions=actions)

    assert not any(terminations.values()), "environment should not be terminated"

    _, _, terminations, _, _ = env.step(actions=actions)

    assert any(terminations.values()), "environment should be terminated after points inspected"
