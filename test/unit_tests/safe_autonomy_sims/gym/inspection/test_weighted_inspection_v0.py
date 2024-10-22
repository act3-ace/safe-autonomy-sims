"""Unit tests for the weighted inspection environment"""

from test.unit_tests.safe_autonomy_sims.gym.inspection.mock import MockInspectionPointSet, StaticDynamics

from safe_autonomy_sims.gym.inspection.weighted_inspection_v0 import WeightedInspectionEnv


def testWeightedInspectionEnvTerminatesOnInspectionThresholdMet():
    """Given a weighted inspection environment and all other variables kept static, when the environment
    is stepped and the threshold for inspection success is met, the environment will report a termination
    event
    """
    env = WeightedInspectionEnv()
    env.reset()

    # This isn't great because we're modifying the internal via a non-public interface, but Python allows it
    # To do this via a public interface, we'd either have to expose things we don't actually want to expose
    # just for testing or rearchitect to allow more modularity, but increases complexity.  Neither option
    # is particularly great right now, so we can live with swapping out internals.  However, this will make
    # this test brittle if the internal workings of the environment changes, such as lazy initialization
    # being done by some other func that is called after reset.
    env.chief._inspection_points = MockInspectionPointSet(parent=env.chief)  # pylint: disable=W0212
    env.deputy._dynamics = StaticDynamics()  # pylint: disable=W0212

    # First step, the environment should not be terminated so we can tell what's causing
    # the environment to terminate on the next step
    _, _, terminated, _, _ = env.step(action=None)
    assert not terminated, "environment should not be terminated"

    _, _, terminated, _, _ = env.step(action=None)

    assert terminated, "environment should be terminated after points inspected"

    assert False
