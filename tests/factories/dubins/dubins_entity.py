import factory

import saferl_sim.dubins.entities as e


class Dubins2dAircraftFactory(factory.Factory):
    class Meta:
        model = e.Dubins2dAircraft

    name = "blue0"
    integration_method = "RK45"
