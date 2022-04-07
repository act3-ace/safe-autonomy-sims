import factory

from saferl.backend.dubins import entities as e


class Dubins2dAircraftFactory(factory.Factory):
    class Meta:
        model = e.Dubins2dAircraft

    name = "blue0"
    integration_method = "RK45"


class Dubins3dAircraftFactory(factory.Factory):
    class Meta:
        model = e.Dubins3dAircraft

    name = "blue0"
    integration_method = "RK45"
