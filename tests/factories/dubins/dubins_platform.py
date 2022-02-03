import factory

import saferl.platforms.dubins.dubins_platform as p
from tests.factories.dubins.dubins_entity import Dubins2dAircraftFactory, Dubins3dAircraftFactory
from tests.factories.base_factories.platform import BasePlatformFactory
import saferl.platforms.dubins.dubins_sensors as s
import saferl.platforms.dubins.dubins_controllers as c


class Dubins2dPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.Dubins2dPlatform

    name = factory.Faker('first_name_nonbinary')
    platform = factory.SubFactory(Dubins2dAircraftFactory, name=name)
    parts_list = [
        (c.AccelerationController, {'axis': 0}),
        (c.YawRateController, {'axis': 1}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
        (s.HeadingSensor, {})
    ]


class Dubins3dPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.Dubins3dPlatform

    platform = factory.SubFactory(Dubins3dAircraftFactory)
    name = factory.Faker('first_name_nonbinary')
    parts_list = [
        (c.AccelerationController, {'axis': 0}),
        (c.YawRateController, {'axis': 1}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
        (s.HeadingSensor, {}),
        (s.FlightPathSensor, {}),
    ]
