import factory

import saferl.platforms.dubins.dubins_platform as p
from tests.factories.dubins.dubins_entity import Dubins2dAircraftFactory
from tests.factories.base_factories.platform import BasePlatformFactory
import saferl.platforms.dubins.dubins_sensors as s
import saferl.platforms.dubins.dubins_controllers as c


class Dubins2dPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.DubinsPlatform

    platform = factory.SubFactory(Dubins2dAircraftFactory)
    name = factory.Faker('first_name_nonbinary')
    parts_list = [
        (c.AccelerationController, {'axis': 0}),
        (c.YawRateController, {'axis': 1}),

    ]
