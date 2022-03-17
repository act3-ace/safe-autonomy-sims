import factory

import saferl.platforms.cwh.cwh_platform as p
from tests.factories.cwh.cwh_entity import CWHSpacecraftFactory
from tests.factories.base_factories.platform import BasePlatformFactory
import saferl.platforms.cwh.cwh_sensors as s
import saferl.platforms.cwh.cwh_controllers as c


class CWHPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.CWHPlatform

    platform_name = factory.Faker('first_name_nonbinary')
    platform = factory.SubFactory(
        CWHSpacecraftFactory,
        name=platform_name,
    )

    platform_config = [
        (c.ThrustController, {'name': 'x_thrust', 'axis': 0}),
        (c.ThrustController, {'name': 'y_thrust', 'axis': 1}),
        (c.ThrustController, {'name': 'z_thrust', 'axis': 2}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
    ]
