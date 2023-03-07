"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------
"""

import factory

import saferl.platforms.dubins.dubins_platform as p
from tests.factories.dubins.dubins_entity import Dubins2dAircraftFactory, Dubins3dAircraftFactory
from tests.factories.base_factories.platform import BasePlatformFactory
import saferl.platforms.dubins.sensors.dubins_sensors as s
import saferl.platforms.common.controllers as c


class Dubins2dPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.Dubins2dPlatform

    platform_name = factory.Faker('first_name_nonbinary')
    platform = factory.SubFactory(
        Dubins2dAircraftFactory,
        name=platform_name,
    )

    parts_list = [
        (c.RateController, {'name': 'Acceleration', 'axis': 0, "property_class": "saferl.platforms.dubins.dubins_properties.AccelerationProp"}),
        (c.RateController, {'name': 'YawRate', 'axis': 1, "property_class": "saferl.platforms.dubins.dubins_properties.YawRateProp"}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
        (s.HeadingSensor, {})
    ]


class Dubins3dPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.Dubins3dPlatform

    platform = factory.SubFactory(Dubins3dAircraftFactory)
    platform_name = factory.Faker('first_name_nonbinary')
    parts_list = [
        (c.RateController, {'name': 'Acceleration', 'axis': 0, "property_class": "saferl.platforms.dubins.dubins_properties.AccelerationProp"}),
        (c.RateController, {'name': 'YawRate', 'axis': 1, "property_class": "saferl.platforms.dubins.dubins_properties.YawRateProp"}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
        (s.HeadingSensor, {}),
        (s.FlightPathSensor, {}),
    ]
