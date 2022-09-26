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

    parts_list = [
        (c.ThrustController, {'name': 'x_thrust', 'axis': 0}),
        (c.ThrustController, {'name': 'y_thrust', 'axis': 1}),
        (c.ThrustController, {'name': 'z_thrust', 'axis': 2}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
    ]
