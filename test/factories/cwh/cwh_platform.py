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

import safe_autonomy_sims.platforms.cwh.cwh_platform as p
from test.factories.cwh.cwh_entity import CWHSpacecraftFactory
from test.factories.base_factories.platform import BasePlatformFactory
import safe_autonomy_sims.platforms.cwh.cwh_sensors as s
import safe_autonomy_sims.platforms.common.controllers as c


class CWHPlatformFactory(BasePlatformFactory):
    class Meta:
        model = p.CWHPlatform

    platform_name = factory.Faker('first_name_nonbinary')
    platform = factory.SubFactory(
        CWHSpacecraftFactory,
        name=platform_name,
    )

    parts_list = [
        (c.RateController, {'name': 'x_thrust', "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", 'axis': 0}),
        (c.RateController, {'name': 'y_thrust', "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", 'axis': 1}),
        (c.RateController, {'name': 'z_thrust', "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", 'axis': 2}),
        (s.PositionSensor, {}),
        (s.VelocitySensor, {}),
    ]
