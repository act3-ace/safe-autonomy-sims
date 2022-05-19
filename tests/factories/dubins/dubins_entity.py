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

import safe_autonomy_dynamics.dubins as e


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
