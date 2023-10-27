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

import typing

import factory
from corl.simulators.base_platform import BasePlatform


class BasePlatformFactory(factory.Factory):
    class Meta:
        model = BasePlatform

    platform_name = None
    platform = None
    parts_list: typing.List = []
