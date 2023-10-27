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

from safe_autonomy_dynamics import cwh as e


class CWHSpacecraftFactory(factory.Factory):
    class Meta:
        model = e.CWHSpacecraft

    name = "blue0"
    integration_method = "RK45"
