"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements an extension of the CoRL Magnitude Glue with the observation_prop property.
"""

from functools import cached_property

from corl.glues.common.magnitude import MagnitudeGlue
from corl.libraries.property import BoxProp, DictProp


class SimsMagnitudeGlue(MagnitudeGlue):

    @cached_property
    def observation_prop(self):
        prop = DictProp(name=self._uname, spaces={self.Fields.MAG: BoxProp(low=[-10000], high=[10000], unit='m/s')})
        return prop
