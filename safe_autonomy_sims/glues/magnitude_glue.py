"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Extension of CoRL Magnitude Glue with implemented observation_prop property.

Author: John McCarroll
"""

from functools import cached_property
from corl.glues.common.magnitude import MagnitudeGlue
from corl.libraries.property import DictProp, BoxProp

class SimsMagnitudeGlue(MagnitudeGlue):

    @cached_property
    def observation_prop(self):
        prop = DictProp(name=self._uname, spaces={self.Fields.MAG: BoxProp(low=[-10000], high=[10000], unit='m/s')})
        return prop

