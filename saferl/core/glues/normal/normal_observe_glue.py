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

Glue that reads measurements from a platform sensor and can normalize the measurements using a defined mu and sigma or the sensor bounds.

Author: Jamie Cunningham
"""

import typing

from corl.glues.base_glue import BaseAgentGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator

from saferl.core.glues.normal.normal_glue import NormalGlue, NormalGlueNormalizationValidator


class NormalObserveSensorGlueValidator(ObserveSensorValidator):
    """
    Validate NormalObserveSensorGlue
    normalization: enable normalization and set mu and sigma or max and min
    """
    normalization = NormalGlueNormalValidator = NormalGlueNormalizationValidator()


class NormalObserveSensorGlue(NormalGlue, ObserveSensor):
    """
    ObserveSensor glue which allows normalization of glue observations using custom mu and sigma.
    """

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        return NormalObserveSensorGlueValidator
