"""
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
