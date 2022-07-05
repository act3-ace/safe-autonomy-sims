"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glues which reads observations from wrapped glues and can normalize the observations using a defined mu and sigma or the
sensor bounds.

Author: Jamie Cunningham
"""

import typing

from corl.glues.base_glue import BaseAgentGlue, BaseAgentGlueValidator
from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue
from corl.glues.base_wrapper import BaseWrapperGlue

from saferl.glues.normal.normal_glue import NormalGlue, NormalGlueValidator


class NormalWrapperGlueValidator(NormalGlueValidator):
    """
    Validator for NormalWrapperGlue config.

    Wrapped: BaseAgentGlue
        The wrapped glue instance.
    """
    wrapped: BaseAgentGlue

    class Config:  # pylint: disable=C0115, R0903
        arbitrary_types_allowed = True


class NormalWrapperGlue(NormalGlue, BaseWrapperGlue):
    """
    Wrapper glue which allows normalization of wrapped glue actions and observations using custom mu and sigma.
    """

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        return NormalWrapperGlueValidator


class NormalMultiWrapperGlueValidator(NormalGlueValidator):
    """
    Wrapped: BaseAgentGlue
        The wrapped glue instance.
    """
    wrapped: typing.List[BaseAgentGlue]

    class Config:  # pylint: disable=C0115, R0903
        arbitrary_types_allowed = True


class NormalMultiWrapperGlue(NormalGlue, BaseMultiWrapperGlue):
    """
    Wrapper glue which allows normalization of wrapped glue actions and observations using custom mu and sigma.
    """

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        return NormalMultiWrapperGlueValidator
