"""
Glues which reads observations from wrapped glues and can normalize the observations using a defined mu and sigma or the sensor bounds.

Author: Jamie Cunningham
"""

import typing

from act3_rl_core.glues.base_glue import BaseAgentGlue, BaseAgentGlueValidator
from act3_rl_core.glues.base_multi_wrapper import BaseMultiWrapperGlue
from act3_rl_core.glues.base_wrapper import BaseWrapperGlue

from saferl.glues.normal.normal_glue import NormalGlue, NormalGlueValidator


class NormalWrapperGlueValidator(NormalGlueValidator):
    """
    wrapped - the wrapped glue instance
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
    wrapped - the wrapped glue instance
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
