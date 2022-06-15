"""
This module defines the base initializer class.

Author: John McCarroll
"""

import typing

from pydantic import BaseModel, PyObject


class InitializerFunctor(BaseModel):
    """
    The functor used by SafeRLSimulator + pydantic to pass values to initializers from the environment config
    """
    functor: PyObject
    config: typing.Dict[str, typing.Any]


class BaseInitializer(BaseModel):
    """
    This class defines the template for Initializer classes. Initializers are responsible
    for providing a dictionary complete with all relevant agent_reset_config values. Initializers
    encapsulatie the  initialization of randomized and conditional (dependant) agent state values.

    TODO: call method docstring
    """

    def __call__(self) -> typing.Dict:
        raise NotImplementedError
