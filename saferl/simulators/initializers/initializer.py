"""
This module defines the base initializer class.

Author: John McCarroll
"""

import typing

from pydantic import BaseModel


class BaseInitializer():
    """
    This class defines the template for Initializer classes. Initializers are responsible
    for providing a dictionary complete with all relevant agent_reset_config values. Initializers
    encapsulatie the  initialization of randomized and conditional (dependant) agent state values.

    TODO: call method docstring
    """

    @property
    def get_validator(self) -> typing.Type[BaseModel]:
        """
        get validator for this Done Functor

        Returns:
            DoneFuncBaseValidator -- validator the done functor will use to generate a configuration
        """
        return BaseModel

    def __call__(self) -> typing.Dict:
        raise NotImplementedError
