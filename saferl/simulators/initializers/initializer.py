"""
This module defines the base initializer class.

Author: John McCarroll
"""

import abc
import typing

from pydantic import BaseModel


class BaseInitializer(abc.ABC):
    """
    This class defines the template for Initializer classes. Initializers are responsible
    for providing a dictionary complete with all relevant agent_reset_config values. Initializers
    encapsulatie the  initialization of randomized and conditional (dependant) agent state values.

    TODO: call method docstring
    """

    def __init__(self, config):
        self.config = self.get_validator(**config)

    @property
    def get_validator(self) -> typing.Type[BaseModel]:
        """
        get validator for this Done Functor

        Returns:
            DoneFuncBaseValidator -- validator the done functor will use to generate a configuration
        """
        return BaseModel

    @abc.abstractmethod
    def __call__(self) -> typing.Dict:
        raise NotImplementedError


class BaseInitializerWithUnits(BaseInitializer):
    """
    This class defines initializers for handling values with units

    def __call__(self, reset_config):

    Parameters
    ----------
    reset_config: dict
        A dictionary containing the reset values for each agent. Agent names are the keys and initialization config dicts
        are the values

    Returns
    -------
    reset_config: dict
        The modified reset config of agent name to initialization values KVPs.
    """

    def __call__(self, reset_config):

        for agent_name, agent_reset_config in reset_config.items():

            reset_config[agent_name] = self.compute_initial_conds_with_units(**agent_reset_config)

        return reset_config

    def compute_initial_conds_with_units(self, **kwargs) -> typing.Dict:
        """
        Computes initial conditions for saferl simulator
        converts parameter provider values with units to raw values and calls compute_initial_conds()

        Returns
        -------
        typing.Dict
            initial conditions for platform
        """
        # TODO convert to correct unit before grabbing value
        kwargs = {k: v.value for k, v in kwargs.items()}
        return self.compute_initial_conds(**kwargs)

    @abc.abstractmethod
    def compute_initial_conds(self, **kwargs) -> typing.Dict:
        """Entry point for calling initializer specific logic with unit stripped values

        Returns
        -------
        typing.Dict
            initial conditions of platform
        """
        raise NotImplementedError
