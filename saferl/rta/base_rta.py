"""
This module contains base classes for RTA filters.
"""

import abc
import typing

import numpy as np
from pydantic import BaseModel


class BaseRTAValidator(BaseModel):
    """
    A validator for the BaseRTA class.

    name : str
        name of RTA class
    """
    name: typing.Optional[str]


class BaseRTA:
    """
    Base class for a RTA filter
    """

    def __init__(self, **kwargs):
        self.config = self.get_validator()(**kwargs)

    @classmethod
    def get_validator(cls):
        """returns the validator for this class

        Returns:
            BaseRTAValidator -- A pydantic validator to be used to validate kwargs
        """
        return BaseRTAValidator

    @abc.abstractmethod
    def filter_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict) -> typing.Dict:
        """
        Filter an action to ensure it fits the given safety constraint for the environment.

        Parameters
        ----------
        action : Union[np.ndarray, tuple, dict]
            The action to be monitored and filtered by the RTA algorithm
        observation : dict
            Observation buffer containing the current observation of the environment

        Returns
        -------
            Union[np.ndarray, tuple, dict]
                The filtered action
        """
        ...
