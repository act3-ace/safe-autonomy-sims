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
    step_size: float = 1


class BaseRTA:
    """
    Base class for a RTA filter
    """

    def __init__(self, **kwargs):
        self.config = self.get_validator()(**kwargs)
        self.enable = True
        self.intervening = False

    @classmethod
    def get_validator(cls):
        """returns the validator for this class

        Returns:
            BaseRTAValidator -- A pydantic validator to be used to validate kwargs
        """
        return BaseRTAValidator

    def filter_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict],
                      observation: typing.Dict) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
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

        if self.enable:
            return self._filter_action(action, observation)

        return action

    @abc.abstractmethod
    def _filter_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict) -> typing.Dict:
        ...


class ConstraintBasedRTA(BaseRTA):
    """TODO"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constraint_list = self._setup_constraints()

    @abc.abstractmethod
    def _setup_constraints(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_state_vector(self, observation):
        ...

    @abc.abstractmethod
    def _pred_state_vector(self, action, state_vec, step_size):
        ...

    @abc.abstractmethod
    def _get_action_vector(self, action):
        ...

    @abc.abstractmethod
    def _get_action_dict(self, action, keys):
        ...


class SimplexModule(ConstraintBasedRTA):
    """TODO"""

    def _filter_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict):
        self.monitor(action, observation)

        if self.intervening:
            return self.backup_control(action, observation)

        return action

    def monitor(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict):
        """TODO"""
        self.intervening = self._monitor(action, observation, self.intervening)

    def backup_control(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict):
        """TODO"""
        return self._backup_control(action, observation)

    @abc.abstractmethod
    def _monitor(self, action, observation, intervening):
        '''
        Returns
        -------
        bool
            True if unsafe
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _backup_control(self, action, observation):
        raise NotImplementedError()


class ExplicitSimplexModule(SimplexModule):
    '''
    Assumes one moving entity
    '''

    def _monitor(self, action, observation, intervening):
        state_vec = self._get_state_vector(observation)
        pred_state_vec = self._pred_state_vector(action, state_vec, self.config.step_size)
        for constraint_name in self.constraint_list:
            c = getattr(self, constraint_name)
            if c.h_x(pred_state_vec) < 0:
                return True

        return False

    @abc.abstractmethod
    def _backup_control(self, action, observation):
        raise NotImplementedError()


class ConstraintModule(abc.ABC):
    """TODO"""

    @abc.abstractmethod
    def h_x(self, state_vec):
        '''
        Safety Constraint (Required for all RTA):
        Function of state variables
        If h(x) >= 0, current state is safe
        Else, current state is unsafe
        Returns a constant
        '''
        raise NotImplementedError()

    def grad(self, state_vec):
        '''
        Gradient of Safety Constraint (Required for ASIF):
        Matrix size n x n, where n is the number of state variables
        Multiply matrix by state vector
        Returns a vector size n
        '''
        raise NotImplementedError()

    def alpha(self, x):
        '''
        Strengthening Function (Required for ASIF):
        Function of h(x)
        Returns a constant
        '''
        raise NotImplementedError()
