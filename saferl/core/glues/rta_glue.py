"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glue containing RTA module for filtering actions.
"""
import abc
import typing
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from corl.glues.common.controller_glue import ControllerGlue
from run_time_assurance.rta import RTAModule

# from saferl.core.rta.cwh.cwh_rta import DockingRTA


def flip_rta(control):
    """
    Simple filter which flips the sign of the input control.

    Parameters
    ----------
    control: dictionary of input controls

    Returns
    -------
    dict
        The filtered control dictionary
    """
    filtered_control = OrderedDict()
    for k, v in control.items():
        filtered_control[k] = -v
    return filtered_control


class RTAGlueValidator(BaseMultiWrapperGlueValidator):
    """
    Validator for RTAGlue class.

    step_size: duration in seconds that agent's action will be applied
    rta: RTA module which filters actions based on a safety function
    state_observation_names: list of keys from observation dict whose direct observation values will be concatenated to form the RTA state
        vector
    """
    step_size: float
    state_observation_names: typing.List[str]


class RTAGlue(BaseMultiWrapperGlue):
    """
    Glue containing RTA module for filtering actions.
    """

    def __init__(self, **kwargs):
        self.config: RTAGlueValidator
        super().__init__(**kwargs)
        self.controller_glues = self._get_controller_glues(self)
        self.rta = self._instantiate_rta_module()
        self.filtered_action = None

    @property
    def get_validator(cls):
        return RTAGlueValidator

    def get_unique_name(self) -> str:
        return "RTAModule"

    def action_space(self) -> gym.spaces.Space:
        action_spaces = [glue.action_space() for glue in self.glues()]
        return gym.spaces.tuple.Tuple(tuple(action_spaces))

    def controller_glue_action_space(self) -> gym.spaces.tuple.Tuple:
        """
        Compiles the action spaces for the terminal control glues for each wrapped chain of glues
        i.e. compiles the action spaces that directly interface with the platform actuators

        Returns
        -------
        gym.spaces.Space
            The gym Space that defines the actions given to the apply_action function for the wrapped terminal controller glues
        """
        action_spaces = [controller_glue.action_space() for controller_glue in self.controller_glues]
        return action_spaces

    def apply_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict) -> None:
        assert isinstance(action, tuple)

        for i in range(len(self.glues())):
            self.glues()[i].apply_action(action[i], observation)

        desired_action = self._get_stored_action()
        filtered_action = self._filter_action(desired_action, observation)

        for controller_glue, controller_filtered_action in zip(self.controller_glues, filtered_action):
            controller_glue.apply_action(controller_filtered_action, observation)

    def _filter_action(self, desired_action: tuple, observation: typing.Dict) -> tuple:
        rta_state_vector = self._get_rta_state_vector(observation)
        rta_action_vector = self._get_action_vector_from_action(desired_action)
        filtered_action_vector = self.rta.filter_control(rta_state_vector, self.config.step_size, rta_action_vector)
        return self._get_action_from_action_vector(filtered_action_vector)

    def _get_rta_state_vector(self, observation: typing.Dict) -> np.ndarray:
        state_obs = []
        for obs_name in self.config.state_observation_names:
            try:
                state_obs.append(observation[obs_name]['direct_observation'])
            except KeyError as e:
                raise KeyError(f"state observation {obs_name} not found. Must be one of {list(observation.keys())}") from e

        state_vec = np.concatenate(state_obs)
        return state_vec

    def _get_action_vector_from_action(self, action: tuple) -> np.ndarray:
        actions_ordered = []
        for controller_action in action:
            actions_ordered += list(controller_action.values())
        control_vector = np.concatenate(actions_ordered)
        return control_vector

    def _get_action_from_action_vector(self, combined_action_vector: np.ndarray) -> tuple:
        combined_action_left = combined_action_vector
        controller_action_spaces = self.controller_glue_action_space()
        action_list = []

        for controller_action_space in controller_action_spaces:
            controller_action = OrderedDict()
            for action_key, action_space in controller_action_space.items():
                action_length = np.prod(action_space.shape)
                action_value = combined_action_left[:action_length]
                combined_action_left = combined_action_left[action_length:]
                controller_action[action_key] = action_value
            action_list.append(controller_action)
        return tuple(action_list)

    @abc.abstractmethod
    def _instantiate_rta_module(self) -> RTAModule:
        raise NotImplementedError

    def observation_space(self):
        return None

    def get_observation(self):
        return None

    def get_info_dict(self):
        return {
            "actual control": self.rta.control_actual,
            "desired control": self.rta.control_desired,
            "intervening": self.rta.intervening,
        }

    def _get_controller_glues(self, glue):
        controller_glues = []
        if isinstance(glue, ControllerGlue):
            controller_glues.append(glue)
        else:
            wrapped_list = list(glue.config.wrapped)
            for wrapped_glue in wrapped_list:
                controller_glues.extend(self._get_controller_glues(glue=wrapped_glue))
        return controller_glues

    def _get_stored_action(self) -> tuple:
        stored_action = []
        for controller_glue in self.controller_glues:
            applied_action = controller_glue.get_applied_control()
            stored_action.append(applied_action)
        return tuple(stored_action)
