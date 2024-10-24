"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a glue for filtering an agent's action through a Run Time Assurance filter.
"""
import abc
import typing
from collections import OrderedDict
from functools import cached_property

import gymnasium
import numpy as np
from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from corl.glues.common.controller_glue import ControllerGlue
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import Prop
from corl.libraries.units import corl_get_ureg
from run_time_assurance.rta import ConstraintBasedRTA, RTAModule

from safe_autonomy_sims.platforms.cwh.cwh_properties import TupleProp


def flip_rta(control):
    """
    Simple filter which flips the sign of the input control.

    Parameters
    ----------
    control: dict
        Dictionary of input controls.

    Returns
    -------
    dict
        The filtered control dictionary.
    """
    filtered_control = OrderedDict()
    for k, v in control.items():
        filtered_control[k] = -v
    return filtered_control


class RTAGlueValidator(BaseMultiWrapperGlueValidator):
    """
    A configuration validator for RTAGlue.

    Attributes
    ----------
    step_size: float
        duration in seconds that agent's action will be applied
    state_observation_names: list[str]
        list of keys from observation dict whose direct observation values will be concatenated to form the RTA state vector
    enabled: bool
        True if RTA is enabled
    """
    step_size: float
    state_observation_names: typing.List[str]
    enabled: bool = True


class RTAGlue(BaseMultiWrapperGlue):
    """
    Glue containing RTA module for filtering actions.
    """

    def __init__(self, **kwargs):
        self.config: RTAGlueValidator
        super().__init__(**kwargs)
        self.controller_glues = self._get_controller_glues(self)
        rta_args = self._get_rta_args()
        singleton = self._get_singleton()
        self.rta = singleton.instance(**rta_args).rta
        self.rta.enable = self.config.enabled
        self.filtered_action = None

    @staticmethod
    def get_validator():
        return RTAGlueValidator

    def get_unique_name(self) -> str:
        return "RTAModule"

    @cached_property
    def action_prop(self) -> typing.Optional[Prop]:
        """
        Build the action property for the controllers that defines the action this glue produces

        Returns
        -------
        typing.Optional[Prop]
            The Property that defines what this glue requires for an action
        """
        # collect action properties of wrapped controllers
        action_props = tuple(glue.action_prop for glue in self.glues())
        prop = TupleProp(spaces=action_props)
        return prop

    @cached_property
    def action_space(self) -> typing.Optional[gymnasium.spaces.Space]:
        action_spaces = [gymnasium.spaces.Dict({glue.action_prop.name: glue.action_space}) for glue in self.glues()]
        return gymnasium.spaces.tuple.Tuple(tuple(action_spaces))

    def controller_glue_action_space(self) -> gymnasium.spaces.tuple.Tuple:
        """
        Compiles the action spaces for the terminal control glues for each wrapped chain of glues
        i.e. compiles the action spaces that directly interface with the platform actuators

        Returns
        -------
        gymnasium.spaces.Space
            The gymnasium Space that defines the actions given to the apply_action function for the wrapped terminal controller glues
        """
        action_spaces = [
            gymnasium.spaces.Dict({controller_glue.action_prop.name: controller_glue.action_space})
            for controller_glue in self.controller_glues
        ]
        return gymnasium.spaces.tuple.Tuple(tuple(action_spaces))

    def apply_action(
        self,
        action: EnvSpaceUtil.sample_type,
        observation: EnvSpaceUtil.sample_type,
        action_space: OrderedDict,
        obs_space: OrderedDict,
        obs_units: OrderedDict
    ) -> None:
        assert isinstance(action, tuple)

        for i in range(len(self.glues())):
            glue = self.glues()[i]
            glue.apply_action(action[i][glue.action_prop.name], observation, action_space, obs_space, obs_units)

        desired_action = self._get_stored_action()
        filtered_action = self._filter_action(desired_action, observation)

        for controller_glue, controller_filtered_action in zip(self.controller_glues, filtered_action):
            controller_glue.apply_action(
                controller_filtered_action[controller_glue.action_prop.name], observation, action_space, obs_space, obs_units
            )

    def _filter_action(self, desired_action: tuple, observation: typing.Dict) -> tuple:
        rta_state_vector = self._get_rta_state_vector(observation)
        rta_action_vector = self._get_action_vector_from_action(desired_action).flatten()
        filtered_action_vector = self.rta.filter_control(rta_state_vector, self.config.step_size, rta_action_vector)
        if isinstance(self.rta, ConstraintBasedRTA):
            self.rta.update_constraint_values(rta_state_vector)
        return self._get_action_from_action_vector(filtered_action_vector)

    def _get_rta_state_vector(self, observation: typing.Dict) -> np.ndarray:
        state_obs = []
        for obs_name in self.config.state_observation_names:
            try:
                state_obs.append(observation[obs_name]['direct_observation'].m)
            except KeyError as e:
                raise KeyError(f"state observation {obs_name} not found. Must be one of {list(observation.keys())}") from e

        state_vec = np.concatenate(state_obs)
        return state_vec

    def _get_action_vector_from_action(self, action: tuple) -> np.ndarray:
        actions_ordered = []
        for controller_action in action:
            actions_ordered.append(controller_action.m)
        control_vector = np.array(actions_ordered)
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
                controller_action[action_key] = corl_get_ureg().Quantity(action_value, "dimensionless")
            action_list.append(controller_action)
        return tuple(action_list)

    @abc.abstractmethod
    def _get_singleton(self):
        """
        Get singleton class containing RTA module.

        Returns
        -------
        RTASingleton
            Custom singleton.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_rta_args(self) -> dict:
        """
        Get RTA arguments.

        Returns
        -------
        dict
            RTA args to be passed at initialization.
        """
        raise NotImplementedError

    @cached_property
    def normalized_action_space(self) -> typing.Optional[gymnasium.spaces.Space]:
        return self.action_space

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        # TODO: Add info to callback for eval metrics
        # obs = {"intervening": int(self.rta.intervening)}
        # if isinstance(self.rta, ConstraintBasedRTA):
        #     info = self.rta.generate_info()['constraints'].items()
        #     n_val = np.array([-1.], dtype=np.float32)
        #     obs['constraints'] = {k: np.array([v], dtype=np.float32) if not np.isnan(v) else n_val for k, v in info}
        # return obs
        return None

    def get_info_dict(self):
        return {
            "actual_control": self.rta.control_actual,
            "desired_control": self.rta.control_desired,
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


class RTASingleton(abc.ABC):
    """
    Implement RTA module as a singleton.
    This prevents reinitialization, which can cause memory leaks due to jit.
    """

    _instance = None

    @classmethod
    def instance(cls, **rta_args):
        """Create singleton"""
        if not cls._instance:
            cls._instance = cls.__new__(cls)
            cls.rta = cls._create_rta_module(cls._instance.__class__, **rta_args)
        return cls._instance

    def __init__(self, **kwargs):
        raise RuntimeError("Use the 'instance' method to initialize RTA objects.")

    @abc.abstractmethod
    def _create_rta_module(self, **rta_args: dict) -> RTAModule:
        """
        Initialize the RTA module

        Parameters
        ----------
        rta_args: dict
            Dictionary of args passed to RTA.

        Returns
        -------
        RTAModule
            Custom RTA module.
        """
        raise NotImplementedError
