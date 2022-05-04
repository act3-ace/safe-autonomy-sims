"""
Glue containing RTA module for filtering actions.
"""
import typing
from collections import OrderedDict

import gym
import numpy as np
from act3_rl_core.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from act3_rl_core.glues.common.controller_glue import ControllerGlue
from pydantic import PyObject

from saferl.core.rta.cwh.cwh_rta import DockingRTA


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

    rta: RTA module which filters actions based on a safety function
    """
    rta: PyObject = DockingRTA


class RTAGlue(BaseMultiWrapperGlue):
    """
    Glue containing RTA module for filtering actions.
    """

    def __init__(self, **kwargs):
        self.config: RTAGlueValidator
        super().__init__(**kwargs)
        self.controller_glues = self._get_controller_glues(self)
        self.config.rta = self.config.rta()

    @property
    def get_validator(cls):
        return RTAGlueValidator

    def action_space(self) -> gym.spaces.Space:
        action_space_dict = {}
        action_spaces = [glue.action_space() for glue in self.glues()]
        action_space_dict[self.config.name] = gym.spaces.tuple.Tuple(tuple(action_spaces))
        return gym.spaces.Dict(action_space_dict)

    def apply_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Dict) -> None:
        assert isinstance(action, dict)  # TODO: Support all action types
        action = next(iter(action.values()))
        for i in range(len(self.glues())):  # TODO: Don't assume glue/action ordering
            self.glues()[i].apply_action(action[i], observation)
        filtered_action = self.rta.filter_action(self._get_stored_action(), observation)
        for controller_glue in self.controller_glues:
            controller_glue.apply_action(filtered_action, observation)

    def observation_space(self):
        return None

    def get_observation(self):
        return None

    def _get_controller_glues(self, glue):
        controller_glues = []
        if isinstance(glue, ControllerGlue):
            controller_glues.append(glue)
        else:
            wrapped_list = list(glue.config.wrapped)
            for wrapped_glue in wrapped_list:
                controller_glues.extend(self._get_controller_glues(glue=wrapped_glue))
        return controller_glues

    def _get_stored_action(self):
        stored_action = OrderedDict()
        for controller_glue in self.controller_glues:
            applied_action = controller_glue.get_applied_control()
            stored_action.update(applied_action)
        return stored_action

    @property
    def rta(self):
        """
        RTA function attached to glue

        Returns
        -------
        RTA function attached to glue.
        """
        return self.config.rta
