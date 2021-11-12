"""
Glue containing RTA module for filtering actions.
"""

import typing

import gym
import numpy as np

from saferl.glues.multi_wrapper import MultiWrapperGlue


class RTAGlue(MultiWrapperGlue):
    """
    Glue containing RTA module for filtering actions.
    """

    def __init__(self):
        super().__init__()
        self.controller_glues = []

    def action_space(self) -> gym.spaces.Space:
        action_space_dict = {}
        action_spaces = [glue.action_space for glue in self.glues()]
        action_space_dict[self.config.name] = gym.spaces.tuple.Tuple(tuple(action_spaces))
        return gym.spaces.Dict(action_space_dict)

    def apply_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict]) -> None:
        pass

    def observation_space(self) -> gym.spaces.Space:
        return None

    def get_observation(self) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        return None
