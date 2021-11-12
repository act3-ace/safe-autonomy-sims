"""
Glue that wraps other glues
"""
import typing

import gym
import numpy as np
from act3_rl_core.glues.base_glue import BaseAgentGlue
from act3_rl_core.glues.base_wrapper import BaseWrapperGlueValidator


class MultiWrapperGlueValidator(BaseWrapperGlueValidator):
    """
    wrapped: a list of wrapped glue instances
    """
    wrapped_glues: typing.List[BaseAgentGlue]


class MultiWrapperGlue(BaseAgentGlue):
    """
    A base object that glues can inherit in order to "wrap" multiple glue instances
    """

    @classmethod
    def get_validator(cls):
        return MultiWrapperGlueValidator

    def glues(self) -> typing.List[BaseAgentGlue]:
        """
        Get the wrapped glue instances.

        Returns
        -------
            list of glue instances
        """
        return self.config.wrapped_glues

    def set_agent_removed(self, agent_removed: bool = True) -> None:
        super().set_agent_removed(agent_removed)
        for glue in self.glues():
            glue.set_agent_removed(agent_removed)

    def action_space(self) -> gym.spaces.Space:
        pass

    def apply_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict]) -> None:
        pass

    def observation_space(self) -> gym.spaces.Space:
        pass

    def get_observation(self) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        pass
