"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements agents capable of using run-time-assurance.
"""

import typing

from corl.agents.base_agent import BaseAgentParser, Functor, FunctorWrapper, TrainableBaseAgent


class FunctorMultiWrapper(Functor):
    """
    A wrapper around one or more functors.

    Attributes
    ----------
    wrapped: Functor, FunctorWrapper
        The functor or functor wrapper configuration wrapped by this functor wrapper.
    """
    wrapped_glues: typing.Union['FunctorMultiWrapper', FunctorWrapper, Functor]

    def create_functor_glue_object(self, platform, agent_name, local_param_storage, world_param_storage):
        """
        Creates a glue functor wrapper object associated with a platform and an agent.

        Parameters
        ----------
        platform : BasePlatform
            The platform instance to be associated with the glue.
        agent_name : str
            The name of the agent to be associated with the glue.
        local_param_storage : [type]
            [description]
        world_param_storage : [type]
            [description]

        Returns
        -------
        MultiWrapperGlue:
            An instance of a glue functor wrapper initialized with the given config.
        """
        wrapped_glues = [
            glue.create_functor_glue_object(platform, agent_name, local_param_storage, world_param_storage) for glue in self.wrapped_glues
        ]
        kwargs = self.config
        kwargs["name"] = self.name
        kwargs["wrapped_glues"] = wrapped_glues
        kwargs["platform"] = platform
        kwargs["agent_name"] = agent_name
        return self.functor(**kwargs)


FunctorMultiWrapper.update_forward_refs()


class ExtendedAgentParser(BaseAgentParser):
    """
    Agent parser extension which parses glue objects.

    Attributes
    ----------
    glues: list
        List of wrapped glues
    """
    glues: typing.List[typing.Union[FunctorMultiWrapper, FunctorWrapper, Functor]]


class RTAAgent(TrainableBaseAgent):
    """
    A trainable RTA Agent

    Attributes
    ----------
    config: ExtendedAgentParser
        The agent configuration
    """

    def __init__(self, **kwargs) -> None:
        self.config: ExtendedAgentParser
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return ExtendedAgentParser
