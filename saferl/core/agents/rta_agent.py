"""
A trainable RTA agent class.
"""

import typing

from corl.agents.base_agent import BaseAgentParser, Functor, FunctorWrapper, TrainableBaseAgent


class FunctorMultiWrapper(Functor):
    """
    wrapped: The functor or functor wrapper configuration wrapped by this functor wrapper.
    """
    wrapped_glues: typing.Union['FunctorMultiWrapper', FunctorWrapper, Functor]

    def create_functor_glue_object(self, platform, agent_name, local_param_storage, world_param_storage):
        """Creates a glue functor wrapper object associated with a platform and an agent.

        Parameters
        ----------
        - platform : [type]
            The platform instance to be associated with the glue.
        - agent_name : [type]
            The name of the agent to be associated with the glue.
        - local_param_storage : [type]
            [description]
        - world_param_storage : [type]
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
    glues: List of wrapped glues
    """
    glues: typing.List[typing.Union[FunctorMultiWrapper, FunctorWrapper, Functor]]


class RTAAgent(TrainableBaseAgent):
    """
    Trainable RTA Agent
    """

    def __init__(self, **kwargs) -> None:
        self.config: ExtendedAgentParser
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        return ExtendedAgentParser
