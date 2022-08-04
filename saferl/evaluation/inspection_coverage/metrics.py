"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.metrics.metric import Metric


class TrueStateMetric(MetricGeneratorTerminalEventScope):
    """Generates DoneStatusCodes value for the agent in an event

    Metric: Discrete
    Scope: Event
    """

    platform_properties: typing.List[str]

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]
        state_history = {agent_id: {}}  # type: typing.Dict[str, dict]

        for step_num, step in enumerate(params.steps):
            # find platform
            current_index = None
            for index, platform in enumerate(step.platforms):
                if platform["name"] == agent_id:
                    current_index = index

            # store state state_history
            state_history[agent_id][step_num] = {}
            for var in self.platform_properties:
                state_history[agent_id][step_num][var] = step.platforms[current_index][var]
                state_history[agent_id][step_num]['reward'] = step.agents[agent_id].total_reward

 
        return state_history



# class FieldOfViewMetric(MetricGeneratorTerminalEventScope):
#     """Generates DoneStatusCodes value for the agent in an event

#     Metric: Discrete
#     Scope: Event
#     """

#     # TODO: change glue names + remove hardcoded deputies
#     # deputies: typing.List[str]

#     def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

#         if "agent_id" not in kwargs:
#             raise RuntimeError("Expecting \"agent_id\" to be provided")

#         agent_id = kwargs["agent_id"].split('.')[0]
#         fov_history = {
#             "blue1_fov": {},
#             "blue2_fov": {},
#             "blue3_fov": {},
#         }  # type: typing.Dict[str, dict]

#         for step_num, step in enumerate(params.steps):
#             # store state fov_history
#             fov_history["blue1_fov"][step_num] = step.agents[agent_id].observations[
#                 "Relative_Position_Blue1FieldOfView.field_of_view_indicator"]
#             fov_history["blue2_fov"][step_num] = step.agents[agent_id].observations[
#                 "Relative_Position_Blue2FieldOfView.field_of_view_indicator"]
#             fov_history["blue3_fov"][step_num] = step.agents[agent_id].observations[
#                 "Relative_Position_Blue3FieldOfView.field_of_view_indicator"]

#         return fov_history

# class EstimatedStateMetric(MetricGeneratorTerminalEventScope):
#     """Generates DoneStatusCodes value for the agent in an event

#     Metric: Discrete
#     Scope: Event
#     """

#     def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

#         if "agent_id" not in kwargs:
#             raise RuntimeError("Expecting \"agent_id\" to be provided")

#         agent_id = kwargs["agent_id"].split('.')[0]
#         ekf_history = {
#             "blue1_ekf": {},
#             "blue2_ekf": {},
#             "blue3_ekf": {},
#         }  # type: typing.Dict[str, dict]

#         for step_num, step in enumerate(params.steps):

#             # TODO: make hardocded agent IDs configurable via list of agent_names
#             # if not agent_id in step.agents:
#             #     raise ValueError('given agent_id, {}, not found in Step.agents'.format(agent_id))

#             # store state ekf_history
#             ekf_history["blue1_ekf"][step_num] = step.agents[agent_id].observations["Relative_Position_Blue1_EKF.extended_kalman_filter"]
#             ekf_history["blue2_ekf"][step_num] = step.agents[agent_id].observations["Relative_Position_Blue2_EKF.extended_kalman_filter"]
#             ekf_history["blue3_ekf"][step_num] = step.agents[agent_id].observations["Relative_Position_Blue3_EKF.extended_kalman_filter"]

#         return ekf_history
