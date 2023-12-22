"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements evaluation metrics for the inspection environment.
"""
import typing

import numpy as np
from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.types.terminals.real import Real
from corl.evaluation.metrics.types.terminals.void import Void


class DeltaV(MetricGeneratorTerminalEventScope):
    """Generates vector of Dicts for the control for each step during an event for an agent
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        arr = 0
        for step in params.steps:

            if agent_id not in step.agents or step.agents[agent_id] is None:
                break

            map_act = step.agents[agent_id].actions

            if map_act is None:
                continue
            # Create a non terminal metric (Dict) that is comprised of the terminal (Real) actions
            real_dict: typing.Dict[str, Metric] = {key: Real(map_act[key]) for key in map_act.keys()}
            arr += np.sum([abs(c.value[0]) for k, c in real_dict.items() if 'thrust' in k]) / 12 * 10  # TODO: HARDCODED

        return Real(arr)


class InspectedPoints(MetricGeneratorTerminalEventScope):
    """Generates single Real indicating the total inspected points for an event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        # Find the last step with observation data from the platform in question
        # If the platform isn't in the episode at all, return Void
        steps_with_platform_in_question = [item for item in params.steps if agent_id in item.agents and item.agents[agent_id] is not None]
        if len(steps_with_platform_in_question) == 0:
            return Void()

        last_step_with_platform_data = steps_with_platform_in_question[-1]

        if last_step_with_platform_data.agents[agent_id] is None:
            raise RuntimeError("Non Op")

        return Real(last_step_with_platform_data.platforms[0]['sensors']['Sensor_InspectedPoints']['measurement'][0])


class EpisodeLength(MetricGeneratorTerminalEventScope):
    """Generates single Real indicating the episode length for an event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        # Find the last step with observation data from the platform in question
        # If the platform isn't in the episode at all, return Void
        steps_with_platform_in_question = [item for item in params.steps if agent_id in item.agents and item.agents[agent_id] is not None]
        if len(steps_with_platform_in_question) == 0:
            return Void()

        return Real(len(steps_with_platform_in_question))


class Success(MetricGeneratorTerminalEventScope):
    """Generates single Real indicating the success percentage for an event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        # Find the last step with observation data from the platform in question
        # If the platform isn't in the episode at all, return Void
        steps_with_platform_in_question = [item for item in params.steps if agent_id in item.agents and item.agents[agent_id] is not None]
        if len(steps_with_platform_in_question) == 0:
            return Void()

        if params.dones['blue0']['SuccessfulInspectionDoneFunction']:  # TODO: Hardcoded platform name
            val = 1
        else:
            val = 0

        return Real(val)


class SafeSuccess(MetricGeneratorTerminalEventScope):
    """Generates single Real indicating the success percentage for an event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        # Find the last step with observation data from the platform in question
        # If the platform isn't in the episode at all, return Void
        steps_with_platform_in_question = [item for item in params.steps if agent_id in item.agents and item.agents[agent_id] is not None]
        if len(steps_with_platform_in_question) == 0:
            return Void()

        if params.dones['blue0']['SafeSuccessfulInspectionDoneFunction']:  # TODO: Hardcoded platform name
            val = 1
        else:
            val = 0

        return Real(val)


class InspectedPointsScore(MetricGeneratorTerminalEventScope):
    """Generates single Real indicating the score for inspected points score for an event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        # Find the last step with observation data from the platform in question
        # If the platform isn't in the episode at all, return Void
        steps_with_platform_in_question = [item for item in params.steps if agent_id in item.agents and item.agents[agent_id] is not None]
        if len(steps_with_platform_in_question) == 0:
            return Void()

        last_step_with_platform_data = steps_with_platform_in_question[-1]

        if last_step_with_platform_data.agents[agent_id] is None:
            raise RuntimeError("Non Op")

        return Real(last_step_with_platform_data.platforms[0]['sensors']['Sensor_InspectedPointsScore']['measurement'][0])
