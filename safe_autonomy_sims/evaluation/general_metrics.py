"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements general observation and action metrics
"""
import typing

import numpy as np
from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.types.nonterminals.dict import Dict
from corl.evaluation.metrics.types.nonterminals.vector import Vector
from corl.evaluation.metrics.types.terminals.real import Real


class ObservationVector(MetricGeneratorTerminalEventScope):
    """
    Generates vector of Dicts for the observations calculated
    for each step during an event for an agent
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        arr: typing.List[Metric] = []
        for step in params.steps:

            if agent_id not in step.agents or step.agents[agent_id] is None:
                break

            map_obs = step.agents[agent_id].observations

            if map_obs is None:
                continue
            # Create a non terminal metric (Dict) that is comprised of the terminal (Real) observations
            real_dict: typing.Dict[str, Metric] = {key: Real(map_obs[key]) for key in map_obs.keys()}
            arr.append(Dict(real_dict))

        return Vector(arr)


class ControlVector(MetricGeneratorTerminalEventScope):
    """
    Generates vector of Dicts for the control for each
    step during an event for an agent
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        arr: typing.List[Metric] = []
        for step in params.steps:

            if agent_id not in step.agents or step.agents[agent_id] is None:
                break

            map_act = step.agents[agent_id].actions

            if map_act is None:
                continue
            # Create a non terminal metric (Dict) that is comprised of the terminal (Real) actions
            real_dict: typing.Dict[str, Metric] = {key: Real(map_act[key]) for key in map_act.keys()}
            arr.append(Dict(real_dict))

        return Vector(arr)


class SafetyViolationRatioMetric(MetricGeneratorTerminalEventScope):
    """
    Generates single Real indicating percentage of steps where safety
    is violated. Must use constraint based RTA.
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]

        arr = []
        for step in params.steps:

            if agent_id not in step.agents or step.agents[agent_id] is None:
                break

            constraints = step.agents[agent_id].observations['RTAModule']['constraints']

            if constraints is None:
                continue

            arr.append(int(np.any([v < 0 for v in constraints.values()])))

        return Real(np.mean(arr) * 100)


class PositionVelocityVector(MetricGeneratorTerminalEventScope):
    """
    Generates vector of Dicts for the Positions calculated for each
    step during an event for an agent
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:

        if "agent_id" not in kwargs:
            raise RuntimeError("Expecting \"agent_id\" to be provided")

        agent_id = kwargs["agent_id"].split('.')[0]
        platform_id = params.agent_to_platforms[agent_id][0]

        arr: typing.List[Metric] = []
        for step in params.steps:

            this_step_platforms = [p['name'] for p in step.platforms]

            if platform_id not in this_step_platforms:
                break

            platform = [p for p in step.platforms if p['name'] == platform_id][0]
            pos = platform['position']
            vel = platform['velocity']

            if pos is None or vel is None:
                continue
            # Create a non terminal metric (Dict) that is comprised of the terminal (Real) observations
            real_dict: typing.Dict[str, Metric] = {'position': pos, 'velocity': vel}
            arr.append(Dict(real_dict))

        return Vector(arr)
