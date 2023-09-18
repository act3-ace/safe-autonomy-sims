"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements the Reward Functions and Reward Validators specific to the inspection task.
"""
import typing
from collections import OrderedDict
from functools import partial

from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.dones.done_func_base import DoneStatusCodes
from corl.rewards.episode_done import EpisodeDoneStateReward, EpisodeDoneStateRewardValidator

from safe_autonomy_sims.dones.cwh.common import MaxDistanceDoneFunction


class MaxDistanceDoneRewardValidator(EpisodeDoneStateRewardValidator):
    """Validation for MaxDistanceDoneReward"""
    done_name: str = MaxDistanceDoneFunction.__name__
    scale: float = -1.0


class MaxDistanceDoneReward(EpisodeDoneStateReward):
    """Reward that only responds to the MaxDistanceDone"""

    def __init__(self, **kwargs) -> None:
        self.donfig: MaxDistanceDoneRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[MaxDistanceDoneRewardValidator]:
        return MaxDistanceDoneRewardValidator

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space,
        observation_units
    ) -> RewardDict:

        reward = RewardDict()
        reward[self.config.agent_name] = 0

        done_state = next_state.agent_episode_state.get(self.config.agent_name, {})
        for done_name, done_code in done_state.items():
            if done_name in self._already_recorded:
                continue
            self._already_recorded.add(done_name)
            self._status_codes[done_code].append(done_name)

        # this will loop starting from win and go down
        consolidate_break = False
        for done_status in DoneStatusCodes:
            for done_name in self._status_codes[done_status]:
                reward[self.config.agent_name] += self.get_scaling_method(
                    observation, action, next_observation, state, next_state, observation_space, observation_units, done_name, done_status
                )(self._counter)
                if self.config.consolidate:
                    consolidate_break = True
                    break
            if consolidate_break:
                break

        self._counter += 1 / next_state.sim_update_rate

        return reward

    def get_scaling_method(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space,
        observation_units,
        done_name,
        done_code
    ) -> typing.Callable[[int], float]:
        # pylint: disable=unused-argument
        # pylint: disable=arguments-differ
        if done_name == self.config.done_name:
            if done_code.name.lower() == 'lose':
                scale = self.config.scale
                return partial(self.constant_scaling, scale=scale)
            return self._status_code_func[done_code.name.lower()]

        return partial(self.constant_scaling, scale=0)
