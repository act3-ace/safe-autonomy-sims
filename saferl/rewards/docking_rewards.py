from collections import OrderedDict

import numpy as np
from act3_rl_core.libraries.env_func_base import EnvFuncBase
from act3_rl_core.libraries.environment_dict import RewardDict
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.rewards.reward_func_base import RewardFuncBase
from numpy_ringbuffer import RingBuffer
from pydantic import BaseModel


class CWHDistanceChangeReward(RewardFuncBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)
        self.scale = -1.0e-05

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> RewardDict:

        reward = RewardDict()
        val = 0

        # question, less brittle way to refer to platforms?
        position = next_state.sim_platforms[0].position
        distance = np.linalg.norm(position)
        self._dist_buffer.append(distance)

        # TODO intialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            val = self.scale * (self._dist_buffer[1] - self._dist_buffer[0])

        reward[self.config.agent_name] = val

        return reward
