import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict


class TimeoutDoneValidator(DoneFuncBaseValidator):
    max_sim_time: float


class TimeoutDoneFunction(DoneFuncBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        return TimeoutDoneValidator

    def __call__(self, observation, action, next_observation, next_state):

        done = DoneDict()

        # get sim time
        sim_time = next(iter(next_state.sim_platforms)).sim_time
        # sim_time = DoneFuncBase._get_platform_time(next(iter(next_state.sim_platforms)))

        done[self.agent] = sim_time > self.config.max_sim_time

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done