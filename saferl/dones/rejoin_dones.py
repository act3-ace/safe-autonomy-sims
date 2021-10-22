import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name

# done conditions
# crash, distance , timeout

# also need SuccessfulRejoin,

class SuccessfulRejoinDoneValidator(DoneFuncBaseValidator):
    rejoin_region_radius: float
    offset_values: typing.List(float,float,float)

class SuccessfulRejoinFunction(DoneFuncBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        return SuccessfulRejoinDoneValidator


    def __call__(self, observation, action, next_observation, next_state):
        # eventually will include velocity constraint
        done = DoneDict()

        lead_aircraft_platform = get_platform_by_name(next_state,self.config.lead)
        wingman_agent_platform = get_platform_by_name(next_state,self.agent)

        # compute the rejoin region , using all three pieces of info

        # all 3 pieces
        rejoin_region_radius = self.config.rejoin_region_radius
        lead_orientation = lead_aircraft_platform.lead_orientation
        offset_vector = np.array(self.config.offset_values)

        # rotate vector then add it to the lead center
        rotated_vector = lead_orientation.apply(offset_vector)
        rejoin_region_center = lead_aircraft_platform.position + rotated_vector

        radial_distance = np.linalg.norm(np.array(position) - rejoin_region_center)
        done[self.agent] = radial_distance <= rejoin_region_center


        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN

        return done


class MaxDistanceDoneValidator(DoneFuncBaseValidator):
    max_distance: float

class MaxDistanceDoneFunction(DoneFuncBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        return MaxDistanceDoneValidator

    def __call__(self, observation, action, next_observation, next_state):

        done = DoneDict()

        wingman_agent_platform = get_platform_by_name(next_state,self.agent)

        dist = np.linalg.norm(wingman_agent_platform.position - lead_aircraft_platform.position)

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


class CrashDoneValidator(DoneFuncBaseValidator):
    safety_margin: float

class CrashDoneFunction(DoneFuncBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        return CrashDoneValidator

    def __call__(self, observation, action, next_observation, next_state):

        done = DoneDict()

        wingman_agent_platform = get_platform_by_name(next_state,self.agent)

        dist = np.linalg.norm(wingman_agent_platform.position - lead_aircraft_platform.position)

        done[self.agent] = dist <= self.config.safety_margin

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done
