import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict

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

        # placeholder til we find out how to find lead_aircraft
        lead_aircraft_id = 8
        wingman_id = 6

        # find lead aircraft
        lead_aircraft = next_state.sim_state[lead_aircraft_id]
        wingman = next_state.sim_state[wingman_id]

        rejoin_region_center = lead_aircraft.rejoin_region_center

        # Rejoin region will change with where the aircraft is

        rejoin_region_radius = self.config.rejoin_region_radius

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

        position = next_state.sim_platforms[0].position

        # compute to origin
        origin = np.array([0, 0, 0])
        dist = np.linalg.norm(origin - np.array(position))

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done
