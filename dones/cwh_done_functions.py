from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
# need to import get_platform_name, WIP
import numpy as np


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

        # compute distance to origin
        #platform = get_platform_name(next_state,self.agent)
        #pos = platform.position

        position = next_state.sim_platforms[0].position

        # compute to origin
        origin = np.array([0,0,0])
        dist = np.linalg.norm(origin-np.array(position))

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


class SuccessfulDockingDoneValidator(DoneFuncBaseValidator):
    docking_region_radius: float


class SuccessfulDockingDoneFunction(DoneFuncBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        return SuccessfulDockingDoneValidator

    def __call__(self,observation,action,next_observation,next_state):
        # eventually will include velocity constraint
        done = DoneDict()
        #platform = get_platform_name(next_state,self.agent)

        #pos = platform.position
        position = next_state.sim_platforms[0].position

        origin = np.array([0,0,0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(position) - origin)
        done[self.agent] = radial_distance <= docking_region_radius

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN

        return done


if __name__ == "__main__":
    from collections import OrderedDict
    from simulators.cwh_simulator import CWHSimulator

    tmp_config = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {
                },
                "platform_config": [
                    (
                        "space.cwh.platforms.cwh_controllers.ThrustController",
                        {"name": "X Thrust", "axis": 0}
                    ),
                    (
                        "space.cwh.platforms.cwh_controllers.ThrustController",
                        {"name": "Y Thrust", "axis": 1}
                    ),
                    (
                        "space.cwh.platforms.cwh_controllers.ThrustController",
                        {"name": "Z Thrust", "axis": 2}
                    ),
                    (
                        "space.cwh.platforms.cwh_sensors.PositionSensor",
                        {}
                    ),
                    (
                        "space.cwh.platforms.cwh_sensors.VelocitySensor",
                        {}
                    )
                ]
            }
        }
    }

    reset_config = {
        "agent_initialization": {
            "blue0": {
                "position": [0, 0, 0],
                "velocity": [0, 0, 0]
            }
        }
    }

    tmp = CWHSimulator(**tmp_config)
    state = tmp.reset(reset_config)

    dist_done_fn = MaxDistanceDoneFunction(agent_name="blue0", max_distance=40000)
    docking_done_fn = SuccessfulDockingDoneFunction(agent_name="blue0", docking_region_radius=0.5)

    for i in range(5):
        state = tmp.step()
        dist_done = dist_done_fn(
            observation=OrderedDict(),
            action=OrderedDict(),
            next_observation=OrderedDict(),
            next_state=state
        )
        docking_done = docking_done_fn(
            observation=OrderedDict(),
            action=OrderedDict(),
            next_observation=OrderedDict(),
            next_state=state
        )
        # print(dist_done)
        print(docking_done)
