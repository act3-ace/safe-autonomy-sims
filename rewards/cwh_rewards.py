from collections import OrderedDict
import numpy as np
from numpy_ringbuffer import RingBuffer
from pydantic import BaseModel

from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.libraries.env_func_base import EnvFuncBase
from act3_rl_core.libraries.environment_dict import RewardDict
from act3_rl_core.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class CWHDistanceChangeRewardValidator(RewardFuncBaseValidator):
    scale: float


class CWHDistanceChangeReward(RewardFuncBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

    @classmethod
    def get_validator(cls):
        return CWHDistanceChangeRewardValidator

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
            val = self.config.scale * (self._dist_buffer[1] - self._dist_buffer[0])

        reward[self.config.agent_name] = val

        return reward


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
                "position": [1, 1, 1],
                "velocity": [-0.5, -0.5, -0.5]
            }
        }
    }

    tmp = CWHSimulator(**tmp_config)
    state = tmp.reset(reset_config)

    reward_fn = CWHDistanceChangeReward(agent_name="blue0", scale=-1e-2)

    for i in range(5):
        state = tmp.step()
        reward = reward_fn(
            observation=OrderedDict(),
            action=OrderedDict(),
            next_observation=OrderedDict(),
            state=StateDict(),
            next_state=state,
            observation_space=StateDict(),
            observation_units=StateDict()
        )
        print("Position: %s\t Velocity: %s" % (
            str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))
        print(reward)
