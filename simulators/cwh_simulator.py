import typing

from pydantic import BaseModel, validator

from act3_rl_core.simulators.base_simulator import BaseSimulator, BaseSimulatorValidator, BaseSimulatorResetValidator
from act3_rl_core.libraries.state_dict import StateDict

from space.cwh.platforms.cwh_platform import CWHPlatform
from space.cwh.cwhspacecraft_sim.platforms.cwh import CWHSpacecraft3d


class CWHSimulatorValidator(BaseSimulatorValidator):
    step_size: int


class CWHPlatformConfigValidator(BaseModel):
    position: typing.List[float]
    velocity: typing.List[float]
    @validator("position", "velocity")
    def check_position_len(cls, v, field):
        if len(v) != 3:
            raise ValueError(f"{field.name} provided to CWHPlatformValidator is not length 3")
        return v


class CWHSimulatorResetValidator(BaseSimulatorResetValidator):
    agent_initialization: typing.Dict[str, CWHPlatformConfigValidator]


class CWHSimulator(BaseSimulator):

    @classmethod
    def get_simulator_validator(cls):
        return CWHSimulatorValidator

    @classmethod
    def get_reset_validator(cls):
        return CWHSimulatorResetValidator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sim_entities = {agent_id: CWHSpacecraft3d(name=agent_id) for agent_id in self.config.agent_configs.keys()}
        self._state = StateDict()

    def reset(self, config):
        self._state.clear()
        config = self.get_reset_validator()(**config)
        for agent_id, entity in self.sim_entities.items():
            i = config.agent_initialization[agent_id]
            self.sim_entities[agent_id].reset(
                **{"x": i.position[0], "y": i.position[1], "z": i.position[2],
                    "x_dot": i.velocity[0], "y_dot": i.velocity[1], "z_dot": i.velocity[2]}
            )
        self._state.sim_platforms = self.get_platforms()
        return self._state

    def get_platforms(self):
        sim_platforms = tuple(CWHPlatform(entity, self.config.agent_configs[agent_id].platform_config) for agent_id, entity in self.sim_entities.items())
        return sim_platforms

    def mark_episode_done(self):
        pass

    def save_episode_information(self, **kwargs):
        pass

    def step(self):
        for platform in self._state.sim_platforms:
            agent_id = platform.name
            action = platform.get_applied_action()
            entity = self.sim_entities[agent_id]
            entity.step_compute(action, self.config.step_size)
            entity.step_apply()
        self._state.sim_platforms = self.get_platforms()
        return self._state


if __name__ == "__main__":
    tmp_config = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {
                },
                "platform_config": []
            }
        }
    }

    reset_config = {
        "agent_initialization": {
            "blue0": {
                "position": [1, 2, 3],
                "velocity": [1, 2, 3]
            }
        }
    }

    tmp = CWHSimulator(**tmp_config)
    state = tmp.reset(reset_config)
    print(state.sim_platforms[0].velocity)
    for i in range(5):
        state = tmp.step()
        print(state.sim_platforms[0].velocity)
