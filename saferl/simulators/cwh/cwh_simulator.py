import typing

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_simulator import BaseSimulatorResetValidator
from pydantic import BaseModel, validator

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.simulators.cwh.backend.platforms.cwh import CWHSpacecraft3d
from saferl.simulators.saferl_simulator import SafeRLSimulator


class CWHPlatformConfigValidator(BaseModel):
    position: typing.List[float]
    velocity: typing.List[float]

    @validator("position", "velocity")
    def check_position_len(cls, v, field):
        if len(v) != 3:
            raise ValueError(f"{field.name} provided to CWHPlatformValidator is not length 3")
        return v


class CWHSimulatorResetValidator(BaseSimulatorResetValidator):
    agent_initialization: typing.Optional[typing.Dict[str, CWHPlatformConfigValidator]] = {
        "blue0": CWHPlatformConfigValidator(position=[0, 1, 2], velocity=[0, 0, 0])
    }


class CWHSimulator(SafeRLSimulator):

    @classmethod
    def get_reset_validator(cls):
        return CWHSimulatorResetValidator

    def get_sim_entities(self):
        return {agent_id: CWHSpacecraft3d(name=agent_id) for agent_id in self.config.agent_configs.keys()}

    def get_platforms(self):
        sim_platforms = tuple(
            CWHPlatform(platform_name=agent_id, platform=entity, platform_config=self.config.agent_configs[agent_id].platform_config)
            for agent_id,
            entity in self.sim_entities.items()
        )
        return sim_platforms

    def reset_sim_entities(self, config):
        config = self.get_reset_validator()(**config)
        for agent_id, entity in self.sim_entities.items():
            i = config.agent_initialization[agent_id]
            self.sim_entities[agent_id].reset(
                **{
                    "x": i.position[0],
                    "y": i.position[1],
                    "z": i.position[2],
                    "x_dot": i.velocity[0],
                    "y_dot": i.velocity[1],
                    "z_dot": i.velocity[2]
                }
            )


PluginLibrary.AddClassToGroup(CWHSimulator, "CWHSimulator", {})

if __name__ == "__main__":
    tmp_config = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {},
                "platform_config": [
                    ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "X Thrust", "axis": 0
                    }), ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "Y Thrust", "axis": 1
                    }), ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "Z Thrust", "axis": 2
                    }), ("saferl.platforms.cwh.cwh_sensors.PositionSensor", {}), ("saferl.platforms.cwh.cwh_sensors.VelocitySensor", {})
                ]
            }
        }
    }

    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "velocity": [0, 0, 0]}}}

    tmp = CWHSimulator(**tmp_config)
    state = tmp.reset(reset_config)
    # print("Position: %s\t Velocity: %s" % (str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))
    for i in range(5):
        state.sim_platforms[0]._controllers[0].apply_control(1)
        # state.sim_platforms[0]._controllers[1].apply_control(2)
        # state.sim_platforms[0]._controllers[2].apply_control(3)
        # print(state.sim_platforms[0]._sensors[1].get_measurement())
        state = tmp.step()
        print("Position: %s\t Velocity: %s" % (str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))
