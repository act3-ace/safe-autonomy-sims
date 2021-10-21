import typing

from pydantic import BaseModel, validator

from act3_rl_core.simulators.base_simulator import BaseSimulator, BaseSimulatorValidator, BaseSimulatorResetValidator
from act3_rl_core.libraries.state_dict import StateDict

from saferl.platforms.dubins.dubins_platform import DubinsPlatform
from air.dubins.dubins_sim.platforms import Dubins3dPlatform


class DubinsSimulatorValidator(BaseSimulatorValidator):
    step_size: int


class DubinsPlatformConfigValidator(BaseModel):
    position: typing.List[float]
    velocity: typing.List[float]

    @validator("position", "velocity")
    def check_position_len(cls, v, field):
        if len(v) != 3:
            raise ValueError(f"{field.name} provided to CWHPlatformValidator is not length 3")
        return v


class DubinsSimulatorResetValidator(BaseSimulatorResetValidator):
    agent_initialization: typing.Dict[str, DubinsPlatformConfigValidator]


class DubinsSimulator(BaseSimulator):
    @classmethod
    def get_simulator_validator(cls):
        return DubinsSimulatorValidator

    @classmethod
    def get_reset_validator(cls):
        return DubinsSimulatorResetValidator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sim_entities = {agent_id: Dubins3dPlatform(name=agent_id) for agent_id in self.config.agent_configs.keys()}
        self._state = StateDict()

    def reset(self, config):
        self._state.clear()
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
                    "z_dot": i.velocity[2],
                }
            )
        self._state.sim_platforms = self.get_platforms()
        self.update_sensor_measurements()
        return self._state

    def get_platforms(self):
        sim_platforms = tuple(
            DubinsPlatform(entity, self.config.agent_configs[agent_id].platform_config) for agent_id, entity in self.sim_entities.items()
        )
        return sim_platforms

    def update_sensor_measurements(self):
        """
        Update and caches all the measurements of all the sensors on each platform
        """
        for plat in self._state.sim_platforms:
            for sensor in plat.sensors:
                sensor.calculate_and_cache_measurement(state=self._state.sim_platforms)

    def mark_episode_done(self):
        pass

    def save_episode_information(self, **kwargs):
        pass

    def step(self):
        for platform in self._state.sim_platforms:
            agent_id = platform.name
            action = platform.get_applied_action()
            entity = self.sim_entities[agent_id]
            entity.step_compute(sim_state=None, action=action, step_size=self.config.step_size)
            entity.step_apply()
        self.update_sensor_measurements()
        return self._state


if __name__ == "__main__":
    tmp_config = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {},
                "platform_config": [
                    # (
                    #     "space.cwh.platforms.cwh_controllers.ThrustController",
                    #     {"name": "X Thrust", "axis": 0}
                    # ),
                    # (
                    #     "space.cwh.platforms.cwh_controllers.ThrustController",
                    #     {"name": "Y Thrust", "axis": 1}
                    # ),
                    # (
                    #     "space.cwh.platforms.cwh_controllers.ThrustController",
                    #     {"name": "Z Thrust", "axis": 2}
                    # ),
                    # (
                    #     "space.cwh.platforms.cwh_sensors.PositionSensor",
                    #     {}
                    # ),
                    # (
                    #     "space.cwh.platforms.cwh_sensors.VelocitySensor",
                    #     {}
                    # )
                ],
            }
        },
    }

    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "velocity": [0, 0, 0]}}}

    tmp = DubinsSimulator(**tmp_config)
    state = tmp.reset(reset_config)
    print("Position: %s\t Velocity: %s" % (str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))
    for i in range(5):
        # state.sim_platforms[0]._controllers[0].apply_control(1)
        # state.sim_platforms[0]._controllers[1].apply_control(2)
        # state.sim_platforms[0]._controllers[2].apply_control(3)
        # print(state.sim_platforms[0]._sensors[1].get_measurement())
        state = tmp.step()
        # print("Position: %s\t Velocity: %s" % (
        #     str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))
