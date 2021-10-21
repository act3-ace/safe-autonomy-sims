import typing

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_simulator import BaseSimulatorResetValidator
from pydantic import BaseModel, validator

import saferl.simulators.dubins.backend.platforms as bp
from saferl.platforms.dubins.dubins_platform import Dubins2dPlatform, Dubins3dPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator


class Dubins2dPlatformConfigValidator(BaseModel):
    position: typing.List[float]
    speed: float
    heading: float

    @validator("position")
    def check_position_len(cls, v, field):
        check_len = 2
        if len(v) != check_len:
            raise ValueError(f"{field.name} provided to DubinsPlatformConfigValidator is not length {check_len}")
        return v


class Dubins2dSimulatorResetValidator(BaseSimulatorResetValidator):
    agent_initialization: typing.Optional[typing.Dict[str, Dubins2dPlatformConfigValidator]] = {
        "blue0": Dubins2dPlatformConfigValidator(position=[0, 1], speed=100, heading=0)
    }


class Dubins2dSimulator(SafeRLSimulator):

    @classmethod
    def get_reset_validator(cls):
        return Dubins2dSimulatorResetValidator

    def get_sim_entities(self):
        return {agent_id: bp.Dubins2dPlatform(name=agent_id) for agent_id in self.config.agent_configs.keys()}

    def get_platforms(self):
        sim_platforms = tuple(
            Dubins2dPlatform(platform_name=agent_id, platform=entity, platform_config=self.config.agent_configs[agent_id].platform_config)
            for agent_id,
            entity in self.sim_entities.items()
        )
        return sim_platforms

    def reset_sim_entities(self, config):
        config = self.get_reset_validator()(**config)
        for agent_id, entity in self.sim_entities.items():
            init_params = config.agent_initialization[agent_id]
            self.sim_entities[agent_id].reset(
                **{
                    "x": init_params.position[0], "y": init_params.position[1], "heading": init_params.heading, "v": init_params.speed
                }
            )


class Dubins3dPlatformConfigValidator(BaseModel):
    position: typing.List[float]
    speed: float
    heading: float
    gamma: float
    roll: float

    @validator("position")
    def check_position_len(cls, v, field):
        check_len = 3
        if len(v) != check_len:
            raise ValueError(f"{field.name} provided to Dubins3dPlatformConfigValidator is not length {check_len}")
        return v


class Dubins3dSimulatorResetValidator(BaseSimulatorResetValidator):
    agent_initialization: typing.Optional[typing.Dict[str, Dubins3dPlatformConfigValidator]] = {
        "blue0": Dubins3dPlatformConfigValidator(position=[0, 1, 2], speed=100, heading=0, gamma=0, roll=0)
    }


class Dubins3dSimulator(SafeRLSimulator):

    @classmethod
    def get_reset_validator(cls):
        return Dubins3dSimulatorResetValidator

    def get_sim_entities(self):
        return {agent_id: bp.Dubins3dPlatform(name=agent_id) for agent_id in self.config.agent_configs.keys()}

    def get_platforms(self):
        sim_platforms = tuple(
            Dubins3dPlatform(platform_name=agent_id, platform=entity, platform_config=self.config.agent_configs[agent_id].platform_config)
            for agent_id,
            entity in self.sim_entities.items()
        )
        return sim_platforms

    def reset_sim_entities(self, config):
        config = self.get_reset_validator()(**config)
        for agent_id, entity in self.sim_entities.items():
            init_params = config.agent_initialization[agent_id]
            self.sim_entities[agent_id].reset(
                **{
                    "x": init_params.position[0],
                    "y": init_params.position[1],
                    "z": init_params.position[2],
                    "heading": init_params.heading,
                    "v": init_params.speed,
                    "gamma": init_params.gamma,
                    "roll": init_params.roll
                }
            )


PluginLibrary.AddClassToGroup(Dubins3dSimulator, "Dubins3dSimulator", {})

if __name__ == "__main__":
    tmp_config_2d = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {},
                "platform_config": [
                    # ("saferl.platforms.dubins.dubins_controllers.CombinedTurnRateAccelerationController", {
                    #     "name": "YawAccControl"
                    # }),
                    ("saferl.platforms.dubins.dubins_controllers.YawRateController", {
                        "name": "YawRateControl", "axis": 0
                    }),
                    ("saferl.platforms.dubins.dubins_controllers.AccelerationController", {
                        "name": "AccelerationControl", "axis": 1
                    }),
                    ("saferl.platforms.dubins.dubins_sensors.PositionSensor", {}),
                    ("saferl.platforms.dubins.dubins_sensors.VelocitySensor", {}),
                    ("saferl.platforms.dubins.dubins_sensors.HeadingSensor", {})
                ]
            }
        }
    }

    tmp_config_3d = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {},
                "platform_config": [
                    ("saferl.platforms.dubins.dubins_controllers.CombinedPitchRollAccelerationController", {
                        "name": "PitchRollAccControl"
                    }),
                    # ("saferl.platforms.dubins.dubins_controllers.PitchRateController", {
                    #     "name": "PitchRateControl", "axis": 0
                    # }),
                    # ("saferl.platforms.dubins.dubins_controllers.RollRateController", {
                    #     "name": "RollRateControl", "axis": 1
                    # }),
                    # ("saferl.platforms.dubins.dubins_controllers.AccelerationController", {
                    #     "name": "AccelerationControl", "axis": 2
                    # }),
                    ("saferl.platforms.dubins.dubins_sensors.PositionSensor", {}),
                    ("saferl.platforms.dubins.dubins_sensors.VelocitySensor", {}),
                    ("saferl.platforms.dubins.dubins_sensors.HeadingSensor", {}),
                    # ("saferl.platforms.dubins.dubins_sensors.FlightPathSensor", {}),
                ]
            }
        }
    }

    # reset_config = {"agent_initialization": {"blue0": {"position": [0, 1], "heading": 0, "speed": 50}}}
    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "heading": 0, "speed": 50, "gamma": 0, "roll": 0}}}

    # tmp = Dubins2dSimulator(**tmp_config_2d)
    tmp = Dubins3dSimulator(**tmp_config_3d)

    state = tmp.reset(reset_config)
    print(
        f"Position: {state.sim_platforms[0].position}\t "
        f"Velocity: {state.sim_platforms[0].velocity}\tHeading: {state.sim_platforms[0].heading}"
    )
    for i in range(5):
        control = [1, 0, 0]
        state.sim_platforms[0]._controllers[0].apply_control(control)
        # state.sim_platforms[0]._controllers[0].apply_control(control[0])
        # state.sim_platforms[0]._controllers[1].apply_control(control[1])
        # state.sim_platforms[0]._controllers[2].apply_control(control[2])
        # print(state.sim_platforms[0]._sensors[1].get_measurement())
        state = tmp.step()
        print(
            f"Position: {state.sim_platforms[0].position}\t "
            f"Velocity: {state.sim_platforms[0].velocity}\tHeading: {state.sim_platforms[0].heading}"
        )
