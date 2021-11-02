"""
This module contains the CWH Simulator for interacting with the CWH Docking task simulator
"""
import typing

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_simulator import BaseSimulatorResetValidator
from pydantic import BaseModel, validator

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator
from saferl_sim.cwh.cwh import CWHSpacecraft


class CWHPlatformConfigValidator(BaseModel):
    """Validator for CWH Platform config

    position: initial position of cwh platform
    velocity: initial velocity of cwh platform
    """

    position: typing.List[float]
    velocity: typing.List[float]

    @validator("position", "velocity")
    def check_3d_vec_len(cls, v, field):
        """checks 3d vector field for length 3

        Parameters
        ----------
        v : typing.List[float]
            vector quantity to check
        field : string
            name of validator field

        Returns
        -------
        typing.List[float]
            v
        """
        if len(v) != 3:
            raise ValueError(f"{field.name} provided to CWHPlatformValidator is not length 3")
        return v


class CWHSimulatorResetValidator(BaseSimulatorResetValidator):
    """Validator for CWH Simulator Reset config

    agent_initialization: Dict of individual platform reset configs
    """

    agent_initialization: typing.Optional[typing.Dict[str, CWHPlatformConfigValidator]] = {
        "blue0": CWHPlatformConfigValidator(position=[0, 1, 2], velocity=[0, 0, 0])
    }


class CWHSimulator(SafeRLSimulator):
    """
    Simulator for CWH Docking Task. Interfaces CWH platforms with underlying CWH entities in Docking simulation.
    """

    @classmethod
    def get_reset_validator(cls):
        return CWHSimulatorResetValidator

    def _construct_platform_map(self) -> dict:
        return {
            'default': (CWHSpacecraft, CWHPlatform),
            'cwh': (CWHSpacecraft, CWHPlatform),
        }


PluginLibrary.AddClassToGroup(CWHSimulator, "CWHSimulator", {})


def main():
    """main"""
    tmp_config = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {
                    "platform": "cwh", "kwargs": {
                        "integration_method": "RK45"
                    }
                },
                "platform_config": [
                    ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "X Thrust", "axis": 0
                    }),
                    ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "Y Thrust", "axis": 1
                    }),
                    ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "Z Thrust", "axis": 2
                    }),
                    ("saferl.platforms.cwh.cwh_sensors.PositionSensor", {}),
                    ("saferl.platforms.cwh.cwh_sensors.VelocitySensor", {}),
                ],
            }
        },
    }

    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "velocity": [0, 0, 0]}}}

    tmp = CWHSimulator(**tmp_config)
    state = tmp.reset(reset_config)
    # print("Position: %s\t Velocity: %s" % (str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))
    for _ in range(5):
        # state.sim_platforms[0]._controllers[0].apply_control(1)
        # state.sim_platforms[0]._controllers[1].apply_control(2)
        # state.sim_platforms[0]._controllers[2].apply_control(3)
        # print(state.sim_platforms[0]._sensors[1].get_measurement())
        state = tmp.step()
        print("Position: %s\t Velocity: %s" % (str(state.sim_platforms[0].position), str(state.sim_platforms[0].velocity)))


if __name__ == "__main__":
    main()
