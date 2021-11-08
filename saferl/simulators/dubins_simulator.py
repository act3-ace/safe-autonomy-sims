"""
Contains the implementations of classes that describe how the simulation is to proceed.
"""

from act3_rl_core.libraries.plugin_library import PluginLibrary

import saferl_sim.dubins.entities as bp
from saferl.platforms.dubins.dubins_platform import Dubins2dPlatform, Dubins3dPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator


class Dubins2dSimulator(SafeRLSimulator):
    """
    A class that contains all essential components of a Dubins2D simulation
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (bp.Dubins2dAircraft, Dubins2dPlatform),
            'dubins2d': (bp.Dubins2dAircraft, Dubins2dPlatform),
        }


class Dubins3dSimulator(SafeRLSimulator):
    """
    A class that contains all essential components of a Dubins 3D simulation
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (bp.Dubins3dAircraft, Dubins3dPlatform),
            'dubins3d': (bp.Dubins3dAircraft, Dubins3dPlatform),
        }


PluginLibrary.AddClassToGroup(Dubins3dSimulator, "Dubins3dSimulator", {})

if __name__ == "__main__":
    tmp_config_2d = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {
                    "platform": "dubins2d"
                },
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
                    ("saferl.platforms.dubins.dubins_sensors.HeadingSensor", {}),
                ],
            }
        },
    }

    tmp_config_3d = {
        "step_size": 1,
        "agent_configs": {
            "blue0": {
                "sim_config": {
                    "platform": "dubins3d"
                },
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
                ],
            }
        },
    }

    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 0], "heading": 0, "speed": 50}}}
    # reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "heading": 0, "speed": 50, "gamma": 0, "roll": 0}}}

    tmp = Dubins2dSimulator(**tmp_config_2d)
    # tmp = Dubins3dSimulator(**tmp_config_3d)

    state = tmp.reset(reset_config)
    print(
        f"Position: {state.sim_platforms[0].position}\t"
        f"Velocity: {state.sim_platforms[0].velocity}\tHeading: {state.sim_platforms[0].heading}"
    )
    for i in range(5):
        control = [1, 0]
        # control = [1, 0, 0]
        # state.sim_platforms[0]._controllers[0].apply_control(control)
        # state.sim_platforms[0]._controllers[0].apply_control(control[0])
        # state.sim_platforms[0]._controllers[1].apply_control(control[1])
        # state.sim_platforms[0]._controllers[2].apply_control(control[2])
        # print(state.sim_platforms[0]._sensors[1].get_measurement())
        state = tmp.step()
        print(
            f"Position: {state.sim_platforms[0].position}\t "
            f"Velocity: {state.sim_platforms[0].velocity}\tHeading: {state.sim_platforms[0].heading}"
        )
