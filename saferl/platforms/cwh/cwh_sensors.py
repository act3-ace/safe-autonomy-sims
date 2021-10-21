from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseSensor

import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh.cwh_simulator import CWHSimulator


class CWHSensor(BaseSensor):

    def _calculate_measurement(self, state):
        raise NotImplementedError


class PositionSensor(CWHSensor):

    def __init__(self, parent_platform, config, measurement_properties=cwh_props.PositionProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

    def _calculate_measurement(self, state):
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)


class VelocitySensor(CWHSensor):

    def __init__(self, parent_platform, config, measurement_properties=cwh_props.VelocityProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

    # state - tuple
    def _calculate_measurement(self, state):
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
