"""
Contains implementations of sensors that can be used in injuction with the CWH platform
"""

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseSensor

import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh_simulator import CWHSimulator


class CWHSensor(BaseSensor):
    """
    Interface for a basic sensor of the CWH platform
    """

    def _calculate_measurement(self, state):
        """
        get measurements from the sensor

        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError


class PositionSensor(CWHSensor):
    """
    Implementation of a sensor designed to give the position at any time
    """

    def __init__(self, parent_platform, config, measurement_properties=cwh_props.PositionProp):
        super().__init__(measurement_properties=measurement_properties, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - position

        Returns
        -------
        list of floats
            position of spacecraft
        """
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)


class VelocitySensor(CWHSensor):
    """
    Implementation of a sensor to give velocity at any time
    """

    def __init__(self, parent_platform, config, measurement_properties=cwh_props.VelocityProp):
        super().__init__(measurement_properties=measurement_properties, parent_platform=parent_platform, config=config)

    # state - tuple
    def _calculate_measurement(self, state):
        """
        Calculate the measurement - velocity

        Returns
        -------
        list of floats
            velocity of spacecraft
        """
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
