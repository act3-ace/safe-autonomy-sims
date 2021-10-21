"""
Contains implementations of sensors that can be used in injuction with the CWH platform
"""

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp
from act3_rl_core.simulators.base_parts import BaseSensor

from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh.cwh_simulator import CWHSimulator


class CWHSensor(BaseSensor):
    """
    Interface for a basic sensor of the CWH platform
    """

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            returns name of the sensor

        """
        return self.__class__.__name__

    @property
    def measurement_properties(self):
        """
        gives the measurement properties of a sensors -  units and bounds

        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError

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

    @property
    def measurement_properties(self):
        """
        retreive the measurement properies

        Returns
        -------
        position_properties : MultiBoxProp
        """
        position_properties = MultiBoxProp(
            name="position", low=[-80000] * 3, high=[80000] * 3, unit=["meters"] * 3, description="position of the spacecraft"
        )
        return position_properties

    def _calculate_measurement(self, state):
        """
        Calculate the position

        Returns
        -------
        list of ints
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

    @property
    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="velocity", low=[-10000] * 3, high=[10000] * 3, unit=["m/s"] * 3, description="velocity of the spacecraft"
        )
        return velocity_properties

    # state - tuple
    def _calculate_measurement(self, state):
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
