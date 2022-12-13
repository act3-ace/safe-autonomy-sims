"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Contains implementations of sensors that can be used in junction with the CWH platform.
"""

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_parts import BaseSensor

import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh_simulator import CWHSimulator
from saferl.simulators.inspection_simulator import InspectionSimulator


class CWHSensor(BaseSensor):
    """
    Interface for a basic sensor of the CWH platform.
    """

    def _calculate_measurement(self, state):
        """
        Get measurements from the sensor.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError


PluginLibrary.AddClassToGroup(CWHSensor, "Sensor_Generic", {"simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH})


class PositionSensor(CWHSensor):
    """
    Implementation of a sensor designed to give the position at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.PositionProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - position.

        Returns
        -------
        list of floats
            Position of spacecraft.
        """
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)

PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": InspectionSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)


class VelocitySensor(CWHSensor):
    """
    Implementation of a sensor to give velocity at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.VelocityProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    # state - tuple
    def _calculate_measurement(self, state):
        """
        Calculate the measurement - velocity.

        Returns
        -------
        list of floats
            Velocity of spacecraft.
        """
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)

PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": InspectionSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)


class InspectedPointsSensor(CWHSensor):
    """
    Implementation of a sensor to give number of points at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.InspectedPointProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - num_inspected_points.

        Returns
        -------
        int
            Number of inspected points.
        """
        return self.parent_platform.num_inspected_points


PluginLibrary.AddClassToGroup(
    InspectedPointsSensor, "Sensor_InspectedPoints", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)

PluginLibrary.AddClassToGroup(
    InspectedPointsSensor, "Sensor_InspectedPoints", {
        "simulator": InspectionSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
