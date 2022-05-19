"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Sensors reporting relative measurements to a paired aircraft.

Author: Jamie Cunningham
"""
import typing

from corl.libraries.plugin_library import PluginLibrary

import saferl.platforms.dubins.sensors.dubins_sensors as d
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator
from saferl.utils import get_rejoin_region_center


class RelativePositionSensor(d.PositionSensor):
    """
    Calculate the relative position between the platform and its partner.
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - relative position to partner.

        Parameters
        ----------
        state: np.ndarray
            Current state.

        Returns
        -------
        np.ndarray
            Quaternion.
        """
        pos = self.parent_platform.position
        ref_pos = self.parent_platform.partner_position
        rel_pos = ref_pos - pos
        return rel_pos


PluginLibrary.AddClassToGroup(
    RelativePositionSensor,
    "Sensor_Relative_Position", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    RelativePositionSensor,
    "Sensor_Relative_Position", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RelativeVelocitySensor(d.VelocitySensor):
    """
    Calculate the relative velocity between the platform and its partner.
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - relative velocity to partner.

        Parameters
        ----------
        state: np.ndarray
            Current state.

        Returns
        -------
        np.ndarray
            Quaternion.
        """
        vel = self.parent_platform.velocity
        ref_vel = self.parent_platform.partner_velocity
        rel_vel = ref_vel - vel
        return rel_vel


PluginLibrary.AddClassToGroup(
    RelativeVelocitySensor,
    "Sensor_Relative_Velocity", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    RelativeVelocitySensor,
    "Sensor_Relative_Velocity", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RelativeRejoinPositionSensorValidator(d.DubinsSensorValidator):
    """
    Validator for the RelativeRejoinPositionSensor.

    offset: float
        The cartesian offset of the rejoin region from the platform's partner.
    """
    offset: typing.List[float]


class RelativeRejoinPositionSensor(d.PositionSensor):
    """
    Calculate the relative position between the platform and the rejoin region.
    """

    @property
    def get_validator(self):
        return RelativeRejoinPositionSensorValidator

    def _raw_measurement(self, state):
        """
        Calculate the measurement - relative position to rejoin region.

        Parameters
        ----------
        state: np.ndarray
            Current state.

        Returns
        -------
        np.ndarray
            Quaternion.
        """
        pos = self.parent_platform.position
        ref_pos = get_rejoin_region_center(self.parent_platform.partner, self.config.offset)
        rel_pos = ref_pos - pos
        return rel_pos


PluginLibrary.AddClassToGroup(
    RelativeRejoinPositionSensor,
    "Sensor_Relative_Rejoin_Position", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    RelativeRejoinPositionSensor,
    "Sensor_Relative_Rejoin_Position", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
