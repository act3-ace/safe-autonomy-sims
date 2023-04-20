"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Contains implementations of sensors that can be used in junction with the six dof CWH platform.
"""
from corl.libraries.plugin_library import PluginLibrary

import safe_autonomy_sims.platforms.cwh.cwh_properties as cwh_props
from safe_autonomy_sims.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from safe_autonomy_sims.platforms.cwh.cwh_sensors import CWHSensor
from safe_autonomy_sims.simulators.cwh_simulator import CWHSimulator
from safe_autonomy_sims.simulators.inspection_simulator import InspectionSimulator


class QuaternionSensor(CWHSensor):
    """
    Implementation of a sensor designed to give the quaternion at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.QuaternionProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - quaternion.

        Returns
        -------
        list of floats
            quaternion of spacecraft.
        """
        return self.parent_platform.quaternion


class AngularVelocitySensor(CWHSensor):
    """
    Implementation of a sensor designed to give the angular velocity at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.AngularVelocityProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - angular_velocity.

        Returns
        -------
        list of floats
            Angular velocity of spacecraft.
        """
        return self.parent_platform.angular_velocity


for sim in [CWHSimulator, InspectionSimulator]:
    for sensor, sensor_name in zip([QuaternionSensor, AngularVelocitySensor], ["Sensor_Quaternion", "Sensor_AngularVelocity"]):
        PluginLibrary.AddClassToGroup(sensor, sensor_name, {"simulator": sim, "platform_type": CWHAvailablePlatformTypes.CWHSixDOF})
