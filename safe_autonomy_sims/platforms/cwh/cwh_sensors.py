"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements sensors for CWH platforms.
"""
import typing

import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.units import corl_get_ureg
from corl.simulators.base_parts import BasePlatformPartValidator, BaseSensor

import safe_autonomy_sims.platforms.cwh.cwh_properties as cwh_props
from safe_autonomy_sims.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from safe_autonomy_sims.simulators.cwh_simulator import CWHSimulator
from safe_autonomy_sims.simulators.inspection_simulator import InspectionSimulator


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
        Quantity
            Position of spacecraft.
        """
        position = np.array(self.parent_platform.position, dtype=np.float32)
        return corl_get_ureg().Quantity(position, "m")


class RelativePositionSensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for RelativePositionSensor

    Attributes
    ----------
    entity_name: str
        The name of the entity the position of which is to be returned.
    """
    entity_name: str = "chief"


class RelativePositionSensor(CWHSensor):
    """
    Implementation of a sensor designed to give the position at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.RelativePositionProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return RelativePositionSensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - position.

        Returns
        -------
        list of floats
            Position of spacecraft.
        """
        relative_position = None
        # handle initialization case
        if self.config.entity_name not in state.sim_entities:
            # raise error if not initialization
            if state.sim_time != 0.0:
                raise ValueError(f"{self.config.entity_name} not found in simulator state!")
            relative_position = np.array([0.0, 0.0, 0.0])
        else:
            relative_position = self.parent_platform.entity_relative_position(state.sim_entities[self.config.entity_name])

        return corl_get_ureg().Quantity(np.array(relative_position, dtype=np.float32), "m")


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
        velocity = np.array(self.parent_platform.velocity, dtype=np.float32)
        return corl_get_ureg().Quantity(velocity, "meter/second")


class RelativeVelocitySensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for RelativeVelocitySensor

    Attributes
    ----------
    entity_name: str
        The name of the entity the velocity of which is to be returned.
    """
    entity_name: str = "chief"


class RelativeVelocitySensor(CWHSensor):
    """
    Implementation of a sensor designed to give the relative velocity at any
    time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.RelativeVelocityProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return RelativeVelocitySensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - position.

        Returns
        -------
        list of floats
            Position of spacecraft.
        """
        relative_velocity = None
        # handle initialization case
        if self.config.entity_name not in state.sim_entities:
            # raise error if not initialization
            if state.sim_time != 0.0:
                raise ValueError(f"{self.config.entity_name} not found in simulator state!")
            relative_velocity = np.array([0.0, 0.0, 0.0])
        else:
            relative_velocity = self.parent_platform.entity_relative_velocity(state.sim_entities[self.config.entity_name])

        return corl_get_ureg().Quantity(np.array(relative_velocity, dtype=np.float32), "m/s")


class InspectedPointsSensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for InspectedPointsSensor

    Attributes
    ----------
    inspector_entity_name: str
        The name of the entity performing inspection.
    """
    inspector_entity_name: str = ""


class InspectedPointsSensor(CWHSensor):
    """
    Implementation of a sensor to give number of points at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.InspectedPointProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return InspectedPointsSensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - num_inspected_points.

        Returns
        -------
        int
            Number of inspected points.
        """
        num_points_inspected = 0
        # handle initialization case
        if self.config.inspector_entity_name not in state.sim_entities:
            # raise error if not initialization
            if state.sim_time != 0.0:
                raise ValueError(f"{self.config.inspector_entity_name} not found in simulator state!")
        else:
            # get inspector entity object
            inspector_entity = state.sim_entities[self.config.inspector_entity_name]

            # count total number of points inspected by
            for points in state.inspection_points_map.values():
                num_points_inspected += points.get_num_points_inspected(inspector_entity=inspector_entity)

        return corl_get_ureg().Quantity(np.array([num_points_inspected], dtype=np.float32), "dimensionless")


class SunAngleSensor(CWHSensor):
    """
    Implementation of a sensor to give the sun angle
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.SunAngleProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - sun angle.

        Returns
        -------
        float
            sun angle
        """
        sun_angle = np.array([state.sun_angle], dtype=np.float32)
        return corl_get_ureg().Quantity(sun_angle, "radians")


class SunVectorSensor(CWHSensor):
    """
    Implementation of a sensor to give the sun unit vector
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.SunVectorProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - sun angle.

        Returns
        -------
        float
            sun angle
        """
        sun_position = np.array([np.cos(state.sun_angle), -np.sin(state.sun_angle), 0], dtype=np.float32)
        sun_position = corl_get_ureg().Quantity(sun_position, "m")
        return sun_position


class UninspectedPointsSensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for InspectedPointsSensor

    Attributes
    ----------
    inspector_entity_name: str
        The name of the entity performing inspection.
    inspection_entity_name: str
        The name of the entity under inspection.
    """
    inspector_entity_name: str = ""
    inspection_entity_name: str = ""


class UninspectedPointsSensor(CWHSensor):
    """
    Implementation of a sensor to give direction from the origin to the location
    of a cluster of uninspected points.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.UninspectedPointProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return UninspectedPointsSensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - cluster_location.

        Returns
        -------
        np.ndarray
            Cluster location of the uninspected points.
        """
        # handle initialization case
        if (
            self.config.inspection_entity_name not in state.inspection_points_map
            or self.config.inspector_entity_name not in state.sim_entities
        ):
            # raise error if not initialization
            if state.sim_time != 0.0:
                raise ValueError(
                    f"{self.config.inspector_entity_name} not found in simulator state \
                        or {self.config.inspection_entity_name} is not an inspectable entity!"
                )
            return corl_get_ureg().Quantity([0., 0., 0.], "meters")

        # get entities
        inspector_entity = state.sim_entities[self.config.inspector_entity_name]

        # get inspection points of object under inspection
        inspector_position = inspector_entity.position
        inspection_points = state.inspection_points_map[self.config.inspection_entity_name]
        cluster_position = inspection_points.kmeans_find_nearest_cluster(inspector_position)
        cluster_position = np.array(cluster_position, dtype=np.float32)
        return corl_get_ureg().Quantity(cluster_position, "meters")


# entity position sensors
class EntitySensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for the various EntitySensors

    Attributes
    ----------
    entity_name : str
        name of the entity to sense
    """
    entity_name: str


class EntityPositionSensor(CWHSensor):
    """
    Implementation of a sensor designed to give the position at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.PositionProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[EntitySensorValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return EntitySensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - position.

        Returns
        -------
        list of floats
            Position of spacecraft.
        """
        return corl_get_ureg().Quantity(state.sim_entities[self.config.entity_name].position, "meters")


class OriginPositionSensor(CWHSensor):
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
        return corl_get_ureg().Quantity(np.array([0, 0, 0]), "meters")


class EntityVelocitySensor(CWHSensor):
    """
    Implementation of a sensor designed to give the position at any time.
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.VelocityProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[EntitySensorValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return EntitySensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - position.

        Returns
        -------
        list of floats
            Position of spacecraft.
        """
        return corl_get_ureg().Quantity(state.sim_entities[self.config.entity_name].velocity, "meter/second")


class BoolArraySensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for BoolArraySensor.

    Attributes
    ----------
    inspection_entity_name: str
        The name of the entity under inspection.
    """
    inspection_entity_name: str = ""


class BoolArraySensor(CWHSensor):
    """
    Implementation of a sensor to give boolean array for all inspected/uninspected points
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.BoolArrayProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return BoolArraySensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - bool_array.

        Returns
        -------
        np.ndarray
            Bool array describing inspected/uninspected points.
        """
        bool_array = None
        # handle initialization case
        if self.config.inspection_entity_name not in state.inspection_points_map:
            # raise error if not initialization
            if state.sim_time != 0.0:
                raise ValueError(f"{self.config.inspection_entity_name} is not an inspectable entity!")
            bool_array = np.array([float(False)])
        else:
            inspection_points = state.inspection_points_map[self.config.inspection_entity_name]
            bool_array = np.array([float(bool(a)) for a in inspection_points.points_inspected_dict.values()])
            if len(bool_array) == 99:  # TODO: Remove hardcoded value
                bool_array = np.concatenate((bool_array, np.zeros(1)))

        return corl_get_ureg().Quantity(bool_array, "dimensionless")


class PriorityVectorSensor(CWHSensor):
    """
    Implementation of a sensor to give the inspected points priority vector
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.PriorityVectorProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - priority vector.

        Returns
        -------
        np.ndarray
            priority vector
        """
        priority_vector = np.array(state.priority_vector, dtype=np.float32)
        return corl_get_ureg().Quantity(priority_vector, "m")


class InspectedPointsScoreSensorValidator(BasePlatformPartValidator):
    """
    A configuration validator for InspectedPointsScoreSensor

    Attributes
    ----------
    inspection_entity_name: str
        The name of the entity under inspection.
    """
    inspector_entity_name: str = ""


class InspectedPointsScoreSensor(CWHSensor):
    """
    Implementation of a sensor to give the inspected points score
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.PointsScoreProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    @staticmethod
    def get_validator() -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return InspectedPointsScoreSensorValidator

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - inspected points score.

        Returns
        -------
        float
            inspected points score.
        """
        # handle initialization case
        if self.config.inspector_entity_name not in state.sim_entities:
            # raise error if not initialization
            if state.sim_time != 0.0:
                raise ValueError(f"{self.config.inspector_entity_name} not found in simulator state!")
            return corl_get_ureg().Quantity(np.array([0.]), "dimensionless")

        # get inspector entity object
        inspector_entity = state.sim_entities[self.config.inspector_entity_name]

        # count total number of points inspected by
        weight = 0.
        for points in state.inspection_points_map.values():
            weight += points.get_total_weight_inspected(inspector_entity=inspector_entity)

        return corl_get_ureg().Quantity(np.array([weight], dtype=np.float32), "dimensionless")


class OrbitStabilitySensor(CWHSensor):
    """
    Implementation of a sensor to give 2nx + v_y, the quantity that determines
    stability of the un-controlled motion
    """

    def __init__(self, parent_platform, config, property_class=cwh_props.OrbitStabilityProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - cluster_location.

        Returns
        -------
        np.ndarray
            Cluster location of the uninspected points.
        """
        pos = self.parent_platform.position
        vel = self.parent_platform.velocity

        n = self.parent_platform._platform.dynamics.n  # pylint: disable=protected-access

        orbit_stability = 2 * pos[0] * n + vel[1]

        return corl_get_ureg().Quantity(np.array([orbit_stability], dtype=np.float32), "dimensionless")


for sim in [CWHSimulator, InspectionSimulator]:
    for sensor, sensor_name in zip(
        [
            CWHSensor,
            PositionSensor,
            VelocitySensor,
            RelativePositionSensor,
            RelativeVelocitySensor,
            InspectedPointsSensor,
            SunAngleSensor,
            SunVectorSensor,
            UninspectedPointsSensor,
            BoolArraySensor,
            EntityPositionSensor,
            EntityVelocitySensor,
            OriginPositionSensor,
            PriorityVectorSensor,
            InspectedPointsScoreSensor,
            OrbitStabilitySensor,
        ],
        [
            "Sensor_Generic",
            "Sensor_Position",
            "Sensor_Velocity",
            "Sensor_RelativePosition",
            "Sensor_RelativeVelocity",
            "Sensor_InspectedPoints",
            "Sensor_SunAngle",
            "Sensor_SunVector",
            "Sensor_UninspectedPoints",
            "Sensor_BoolArray",
            "Sensor_EntityPosition",
            "Sensor_EntityVelocity",
            "Sensor_OriginPosition",
            "Sensor_PriorityVector",
            "Sensor_InspectedPointsScore",
            "Sensor_OrbitStability"
        ]
    ):
        PluginLibrary.AddClassToGroup(sensor, sensor_name, {"simulator": sim, "platform_type": CWHAvailablePlatformTypes})
