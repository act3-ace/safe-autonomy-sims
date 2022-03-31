"""
This module defines the measurement and control properties for Dubins aircraft sensors and controllers.
"""

import typing

import numpy as np
from act3_rl_core.libraries.property import BoxProp
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated


class AccelerationProp(BoxProp):
    """
    Acceleration control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "acceleration"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-96.5]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [96.5]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["ft/s/s"]
    description: str = "Direct Acceleration Control"


class YawRateProp(BoxProp):
    """
    Yaw rate control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "yaw_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(-10)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(10)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad/s"]
    description: str = "Direct Yaw Rate Control"


class PitchRateProp(BoxProp):
    """
    Pitch rate control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "pitch_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(-10)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(10)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad/s"]
    description: str = "Direct Pitch Rate Control"


class RollRateProp(BoxProp):
    """
    Roll rate control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "roll_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(-10)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(10)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad/s"]
    description: str = "Direct Roll Rate Control"


class YawAndAccelerationProp(BoxProp):
    """
    Combined Yaw Rate and Acceleration control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "yaw_acceleration"
    low: Annotated[typing.List[StrictFloat], Field(min_items=2, max_items=2)] = [np.deg2rad(-10), -96.5]
    high: Annotated[typing.List[StrictFloat], Field(min_items=2, max_items=2)] = [np.deg2rad(10), 96.5]
    unit: Annotated[typing.List[StrictStr], Field(min_items=2, max_items=2)] = ["rad/s", "ft/s/s"]
    description: str = "Direct Yaw Rate and Acceleration Control"


class PitchRollAndAccelerationProp(BoxProp):
    """
    Combined Pitch Rate, Roll Rate, and Acceleration control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "pitch_roll_acceleration"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [np.deg2rad(-5), np.deg2rad(-10), -96.5]
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [np.deg2rad(5), np.deg2rad(10), 96.5]
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad/s", "rad/s", "ft/s/s"]
    description: str = "Direct Pitch Rate, Roll Rate and Acceleration Control"


class PositionProp(BoxProp):
    """
    Position sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-500000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [500000.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["ft"] * 3
    description: str = "Position Sensor Properties"


class VelocityProp(BoxProp):
    """
    Velocity sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-2000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [2000.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["ft/s"] * 3
    description: str = "Velocity Sensor Properties"


class HeadingProp(BoxProp):
    """
    Heading sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "heading"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-np.pi]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.pi]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad"]
    description: str = "Heading Sensor Properties"


class FlightPathProp(BoxProp):
    """
    Flight path sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "flight_path"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-np.pi]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.pi]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad"]
    description: str = "Flight Path Sensor Properties"


class RollProp(BoxProp):
    """
    Roll sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "roll"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-np.pi]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.pi]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad"]
    description: str = "Roll Sensor Properties"


class QuaternionProp(BoxProp):
    """
    Quaternion sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "quaternion"
    low: Annotated[typing.List[StrictFloat], Field(min_items=4, max_items=4)] = [-1.] * 4
    high: Annotated[typing.List[StrictFloat], Field(min_items=4, max_items=4)] = [1.] * 4
    unit: Annotated[typing.List[StrictStr], Field(min_items=4, max_items=4)] = ["None"] * 4
    description: str = "Quaternion Sensor Properties"


class SpeedProp(BoxProp):
    """
    Speed sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "speed"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [200]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [400]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["ft/s"]
    description: str = "Speed Sensor Properties"
