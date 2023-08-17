"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines the measurement and control properties for CWH spacecraft sensors and controllers.
"""
import typing

from corl.libraries.property import BoxProp, DiscreteProp
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated


class ThrustProp(BoxProp):
    """
    Thrust control properties.

    name : str
        Control property name.
    low : list[float]
        Minimum bounds of control input.
    high : list[float]
        Maximum bounds of control input.
    unit : str
        Unit of measurement for control input.
    description : str
        Description of control properties.
    """

    name: str = "thrust"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-1.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["newtons"]
    description: str = "Direct Thrust Control"


class MomentProp(BoxProp):
    """
    Moment control properties.

    name : str
        Control property name.
    low : list[float]
        Minimum bounds of control input.
    high : list[float]
        Maximum bounds of control input.
    unit : str
        Unit of measurement for control input.
    description : str
        Description of control properties.
    """

    name: str = "moment"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-0.001]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.001]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["none"]
    description: str = "Direct Moment Control"


class PositionProp(BoxProp):
    """
    Position sensor properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-10000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [10000.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m"] * 3
    description: str = "Position Sensor Properties"


class VelocityProp(BoxProp):
    """
    Velocity sensor properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-1000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [1000.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m/s"] * 3
    description: str = "Velocity Sensor Properties"


class InspectedPointProp(DiscreteProp):
    """
    Inspected points sensor properties.

    name : str
        Sensor property name.
    unit : str
        Unit of measurement for sensor output.
    n : int

    description : str
        Description of sensor properties.
    """

    name: str = "inspected_points"
    n: int = 101
    description: str = "Inspected Points Sensor Properties"


class SunAngleProp(BoxProp):
    """
    Sun angle sensor properties.

    name : str
        Sensor property name.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "SunAngle"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-10000.]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [10000.]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad"]
    description: str = "Sun Angle Sensor Properties"


class UninspectedPointProp(BoxProp):
    """
    Uninspected points cluster location properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "uninspected_points"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-1.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [1.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m"] * 3
    description: str = "Uninspected points cluster location"


class BoolArrayProp(BoxProp):
    """
    bool array properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "bool_array"
    num_points: int = 100
    low: Annotated[typing.List[StrictFloat], Field(min_items=num_points, max_items=num_points)] = [0.0] * num_points
    high: Annotated[typing.List[StrictFloat], Field(min_items=num_points, max_items=num_points)] = [1.0] * num_points
    unit: Annotated[typing.List[StrictStr], Field(min_items=num_points, max_items=num_points)] = ["None"] * num_points
    description: str = "Boolean array"


class QuaternionProp(BoxProp):
    """
    Quaternion sensor properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "quaternion"
    low: Annotated[typing.List[StrictFloat], Field(min_items=4, max_items=4)] = [-1.0] * 4
    high: Annotated[typing.List[StrictFloat], Field(min_items=4, max_items=4)] = [1.0] * 4
    unit: Annotated[typing.List[StrictStr], Field(min_items=4, max_items=4)] = ["N/A"] * 4
    description: str = "Quaternion Sensor Properties"


class OrientationVectorProp(BoxProp):
    """
    Orientation Unit Vector sensor properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "orientation_unit_vector"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-1.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [1.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["N/A"] * 3
    description: str = "Orientation Unit Vector Sensor Properties"


class AngularVelocityProp(BoxProp):
    """
    Angular Velocity sensor properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "angular velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-10.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [10.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad/s"] * 3
    description: str = "Angular Velocity Sensor Properties"


class PriorityVectorProp(BoxProp):
    """
    Priority Vector sensor properties.

    name : str
        Sensor property name.
    low : list[float]
        Minimum bounds of sensor output.
    high : list[float]
        Maximum bounds of sensor output.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "priority vector"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-1.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [1.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m"] * 3
    description: str = "Priority Vector Sensor Properties"


class PointsScoreProp(BoxProp):
    """
    Inspected points score sensor properties.

    name : str
        Sensor property name.
    unit : str
        Unit of measurement for sensor output.
    description : str
        Description of sensor properties.
    """

    name: str = "inspected points score"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1.]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["None"]
    description: str = "Inspected Points Score Sensor Properties"
