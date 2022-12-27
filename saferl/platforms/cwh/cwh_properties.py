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
