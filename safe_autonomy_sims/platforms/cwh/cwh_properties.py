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

import gymnasium
import numpy as np
from corl.libraries.property import BoxProp, NestedQuantity, Prop, Quantity
from pydantic import Field, StrictFloat
from typing_extensions import Annotated


class ThrustProp(BoxProp):
    """
    Thrust control properties.

    Attributes
    ----------
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
    unit: str = "newtons"
    description: str = "Direct Thrust Control"


class MomentProp(BoxProp):
    """
    Moment control properties.

    Attributes
    ----------
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
    unit: str = "dimensionless"
    description: str = "Direct Moment Control"


class PositionProp(BoxProp):
    """
    Position sensor properties.

    Attributes
    ----------
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
    unit: str = "m"
    description: str = "Position Sensor Properties"


class RelativePositionProp(BoxProp):
    """
    Relative position sensor properties.

    Attributes
    ----------
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

    name: str = "relative_position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-20000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [20000.0] * 3
    unit: str = "m"
    description: str = "Relative Position Sensor Properties"


class VelocityProp(BoxProp):
    """
    Velocity sensor properties.

    Attributes
    ----------
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
    unit: str = "m/s"
    description: str = "Velocity Sensor Properties"


class RelativeVelocityProp(BoxProp):
    """
    Relative velocity sensor properties.

    Attributes
    ----------
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

    name: str = "relative_velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-2000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [2000.0] * 3
    unit: str = "m/s"
    description: str = "Relative Velocity Sensor Properties"


class InspectedPointProp(BoxProp):
    """
    Inspected points sensor properties.

    Attributes
    ----------
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

    name: str = "inspected_points"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [100.]
    unit: str = "dimensionless"
    description: str = "Inspected Points Sensor Properties"


class SunAngleProp(BoxProp):
    """
    Sun angle sensor properties.

    Attributes
    ----------
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
    unit: str = "rad"
    description: str = "Sun Angle Sensor Properties"


class SunVectorProp(BoxProp):
    """
    Sun vector sensor properties.

    Attributes
    ----------
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

    name: str = "SunVector"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-1.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [1.0] * 3
    unit: str = "m"
    description: str = "Sun Unit Vector Properties"


class UninspectedPointProp(BoxProp):
    """
    Uninspected points cluster location properties.

    Attributes
    ----------
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
    unit: str = "m"
    description: str = "Uninspected points cluster location"


class BoolArrayProp(BoxProp):
    """
    bool array properties.

    Attributes
    ----------
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
    unit: str = "dimensionless"
    description: str = "Boolean array"


class QuaternionProp(BoxProp):
    """
    Quaternion sensor properties.

    Attributes
    ----------
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
    unit: str = "dimensionless"
    description: str = "Quaternion Sensor Properties"


class OrientationVectorProp(BoxProp):
    """
    Orientation Unit Vector sensor properties.

    Attributes
    ----------
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
    unit: str = "dimensionless"
    description: str = "Orientation Unit Vector Sensor Properties"


class RotatedAxesProp(BoxProp):
    """
    Coordinate axis unit vectors rotated into agent frame

    Attributes
    ----------
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

    name: str = "rotated_axes_unit_vectors"
    low: Annotated[typing.List[StrictFloat], Field(min_items=6, max_items=6)] = [-1.0] * 6
    high: Annotated[typing.List[StrictFloat], Field(min_items=6, max_items=6)] = [1.0] * 6
    unit: str = "dimensionless"
    description: str = "Rotated Axes Unit Vectors Sensor Properties"


class AngularVelocityProp(BoxProp):
    """
    Angular Velocity sensor properties.

    Attributes
    ----------
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
    unit: str = "rad/s"
    description: str = "Angular Velocity Sensor Properties"


class PriorityVectorProp(BoxProp):
    """
    Priority Vector sensor properties.

    Attributes
    ----------
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
    unit: str = "m"
    description: str = "Priority Vector Sensor Properties"


class PointsScoreProp(BoxProp):
    """
    Inspected points score sensor properties.

    Attributes
    ----------
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
    unit: str = "dimensionless"
    description: str = "Inspected Points Score Sensor Properties"


class TupleProp(Prop):
    """
    Tuple space properties

    spaces : tuple
        set of subspaces to combine into a single Tuple space
    """
    spaces: typing.Tuple

    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates MultiDiscrete gymnasium space
        """
        subspaces = tuple(sub_prop.create_space(seed=seed) for sub_prop in self.spaces)
        return gymnasium.spaces.Tuple(subspaces)

    def get_units(self):
        return tuple({prop.name: prop.get_units()} for prop in self.spaces)

    def create_unit_converted_prop(self, unit: str):
        raise RuntimeError(
            f"Prop was told to try and convert to unit {unit} but the only but a Prop is not "
            f"capable of having a unit as it is not a leaf node.  this error should never happen and is just a sanity check"
            "to show this code path is not supported"
        )

    def zero_mean(self) -> Prop:
        raise NotImplementedError

    def scale(self, scale) -> Prop:
        raise NotImplementedError

    def create_quantity(self, value: dict | (float | (int | (list | np.ndarray)))) -> Quantity | NestedQuantity:
        """
        This function taskes in values and will attempt to create either a Quantity or NestedQuantity
        from it, properly applying units along the way
        """
        raise NotImplementedError

    def create_zero_sample(self) -> Quantity | NestedQuantity:
        """
        This function will attempt to return 0 for each leaf node for all properties
        in a tree, as this is usually a safe default.  however if 0 is not in the low
        or high for this space it will return the lowest value possible
        Discrete values will always be the lowest, which should be 0
        """
        raise NotImplementedError

    def create_low_sample(self) -> Quantity | NestedQuantity:
        """
        This function will attempt to return the lowest possible value for each leaf node
        for all properties in a tree.
        """
        raise NotImplementedError


class OrbitStabilityProp(BoxProp):
    """
    Property to hold the orbit stability quantity 2*n*x + v_y
    Highs and lows are obtained manually from this equation and the highs/lows
    of the position and velocity properties and rounded

    Attributes
    ----------
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

    name: str = "orbit stability"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-1025.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1025.0]
    unit: str = "m/s"
    description: str = "Orbit Stability Sensor Property"
