import math
import typing

import numpy as np
from act3_rl_core.libraries.property import BoxProp
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated


class AccelerationProp(BoxProp):
    name: str = "acceleration"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-96.5]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [96.5]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["ft/s/s"]
    description: str = "Direct Acceleration Control"


class YawRateProp(BoxProp):
    name: str = "yaw_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(-10)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(10)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad/s"]
    description: str = "Direct Yaw Rate Control"


class PitchRateProp(BoxProp):
    name: str = "pitch_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(-10)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(10)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad/s"]
    description: str = "Direct Pitch Rate Control"


class RollRateProp(BoxProp):
    name: str = "roll_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(-10)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.deg2rad(10)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad/s"]
    description: str = "Direct Roll Rate Control"


class YawAndAccelerationProp(BoxProp):
    name: str = "yaw_acceleration"
    low: Annotated[typing.List[StrictFloat], Field(min_items=2, max_items=2)] = [np.deg2rad(-10), -96.5]
    high: Annotated[typing.List[StrictFloat], Field(min_items=2, max_items=2)] = [np.deg2rad(10), 96.5]
    unit: Annotated[typing.List[StrictStr], Field(min_items=2, max_items=2)] = ["rad/s", "ft/s/s"]
    description: str = "Direct Yaw Rate and Acceleration Control"


class PitchRollAndAccelerationProp(BoxProp):
    name: str = "pitch_roll_acceleration"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [np.deg2rad(-5), np.deg2rad(-10), -96.5]
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [np.deg2rad(5), np.deg2rad(10), 96.5]
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad/s", "rad/s", "ft/s/s"]
    description: str = "Direct Pitch Rate, Roll Rate and Acceleration Control"


class PositionProp(BoxProp):
    name: str = "position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-math.inf] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [math.inf] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["ft"] * 3
    description: str = "Position Sensor Properties"


class VelocityProp(BoxProp):
    name: str = "velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-math.inf] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [math.inf] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["ft/s"] * 3
    description: str = "Velocity Sensor Properties"


class HeadingProp(BoxProp):
    name: str = "heading"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-math.pi * 2]
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-math.pi * 2]
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad"]
    description: str = "Heading Sensor Properties"


class FlightPathProp(BoxProp):
    name: str = "flight_path"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-math.pi * 2]
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-math.pi * 2]
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad"]
    description: str = "Flight Path Sensor Properties"
