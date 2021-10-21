import typing

from act3_rl_core.libraries.property import BoxProp
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated


class ThrustProp(BoxProp):
    name: str = "thrust"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-1]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["newtons"]
    description: str = "Direct Thrust Control"


class PositionProp(BoxProp):
    name: str = "position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-80000] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [80000] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m"] * 3
    description: str = "Position Sensor Properties"


class VelocityProp(BoxProp):
    name: str = "velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-10000] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [10000] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m/s"] * 3
    description: str = "Velocity Sensor Properties"
