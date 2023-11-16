"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains various utility functions.
"""
import typing

import numpy as np
from corl.simulators.base_platform import BasePlatform
from corl.simulators.common_platform_utils import get_sensor_by_name
from pydantic import BaseModel


def shallow_dict_merge(a: typing.Dict, b: typing.Dict, in_place: bool = False, allow_collisions: bool = True):
    """
    Merges dictionaries a and b. Keeps entries from input a if collisions as ignored.

    This is a shallow merge, values in a and b are not copied

    Parameters
    ----------
    a : dict
        Merged into output dictionary. Values preferred in key collision if allow_collisions is false. Modified in-place if in_place True
    b : dict
        Merged into output dictionary. Values ignored if key collides with a and allow_collisions is false.
    in_place: bool
        Modifies input dict a in-place if True, by default False
    allow_collisions : bool, optional
        If True, takes values from input dict a when keys collide.
        If False, raises KeyCollisionError when keys collide.chief_position
        Formed by merging dictionaries a and b
    """

    if in_place:
        output = a
    else:
        output = {**a}

    for k, v in b.items():
        if k in output:
            if allow_collisions:
                continue

            raise KeyCollisionError(k)

        output[k] = v

    return output


class KeyCollisionError(Exception):
    """Custom exception for dictionary merging key collision errors

    Parameters
    ----------
    key : Any
        colliding key
    message: str
        Optional error message
    """

    def __init__(self, key, message: str = None):
        self.key = key
        if message is None:
            message = f"Keys '{key}' collided"
        super().__init__(message)


def get_relative_position(platform: BasePlatform, reference_position_sensor_name: str):
    """
    Finds the relative position between a platform and its reference_position Sensor's returned position.

    Parameters
    ----------
    platform: BasePlatform
        The name of the platform whose relative position will be returned
    reference_position_sensor_name: str
        The name of the sensor on the platform responsible for tracking the absolute position of a reference entity

    Returns
    -------
    relative_position: numpy.ndarray
        The x,y,z length vector describing the relative position of the platform from the reference entity.
    """

    position = platform.position
    reference_position = get_sensor_by_name(platform, reference_position_sensor_name).get_measurement()
    relative_position = np.array(position) - reference_position.m

    return relative_position


def get_relative_velocity(platform: BasePlatform, reference_velocity_sensor_name: str):
    """
    Finds the relative velocity between a platform and its reference_position Sensor's returned velocity.

    Parameters
    ----------
    platform: BasePlatform
        The name of the platform whose relative position will be returned
    reference_velocity_sensor_name: str
        The name of the sensor on the platform responsible for tracking the absolute velocity of a reference entity

    Returns
    -------
    relative_velocity: numpy.ndarray
        The x,y,z length vector describing the relative velocity of the platform from the reference entity.
    """

    velocity = platform.velocity
    reference_velocity = get_sensor_by_name(platform, reference_velocity_sensor_name).get_measurement()
    relative_velocity = np.array(velocity) - reference_velocity.m

    return relative_velocity


def velocity_limit(distance, velocity_threshold, threshold_distance, mean_motion, slope=2.0):
    """
    Get the velocity limit from the platform's current distance from the docking region.

    Parameters
    ----------
    distance: numpy.ndarray
        The x,y,z length vector describing the distance of the platform from the docking region.
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance: float
        The radius of the docking region.
    mean_motion: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region.

    Returns
    -------
    float
        The velocity limit given the platform's position.
    """

    vel_limit = velocity_threshold
    if distance > threshold_distance:
        vel_limit += slope * mean_motion * (distance - threshold_distance)
    return vel_limit


def max_vel_violation(relative_position, relative_velocity, velocity_threshold, threshold_distance, mean_motion, lower_bound, slope=2.0):
    """
    Get the magnitude of a velocity limit violation if one has occurred.

    Parameters
    ----------
    relative_position: numpy.ndarray
        The x,y,z length vector describing the distance of the platform from the docking region.
    relative_velocity: numpy.ndarray
        The x,y,z length vector describing the velocity of the platform with respect to the docking region.
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance: float
        The radius of the docking region.
    mean_motion: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    lower_bound: bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region.

    Returns
    -------
    violated: bool
        Boolean value indicating if the velocity limit has been violated.
    violation: float
        The magnitude of the velocity limit violation.
    """
    distance = np.linalg.norm(relative_position)
    relative_velocity_magnitude = np.linalg.norm(relative_velocity)

    vel_limit = velocity_limit(distance, velocity_threshold, threshold_distance, mean_motion, slope=slope)

    violation = relative_velocity_magnitude - vel_limit
    violated = bool(relative_velocity_magnitude > vel_limit)
    if lower_bound:
        violation *= -1
        violated = bool(relative_velocity_magnitude < vel_limit)

    return violated, violation


class VelocityConstraintValidator(BaseModel):
    """
    A configuration validator for velocity constraint configuration options.

    Attributes
    ----------
    velocity_threshold : float
        The maximum tolerated velocity within crashing region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the crashing region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0


def get_closest_fft_distance(state: np.ndarray, n: float, time_array: typing.Union[np.ndarray, list]) -> float:
    """
    Get the closest Free Flight Trajectory (FFT) distance over a given time horizon.
    Use closed form CWH dynamics to calculate future states in the absence of control

    Parameters
    ----------
    state: np.ndarray
        Initial state
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    time_array: Union[np.ndarray, list]
        Array containing each point in time to check the FFT trajectory

    Returns
    -------
    float
        Closest relative distance to the origin achieved during the FFT
    """

    distances = []
    for t in time_array:
        x = (4 - 3 * np.cos(n * t)) * state[0] + np.sin(n * t) * state[3] / n + 2 / n * (1 - np.cos(n * t)) * state[4]
        y = 6 * (np.sin(n * t) - n * t) * state[0] + state[1] - 2 / n * (1 - np.cos(n * t)) * state[3] + (4 * np.sin(n * t) -
                                                                                                          3 * n * t) * state[4] / n
        z = state[2] * np.cos(n * t) + state[5] / n * np.sin(n * t)
        distances.append(np.linalg.norm([x, y, z]))
    return float(min(distances))
