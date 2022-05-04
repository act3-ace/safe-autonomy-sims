"""
This module contains various utility functions.

Author: Jamie Cunningham
"""

import numpy as np
from corl.simulators.common_platform_utils import get_platform_by_name


def velocity_limit(state, agent_name, velocity_threshold, threshold_distance, mean_motion, slope=2.0):
    """
    Get the velocity limit from the agent's current position.

    Parameters
    ----------
    state: StateDict
        The current state of the system.
    agent_name: str
        The name of the agent platform
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing
    threshold_distance: float
        The radius of the docking region
    mean_motion: float
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region

    Returns
    -------
    float
        The velocity limit given the agent's position.
    """
    deputy = get_platform_by_name(state, agent_name)
    dist = np.linalg.norm(deputy.position)
    vel_limit = velocity_threshold
    if dist > threshold_distance:
        vel_limit += slope * mean_motion * (dist - threshold_distance)
    return vel_limit


def max_vel_violation(state, agent_name, velocity_threshold, threshold_distance, mean_motion, lower_bound, slope=2.0):
    """
    Get the magnitude of a velocity limit violation if one has occurred.

    Parameters
    ----------
    state: StateDict
        The current state of the system.
    agent_name: str
        The name of the agent platform
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing
    threshold_distance: float
        The radius of the docking region
    mean_motion: float
    lower_bound: bool
        If True, the function enforces a minimum velocity constraint on the agent's platform
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region

    Returns
    -------
    violated: bool
        Boolean value indicating if the velocity limit has been violated
    violation: float
        The magnitude of the velocity limit violation.
    """
    deputy = get_platform_by_name(state, agent_name)
    rel_vel = deputy.velocity
    rel_vel_mag = np.linalg.norm(rel_vel)

    vel_limit = velocity_limit(state, agent_name, velocity_threshold, threshold_distance, mean_motion, slope=slope)

    violation = rel_vel_mag - vel_limit
    violated = rel_vel_mag > vel_limit
    if lower_bound:
        violation *= -1
        violated = rel_vel_mag < vel_limit

    return violated, violation


def get_rejoin_region_center(ref, offset):
    """
    Get the position of the rejoin region's center

    Parameters
    ----------
    ref: BasePlatform
        The reference platform for the rejoin region
    offset: np.ndarray (length <= 3)
        The cartesian offset of the center of the rejoin region from the reference platform

    Returns
    -------
    center: np.ndarray
        The [x, y, z] position of the rejoin region's center
    """
    full_offset = np.zeros(3)
    full_offset[:len(offset)] = offset
    ref_orientation = ref.orientation
    full_offset = ref_orientation.apply(full_offset)
    center = ref.position + full_offset
    return center


def in_rejoin(wingman, lead, radius, offset):
    """
    Determines if the wingman platform is within the rejoin region relative to the lead platform.

    Parameters
    ----------
    wingman: BasePlatform
        The wingman platform
    lead: BasePlatform
        The lead platform
    radius: float
        The radius of the rejoin region
    offset: np.ndarray (length <= 3)
        The cartesian offset of the rejoin region's center from the lead platform

    Returns
    -------
    in_rejoin: bool
        Value is true if wingman platform is within the rejoin region
    distance: float
        Distance from center of rejoin region
    """
    rejoin_center = get_rejoin_region_center(lead, offset)
    distance = np.linalg.norm(wingman.position - rejoin_center)
    in_region = distance <= radius
    return in_region, distance
