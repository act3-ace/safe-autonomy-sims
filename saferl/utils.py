"""
This module contains various utility functions.

Author: Jamie Cunningham
"""
import copy
import typing
from collections import OrderedDict

import gym
import numpy as np
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name


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


def normalize_space_from_mu_sigma(
    space_likes: typing.Tuple[gym.spaces.Space],
    mu: float = 0.0,
    sigma: float = 1.0,
) -> gym.spaces.Space:
    """
    Normalizes a given gym box using the provided mu and sigma

    Parameters
    ----------
    space_likes: typing.Tuple[gym.spaces.Space]
        the gym space to turn all boxes into the scaled space
    mu: float = 0.0
        mu for normalization
    sigma: float = 1.0
        sigma for normalization

    Returns
    -------
    gym.spaces.Space:
        the new gym spaces where all boxes have had their bounds changed
    """
    space_arg = space_likes[0]
    if isinstance(space_arg, gym.spaces.Box):
        low = np.divide(np.subtract(space_arg.low, mu), sigma)
        high = np.divide(np.subtract(space_arg.high, mu), sigma)
        return gym.spaces.Box(low=low, high=high, shape=space_arg.shape, dtype=np.float32)
    return copy.deepcopy(space_arg)


def normalize_sample_from_mu_sigma(
    space_likes: typing.Tuple[gym.spaces.Space, typing.Union[OrderedDict, dict, tuple, np.ndarray, list]],
    mu: float = 0.0,
    sigma: float = 1,
) -> typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
    """
    This normalizes a sample from a box space using the mu and sigma arguments

    Parameters
    ----------
    space_likes: typing.Tuple[gym.spaces.Space, sample_type]
        the first is the gym space
        the second is the sample of this space to scale
    mu: float
        the mu used for normalizing the sample
    sigma: float
        the sigma used for normalizing the sample

    Returns
    -------
    typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
        the normalized sample
    """
    (space_arg, space_sample_arg) = space_likes
    if isinstance(space_arg, gym.spaces.Box):
        val = np.array(space_sample_arg)
        norm_value = np.subtract(val, mu)
        norm_value = np.divide(norm_value, sigma)
        return norm_value.astype(np.float32)
    return copy.deepcopy(space_sample_arg)


def unnormalize_sample_from_mu_sigma(
    space_likes: typing.Tuple[gym.spaces.Space, typing.Union[OrderedDict, dict, tuple, np.ndarray, list]],
    mu: float = 0.0,
    sigma: float = 1,
) -> typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
    """
    This unnormalizes a sample from a box space using the mu and sigma arguments

    Parameters
    ----------
    space_likes: typing.Tuple[gym.spaces.Space, sample_type]
        the first is the gym space
        the second is the sample of this space to scale
    mu: float
        the mu used for unnormalizing the sample
    sigma: float
        the sigma used for unnormalizing the sample

    Returns
    -------
    typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
        the unnormalized sample
    """
    (space_arg, space_sample_arg) = space_likes
    if isinstance(space_arg, gym.spaces.Box):
        val = np.array(space_sample_arg)
        norm_value = np.add(np.multiply(val, sigma), mu)
        return norm_value.astype(np.float32)
    return copy.deepcopy(space_sample_arg)
