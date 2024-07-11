"""Reward functions for the docking environment"""

import math
from safe_autonomy_sims.gym.docking.utils import rel_dist, rel_vel, delta_v


def distance_pivot_reward(
    state: dict,
    prev_state: dict,
    c: float = 2.0,
    pivot_ratio: float = 2.0,
    pivot: float = 100,
):
    """A dense exponential reward based on the distance
    between the chief and the deputy.

    The reward is calculated from the current
    and previous state of the system and adjusted
    based on a given pivot value.

    $r_t = c(e^(-ad_t)-e^(ad_(t-1)))$

    where
    * $c$ is a scale factor
    * $a$ is the exponential coefficient given by
    $a = ln(pivot_ratio) / pivot $
    * $d_t$ is the distance between the chief and deputy at time $t$

    The range of the reward is adjusted based on whether the distance
    between the chief and the deputy is greater than or
    less than the pivot value.

    Parameters
    ----------
    state : dict
        current simulation state
    prev_state : dict
        previous simulation state
    c : float, optional
        scale factor, by default 2.0
    pivot_ratio : float, optional
        reward adjustment ratio when d>pivot, by default 2.0
    pivot : int, optional
        distance where the reward gets adjusted, by default 100

    Returns
    -------
    float
        reward value
    """
    a = math.log(pivot_ratio) / pivot
    r = c * (math.exp(-a * rel_dist(state)) - math.exp(-a * rel_dist(prev_state)))
    return r


def delta_v_reward(state: dict, prev_state: dict, m: float = 12.0, b: float = 0.0):
    """A dense reward based on the deputy's fuel
    use (change in velocity).

    $r_t = -((\deltav / m) + b)$

    where
    * $\deltav$ is the change in velocity
    * $m$ is the mass of the deputy
    * $b$ is a tunable bias term

    Parameters
    ----------
    state : dict
        current simulation state
    prev_state : dict
        previous simulation state
    m : float, optional
        deputy mass, by default 12.0
    b : float, optional
        bias term, by default 0.0

    Returns
    -------
    float
        reward value
    """
    r = -((delta_v(state, prev_state) / m) + b)
    return r


def velocity_constraint_reward(state: dict, v_limit: float):
    """A dense reward that punishes the deputy
    for violating a distance-based velocity constraint.

    $r_t = min(-(v - v_{limit}), 0)$

    where:
    * $v$ is the deputy's velocity
    * $v_{limit}$ is a velocity constraint

    Parameters
    ----------
    state : dict
        current simulation state
    v_limit : float
        velocity limit

    Returns
    -------
    float
        reward value
    """
    r = min(-(rel_vel(state=state) - v_limit), 0)
    return r


def docking_success_reward(
    state: dict,
    t: float,
    vel_limit: float,
    docking_radius: float = 0.2,
    max_time: float = 2000,
):
    """A sparse reward based on the amount of time
    it took the deputy to successfully dock with
    the chief.

    If docked at a safe velocity:
    $r_t = 1 + (1-t/t_{max})$
    else:
    $r_t = 0$

    Parameters
    ----------
    state : dict
        current simulation state
    t : float
        current timestep
    vel_limit : float
        maximum safe velocity
    docking_radius : float, optional
        radius of the docking region, by default 0.2
    max_time : int, optional
        maximum allowed time to dock, by default 2000

    Returns
    -------
    float
        reward value
    """
    in_docking = rel_dist(state=state) < docking_radius
    safe_velocity = rel_vel(state=state) < vel_limit
    if in_docking and safe_velocity:
        r = 1.0 + (1 - t / max_time)
    else:
        r = 0
    return r


def timeout_reward(t: float, max_time: float = 2000):
    """A sparse reward that punishes the agent
    for not completing the task within the time
    limit.

    Parameters
    ----------
    t : float
        current simulation time
    max_time : float, optional
        maximum time allowed to complete the task, by default 2000

    Returns
    -------
    float
        reward value
    """
    r = 0
    if t > max_time:
        r = -1.0
    return r


def crash_reward(state: dict, vel_limit: float, docking_radius: float = 0.2):
    """A sparse reward that punishes the agent
    for entering the docking region at an unsafe velocity
    (crashing).

    Parameters
    ----------
    state : dict
        current simulation state
    vel_limit : float
        maximum safe velocity
    docking_radius : float, optional
        radius of the docking region, by default 0.2

    Returns
    -------
    float
        reward value
    """
    in_docking = rel_dist(state=state) < docking_radius
    safe_velocity = rel_vel(state=state) < vel_limit
    if in_docking and not safe_velocity:
        r = -1.0
    else:
        r = 0
    return r


def out_of_bounds_reward(state: float, max_distance: float = 10000):
    """A sparse reward that punishes the agent
    for going out of bounds.

    Parameters
    ----------
    state : dict
        current simulation state
    max_distance : int, optional
        maximum allowed distance from the chief, by default 10000

    Returns
    -------
    float
        reward value
    """
    d = rel_dist(state=state)
    if d > max_distance:
        r = -1.0
    else:
        r = 0
    return r
