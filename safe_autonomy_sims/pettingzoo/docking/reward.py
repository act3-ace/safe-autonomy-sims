"""Reward functions for the docking environment"""

import math
import numpy as np
import safe_autonomy_simulation.entities as e
import safe_autonomy_sims.pettingzoo.docking.utils as utils


def distance_pivot_reward(
    rel_dist: float,
    rel_dist_prev: float,
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
    rel_dist : float
        current relative distance
    rel_dist_prev : float
        previous relative distance
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
    r = c * (math.exp(-a * rel_dist) - math.exp(-a * rel_dist_prev))
    return r


def delta_v_reward(v: np.ndarray, prev_v: np.ndarray, m: float = 12.0, b: float = 0.0):
    """A dense reward based on the deputy's fuel
    use (change in velocity).

    $r_t = -((\deltav / m) + b)$

    where
    * $\deltav$ is the change in velocity
    * $m$ is the mass of the deputy
    * $b$ is a tunable bias term

    Parameters
    ----------
    v : np.ndarray
        current velocity
    prev_v : np.ndarray
        previous velocity
    m : float, optional
        deputy mass, by default 12.0
    b : float, optional
        bias term, by default 0.0

    Returns
    -------
    float
        reward value
    """
    r = -((utils.delta_v(v=v, prev_v=prev_v) / m) + b)
    return r


def velocity_constraint_reward(v1: np.ndarray, v2: np.ndarray, v_limit: float):
    """A dense reward that punishes the deputy
    for violating a distance-based velocity constraint.

    $r_t = min(-(|v1 - v2| - v_{limit}), 0)$

    where:
    * $v1$ is the deputy's velocity
    * $v2$ is the chief's velocity
    * $v_{limit}$ is a velocity constraint

    Parameters
    ----------
    v1 : np.ndarray
        deputy velocity
    v2 : np.ndarray
        chief velocity
    v_limit : float
        velocity limit

    Returns
    -------
    float
        reward value
    """
    r = min(-(utils.rel_vel(vel1=v1, vel2=v2) - v_limit), 0)
    return r


def docking_success_reward(
    chief: e.PhysicalEntity,
    deputy: e.PhysicalEntity,
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
    chief : e.PhysicalEntity
        chief entity
    deputy : e.PhysicalEntity
        deputy entity
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
    in_docking = (
        utils.rel_dist(pos1=chief.position, pos2=deputy.position) < docking_radius
    )
    safe_velocity = utils.rel_vel(vel1=chief.velocity, vel2=deputy.velocity) < vel_limit
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


def crash_reward(
    chief: e.PhysicalEntity,
    deputy: e.PhysicalEntity,
    vel_limit: float,
    docking_radius: float = 0.2,
):
    """A sparse reward that punishes the agent
    for entering the docking region at an unsafe velocity
    (crashing).

    Parameters
    ----------
    chief : e.PhysicalEntity
        chief entity
    deputy : e.PhysicalEntity
        deputy entity
    vel_limit : float
        maximum safe velocity
    docking_radius : float, optional
        radius of the docking region, by default 0.2

    Returns
    -------
    float
        reward value
    """
    in_docking = (
        utils.rel_dist(pos1=chief.position, pos2=deputy.position) < docking_radius
    )
    safe_velocity = utils.rel_vel(vel1=chief.velocity, vel2=deputy.velocity) < vel_limit
    if in_docking and not safe_velocity:
        r = -1.0
    else:
        r = 0
    return r


def out_of_bounds_reward(
    chief_pos: np.ndarray, deputy_pos: np.ndarray, max_distance: float = 10000
):
    """A sparse reward that punishes the agent
    for going out of bounds.

    Parameters
    ----------
    chief_pos : np.ndarray
        chief position
    deputy_pos : np.ndarray
        deputy position
    max_distance : int, optional
        maximum allowed distance from the chief, by default 10000

    Returns
    -------
    float
        reward value
    """
    d = utils.rel_dist(pos1=chief_pos, pos2=deputy_pos)
    if d > max_distance:
        r = -1.0
    else:
        r = 0
    return r
