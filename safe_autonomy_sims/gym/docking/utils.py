"""Utility functions for the docking environment"""

import numpy as np


def polar_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """Convert polar coordinates to cartesian coordinates.

    Parameters
    ----------
    r : float
        radial distance
    theta : float
        azimuthal angle
    phi : float
        polar angle

    Returns
    -------
    np.ndarray
        cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def rel_dist(state: dict):
    """The relative distance between the chief and the deputy.

    Parameters
    ----------
    state : dict
        the current state of the system

    Returns
    -------
    float
        the euclidean distance between the chief and the deputy
    """
    chief_pos = state["chief"][:3]
    deputy_pos = state["deputy"][:3]
    rel_d = np.linalg.norm(chief_pos - deputy_pos)
    return rel_d


def rel_vel(state: dict):
    """The relative velocity between the chief and the deputy.

    Parameters
    ----------
    state : dict
        the current state of the system

    Returns
    -------
    float
        the relative velocity between the chief and the deputy
    """
    chief_v = state["chief"][3:6]
    deputy_v = state["deputy"][3:6]
    rel_v = np.linalg.norm(chief_v - deputy_v)
    return rel_v


def delta_v(state: dict, prev_state: dict):
    """The change in velocity of the deputy.

    Parameters
    ----------
    state : dict
        the current simulation state
    prev_state : dict
        the previous simulation state

    Returns
    -------
    float
        the deputy's change in velocity
    """
    v = np.linalg.norm(state["deputy"][3:6])
    prev_v = np.linalg.norm(prev_state["deputy"][3:6])
    return v - prev_v


def v_limit(
    state: dict,
    a: float = 2.0,
    n: float = 0.001027,
    v_max: float = 0.2,
    docking_radius: float = 0.5,
):
    """A linear velocity limit based on the distance between the chief
    and the deputy.

    $v_{limit} = v_{max} + an(d-r)$

    where:
    * $v_max$ is a maximum allowable velocity
    * $a$ is the slope of the linear velocity limit
    * $d$ is the distance between the chief and the deputy
    * $r$ is the radius of the docking region around the chief

    Parameters
    ----------
    state : dict
        the current simulation state
    a : float, optional
        slope of the linear velocity limit, by default 2.0
    n : float, optional
        deputy mean motion, by default 0.001027
    v_max : float, optional
        maximum allowable velocity, by default 0.2
    docking_radius : float, optional
        the radius of the docking region around the chief, by default 0.5

    Returns
    -------
    float
        velocity limit
    """
    v_limit = v_max + (a * n * (rel_dist(state=state) - docking_radius))
    return v_limit
