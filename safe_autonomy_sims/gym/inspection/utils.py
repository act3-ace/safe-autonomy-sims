"""Utility functions for the inspection environment"""

import numpy as np
import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.sims.inspection as sim


def polar_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """Convert polar coordinates to cartesian coordinates.

    Parameters
    ----------
    r : float
        radial distance
    theta : float
        polar angle
    phi : float
        azimuthal angle

    Returns
    -------
    np.ndarray
        cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rel_dist(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """The relative distance between two positions.

    Parameters
    ----------
    pos1 : np.ndarray
        the first position
    pos2 : np.ndarray
        the second position

    Returns
    -------
    float
        the euclidean distance between the two positions
    """
    rel_d = np.linalg.norm(pos1 - pos2)
    return rel_d


def rel_vel(vel1: np.ndarray, vel2: np.ndarray) -> float:
    """The relative velocity between two velocities.

    Parameters
    ----------
    vel1 : np.ndarray
        the first velocity
    vel2 : np.ndarray
        the second velocity

    Returns
    -------
    float
        the relative velocity between the two velocities
    """
    rel_v = np.linalg.norm(vel1 - vel2)
    return rel_v


def delta_v(v: np.ndarray, prev_v: np.ndarray) -> np.ndarray:
    """The change in velocity

    Parameters
    ----------
    v : np.ndarray
        the current velocity
    prev_v : np.ndarray
        the previous velocity

    Returns
    -------
    float
        the change in velocity
    """
    v_norm = np.linalg.norm(v)
    prev_v_norm = np.linalg.norm(prev_v)
    return v_norm - prev_v_norm


def v_limit(
    chief_pos: np.ndarray,
    deputy_pos: np.ndarray,
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
    chief_pos : np.ndarray
        the position of the chief
    deputy_pos : np.ndarray
        the position of the deputy
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
    v_limit = v_max + (
        a * n * (rel_dist(pos1=chief_pos, pos2=deputy_pos) - docking_radius)
    )
    return v_limit


def closest_fft_distance(
    chief: sim.Target, deputy: sim.Inspector, n: float = 0.001027, time_step: int = 1
) -> float:
    """
    Get the closest Free Flight Trajectory (FFT) distance between the deputy
    and the chief over one orbit using closed form CWH dynamics to calculate
    future states

    Parameters
    ----------
    chief : sim.Target
        chief spacecraft under inspection
    deputy : sim.Inspector
        deputy spacecraft performing inspection
    n: float, optional
        orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    time_step: int
        time step in seconds to calculate FFT over, by default 1

    Returns
    -------
    float
        closest relative distance between deputy and chief during the FFT
    """

    def get_pos(platform: e.PhysicalEntity, t: int):
        x = (
            (4 - 3 * np.cos(n * t)) * platform.x
            + np.sin(n * t) * platform.x_dot / n
            + 2 / n * (1 - np.cos(n * t)) * platform.y_dot
        )
        y = (
            6 * (np.sin(n * t) - n * t) * platform.x
            + platform.y
            - 2 / n * (1 - np.cos(n * t)) * platform.x_dot
            + (4 * np.sin(n * t) - 3 * n * t) * platform.y_dot / n
        )
        z = platform.z * np.cos(n * t) + platform.z_dot / n * np.sin(n * t)
        return np.array([x, y, z])

    distances = []
    times = np.arange(0, 2 * np.pi / n, time_step)
    for time in times:
        dep_pos = get_pos(platform=deputy, t=time)
        chief_pos = get_pos(platform=chief, t=time)
        distances.append(np.linalg.norm(chief_pos - dep_pos))
    return float(min(distances))
