"""Reward functions for the inspection tasks"""

import numpy as np
import safe_autonomy_simulation.sims.inspection as sim
from scipy.spatial.transform import Rotation

from safe_autonomy_sims.pettingzoo.inspection.utils import delta_v, rel_dist


def observed_points_reward(chief: sim.Target, prev_num_inspected: int) -> float:
    """A dense reward which rewards the agent for inspecting
    new points during each step of the episode.

    $r_t = 0.01 * (p_t - p_{t-1})$

    where $p_t$ is the total number of inspected points at
    time $t$.

    Parameters
    ----------
    chief : Target
        chief spacecraft under inspection
    prev_num_inspected : int
        number of previously inspected points

    Returns
    -------
    float
        reward value
    """
    current_num_inspected = chief.inspection_points.get_num_points_inspected()
    step_inspected = current_num_inspected - prev_num_inspected
    r = 0.01 * step_inspected
    return r


def weighted_observed_points_reward(chief: sim.Target, prev_weight_inspected: float) -> float:
    """A dense reward which rewards the agent for inspecting
    new points during each step of the episode conditioned by
    individual point weights.

    $r_t = 1.0 * (w_t - w_{t-1})$

    where $w_t$ is the total weight of inspected points at
    time $t$.

    Parameters
    ----------
    chief : Target
        chief spacecraft under inspected
    prev_weight_inspected : float
        weight of previously inspected points

    Returns
    -------
    float
        reward value
    """
    current_weight_inspected = chief.inspection_points.get_total_weight_inspected()
    step_inspected = current_weight_inspected - prev_weight_inspected
    r = 1.0 * step_inspected
    return r


def inspection_success_reward(chief: sim.Target, total_points: int) -> float:
    """A sparse reward applied when the agent successfully
    inspects every point.

    $r_t = 1 if p_t == p_{total}, else 0$

    where $p_t$ is the number of inspected points at time
    $t$ and $p_{total}$ is the total number of points to be
    inspected.

    Parameters
    ----------
    chief : Target
        chief spacecraft under inspection
    total_points : int
        total number of points to be inspected

    Returns
    -------
    float
        reward value
    """
    num_inspected = chief.inspection_points.get_num_points_inspected()
    if num_inspected == total_points:
        r = 1.0
    else:
        r = 0.0
    return r


def weighted_inspection_success_reward(chief: sim.Target, total_weight: float):
    r"""A sparse reward applied when the agent successfully inspects
    point weights above the given threshold.

    $r_t = 1 if w_t \geq w_s else 0$

    where $w_t$ is the total weight of inspected points at time $t$
    and $w_s$ is the total weight of inspected points required for
    successful inspection.

    Parameters
    ----------
    chief : Target
        spacecraft under inspection
    total_weight : float
        inspected weight threshold for success

    Returns
    -------
    float
        reward value
    """
    weight_inspected = chief.inspection_points.get_total_weight_inspected()
    if weight_inspected >= total_weight:
        r = 1.0
    else:
        r = 0.0
    return r


def delta_v_reward(control: np.ndarray, m: float = 12.0, b: float = 0.0, scale: float = -0.01):
    """A dense reward based on the deputy's fuel
    use (change in velocity).

    $r_t = \scale * ((\deltav + b)$

    where
    # $\scale$ is the scalar of the reward
    * $\deltav$ is the total thrust divided by deputy's mass
    * $b$ is a tunable bias term

    Parameters
    ----------
    control : np.ndarray
        the control vector of the deputy's thrust outputs
    m : float, optional
        deputy mass, by default 12.0
    b : float, optional
        bias term, by default 0.0
    scale : float, optional
        scalar to modify the magnitude and sign of the reward

    Returns
    -------
    float
        reward value
    """
    dv = delta_v(control=control, m=m)
    r = scale * dv + b
    return r


def crash_reward(chief: sim.Target, deputy: sim.Inspector, crash_radius: float):
    """A sparse reward that punishes the agent
    for intersecting with the chief (crashing).

    $r_t = d_c < r$

    where $d_c$ is the distance between the deputy and
    the chief and $r$ is the radius of the crash region.

    Parameters
    ----------
    chief : Target
        chief spacecraft under inspection
    deputy : Inspector
        deputy spacecraft performing inspection
    crash_radius : float
        distance from chief which triggers a crash

    Returns
    -------
    float
        reward value
    """
    if rel_dist(pos1=chief.position, pos2=deputy.position) < crash_radius:
        r = -1.0
    else:
        r = 0
    return r


def facing_chief_reward(chief: sim.Target, deputy: sim.Inspector, epsilon: float):
    r"""A dense gaussian decaying reward which reward the agent
    for facing the chief.

    $r_t = 0.0005 * e^(-|\delta(f, 1)^2 / \espilon|)$

    where
    * $\delta(f, 1)$ is the difference between 1 and the dot
    product of the camera orientation and the relative position
    between the deputy and the chief
    * $\epsilon$ is the length of the exponential decay curve

    Parameters
    ----------
    chief : Target
        chief spacecraft under inspection
    deputy : Inspector
        deputy spacecraft performing inspection
    epsilon : float
        length of gaussian decay curve

    Returns
    -------
    float
        reward value
    """
    rel_pos = chief.position - deputy.position
    rel_pos = rel_pos / np.linalg.norm(rel_pos)
    gaussian_decay = np.exp(-np.abs(((np.dot(
        Rotation.from_quat(deputy.camera.orientation).as_euler("XYZ"),
        rel_pos,
    ) - 1)**2) / epsilon))
    reward = 0.0005 * gaussian_decay
    return reward


def live_timestep_reward(t: float, t_max: float):
    """A dense reward which rewards the agent for
    each timestep it remains active in the simulation.

    $r_t = 0.001 if t <= t_{max}$

    where $t$ is the current time step and $t_{max}$
    is the maximum allowable time for the episode.

    Parameters
    ----------
    t : float
        current time step
    t_max : float
        max time step allowed for episode

    Returns
    -------
    float
        reward value
    """
    reward = 0.0
    if t <= t_max:
        reward = 0.001
    return reward


def max_distance_reward(chief: sim.Target, deputy: sim.Inspector, max_distance: float, scale: float = -1.0):
    # TODO: docstring
    d = rel_dist(pos1=deputy.position, pos2=chief.position)
    reward = scale if d > max_distance else 0.0
    return reward