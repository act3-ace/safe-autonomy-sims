"""Reward functions for the inspection tasks"""

from safe_autonomy_sims.gymnasium.inspection.utils import delta_v, rel_dist, rel_vel


def observed_points_reward(state: dict, num_inspected: int) -> float:
    """A dense reward which rewards the agent for inspecting
    new points during each step of the episode.

    $r_t = 0.01 * (p_t - p_{t-1})$

    where $p_t$ is the total number of inspected points at
    time $t$.

    Parameters
    ----------
    state : dict
        current simulation state
    num_inspected : int
        number of previously inspected points

    Returns
    -------
    float
        reward value
    """
    current_num_inspected = state["inspection_points"].get_num_points_inspected()
    step_inspected = num_inspected - current_num_inspected
    r = 0.01 * step_inspected
    return r


def inspection_success_reward(state: dict, total_points: int) -> float:
    """A sparse reward applied when the agent successfully
    inspects every point.

    $r_t = 1 if p_t == p_{total}, else 0$

    where $p_t$ is the number of inspected points at time
    $t$ and $p_{total}$ is the total number of points to be
    inspected.

    Parameters
    ----------
    state : dict
        current simulation state
    total_points : int
        total number of points to be inspected

    Returns
    -------
    float
        reward value
    """
    num_inspected = state["inspection_points"].get_num_points_inspected()
    if num_inspected == total_points:
        r = 1.0
    else:
        r = 0.0
    return r


def delta_v_reward(state: dict, prev_state: dict, m: float = 12.0, b: float = 0.0):
    """A dense reward based on the deputy's fuel
    use (change in velocity).

    $r_t = -0.1((\deltav / m) + b)$

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
    r = -0.1 * ((delta_v(state, prev_state) / m) + b)
    return r


def crash_reward(state: dict, crash_radius: float):
    """A sparse reward that punishes the agent
    for intersecting with the chief (crashing).

    $r_t = d_c < r$

    where $d_c$ is the distance between the deputy and
    the chief and $r$ is the radius of the crash region.

    Parameters
    ----------
    state : dict
        current simulation state
    crash_radius : float
        distance from chief which triggers a crash

    Returns
    -------
    float
        reward value
    """
    if rel_dist(state=state) < crash_radius:
        r = -1.0
    else:
        r = 0
    return r
