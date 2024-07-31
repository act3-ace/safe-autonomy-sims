"""A gymnasium environment for training agents to dock a deputy spacecraft onto a chief spacecraft"""

from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
import safe_autonomy_simulation
from gymnasium import spaces
import safe_autonomy_sims.gym.docking.reward as r
from safe_autonomy_sims.gym.docking.utils import v_limit, rel_dist, rel_vel, polar_to_cartesian


class DockingEnv(gym.Env):
    r"""
    ## Description

    This environment is a spacecraft docking problem.
    The goal is for a single deputy spacecraft to navigate
    towards and dock onto a chief spacecraft.

    ## Action Space

    Actions are thruster force values for each of the 3 bi-directional thrusters
    on the x-, y-, and z-axis of the deputy spacecraft's body frame.

    The action space is a `Box(-1, 1, shape=(3,))` where each value in the array
    represents the force applied to the x-, y-, and z-axis of the deputy spacecraft.

    | Index | Action        | Control Min | Control Max | Type (units) |
    |-------|---------------|-------------|-------------|--------------|
    | 0     | x-axis thrust | -1          | 1           | Force (N)    |
    | 1     | y-axis thrust | -1          | 1           | Force (N)    |
    | 2     | z-axis thrust | -1          | 1           | Force (N)    |

    ### Observation Space

    At each timestep, the agent receives the observation,
    $o = [x, y, z, v_x, v_y, v_z, s, v_{limit}]$, where:

    * $x, y,$ and $z$ represent the deputy's position in the Hill's frame,
    * $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity in the Hill's frame,
    * $s$ is the speed of the deputy,
    * $v_{limit}$ is the safe velocity limit given by: $v_{max} + an(d_{chief} - r_{docking})$
        * $v_{max}$ is the maximum allowable velocity of the deputy within the docking region
        * $a$ is the slope of the linear velocity limit as a function of distance from the docking region
        * $n$ is the mean motion constant
        * $d_{chief}$ is the deputy's distance from the chief
        * $r_{docking}$ is the radius of the docking region

    The observation space is a `Box([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0], np.inf, shape=(8,))`.

    | Index | Observation                              | Min     | Max    | Type (units)   |
    |-------|------------------------------------------|---------|--------|----------------|
    | 0     | x position of the deputy in Hill's frame | -np.inf | np.inf | position (m)   |
    | 1     | y position of the deputy in Hill's frame | -np.inf | np.inf | position (m)   |
    | 2     | z position of the deputy in Hill's frame | -np.inf | np.inf | position (m)   |
    | 3     | x velocity of the deputy                 | -np.inf | np.inf | velocity (m/s) |
    | 4     | y velocity of the deputy                 | -np.inf | np.inf | velocity (m/s) |
    | 5     | z velocity of the deputy                 | -np.inf | np.inf | velocity (m/s) |
    | 6     | speed of the deputy                      | -np.inf | np.inf | speed (m/s)    |
    | 7     | safe velocity limit                      | -np.inf | np.inf | velocity (m/s) |


    ## State Transition Dynamics

    The relative motion between the deputy and chief is represented by
    linearized Clohessy-Wiltshire equations [[1]](#1), given by

    $$
        \dot{\boldsymbol{x}} = A {\boldsymbol{x}} + B\boldsymbol{u},
    $$

    where:
    * state $\boldsymbol{x}=[x,y,z,\dot{x},\dot{y},\dot{z}]^T \in \mathcal{X}=\mathbb{R}^6$
    * control $\boldsymbol{u}= [F_x,F_y,F_z]^T \in \mathcal{U} = [-1N, 1N]^3$

    * $$
        A =
    \begin{bmatrix}
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    3n^2 & 0 & 0 & 0 & 2n & 0 \\
    0 & 0 & 0 & -2n & 0 & 0 \\
    0 & 0 & -n^2 & 0 & 0 & 0 \\
    \end{bmatrix},
        B =
    \begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \frac{1}{m} & 0 & 0 \\
    0 & \frac{1}{m} & 0 \\
    0 & 0 & \frac{1}{m} \\
    \end{bmatrix},
    $$

    * mean motion constant $n = 0.001027 rad/s$

    ## Rewards

    The reward is the sum of the following terms:

    * $r_t += c(e^{-ad_t} - e^{-ad_{t-1}})$
        * this is a dense reward for approaching the chief
        * $c$ is a scale factor
        * $a$ is the exponential coefficent (can be calculated from a pivot value)
        * $d_t$ is the distance from the chief at time $t$
    * $r_t += -((\delta{v} / m) + b)$
        * this is a dense reward for minimizing fuel usage
        * $\delta{v}$ is the change in velocity
        * $m$ is the mass of the deputy
        * $b$ is a tunable bia term
    * $r_t += v - v_{limit}$ if $v_{limit} < v$, else 0
        * this is a dense reward that punishes velocity constraint violations
        * $v_{limit}$ is the safe velocity limit given by: $v_{max} + an(d_{chief} - r_{docking})$, described above
    * $r += 1 + (1 - (t/t_{max}))$ if docking is successful, else 0
        * this is a sparse reward for minimizing the time required to dock onto the chief
        * $t$ is the current time
        * $t_{max}$ is the maximum episode length before timeout
    * $r += -1.0$ if the agent times out, crashes, or goes out of bounds, else 0
        * this is a sparse reward that punishes the agent for failing to dock onto the chief

    ## Starting State

    At the start of any episode, the state is randomly initialized with the following conditions:

    * chief $(x,y,z)$ = $(0, 0, 0)$
    * docking radius = $0.5 m$
    * deputy position $(x, y, z)$ is converted after randomly selecting the position in polar notation $(r, \phi, \psi)$ using a uniform distribution with
        * $r \in [100, 150] m$
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $x = r \cos{\psi} \cos{\phi}$
        * $y = r \sin{\psi} \cos{\phi}$
        * $z = r \sin{\phi}$
    * deputy $(v_x, v_y, v_z)$ is converted after randomly selecting the velocity in polar notation $(v, \phi, \psi)$ using a Gaussian distribution with
        * $v \in [0, 0.8]$ m/s
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $v_x = v \cos{\psi} \cos{\phi}$
        * $v_y = v \sin{\psi} \cos{\phi}$
        * $v_z = v \sin{\phi}$

    ## Episode End

    An episode will end if any of the following conditions are met:

    * Termination: the agent enters the docking region at a safe velocity (docked),
    * Termination: the agent violates the velocity constraint within the docking region (crash),
    * Truncation: the agent exceeds a `max_distance = 10000` meter radius away from the chief,
    * Truncation: the velocity limit penalty exceeds -5
    * Truncation: the maximum number of timesteps, `max_timesteps = 2000`, is reached

    The episode is considered done and successful if and only if the agent maneuvers the deputy
    within the docking region while maintaining a safe velocity.


    ## References

    <a id="1">[1]</a>
    Clohessy, W., and Wiltshire, R., “Terminal Guidance System for Satellite Rendezvous,”
    *Journal of the Aerospace Sciences*, Vol. 27, No. 9, 1960, pp. 653–658.
    """

    def __init__(
        self,
        docking_radius: float = 0.2,
        max_time: int = 2000,
        max_distance: float = 10000,
        max_v_violation: int = 5,
    ) -> None:
        # Each spacecraft obs = [x, y, z, v_x, v_y, v_z, s, v_limit]
        self.observation_space = spaces.Box(
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0]),
            np.inf,
            shape=(8,),
        )

        self.action_space = spaces.Box(-1, 1, shape=(3,))

        # Environment parameters
        self.docking_radius = docking_radius
        self.max_time = max_time
        self.max_distance = max_distance
        self.max_v_violation = max_v_violation

        # Episode level information
        self.prev_state = None
        self.episode_v_violations = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        obs, info = self._get_obs(), self._get_info()
        self.prev_state = None
        self.episode_v_violations = 0
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Store previous simulator state
        self.prev_state = self.sim_state.copy()

        # Update simulator state
        self.deputy.add_control(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        return observation, reward, terminated, truncated, info

    def _init_sim(self):
        # Initialize simulator with chief and deputy spacecraft
        self.chief = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
            name="chief"
        )
        self.deputy = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
            name="deputy",
            position=polar_to_cartesian(
                r=self.np_random.uniform(100, 150),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
                theta=self.np_random.uniform(0, 2 * np.pi),
            ),
            velocity=polar_to_cartesian(
                r=self.np_random.uniform(0, 0.8),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
                theta=self.np_random.uniform(0, 2 * np.pi),
            )
        )
        self.simulator = safe_autonomy_simulation.Simulator(
            frame_rate=1, entities=[self.chief, self.deputy]
        )

    def _get_obs(self):
        v_lim = v_limit(state=self.sim_state)
        s = np.linalg.norm(self.deputy.velocity)
        obs = self.observation_space.sample()
        obs[:3] = self.deputy.position
        obs[3:6] = self.deputy.velocity
        obs[6] = s
        obs[7] = v_lim
        return obs

    def _get_info(self):
        return {}

    def _get_reward(self):
        reward = 0

        # Dense rewards
        reward += r.distance_pivot_reward(
            state=self.sim_state, prev_state=self.prev_state
        )
        reward += r.delta_v_reward(state=self.sim_state, prev_state=self.prev_state)
        reward += r.velocity_constraint_reward(
            state=self.sim_state, v_limit=v_limit(self.sim_state)
        )

        # Sparse rewards
        reward += r.docking_success_reward(
            state=self.sim_state,
            t=self.simulator.sim_time,
            vel_limit=v_limit(state=self.sim_state),
            docking_radius=self.docking_radius,
            max_time=self.max_time,
        )
        reward += r.timeout_reward(t=self.simulator.sim_time, max_time=self.max_time)
        reward += r.crash_reward(
            state=self.sim_state,
            vel_limit=v_limit(state=self.sim_state),
            docking_radius=self.docking_radius,
        )
        reward += r.out_of_bounds_reward(
            state=self.sim_state, max_distance=self.max_distance
        )
        return reward

    def _get_terminated(self):
        # Get state info
        d = rel_dist(state=self.sim_state)
        v = rel_vel(state=self.sim_state)
        vel_limit = v_limit(state=self.sim_state)
        in_docking = d < self.docking_radius
        safe_v = v < vel_limit
        self.episode_v_violations += -r.velocity_constraint_reward(
            state=self.sim_state, v_limit=vel_limit
        )

        # Determine if in terminal state
        crash = in_docking and not safe_v
        docked = in_docking and safe_v

        return crash or docked

    def _get_truncated(self):
        max_time = self.simulator.sim_time > self.max_time
        oob = rel_dist(state=self.sim_state) > self.max_distance
        max_v_violation = self.episode_v_violations > self.max_v_violation
        return max_time or oob or max_v_violation

    @property
    def sim_state(self) -> dict:
        return {"chief": self.chief.state, "deputy": self.deputy.state}
