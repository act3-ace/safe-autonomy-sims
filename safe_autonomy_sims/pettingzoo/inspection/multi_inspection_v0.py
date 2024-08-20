"""A gymnasium environment for training agents to dock a deputy spacecraft onto a chief spacecraft"""

import copy
import typing
import functools
import pettingzoo
import numpy as np
import gymnasium as gym
import safe_autonomy_simulation.sims.inspection as sim
import safe_autonomy_sims.pettingzoo.inspection.reward as r
import safe_autonomy_sims.pettingzoo.inspection.utils as utils


class MultiInspectionEnv(pettingzoo.ParallelEnv):
    r"""
    ## Description

    This environment represents a spacecraft inspection problem
    The goal is for a single deputy spacecraft to navigate around and inspect
    the entire surface of a chief spacecraft.

    The chief is covered in 100 inspection points that the agent must
    observe while they are illuminated by the moving sun. The optimal
    policy will inspect all 100 points within 2 revolutions of the sun
    while using as little fuel as possible.
    
    In this inspection environment, the agent only controls its translational
    motion and is always assumed to be pointing at the chief spacecraft.
    
    __Note: the policy selects a new action every 10 seconds__

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

    ## Observation Space

    At each timestep, the agent receives the observation, $o = [x, y, z, v_x, v_y, v_z, \theta_{sun}, n, x_{ups}, y_{ups}, z_{ups}]$, where:

    * $x, y,$ and $z$ represent the deputy's position in the Hill's frame,
    * $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity in the Hill's frame,
    * $\theta_{sun}$ is the angle of the sun,
    * $n$ is the number of points that have been inspected so far and,
    * $x_{ups}, y_{ups},$ and $z_{ups}$ are the unit vectors pointing to the nearest large cluster of unispected points (found via k-means)

    The observation space is a `Box` with the following bounds:

    | Index | Observation                                            | Min  | Max    | Type (units)   |
    |-------|--------------------------------------------------------|------|--------|----------------|
    | 0     | x position of the deputy in Hill's frame               | -inf | inf    | position (m)   |
    | 1     | y position of the deputy in Hill's frame               | -inf | inf    | position (m)   |
    | 2     | z position of the deputy in Hill's frame               | -inf | inf    | position (m)   |
    | 3     | x velocity of the deputy in Hill's frame               | -inf | inf    | velocity (m/s) |
    | 4     | y velocity of the deputy in Hill's frame               | -inf | inf    | velocity (m/s) |
    | 5     | z velocity of the deputy in Hill's frame               | -inf | inf    | velocity (m/s) |
    | 6     | angle of the sun                                       | 0    | $2\pi$ | angle (rad)    |
    | 7     | number of points inspected                             | 0    | 100    | count          |
    | 8     | x component of unit vector pointing to nearest cluster | -1   | 1      | scalar         |
    | 9     | y component of unit vector pointing to nearest cluster | -1   | 1      | scalar         |
    | 10    | z component of unit vector pointing to nearest cluster | -1   | 1      | scalar         |

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

    The reward $r_t$ at each time step is the sum of the following terms:

    * $r_t += 0.01(num\_inspected\_points_t - num\_inspected\_points_{t-1})$
        * a dense reward for observing new points
    * $r $= 1$ if $num\_inspected\_points_i == 100$, else 0
        * a sparse reward for successfully inspecting all points
    * $r += -1$ if $radius < crash\_region\_radius$, else 0
        * a sparse reward punishing the agent for crashing with the chief
    * $r += -0.1||\boldsymbol{u}||$
        * a dense reward for minimizing fuel cost
        * fuel cost is represented as the norm of the control input (action)

    ## Starting State

    At the start of an episode, the state is randomly initialized with the following conditions:

    * chief $(x,y,z)$ = $(0, 0, 0)$
    * chief radius = $10 m$
    * chief # of points = $100$
    * deputy position $(x, y, z)$ is converted after randomly selecting the position in polar notation $(r, \phi, \psi)$ using a uniform distribution with
        * $r \in [50, 100] m$
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $x = r \cos{\psi} \cos{\phi}$
        * $y = r \sin{\psi} \cos{\phi}$
        * $z = r \sin{\phi}$
    * deputy $(v_x, v_y, v_z)$ is converted after randomly selecting the velocity in polar notation $(r, \phi, \psi)$ using a Gaussian distribution with
        * $v \in [0, 0.8] * v_{\rm max}, \quad v_{\rm max} = 2nr$ m/s
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $v_x = v \cos{\psi} \cos{\phi}$
        * $v_y = v \sin{\psi} \cos{\phi}$
        * $v_z = v \sin{\phi}$
    * Initial sun angle is randomly selected using a uniform distribution
        * $\theta_{sun} \in [0, 2\pi] rad$

    ## Episode End

    An episode will end if any of the following conditions are met:

    * Termination: the agent exceeds a `max_distance = 800` meter radius away from the chief
    * Termination: the agent moves within a `crash_region_radius = 10` meter radius around the chief
    * Termination: all 100 points around the chief have been inspected
    * Truncation: the maximum number of timesteps, `max_timesteps = 1224`, is reached

    The episode is considered done and successful if and only if all 100 points have been inspected.

    ## References

    <a id="1">[1]</a>
    Clohessy, W., and Wiltshire, R., “Terminal Guidance System for Satellite Rendezvous,” *Journal of the Aerospace Sciences*, Vol. 27, No. 9, 1960, pp. 653–658.

    <a id="2">[2]</a>
    Dunlap, K., Mote, M., Delsing, K., and Hobbs, K. L., “Run Time Assured Reinforcement Learning for Safe Satellite
    Docking,” *Journal of Aerospace Information Systems*, Vol. 20, No. 1, 2023, pp. 25–36. [https://doi.org/10.2514/1.I011126](https://doi.org/10.2514/1.I011126).

    <a id="3">[3]</a>
    Gaudet, B., Linares, R., and Furfaro, R., “Adaptive Guidance and Integrated Navigation with Reinforcement Meta-Learning,”
    *CoRR*, Vol. abs/1904.09865, 2019. URL [http://arxiv.org/abs/1904.09865](http://arxiv.org/abs/1904.09865).

    <a id="4">[4]</a>
    Battin, R. H., “An introduction to the mathematics and methods of astrodynamics,” 1987.

    <a id="5">[5]</a>
    Campbell, T., Furfaro, R., Linares, R., and Gaylor, D., “A Deep Learning Approach For Optical Autonomous Planetary Relative
    Terrain Navigation,” 2017.

    <a id="6">[6]</a>
    Furfaro, R., Bloise, I., Orlandelli, M., Di Lizia, P., Topputo, F., and Linares, R., “Deep Learning for Autonomous Lunar
    Landing,” 2018.

    <a id="7">[7]</a>
    Lei, H. H., Shubert, M., Damron, N., Lang, K., and Phillips, S., “Deep reinforcement Learning for Multi-agent Autonomous
    Satellite Inspection,” *AAS Guidance Navigation and Control Conference*, 2022.

    <a id="8">[8]</a>
    Aurand, J., Lei, H., Cutlip, S., Lang, K., and Phillips, S., “Exposure-Based Multi-Agent Inspection of a Tumbling Target Using
    Deep Reinforcement Learning,” *AAS Guidance Navigation and Control Conference*, 2023.

    <a id="9">[9]</a>
    Brandonisio, A., Lavagna, M., and Guzzetti, D., “Reinforcement Learning for Uncooperative Space Objects Smart Imaging
    Path-Planning,” The *Journal of the Astronautical Sciences*, Vol. 68, No. 4, 2021, pp. 1145–1169. [https://doi.org/10.1007/s40295-021-00288-7](https://doi.org/10.1007/s40295-021-00288-7).
    """

    def __init__(
        self,
        num_agents: int = 2,
        success_threshold: float = 100,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 1000,
    ) -> None:
        self.possible_agents = [f"deputy_{i}" for i in range(num_agents)]

        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state = None
        self.prev_num_inspected = 0

    def reset(
        self, seed: int | None = None, options: dict[str, typing.Any] | None = None
    ) -> tuple[typing.Any, dict[str, typing.Any]]:
        self.rng = np.random.default_rng(seed)
        self.agents = copy.copy(self.possible_agents)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: self._get_info(a) for a in self.agents}
        self.prev_state = None
        self.prev_num_inspected = 0
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[typing.Any, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        self.prev_num_inspected = (
            self.chief.inspection_points.get_num_points_inspected()
        )

        # Update simulator state
        for a, action in actions.items():
            self.deputies[a].add_control(action)
        self.simulator.step()

        # Get info from simulator
        observations = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: self._get_reward(a) for a in self.agents}
        infos = {a: self._get_info(a) for a in self.agents}
        terminations = {a: self._get_terminated(a) for a in self.agents}
        truncations = {
            a: False for a in self.agents
        }  # used to signal episode ended unexpectedly

        # End episode if any agent is terminated or truncated
        if any(terminations.values() or any(truncations.values())):
            truncations = {a: True for a in self.agents}
            terminations = {a: True for a in self.agents}
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        self.chief = sim.Target(
            name="chief",
            num_points=100,
            radius=1,
        )
        self.deputies = {
            a: sim.Inspector(
                name=a,
                position=utils.polar_to_cartesian(
                    r=self.rng.uniform(50, 100),
                    theta=self.rng.uniform(0, 2 * np.pi),
                    phi=self.rng.uniform(-np.pi / 2, np.pi / 2),
                ),
                velocity=utils.polar_to_cartesian(
                    r=self.rng.uniform(0, 0.8) * 2 * 0.001027 * 1,  # max = 0.8 * 2nr
                    theta=self.rng.uniform(0, 2 * np.pi),
                    phi=self.rng.uniform(-np.pi / 2, np.pi / 2),
                ),
            )
            for a in self.agents
        }
        self.sun = sim.Sun(theta=self.rng.uniform(0, 2 * np.pi))
        self.simulator = sim.InspectionSimulator(
            frame_rate=1,
            inspectors=list(self.deputies.values()),
            targets=[self.chief],
            sun=self.sun,
        )

    def _get_obs(self, agent: str) -> np.ndarray:
        deputy = self.deputies[agent]
        obs = self.observation_space(agent).sample()
        obs[:3] = deputy.position
        obs[3:6] = deputy.velocity
        obs[6] = self.sun.theta
        obs[7] = self.chief.inspection_points.get_num_points_inspected()
        obs[8:11] = self.chief.inspection_points.kmeans_find_nearest_cluster(
            camera=deputy.camera, sun=self.sun
        )
        return obs

    def _get_info(self, agent):
        return {}

    def _get_reward(self, agent):
        reward = 0
        deputy = self.deputies[agent]

        # Dense rewards
        reward += r.observed_points_reward(
            chief=self.chief, prev_num_inspected=self.prev_num_inspected
        )
        reward += r.delta_v_reward(
            v=deputy.velocity, prev_v=self.prev_state[agent][3:6]
        )

        # Sparse rewards
        reward += r.inspection_success_reward(
            chief=self.chief,
            total_points=self.success_threshold,
        )
        reward += r.crash_reward(
            chief=self.chief,
            deputy=deputy,
            crash_radius=self.crash_radius,
        )
        return reward

    def _get_terminated(self, agent):
        deputy = self.deputies[agent]

        # Get state info
        d = utils.rel_dist(pos1=self.chief.position, pos2=deputy.position)

        # Determine if in terminal state
        oob = d > self.max_distance
        crash = d < self.crash_radius
        timeout = self.simulator.sim_time > self.max_time
        all_inspected = (
            self.chief.inspection_points.get_num_points_inspected()
            == self.success_threshold
        )

        return oob or crash or timeout or all_inspected

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: typing.Any) -> gym.Space:
        return gym.spaces.Box(
            np.concatenate(
                (
                    [-np.inf] * 3,  # position
                    [-np.inf] * 3,  # velocity
                    [0],  # sun angle
                    [0],  # num inspected
                    [-1] * 3,  # nearest cluster
                )
            ),
            np.concatenate(
                (
                    [np.inf] * 3,  # position
                    [np.inf] * 3,  # velocity
                    [2 * np.pi],  # sun angle
                    [100],  # num inspected
                    [1] * 3,  # nearest cluster
                )
            ),
            shape=(11,),
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: typing.Any) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(3,))

    @property
    def sim_state(self) -> dict:
        state = {
            "chief": self.chief.state,
        }
        for a in self.agents:
            state[a] = self.deputies[a].state
        return state
