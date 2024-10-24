"""Module for the V0 version of the Weighted Multiagent Inspection environment"""
import copy
import functools
import typing

import gymnasium as gym
import numpy as np
import pettingzoo
import safe_autonomy_simulation.sims.inspection as sim

import safe_autonomy_sims.pettingzoo.inspection.reward as r
from safe_autonomy_sims.gym.inspection.utils import closest_fft_distance, polar_to_cartesian, rel_dist


class WeightedMultiInspectionEnv(pettingzoo.ParallelEnv):
    # pylint:disable=C0301
    r"""
    In this weighted inspection environment, the goal is for a single deputy spacecraft
    to navigate around and inspect the entire surface of a chief spacecraft.

    The chief is covered in 100 inspection points that the agent must observe
    while they are illuminated by the moving sun. The points are weighted by
    priority, such that it is more important to inspect some points than others.
    A unit vector is used to indicate the direction of highest importance, where
    points are weighted based on their angular distance to this vector. All
    point weights add up to a value of one. The optimal policy will inspect
    points whose cumulative weight exceeds 0.95 within 2 revolutions of the sun
    while using as little fuel as possible.

    In this weighted inspection environment, the agent only controls its
    translational motion and is always assumed to be pointing at the chief spacecraft.

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

    At each timestep, the agent receives the observation,
    $o = [x, y, z, v_x, v_y, v_z, \theta_{sun}, n, x_{ups}, y_{ups}, z_{ups}, x_{pv}, y_{pv}, z_{pv}, w_{points}]$, where:

    * $x, y,$ and $z$ represent the deputy's position in the Hill's frame,
    * $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity in the Hill's frame,
    * $\theta_{sun}$ is the angle of the sun,
    * $n$ is the number of points that have been inspected so far
    * $x_{ups}, y_{ups},$ and $z_{ups}$ are the unit vector elements pointing to the nearest large cluster of unispected points
    * $x_{pv}, y_{pv},$ and $z_{pv}$ are the unit vector elements pointing to the priority vector indicating point priority
    * $w_{points}$ is the cumulative weight of inpsected points

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
    | 11    | x component of unit vector pointing to priority vector | -1   | 1      | scalar         |
    | 12    | y component of unit vector pointing to priority vector | -1   | 1      | scalar         |
    | 13    | z component of unit vector pointing to priority vector | -1   | 1      | scalar         |
    | 14    | cumulative weight of inspected points                  | 0    | 1      | scalar         |

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

    * $r_t += 1.0(weight\_inspected\_points_t - weight\_inspected\_points_{t-1})$
        * a dense reward for observing inspection points
    * $r += 1$ if $weight\_inspected\_points_i \geq 0.95$ and $FFT_radius \geq crash\_region\_radius$, $r = -1$ if $weight\_inspected\_points_i \geq 0.95$ and $FFT_radius < crash\_region\_radius$, else 0
        * a sparse reward for successfully inspecting the chief spacecraft
        * positive reward if the agent inspects 95% of the points and is on a safe trajectory
        * negative reward if the agent inspects 95% of the points and is on a collision course with the chief spacecraft
    * $r += -1$ if $radius < crash\_region\_radius$, else 0
        * a sparse reward punishing the agent for crashing with the chief spacecraft
    * $r += -0.1||\boldsymbol{u}||$
        * a dense reward for minimizing fuel usage
        * fuel use is represented by the norm of the control input (action)

    ## Starting State

    At the start of an episode, the state is randomly initialized with the following conditions:

    * chief $(x,y,z)$ = $(0, 0, 0)$
    * chief radius = $10 m$
    * chief # of points = $100$
    * priority unit vector orientation for point weighting is randomly sampled from a uniform distribution using polar notation $(\phi, \psi)$
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
    * deputy position $(x, y, z)$ is converted after randomly selecting the position in polar notation $(r, \phi, \psi)$ using a uniform distribution with
        * $r \in [50, 100] m$
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $x = r \cos{\psi} \cos{\phi}$
        * $y = r \sin{\psi} \cos{\phi}$
        * $z = r \sin{\phi}$
    * deputy $(v_x, v_y, v_z)$ is converted after randomly selecting the velocity in polar notation $(r, \phi, \psi)$ using a Gaussian distribution with
        * $v \in [0, 0.3]$ m/s
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $v_x = v \cos{\psi} \cos{\phi}$
        * $v_y = v \sin{\psi} \cos{\phi}$
        * $v_z = v \sin{\phi}$
    * Initial sun angle is randomly selected using a uniform distribution
        * $\theta_{sun} \in [0, 2\pi] rad$
        * If the deputy is initialized where it's sensor points within 60 degrees of the sun, its position is negated such that the sensor points away from the sun.

    ## Episode End

    An episode will end if any of the following conditions are met:

    * Terminated: the agent exceeds a `max_distance = 800` meter radius away from the chief
    * Terminated: the agent moves within a `crash_region_radius = 10` meter radius around the chief
    * Terminated: the cumulative weight of inspected points exceeds 0.95
    * Truncated: the maximum number of timesteps, `max_timesteps = 1224`

    The episode is considered done and successful if and only if the cumulative weight of inspected points exceeds 0.95.

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
    """  # noqa:E501

    # pylint:enable=C0301

    def __init__(
        self,
        num_agents: int = 2,
        success_threshold: float = 0.95,
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
        # For some reason, mypy complains about prev_state possibly being None in the multi inspection environments
        # but not single agent.
        self.prev_state: dict[typing.Any, typing.Any] = {}
        self.prev_num_inspected = 0
        self.prev_weight_inspected = 0.0

        # Lazy initialized items
        self.rng: np.random.Generator
        self.chief: sim.SixDOFTarget
        self.deputies: dict[str, sim.SixDOFInspector]
        self.sun: sim.Sun
        self.simulator: sim.InspectionSimulator

    def reset(self, seed: int | None = None, options: dict[str, typing.Any] | None = None) -> tuple[typing.Any, dict[str, typing.Any]]:
        self.agents = copy.copy(self.possible_agents)
        self.rng = np.random.default_rng(seed)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: self._get_info(a) for a in self.agents}
        self.prev_state = {}
        self.prev_num_inspected = 0
        self.prev_weight_inspected = 0.0
        return observations, infos

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        priority_vector = self.rng.uniform(-1, 1, size=3)
        priority_vector /= np.linalg.norm(priority_vector)  # convert to unit vector
        self.chief = sim.Target(
            name="chief",
            num_points=100,
            radius=1,
            priority_vector=priority_vector,
        )
        self.deputies = {
            a:
            sim.Inspector(
                name=a,
                position=polar_to_cartesian(
                    r=self.rng.uniform(50, 100),
                    phi=self.rng.uniform(-np.pi / 2, np.pi / 2),
                    theta=self.rng.uniform(0, 2 * np.pi),
                ),
                velocity=polar_to_cartesian(
                    r=self.rng.uniform(0, 0.3),
                    phi=self.rng.uniform(-np.pi / 2, np.pi / 2),
                    theta=self.rng.uniform(0, 2 * np.pi),
                ),
            )
            for a in self.agents
        }
        self.sun = sim.Sun(theta=self.rng.uniform(0, 2 * np.pi))
        self.simulator = sim.InspectionSimulator(
            frame_rate=10,
            inspectors=list(self.deputies.values()),
            targets=[self.chief],
            sun=self.sun,
        )

    def step(
        self, actions: dict[str, typing.Any]
    ) -> tuple[dict[str, typing.Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        self.prev_num_inspected = (self.chief.inspection_points.get_num_points_inspected())
        self.prev_weight_inspected = (self.chief.inspection_points.get_total_weight_inspected())

        # Update simulator state
        for agent, action in actions.items():
            self.deputies[agent].add_control(action)
        self.simulator.step()

        # Get info from simulator
        observations = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: self._get_reward(a) for a in self.agents}
        infos = {a: self._get_info(a) for a in self.agents}
        terminations = {a: self._get_terminated(a) for a in self.agents}
        truncations = {a: False for a in self.agents}  # used to signal episode ended unexpectedly

        # End episode if any agent is terminated or truncated
        if any(terminations.values()) or any(truncations.values()):
            truncations = {a: True for a in self.agents}
            terminations = {a: True for a in self.agents}
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent: typing.Any) -> np.ndarray:
        deputy = self.deputies[agent]
        obs = self.observation_space(agent).sample()
        obs[:3] = deputy.position
        obs[3:6] = deputy.velocity
        obs[6] = self.sun.theta
        obs[7] = self.chief.inspection_points.get_num_points_inspected()
        obs[8:11] = self.chief.inspection_points.kmeans_find_nearest_cluster(camera=deputy.camera, sun=self.sun)
        obs[11:14] = self.chief.inspection_points.priority_vector
        obs[14] = self.chief.inspection_points.get_total_weight_inspected()
        return obs

    def _get_info(self, agent: typing.Any) -> dict:
        return {agent: None}

    def _get_reward(self, agent: typing.Any) -> float:
        reward = 0.0
        deputy = self.deputies[agent]

        # Dense rewards
        reward += r.weighted_observed_points_reward(chief=self.chief, prev_weight_inspected=self.prev_weight_inspected)
        reward += r.delta_v_reward(
            v=deputy.velocity,
            prev_v=self.prev_state[agent][3:6],
        )

        # Sparse rewards
        success_reward = r.weighted_inspection_success_reward(chief=self.chief, total_weight=self.success_threshold)
        if (success_reward > 0 and closest_fft_distance(chief=self.chief, deputy=deputy) < self.crash_radius):
            success_reward = -1.0
        reward += success_reward
        reward += r.crash_reward(chief=self.chief, deputy=deputy, crash_radius=self.crash_radius)

        return reward

    def _get_terminated(self, agent: typing.Any) -> bool:
        deputy = self.deputies[agent]

        # Get state info
        d = rel_dist(pos1=self.chief.position, pos2=deputy.position)

        # Determine if in terminal state
        oob = d > self.max_distance
        crash = d < self.crash_radius
        timeout = self.simulator.sim_time > self.max_time
        all_inspected = self.chief.inspection_points.get_total_weight_inspected() >= self.success_threshold

        return oob or crash or timeout or all_inspected

    # Pylint warns that self will never be garbage collected due to the use of the lru_cache, but the environment
    # should never be garabage collected when used
    @functools.lru_cache(maxsize=None)  # pylint:disable=W1518
    def observation_space(self, agent: typing.Any) -> gym.Space:
        return gym.spaces.Box(
            np.concatenate(
                (
                    [-np.inf] * 3,  # position
                    [-np.inf] * 3,  # velocity
                    [0],  # sun angle
                    [0],  # num inspected
                    [-1] * 3,  # nearest cluster unit vector
                    [-1] * 3,  # priority vector unit vector
                    [0],  # weight inspected
                )
            ),
            np.concatenate(
                (
                    [np.inf] * 3,  # position
                    [np.inf] * 3,  # velocity
                    [2 * np.pi],  # sun angle
                    [100],  # num inspected
                    [1] * 3,  # nearest cluster unit vector
                    [1] * 3,  # priority vector unit vector
                    [1],  # weight inspected
                )
            ),
            shape=(15, ),
        )

    # Pylint warns that self will never be garbage collected due to the use of the lru_cache, but the environment
    # should never be garabage collected when used
    @functools.lru_cache(maxsize=None)  # pylint:disable=W1518
    def action_space(self, agent: typing.Any) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(3, ))

    @property
    def sim_state(self) -> dict[str, np.ndarray]:
        """Provides the state of the simulator

        Returns
        -------
        dict
            A dictionary containing the state of the deputies and the state of the chief
        """
        state = {
            "chief": self.chief.state,
        }
        for a in self.agents:
            state[a] = self.deputies[a].state
        return state

    def render(self) -> None | np.ndarray | str | list:
        raise NotImplementedError

    def state(self) -> np.ndarray:
        agent_states = [deputy.state for deputy in list(self.deputies.values())]
        return np.vstack((self.chief.state, agent_states))
