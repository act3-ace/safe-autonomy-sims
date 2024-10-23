"""Module for the V0 version of the Weighted 6DOF Multiagent Inspection environment"""
import copy
import functools
import typing

import gymnasium as gym
import numpy as np
import pettingzoo
import safe_autonomy_simulation.sims.inspection as sim
from scipy.spatial.transform import Rotation

import safe_autonomy_sims.pettingzoo.inspection.reward as r
from safe_autonomy_sims.gym.inspection.utils import closest_fft_distance, polar_to_cartesian, rel_dist


class WeightedSixDofMultiInspectionEnv(pettingzoo.ParallelEnv):
    # pylint:disable=C0301
    r"""
    ## Environment

    In this weighted six dof inspection environment, the goal is for a single deputy spacecraft
    to navigate around and inspect the entire surface of a chief spacecraft.

    The chief is covered in 100 inspection points that the agent must observe
    while they are illuminated by the moving sun. The points are weighted by
    priority, such that it is more important to inspect some points than others.
    A unit vector is used to indicate the direction of highest importance, where
    points are weighted based on their angular distance to this vector. All
    point weights add up to a value of one. The optimal policy will inspect
    points whose cumulative weight exceeds 0.95 within 2 revolutions of the sun
    while using as little fuel as possible.

    In this six DOF inspection environment, the agent controls its translational
    and rotational movement, requiring it to orient itself towards the chief for
    inspection.

    __Note: the policy selects a new action every 10 seconds__

    ## Action Space

    Actions are thruster force values for each of the 3 bi-directional thrusters
    on the x-, y-, and z-axis of the deputy spacecraft's body frame and the
    reaction wheel abstraction controlling the deputy's x, y, and z moments with
    scalar values. These controls are able to move and rotate the spacecraft in
    any direction.

    The action space is a `Box` with the following bounds:

    | Index | Action        | Control Min | Control Max | Type (units) |
    |-------|---------------|-------------|-------------|--------------|
    | 0     | x-axis thrust | -1          | 1           | Force (N)    |
    | 1     | y-axis thrust | -1          | 1           | Force (N)    |
    | 2     | z-axis thrust | -1          | 1           | Force (N)    |
    | 3     | x-axis moment | -0.001      | 0.001       | Moment (Nm)  |
    | 4     | y-axis moment | -0.001      | 0.001       | Moment (Nm)  |
    | 5     | z-axis moment | -0.001      | 0.001       | Moment (Nm)  |

    ## Observation Space

    At each timestep, the agent receives the observation,
    $o = [x, y, z, |pos|, |x|, |y|, |z|, v_x, v_y, v_z, |v|, |v_x|, |v_y|, |v_z|, \omega_{x}, \omega_{y}, \omega_{z}, \theta_{x}, \theta_{y}, \theta_{z}, f, \theta_{sun}, n, x_{ups}, y_{ups}, z_{ups}, x_{pv}, y_{pv}, z_{pv}, w_{points}, p_o]$, where:

    * $x, y,$ and $z$ represent the deputy's position relative to the chief,
    * $|pos|, |x|, |y|, |z|$ is the magnorm representation of the deputy's position relative to the chief
    * $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity relative to the chief,
    * $|v|, |v_x|, |v_y|, |v_z|$ is the magnorm representation of the deputy's velocity relative to the chief
    * $\omega_x, \omega_y, \omega_z$ are the components of the deputy's angular velocity
    * $\theta_{cam}$ is the camera's orientation in Hill's frame
    * $\theta_{x}, \theta_{y}, \theta_{z}$ are the deputy axis coordinates in Hill's frame
    * $f$ is the dot-product between the camera orientation vector and the relative position between the deputy and the chief. This value is 1 when the camera is pointing at the chief.
    * $\theta_{sun}$ is the angle of the sun,
    * $n$ is the number of points that have been inspected so far and,
    * $x_{ups}, y_{ups},$ and $z_{ups}$ are the unit vector elements pointing to the nearest large cluster of unispected points as determined by the *Uninspected Points Sensor*.
    * $x_{pv}, y_{pv},$ and $z_{pv}$ are the unit vector elements pointing to the priority vector indicating point priority.
    * $w_{points}$ is the cumulative weight of inpsected points
    * $p_o$ is the dot-product between the uninspected points cluster given by the Uninspected Points Sensor and the deputy's position. This signals if the uninspected points are occluded for if the camera is facing the points but they are not being inspected, there is occlusion.

    The observation space is a `Box` with the following bounds:

    | Index | Observation                                                 | Min | Max | Type (units) |
    |-------|-------------------------------------------------------------|-----|-----|--------------|
    | 0     | x position of the deputy in Hill's frame                    | -inf| inf | Position (m) |
    | 1     | y position of the deputy in Hill's frame                    | -inf| inf | Position (m) |
    | 2     | z position of the deputy in Hill's frame                    | -inf| inf | Position (m) |
    | 3     | |pos| magnitude of the deputy's position                    | -inf| inf | Position (m) |
    | 4     | |x| magnitude of the deputy's x position                    | -inf| inf | Position (m) |
    | 5     | |y| magnitude of the deputy's y position                    | -inf| inf | Position (m) |
    | 6     | |z| magnitude of the deputy's z position                    | -inf| inf | Position (m) |
    | 7     | x component of the deputy's velocity                        | -inf| inf | Velocity (m/s) |
    | 8     | y component of the deputy's velocity                        | -inf| inf | Velocity (m/s) |
    | 9     | z component of the deputy's velocity                        | -inf| inf | Velocity (m/s) |
    | 10    | |v| magnitude of the deputy's velocity                      | -inf| inf | Velocity (m/s) |
    | 11    | |v_x| magnitude of the x component of the deputy's velocity | -inf| inf | Velocity (m/s) |
    | 12    | |v_y| magnitude of the y component of the deputy's velocity | -inf| inf | Velocity (m/s) |
    | 13    | |v_z| magnitude of the z component of the deputy's velocity | -inf| inf | Velocity (m/s) |
    | 14    | x component of the deputy's angular velocity                | -inf| inf | Angular Velocity (rad/s) |
    | 15    | y component of the deputy's angular velocity                | -inf| inf | Angular Velocity (rad/s) |
    | 16    | z component of the deputy's angular velocity                | -inf| inf | Angular Velocity (rad/s) |
    | 17    | x component of the deputy's orientation                     | 0   | 2pi | Orientation (rad) |
    | 18    | y component of the deputy's orientation                     | 0   | 2pi | Orientation (rad) |
    | 19    | z component of the deputy's orientation                     | 0   | 2pi | Orientation (rad) |
    | 20    | facing chief dot product                                    | -1  | 1   | Scalar |
    | 21    | sun angle                                                   | 0   | 2pi | Angle (rad) |
    | 22    | number of inspected points                                  | 0   | 100 | Scalar |
    | 23    | x component of unit vector pointing to the nearest cluster  | -1  | 1   | Scalar |
    | 24    | y component of unit vector pointing to the nearest cluster  | -1  | 1   | Scalar |
    | 25    | z component of unit vector pointing to the nearest cluster  | -1  | 1   | Scalar |
    | 26    | x component of unit vector pointing to the priority vector  | -1  | 1   | Scalar |
    | 27    | y component of unit vector pointing to the priority vector  | -1  | 1   | Scalar |
    | 28    | z component of unit vector pointing to the priority vector  | -1  | 1   | Scalar |
    | 29    | cumulative weight of inspected points                       | 0   | 1   | Scalar |
    | 30    | dot product between nearest cluster and deputy's position   | -1  | 1   | Scalar |

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

    The body frame rotational motion state transition of each spacecraft
    given its quaternion orientation and angular velocity
    $[q_1, q_2, q_3, q_4, \omega_x, \omega_y, \omega_z]$ is defined by

    $$
    \begin{bmatrix}
    \dot{q_1} \\
    \dot{q_2} \\
    \dot{q_3} \\
    \dot{q_4} \\
    \dot{\omega_x} \\
    \dot{\omega_y} \\
    \dot{\omega_z} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    \frac{1}{2}(q_4\omega_x - q_3\omega_y + q_2\omega_z) \\
    \frac{1}{2}(q_3\omega_x + q_4\omega_y - q_1\omega_z) \\
    \frac{1}{2}(-q_2\omega_x + q_1\omega_y + q_4\omega_z) \\
    \frac{1}{2}(-q_1\omega_x - q_2\omega_y - q_3\omega_z) \\
    J_1^{-1}((J_2 - J_3)\omega_y\omega_z) \\
    J_2^{-1}((J_3 - J_1)\omega_x\omega_z) \\
    J_3^{-1}((J_1 - J_2)\omega_x\omega_y) \\
    \end{bmatrix}
    $$

    where

    $$
    J =
    \begin{bmatrix}
    0.0573 & 0.0 & 0.0 \\
    0.0 & 0.0573 & 0.0 \\
    0.0 & 0.0 & 0.0573
    \end{bmatrix}
    $$

    is an inertial matrix.

    ## Rewards

    The reward $r_t$ at each time step is the sum of the following terms:
    * $r_t += 1.0(weight\_inspected\_points_t - weight\_inspected\_points_{t-1})$.
        * a dense reward for inspecting new points
    * $r_t += 1$ if $weight\_inspected\_points_i \geq 0.95$ and $FFT_radius \geq crash\_region\_radius$, $r = -1$ if $weight\_inspected\_points_i \geq 0.95$ and $FFT_radius < crash\_region\_radius$, else 0
        * a sparse reward for successfully inspecting the chief
        * positive reward for successful inspection and safe trajectory
        * negative reward for successful inspection and unsafe trajectory (FFT collision with chief)
    * $r_t += -1$ if $radius < crash\_region\_radius$, else 0
        * a sparse reward punishing the agent for crashing with the chief
    * $r_t += -1$ if $||pos_{chief} - pos_{deputy}|| > distance_{max}$, else 0.
        * a sparse reward punishing the agent for moving too far from the chief
    * $r_t += 0.001$ if $t < t_{max}$, else 0
        * a dense reward for staying alive
    * $r_t += 0.0005 * e^{-|\delta_t(f, 1)^2 / \epsilon|}$
        * a dense reward for facing the chief
        * $\delta_t(f, 1)$ is the difference between the $f$, dot-product between the camera orientation vector and the relative position between the deputy and the chief, and 1.
        * $\epsilon$ is the length of the reward curve for the exponential decay (configurable)
    * $r_t += -0.1||\boldsymbol{u}||$
        * a dense reward for minimizing thruster usage (fuel cost)

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
        * $v \in [0, 0.3]$ m/s
        * $\psi \in [0, 2\pi] rad$
        * $\phi \in [-\pi/2, \pi/2] rad$
        * $v_x = v \cos{\psi} \cos{\phi}$
        * $v_y = v \sin{\psi} \cos{\phi}$
        * $v_z = v \sin{\phi}$
    * deputy $(\omega_x, \omega_y, \omega_z)$ is sampled from a uniform distribution between $[-0.01, -0.01, -0.01]$ rad/s and $[0.01, 0.01, 0.01]$ rad/s
    * Initial sun angle is randomly selected using a uniform distribution
        * $\theta_{sun} \in [0, 2\pi] rad$
        * If the deputy is initialized where it's sensor points within 60 degrees of the sun, its position is negated such that the sensor points away from the sun.

    ## Episode End

    An episode will terminate if any of the following conditions are met:

    * Termination: the agent exceeds a `max_distance = 800` meter radius away from the chief
    * Termination: the agent moves within a `crash_region_radius = 15` meter radius around the chief
    * Termination: the cumulative weight of inspected points exceeds 0.95
    * Truncation: the maximum number of timesteps, `max_timesteps = 1224`

    The episode is considered done and successful if and only if the cumulative weight of
    inspected points exceeds 0.95 while the deputy remains on a safe trajectory
    (not on a collision course with the chief).

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

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        priority_vector = self.rng.uniform(-1, 1, size=3)
        priority_vector /= np.linalg.norm(priority_vector)  # convert to unit vector
        self.chief = sim.SixDOFTarget(
            name="chief",
            num_points=100,
            radius=1,
            priority_vector=priority_vector,
        )
        self.deputies = {
            a:
            sim.SixDOFInspector(
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

    def _get_obs(self, agent: typing.Any) -> np.ndarray:
        deputy = self.deputies[agent]
        obs = self.observation_space(agent).sample()
        obs[:3] = deputy.position
        obs[3] = np.linalg.norm(deputy.position)
        obs[4:7] = np.abs(deputy.position)
        obs[7:10] = deputy.velocity
        obs[10] = np.linalg.norm(deputy.velocity)
        obs[11:14] = np.abs(deputy.velocity)
        obs[14:17] = deputy.angular_velocity
        obs[17:20] = Rotation.from_quat(deputy.orientation).as_euler("XYZ")
        obs[20] = np.dot(
            Rotation.from_quat(deputy.camera.orientation).as_euler("XYZ"),
            (self.chief.position - deputy.position) / np.linalg.norm(self.chief.position - deputy.position),
        )
        obs[21] = self.sun.theta
        obs[22] = self.chief.inspection_points.get_num_points_inspected()
        obs[23:26] = self.chief.inspection_points.kmeans_find_nearest_cluster(camera=deputy.camera, sun=self.sun)
        obs[26:29] = self.chief.inspection_points.priority_vector
        obs[29] = self.chief.inspection_points.get_total_weight_inspected()
        obs[30] = np.dot(
            Rotation.from_quat(deputy.camera.orientation).as_euler("XYZ"),
            self.chief.inspection_points.kmeans_find_nearest_cluster(camera=deputy.camera, sun=self.sun),
        )
        return obs

    def _get_info(self, agent: typing.Any) -> dict[str, typing.Any]:
        return {agent: None}

    def _get_reward(self, agent: typing.Any) -> float:
        reward = 0.0
        deputy = self.deputies[agent]

        # Dense rewards
        reward += r.weighted_observed_points_reward(chief=self.chief, prev_weight_inspected=self.prev_weight_inspected)
        reward += r.delta_v_reward(v=deputy.velocity, prev_v=self.prev_state[agent][3:6])

        reward += r.live_timestep_reward(t=self.simulator.sim_time, t_max=self.max_time)
        reward += r.facing_chief_reward(chief=self.chief, deputy=deputy, epsilon=0.01)

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
                    [-np.inf] * 3,  # deputy position
                    [-np.inf] * 4,  # deputy position magnorm
                    [-np.inf] * 3,  # deputy velocity
                    [-np.inf] * 4,  # deputy velocity magnorm
                    [-np.inf] * 3,  # deputy angular velocity
                    [-2 * np.pi] * 3,  # deputy orientation (euler)
                    [-1],  # facing chief dot product
                    [0],  # sun angle
                    [0],  # number of inspected points
                    [-1] * 3,  # nearest cluster unit vector
                    [-1] * 3,  # priority vector unit vector
                    [0],  # cumulative weight of inspected points
                    [-1],  # facing cluster dot product
                )
            ),
            np.concatenate(
                (
                    [np.inf] * 3,  # deputy position
                    [np.inf] * 4,  # deputy position magnorm
                    [np.inf] * 3,  # deputy velocity
                    [np.inf] * 4,  # deputy velocity magnorm
                    [np.inf] * 3,  # deputy angular velocity
                    [2 * np.pi] * 3,  # deputy orientation
                    [1],  # facing chief dot product
                    [2 * np.pi],  # sun angle
                    [100],  # number of inspected points
                    [1] * 3,  # nearest cluster unit vector
                    [1] * 3,  # priority vector unit vector
                    [1],  # cumulative weight of inspected points
                    [1],  # facing cluster dot product
                )
            ),
            shape=(31, ),
            dtype=np.float64,
        )

    # Pylint warns that self will never be garbage collected due to the use of the lru_cache, but the environment
    # should never be garabage collected when used
    @functools.lru_cache(maxsize=None)  # pylint:disable=W1518
    def action_space(self, agent: typing.Any) -> gym.Space:
        return gym.spaces.Box(
            np.array([-1, -1, -1, -0.001, -0.001, -0.001]),
            np.array([1, 1, 1, 0.001, 0.001, 0.001]),
            shape=(6, ),
            dtype=np.float64,
        )

    @property
    def sim_state(self) -> dict:
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
