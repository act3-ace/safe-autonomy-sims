"""Module for the V0 version of the Weighted 6DOF Inspection environment"""
import typing

import gymnasium as gym
import numpy as np
import copy
import safe_autonomy_simulation.sims.inspection as sim
from gymnasium.core import RenderFrame
from scipy.spatial.transform import Rotation

import safe_autonomy_sims.gym.inspection.reward as r
from safe_autonomy_sims.gym.inspection.utils import closest_fft_distance, polar_to_cartesian, rel_dist


class WeightedSixDofInspectionEnv(gym.Env):
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
    | 3     | x component of the deputy's velocity                        | -inf| inf | Velocity (m/s) |
    | 4     | y component of the deputy's velocity                        | -inf| inf | Velocity (m/s) |
    | 5     | z component of the deputy's velocity                        | -inf| inf | Velocity (m/s) |
    | 6     | x component of the deputy's angular velocity                | -inf| inf | Angular Velocity (rad/s) |
    | 7     | y component of the deputy's angular velocity                | -inf| inf | Angular Velocity (rad/s) |
    | 8     | z component of the deputy's angular velocity                | -inf| inf | Angular Velocity (rad/s) |
    | 9     | x component of the deputy's quaternion orientation          | -inf| inf | Orientation |
    | 10    | y component of the deputy's quaternion orientation          | -inf| inf | Orientation |
    | 11    | z component of the deputy's quaternion orientation          | -inf| inf | Orientation |
    | 12    | w component of the deputy's quaternion orientation          | -inf| inf | Orientation |
    | 13    | sun angle                                                   | 0   | 2pi | Angle (rad) |
    | 14    | number of inspected points                                  | 0   | 100 | Scalar |
    | 15    | x component of unit vector pointing to the nearest cluster  | -1  | 1   | Scalar |
    | 16    | y component of unit vector pointing to the nearest cluster  | -1  | 1   | Scalar |
    | 17    | z component of unit vector pointing to the nearest cluster  | -1  | 1   | Scalar |
    | 18    | x component of unit vector pointing to the priority vector  | -1  | 1   | Scalar |
    | 19    | y component of unit vector pointing to the priority vector  | -1  | 1   | Scalar |
    | 20    | z component of unit vector pointing to the priority vector  | -1  | 1   | Scalar |
    | 21    | cumulative weight of inspected points                       | 0   | 1   | Scalar |

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
    * deputy camera parameters:
        * field of view = $\pi$ rad
        * focal length = $1 m$
    * Initial sun angle is randomly selected using a uniform distribution
        * $\theta_{sun} \in [0, 2\pi] rad$
    * Simulator frame rate = $0.1 Hz$

    ## Episode End

    An episode will terminate if any of the following conditions are met:

    * Termination: the agent moves within a `crash_region_radius = 15` meter radius around the chief
    * Termination: the cumulative weight of inspected points exceeds 0.95
    * Truncation: the maximum number of timesteps, `max_timesteps = 12236`
    * Truncation: the agent exceeds a `max_distance = 800` meter radius away from the chief

    The episode is considered done and successful if and only if the cumulative weight of
    inspected points exceeds 0.95 while the deputy remains on a safe trajectory
    (not on a collision course with the chief).

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    * reward_components - a dict of reward component string names to their last computed float values.
    * status - a string descirbing the status of the current episode.

    Statuses for the episode are either "Running" or describe a unique terminal state. Terminal
    states can be one of the following: "Success", "Crash", "Out of Bounds", "Timeout".

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
        success_threshold: float = 0.95,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 12236,
    ) -> None:
        self.observation_space = gym.spaces.Box(
            np.concatenate(
                (
                    [-np.inf] * 3,  # deputy position
                    [-np.inf] * 3,  # deputy velocity
                    [-np.inf] * 3,  # deputy angular velocity
                    [-2 * np.pi] * 4,  # deputy orientation (quaternion)
                    [0],  # sun angle
                    [0],  # number of inspected points
                    [-1] * 3,  # nearest cluster unit vector
                    [-1] * 3,  # priority vector unit vector
                    [0],  # cumulative weight of inspected points
                )
            ),
            np.concatenate(
                (
                    [np.inf] * 3,  # deputy position
                    [np.inf] * 3,  # deputy velocity
                    [np.inf] * 3,  # deputy angular velocity
                    [2 * np.pi] * 4,  # deputy orientation
                    [2 * np.pi],  # sun angle
                    [100],  # number of inspected points
                    [1] * 3,  # nearest cluster unit vector
                    [1] * 3,  # priority vector unit vector
                    [1],  # cumulative weight of inspected points
                )
            ),
            shape=(22, ),
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            np.array([-1, -1, -1, -0.001, -0.001, -0.001]),
            np.array([1, 1, 1, 0.001, 0.001, 0.001]),
            shape=(6, ),
            dtype=np.float64,
        )

        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state: dict[typing.Any, typing.Any] | None = None
        self.prev_num_inspected = 0
        self.prev_weight_inspected = 0.0
        self.reward_components = {}
        self.status = "Running"

        # Lazy initialized
        self.chief: sim.Target
        self.deputy: sim.Inspector
        self.sun: sim.Sun
        self.simulator: sim.InspectionSimulator

    def reset(self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None) -> tuple[typing.Any, dict[str, typing.Any]]:
        super().reset(seed=seed, options=options)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        self.reward_components = {}
        self.status = "Running"
        obs, info = self._get_obs(), self._get_info()
        self.prev_state = None
        self.prev_num_inspected = 0
        self.prev_weight_inspected = 0.0
        return obs, info

    def step(self, action: typing.Any) -> tuple[typing.Any, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
        assert self.action_space.contains(
            action
        ), f"given action {action} is not contained in action space {self.action_space}"

        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        if self.simulator.sim_time > 0:
            self.prev_num_inspected = (self.chief.inspection_points.get_num_points_inspected())
            self.prev_weight_inspected = (self.chief.inspection_points.get_total_weight_inspected())

        # Update simulator state
        self.deputy.add_control(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        priority_vector = self.np_random.uniform(-1, 1, size=3)
        priority_vector /= np.linalg.norm(priority_vector)  # convert to unit vector
        self.chief = sim.SixDOFTarget(
            name="chief",
            num_points=100,
            radius=10,
            priority_vector=priority_vector,
        )
        self.deputy = sim.SixDOFInspector(
            name="deputy",
            position=polar_to_cartesian(
                r=self.np_random.uniform(50, 100),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
                theta=self.np_random.uniform(0, 2 * np.pi),
            ),
            velocity=polar_to_cartesian(
                r=self.np_random.uniform(0, 0.3),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
                theta=self.np_random.uniform(0, 2 * np.pi),
            ),
            fov=np.pi/3,
            focal_length=9.6e-3,
        )
        self.sun = sim.Sun(theta=self.np_random.uniform(0, 2 * np.pi))
        self.simulator = sim.InspectionSimulator(
            frame_rate=0.1,
            inspectors=[self.deputy],
            targets=[self.chief],
            sun=self.sun,
        )

    def _get_obs(self):
        obs = self.observation_space.sample()
        obs[0:3] = self.deputy.position
        obs[3:6] = self.deputy.velocity
        obs[6:9] = self.deputy.angular_velocity
        obs[9:13] = self.deputy.orientation
        obs[13] = self.sun.theta
        obs[14] = self.chief.inspection_points.get_num_points_inspected()
        obs[15:18] = self.chief.inspection_points.kmeans_find_nearest_cluster(camera=self.deputy.camera, sun=self.sun)
        obs[18:21] = self.chief.inspection_points.priority_vector
        obs[21] = self.chief.inspection_points.get_total_weight_inspected()

        return obs

    def _get_info(self):
        return {
            "reward_components": copy.copy(self.reward_components),
            "status": copy.copy(self.status)
        }

    def _get_reward(self):
        reward = 0

        # Dense rewards
        points_reward = r.weighted_observed_points_reward(chief=self.chief, prev_weight_inspected=self.prev_weight_inspected)
        self.reward_components["observed_points"] = points_reward
        reward += points_reward

        step_size = 1 / self.simulator.frame_rate
        delta_v_reward = r.delta_v_reward(control=self.deputy.last_control, scale=-0.005, step_size=step_size)
        self.reward_components["delta_v"] = delta_v_reward
        reward += delta_v_reward

        live_timestep_reward = r.live_timestep_reward(t=self.simulator.sim_time, t_max=3000.0)
        self.reward_components["live_timestep"] = live_timestep_reward
        reward += live_timestep_reward

        facing_chief_reward = r.facing_chief_reward(chief=self.chief, deputy=self.deputy, epsilon=0.15)
        self.reward_components["facing_chief"] = facing_chief_reward
        reward += facing_chief_reward

        # Sparse rewards
        success_reward = r.weighted_inspection_success_reward(chief=self.chief, total_weight=self.success_threshold)
        if (success_reward > 0 and closest_fft_distance(chief=self.chief, deputy=self.deputy) < self.crash_radius):
            success_reward = -1.0
        self.reward_components["success"] = success_reward
        reward += success_reward

        crash_reward = r.crash_reward(chief=self.chief, deputy=self.deputy, crash_radius=self.crash_radius)
        self.reward_components["crash"] = crash_reward
        reward += crash_reward

        max_distance_reward = r.max_distance_reward(chief=self.chief, deputy=self.deputy, max_distance=self.max_distance)
        self.reward_components["max_distance"] = max_distance_reward
        reward += max_distance_reward

        return reward

    def _get_terminated(self):
        # Get state info
        d = rel_dist(pos1=self.deputy.position, pos2=self.chief.position)

        # Determine if in terminal state
        crash = d < self.crash_radius
        all_inspected = self.chief.inspection_points.get_total_weight_inspected() >= self.success_threshold

        # Update Status
        if crash:
            self.status = "Crash"
        elif all_inspected:
            self.status = "Success"

        return crash or all_inspected

    def _get_truncated(self):
        d = rel_dist(pos1=self.deputy.position, pos2=self.chief.position)
        oob = d > self.max_distance
        timeout = self.simulator.sim_time > self.max_time

        # Update Status
        if oob:
            self.status = "Out of Bounds"
        elif timeout:
            self.status = "Timeout"

        return oob or timeout

    @property
    def sim_state(self) -> dict:
        """Provides the state of the simulator

        Returns
        -------
        dict
            A dictionary containing the state of the deputy and the state of the chief
        """
        state = {
            "deputy": self.deputy.state,
            "chief": self.chief.state,
        }
        return state

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError
