import typing
import numpy as np
import gymnasium as gym
import safe_autonomy_simulation.sims.inspection as sim
import safe_autonomy_sims.gym.inspection.reward as r
import safe_autonomy_sims.gym.inspection.utils as utils


class WeightedInspectionEnv(gym.Env):
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
    * deputy camera parameters:
        * field of view = $\pi$ rad
        * focal length = $1 m$
    * Initial sun angle is randomly selected using a uniform distribution
        * $\theta_{sun} \in [0, 2\pi] rad$
    * Simulator frame rate = $0.1 Hz$

    ## Episode End

    An episode will end if any of the following conditions are met:

    * Terminated: the agent moves within a `crash_region_radius = 10` meter radius around the chief
    * Terminated: the cumulative weight of inspected points exceeds 0.95
    * Truncated: the maximum number of timesteps, `max_timesteps = 12236`
    * Truncated: the agent exceeds a `max_distance = 800` meter radius away from the chief

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
    """

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
            shape=(15,),
        )

        self.action_space = gym.spaces.Box(-1, 1, shape=(3,))

        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state = None
        self.prev_num_inspected = 0
        self.prev_weight_inspected = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None
    ) -> tuple[typing.Any, dict[str, typing.Any]]:
        super().reset(seed=seed, options=options)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        obs, info = self._get_obs(), self._get_info()
        self.prev_state = None
        self.prev_num_inspected = 0
        self.prev_weight_inspected = 0.0
        return obs, info

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        priority_vector = self.np_random.uniform(-1, 1, size=3)
        priority_vector /= np.linalg.norm(priority_vector)  # convert to unit vector
        self.chief = sim.Target(
            name="chief",
            num_points=100,
            radius=1,
            priority_vector=priority_vector,
        )
        self.deputy = sim.Inspector(
            name="deputy",
            position=utils.polar_to_cartesian(
                r=self.np_random.uniform(50, 100),
                theta=self.np_random.uniform(0, 2 * np.pi),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
            ),
            velocity=utils.polar_to_cartesian(
                r=self.np_random.uniform(0, 0.3),
                theta=self.np_random.uniform(0, 2 * np.pi),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
            ),
            fov=np.pi,
            focal_length=1,
        )
        self.sun = sim.Sun(theta=self.np_random.uniform(0, 2 * np.pi))
        self.simulator = sim.InspectionSimulator(
            frame_rate=0.1,
            inspectors=[self.deputy],
            targets=[self.chief],
            sun=self.sun,
        )

    def step(
        self, action: typing.Any
    ) -> tuple[typing.Any, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        self.prev_num_inspected = (
            self.chief.inspection_points.get_num_points_inspected()
        )
        self.prev_weight_inspected += (
            self.chief.inspection_points.get_total_weight_inspected()
        )

        # Update simulator state
        self.deputy.add_control(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        terminated = self._get_terminated()
        truncated = self.simulator.sim_time > self.max_time
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = self.observation_space.sample()
        obs[:3] = self.deputy.position
        obs[3:6] = self.deputy.velocity
        obs[6] = self.sun.theta
        obs[7] = self.chief.inspection_points.get_num_points_inspected()
        obs[8:11] = self.chief.inspection_points.kmeans_find_nearest_cluster(
            camera=self.deputy.camera, sun=self.sun
        )
        obs[11:14] = self.chief.inspection_points.priority_vector
        obs[14] = self.chief.inspection_points.get_total_weight_inspected()
        return obs

    def _get_info(self):
        return {}

    def _get_reward(self):
        reward = 0

        # Dense rewards
        reward += r.weighted_observed_points_reward(
            chief=self.chief, prev_weight_inspected=self.prev_weight_inspected
        )
        reward += r.delta_v_reward(
            v=self.deputy.velocity, prev_v=self.prev_state["deputy"][3:6]
        )

        # Sparse rewards
        reward += (
            r.weighted_inspection_success_reward(
                chief=self.chief, total_weight=self.success_threshold
            )
            if utils.closest_fft_distance(chief=self.chief, deputy=self.deputy)
            < self.crash_radius
            else -1.0
        )
        reward += r.crash_reward(
            chief=self.chief, deputy=self.deputy, crash_radius=self.crash_radius
        )

        return reward

    def _get_terminated(self):
        # Get state info
        d = utils.rel_dist(pos1=self.chief.position, pos2=self.deputy.position)

        # Determine if in terminal state
        crash = d < self.crash_radius
        all_inspected = self.prev_weight_inspected >= self.success_threshold

        return crash or all_inspected

    def _get_truncated(self):
        d = utils.rel_dist(pos1=self.chief.position, pos2=self.deputy.position)
        oob = d > self.max_distance
        timeout = self.simulator.sim_time > self.max_time
        return oob or timeout

    @property
    def sim_state(self) -> dict:
        state = {
            "deputy": self.deputy.state,
            "chief": self.chief.state,
        }
        return state
