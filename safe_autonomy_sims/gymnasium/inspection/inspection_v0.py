"""A gymnasium environment for training agents to dock a deputy spacecraft onto a chief spacecraft"""

from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from safe_autonomy_simulation.sims.inspection import (
    InspectionSimulator,
    Inspector,
    Target,
    Camera,
    Sun,
)
import safe_autonomy_sims.gymnasium.inspection.reward as r
from safe_autonomy_sims.gymnasium.inspection.utils import rel_dist


class InspectionEnv(gym.Env):
    def __init__(
        self,
        success_threshold: float = 100,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 1000,
    ) -> None:
        # Each spacecraft obs = [x, y, z, v_x, v_y, v_z, theta_sun, n, x_ups, y_ups, z_ups]
        self.observation_space = spaces.Box(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                0,
                0,
                -1,
                -1,
                -1,
            ],
            [
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                2 * np.pi,
                100,
                1,
                1,
                1,
            ],
            shape=(11,),
        )

        # Each spacecraft is controlled by [xdot, ydot, zdot]
        self.action_space = spaces.Box(
            -1, 1, shape=(3,)
        )  # only the deputy is controlled

        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state = None
        self.prev_num_inspected = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        obs, info = self._get_obs(), self._get_info()
        self.prev_state = None
        self.prev_num_inspected = 0
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        self.prev_num_inspected = (
            self.chief.inspection_points.get_num_points_inspected()
        )

        # Update simulator state
        self.deputy.add_control(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        terminated = self._get_terminated()
        truncated = False  # used to signal episode ended unexpectedly
        return observation, reward, terminated, truncated, info

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        self.chief = Target(
            name="chief",
            num_points=100,
            radius=1,
        )
        self.deputy = Inspector(
            name="deputy",
            camera=Camera(
                name="deputy_camera",
                fov=90,
                resolution=(640, 480),
                pixel_pitch=1e-6,
            ),
            position=self.np_random.uniform(-100, 100, size=3),
            velocity=self.np_random.uniform(-1, 1, size=3),
        )
        self.sun = Sun(theta=self.np_random.uniform(0, 2 * np.pi))
        self.simulator = InspectionSimulator(
            frame_rate=1,
            inspectors=[self.deputy],
            targets=[self.chief],
            sun=self.sun,
        )

    def _get_obs(self):
        obs = self.observation_space.sample()
        obs[:3] = self.deputy.position
        obs[3:6] = self.deputy.velocity
        obs[6] = self.sun.theta
        obs[7] = self.chief.inspection_points.get_num_points_inspected()
        obs[8:11] = self.chief.inspection_points.kmeans_find_nearest_cluster(
            camera=self.deputy.camera, sun=self.sun
        )
        return obs

    def _get_info(self):
        pass

    def _get_reward(self):
        reward = 0

        # Dense rewards
        reward += r.observed_points_reward(
            chief=self.chief, num_inspected=self.num_inspected
        )
        reward += r.delta_v_reward(state=self.sim_state, prev_state=self.prev_state)

        # Sparse rewards
        reward += r.inspection_success_reward(
            chief=self.chief,
            total_points=self.success_threshold,
        )
        reward += r.crash_reward(
            state=self.sim_state,
            crash_radius=self.crash_radius,
        )
        return reward

    def _get_terminated(self):
        # Get state info
        d = rel_dist(state=self.sim_state)

        # Determine if in terminal state
        oob = d > self.max_distance
        crash = d < self.crash_radius
        timeout = self.simulator.sim_time > self.max_time
        all_inspected = self.num_inspected == self.success_threshold

        return oob or crash or timeout or all_inspected

    @property
    def sim_state(self) -> dict:
        state = {
            "deputy": self.deputy.state,
            "chief": self.chief.state,
        }
        return state
