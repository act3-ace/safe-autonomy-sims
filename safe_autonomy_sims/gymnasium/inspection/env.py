"""A gymnasium environment for training agents to dock a deputy spacecraft onto a chief spacecraft"""

from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from safe_autonomy_simulation.inspection.inspection_simulator import InspectionSimulator
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft
import safe_autonomy_sims.gymnasium.inspection.reward as r
from safe_autonomy_sims.gymnasium.inspection.utils import (
    rel_dist,
    closest_fft_distance,
)


class TranslationalInspectionEnv(gym.Env):
    def __init__(
        self,
        chief_init: dict,
        deputy_init: dict,
        inspection_point_config: dict,
        success_threshold: float = 100,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 1000,
    ) -> None:
        # Initialize simulator with chief and deputy spacecraft
        self.simulator = InspectionSimulator(
            entities={
                "chief": CWHSpacecraft(name="chief", **chief_init),
                "deputy": CWHSpacecraft(name="deputy", **deputy_init),
            },
            inspectors=["deputy"],
            inspection_point_map={"deputy": inspection_point_config},
            frame_rate=1,
        )

        # Each spacecraft obs = [x, y, z, v_x, v_y, v_z, theta_sun, n, x_ups, y_ups, z_ups]
        self.observation_space = spaces.Dict(
            {
                "chief": spaces.Box(
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
                    shape=(8,),
                ),
                "deputy": spaces.Box(
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
                    shape=(8,),
                ),
            }
        )

        # Each spacecraft is controlled by [xdot, ydot, zdot]
        self.action_space = spaces.Dict(
            {
                "deputy": spaces.Box(-1, 1, shape=(3,))  # only the deputy is controlled
            }
        )

        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state = None
        self.num_inspected = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.simulator.reset()
        obs, info = self._get_obs(), self._get_info()
        self.prev_state = None
        self.num_inspected = 0
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        self.num_inspected = self.sim_state[
            "inspection_points"
        ].get_num_points_inspected()

        # Update simulator state
        self.simulator.add_controls(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        terminated = self._get_terminated()
        truncated = False  # used to signal episode ended unexpectedly
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = self.simulator.info
        return obs

    def _get_info(self):
        pass

    def _get_reward(self):
        reward = 0

        # Dense rewards
        reward += r.observed_points_reward(
            state=self.sim_state, num_inspected=self.num_inspected
        )
        reward += r.delta_v_reward(state=self.sim_state)

        # Sparse rewards
        reward += r.inspection_success_reward(
            state=self.sim_state,
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
        return self.simulator.info


class WeightedTranslationalInspectionEnv(TranslationalInspectionEnv):
    def __init__(
        self,
        chief_init: dict,
        deputy_init: dict,
        inspection_point_config: dict,
        success_threshold: float = 0.95,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 1000,
    ) -> None:
        super().__init__(
            chief_init,
            deputy_init,
            inspection_point_config,
            success_threshold,
            crash_radius,
            max_distance,
            max_time,
        )
        self.observation_space = spaces.Dict(
            {
                "chief": spaces.Box(
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
                        -1,
                        -1,
                        -1,
                        0,
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
                        1,
                        1,
                        1,
                        1,
                    ],
                    shape=(15,),
                ),
                "deputy": spaces.Box(
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
                        -1,
                        -1,
                        -1,
                        0,
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
                        1,
                        1,
                        1,
                        1,
                    ],
                    shape=(15,),
                ),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.weight_inspected = 0.0
        return super().reset(seed=seed, options=options)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.weight_inspected += self.sim_state[
            "inspection_points"
        ].get_total_weight_inspected()
        return super().step(action)

    def _get_reward(self):
        reward = 0

        # Dense rewards
        reward += r.weighted_observed_points_reward(
            state=self.sim_state, weight_inspected=self.weight_inspected
        )
        reward += r.delta_v_reward(state=self.sim_state, prev_state=self.prev_state)

        # Sparse rewards
        reward += (
            r.weighted_inspection_success_reward(
                state=self.sim_state, total_weight=self.success_threshold
            )
            if closest_fft_distance(state=self.sim_state) < self.crash_radius
            else -1.0
        )
        reward += r.crash_reward(state=self.sim_state, crash_radius=self.crash_radius)

        return reward

    def _get_terminated(self):
        # Get state info
        d = rel_dist(state=self.sim_state)

        # Determine if in terminal state
        oob = d > self.max_distance
        crash = d < self.crash_radius
        timeout = self.simulator.sim_time > self.max_time
        all_inspected = self.weight_inspected >= self.success_threshold

        return oob or crash or timeout or all_inspected


class WeightedSixDofInspectionEnv(WeightedTranslationalInspectionEnv):
    def __init__(
        self,
        chief_init: dict,
        deputy_init: dict,
        inspection_point_config: dict,
        success_threshold: float = 0.95,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 1000,
    ) -> None:
        super().__init__(
            chief_init,
            deputy_init,
            inspection_point_config,
            success_threshold,
            crash_radius,
            max_distance,
            max_time,
        )
        self.observation_space = spaces.Dict(
            {
                "chief": spaces.Box(
                    [
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -1,
                        -1,
                        -1,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -1,
                        -1,
                        -1,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        0,
                        0,
                    ],
                    [
                        np.inf,
                        np.inf,
                        np.inf,
                        np.inf,
                        1,
                        1,
                        1,
                        np.inf,
                        np.inf,
                        np.inf,
                        np.inf,
                        1,
                        1,
                        1,
                        np.inf,
                        np.inf,
                        np.inf,
                        2 * np.pi,
                        2 * np.pi,
                        2 * np.pi,
                        2 * np.pi,
                        1,
                        2 * np.pi,
                        100,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ],
                    shape=(32,),
                ),
                "deputy": spaces.Box(
                    [
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -1,
                        -1,
                        -1,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -1,
                        -1,
                        -1,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        0,
                        0,
                    ],
                    [
                        np.inf,
                        np.inf,
                        np.inf,
                        np.inf,
                        1,
                        1,
                        1,
                        np.inf,
                        np.inf,
                        np.inf,
                        np.inf,
                        1,
                        1,
                        1,
                        np.inf,
                        np.inf,
                        np.inf,
                        2 * np.pi,
                        2 * np.pi,
                        2 * np.pi,
                        2 * np.pi,
                        1,
                        2 * np.pi,
                        100,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ],
                    shape=(32,),
                ),
            }
        )
        self.action_space = spaces.Dict(
            {
                "deputy": spaces.Box(
                    [-1, -1, -1, -0.001, -0.001, -0.001],
                    [1, 1, 1, 0.001, 0.001, 0.001],
                    shape=(6,),
                )  # only the deputy is controlled
            }
        )

    def _get_reward(self):
        reward = super()._get_reward()
        reward += r.live_timestep_reward(t=self.simulator.sim_time, t_max=self.max_time)
        reward += r.facing_chief_reward(state=self.sim_state)
        return reward
