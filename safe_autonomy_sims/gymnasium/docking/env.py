"""A gymnasium environment for training agents to dock a deputy spacecraft onto a chief spacecraft"""

from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from safe_autonomy_simulation.docking_simulator import DockingSimulator
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft
import safe_autonomy_sims.gymnasium.docking.reward as r
from safe_autonomy_sims.gymnasium.docking.utils import v_limit, rel_dist, rel_vel


class DockingEnv(gym.Env):
    def __init__(
        self,
        chief_init: dict,
        deputy_init: dict,
        docking_radius: float = 0.2,
        max_time: int = 2000,
        max_distance: float = 10000,
        max_v_violation: int = 5,
    ) -> None:
        # Initialize simulator with chief and deputy spacecraft
        self.simulator = DockingSimulator(
            entities={
                "chief": CWHSpacecraft(name="chief", **chief_init),
                "deputy": CWHSpacecraft(name="deputy", **deputy_init),
            },
            frame_rate=1,
        )

        # Each spacecraft obs = [x, y, z, v_x, v_y, v_z, s, v_limit]
        self.observation_space = spaces.Dict(
            {
                "chief": spaces.Box(
                    [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0],
                    np.inf,
                    shape=(8,),
                ),
                "deputy": spaces.Box(
                    [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0],
                    np.inf,
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
        reward += r.distance_pivot_reward(
            state=self.sim_state, prev_state=self.prev_state
        )
        reward += r.delta_v_reward(state=self.sim_state)
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
        oob = d > self.max_distance
        crash = in_docking and not safe_v
        max_v_violation = self.episode_v_violations > self.max_v_violation
        timeout = self.simulator.sim_time > self.max_time
        docked = in_docking and safe_v

        return oob or crash or max_v_violation or timeout or docked

    @property
    def sim_state(self) -> dict:
        return self.simulator.info
