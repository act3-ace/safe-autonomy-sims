import abc
import copy

import gym
import numpy as np
from numpy.lib.arraysetops import isin
import scipy.integrate
import scipy.spatial
from typing import Union, Tuple


class BaseEnvObj(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name):
        self.name = name

    @property
    @abc.abstractmethod
    def x(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def z(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def orientation(self) -> scipy.spatial.transform.Rotation:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def velocity(self):
        raise NotImplementedError


class BasePlatform(BaseEnvObj):
    def __init__(self, name, dynamics, control_default, control_min=-np.inf, control_max=np.inf, control_map=None):

        super().__init__(name)
        self.dependent_objs = []

        self.dynamics = dynamics

        self.control_default = control_default
        self.control_min = control_min
        self.control_max = control_max
        self.control_map = control_map

        self.reset()

    def reset(self, state=None, **kwargs):
        assert state is None or isinstance(state, np.ndarray)
        if state:
            self._state = state.copy()
        else:
            self._state = self.build_state(**kwargs)

        self.state_dot = np.zeros_like(self._state)

    def step(self, step_size, action=None):

        if action is None:
            control = self.control_default.copy()
        else:
            if isinstance(action, dict):
                assert self.control_map is not None, "Cannot use dict-type action without a control_map (see platform __init__())"
                control = self.control_default.copy()
                for action_name, action_value in action.items():
                    if action_name not in self.control_map:
                        raise KeyError(
                            f"action '{action_name}' not found in platform's control_map, "
                            f"please use one of: {[k for k in self.control_map.keys()]}"
                        )
                    else:
                        control[self.control_map[action_name]] = action_value
            elif isinstance(action, list):
                control = np.array(action, dtype=np.float32)
            elif isinstance(action, np.ndarray):
                control = action.copy()
            else:
                raise ValueError("action must be type dict, list, or np.ndarray")

        # enforce control bounds
        control = np.clip(control, self.control_min, self.control_max)

        # compute new state if dynamics were applied
        self.state, self.state_dot = self.dynamics.step(step_size, self.state, control)

        for obj in self.dependent_objs:
            obj.step(step_size, action=action)

    def register_dependent_obj(self, obj):
        self.dependent_objs.append(obj)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy()


class BaseDynamics(abc.ABC):
    def __init__(
        self, 
        state_min: Union[float, np.ndarray] = -np.inf, 
        state_max: Union[float, np.ndarray] = np.inf, 
        angle_wrap_centers: np.ndarray = None,
    ):
        self.state_min = state_min
        self.state_max = state_max
        self.angle_wrap_centers = angle_wrap_centers

    def step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        next_state, state_dot = self._step(step_size, state, control)
        next_state = np.clip(next_state, self.state_min, self.state_max)
        next_state = self.wrap_angles(next_state)
        return next_state, state_dot

    def wrap_angles(self, state):
        wrapped_state = state.copy()
        if self.angle_wrap_centers is not None:
            wrap_idxs = np.logical_not(np.isnan(self.angle_wrap_centers))

            wrapped_state[wrap_idxs] = \
                ((wrapped_state[wrap_idxs] + np.pi) % (2*np.pi)) - np.pi + self.angle_wrap_centers[wrap_idxs]

        return wrapped_state

    @abc.abstractmethod
    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):
    def __init__(self, integration_method="Euler", **kwargs):
        self.integration_method = integration_method
        super().__init__(**kwargs)

    def compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        state_dot = self._compute_state_dot(t, state, control)
        state_dot = self.clip_state_dot_by_state_limits(state, state_dot)
        return state_dot

    @abc.abstractmethod
    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def clip_state_dot_by_state_limits(self, state, state_dot):
        lower_bounded_states = state <= self.state_min
        upper_bounded_state = state >= self.state_max

        state_dot[lower_bounded_states] = np.clip(state_dot[lower_bounded_states], 0, np.inf)
        state_dot[upper_bounded_state] = np.clip(state_dot[upper_bounded_state], -np.inf, 0)

        return state_dot

    def _step(self, step_size, state, control):

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.compute_state_dot, (0, step_size), state, args=(control,))

            next_state = sol.y[:, -1]  # save last timestep of integration solution
            state_dot = self.compute_state_dot(step_size, next_state, control)
        elif self.integration_method == "Euler":
            state_dot = self.compute_state_dot(0, state, control)
            next_state = state + step_size * state_dot
            state_dot = state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return next_state, state_dot


class BaseLinearODESolverDynamics(BaseODESolverDynamics):
    def __init__(self, **kwargs):
        self.A, self.B = self.gen_dynamics_matrices()
        super().__init__(**kwargs)

    @abc.abstractmethod
    def gen_dynamics_matrices(self):
        raise NotImplementedError

    def update_dynamics_matrices(self, state):
        pass

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray):
        self.update_dynamics_matrices(state)
        state_dot = np.matmul(self.A, state) + np.matmul(self.B, control)
        return state_dot
