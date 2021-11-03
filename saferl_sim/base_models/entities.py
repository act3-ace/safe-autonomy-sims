"""
This module provides base implentations for entities in the saferl simulator
"""

import abc
from typing import Tuple, Union

import numpy as np
import scipy.integrate
import scipy.spatial
from pydantic import BaseModel


class BaseEntityValidator(BaseModel):
    """Validator for BaseEntity's config member

    Parameters
    ----------
    name : str
        name of entity
    """
    name: str


class BaseEntity(abc.ABC):
    """Base implementation of a dynamics controlled entity within the saferl sim

    Parameters
    ----------
    dynamics : BaseDynamics
        dynamics object for computing state transitions
    control_default: np.ndarray
        default control vector used when no action is passed to step(). Typically 0 or neutral for each actuator.
    control_min: np.ndarray
        minimum allowable control vector values. Control vectors that exceed this limit are clipped.
    control_max: np.ndarray
        maximum allowable control vector values. Control vectors that exceed this limit are clipped.
    control_map: dict
        Optional mapping for actuator names to their indices in the state vector.
        Allows dictionary action inputs in step().
    """

    def __init__(self, dynamics, control_default, control_min=-np.inf, control_max=np.inf, control_map=None, **kwargs):
        self.config = self._get_config_validator()(**kwargs)
        self.name = self.config.name
        self.dynamics = dynamics

        self.control_default = control_default
        self.control_min = control_min
        self.control_max = control_max
        self.control_map = control_map

        self._state = self._build_state()
        self.state_dot = np.zeros_like(self._state)

    @classmethod
    def _get_config_validator(cls):
        return BaseEntityValidator

    @abc.abstractmethod
    def _build_state(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, step_size, action=None):
        """Executes a state transition simulation step for the entity

        Parameters
        ----------
        step_size : float
            duration of simulation step in seconds
        action : Union(dict, list, np.ndarray), optional
            Control action taken by entity, by default None resulting in a control of control_default
            When list or ndarray, directly used and control vector for dynamics model
            When dict, unpacked into control vector. Requires control_map to be defined.
        Raises
        ------
        KeyError
            Raised when action dict key not found in control map
        ValueError
            Raised when action is not one of the required types
        """

        if action is None:
            control = self.control_default.copy()
        else:
            if isinstance(action, dict):
                assert self.control_map is not None, "Cannot use dict-type action without a control_map (see BaseEntity __init__())"
                control = self.control_default.copy()
                for action_name, action_value in action.items():
                    if action_name not in self.control_map:
                        raise KeyError(
                            f"action '{action_name}' not found in entity's control_map, "
                            f"please use one of: {self.control_map.keys()}"
                        )

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

    @property
    def state(self) -> np.ndarray:
        """Returns copy of entity's state vector

        Returns
        -------
        np.ndarray
            copy of state vector
        """
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy()

    @property
    @abc.abstractmethod
    def x(self):
        """get x"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self):
        """get y"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def z(self):
        """get z"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self) -> np.ndarray:
        """get 3d position vector"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def orientation(self) -> scipy.spatial.transform.Rotation:
        """get orientation of entity

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation tranformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def velocity(self):
        """get 3d velocity vector"""
        raise NotImplementedError


class BaseDynamics(abc.ABC):
    """
    State transition implementation for a physics dynamics model. Used by entities to compute their next state while stepping.

    Parameters
    ----------
    state_min : float or np.ndarray
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    state_min : float or np.ndarray
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    angle_wrap_centers: np.ndarray
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
    """

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
        """Computes the dynamics state transition from the current state and control input

        Parameters
        ----------
        step_size : float
            Duration of the simation step in seconds.
        state : np.ndarray
            Current state of the system at the beginning of the simulation step.
        control : np.ndarray
            Control vector of the dynamics model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            tuple of the systems's next state and the state's instantaneous time derivative at the end of the step
        """
        next_state, state_dot = self._step(step_size, state, control)
        next_state = np.clip(next_state, self.state_min, self.state_max)
        next_state = self._wrap_angles(next_state)
        return next_state, state_dot

    def _wrap_angles(self, state):
        wrapped_state = state.copy()
        if self.angle_wrap_centers is not None:
            wrap_idxs = np.logical_not(np.isnan(self.angle_wrap_centers))

            wrapped_state[wrap_idxs] = \
                ((wrapped_state[wrap_idxs] + np.pi) % (2 * np.pi)) - np.pi + self.angle_wrap_centers[wrap_idxs]

        return wrapped_state

    @abc.abstractmethod
    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):
    """
    State transition implementation for generic Ordinary Differential Equation dynamics models.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    integration_method : string
        Numerical integration method used by dyanmics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate.
        'Euler' is fast but very inaccurate.
    kwargs
        Additional keyword arguments passed to parent BaseDynamics constructor.
    """

    def __init__(self, integration_method="RK45", **kwargs):
        self.integration_method = integration_method
        super().__init__(**kwargs)

    def compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Computes the instataneous time derivative of the state vector

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : np.ndarray
            Current state vector at time t.
        control : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            Instantaneous time derivative of the state vector.
        """
        state_dot = self._compute_state_dot(t, state, control)
        state_dot = self._clip_state_dot_by_state_limits(state, state_dot)
        return state_dot

    @abc.abstractmethod
    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _clip_state_dot_by_state_limits(self, state, state_dot):
        lower_bounded_states = state <= self.state_min
        upper_bounded_state = state >= self.state_max

        state_dot[lower_bounded_states] = np.clip(state_dot[lower_bounded_states], 0, np.inf)
        state_dot[upper_bounded_state] = np.clip(state_dot[upper_bounded_state], -np.inf, 0)

        return state_dot

    def _step(self, step_size, state, control):

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.compute_state_dot, (0, step_size), state, args=(control, ))

            next_state = sol.y[:, -1]  # save last timestep of integration solution
            state_dot = self.compute_state_dot(step_size, next_state, control)
        elif self.integration_method == "Euler":
            state_dot = self.compute_state_dot(0, state, control)
            next_state = state + step_size * state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return next_state, state_dot


class BaseLinearODESolverDynamics(BaseODESolverDynamics):
    """
    State transition implementation for generic Linear Ordinary Differential Equation dynamics models of the form dx/dt = Ax+Bu.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    kwargs
        Additional keyword arguments passed to parent BaseODESolverDynamics constructor.
    """

    def __init__(self, **kwargs):
        self.A, self.B = self._gen_dynamics_matrices()
        super().__init__(**kwargs)

    @abc.abstractmethod
    def _gen_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes the linear ODE matrices A, B.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            tuple of the systems's dynamics matrices A, B.
        """
        raise NotImplementedError

    def _update_dynamics_matrices(self, state):
        """
        Updates the linear ODE matrices A, B with current system state before computing derivative.
        Allows non-linear dynamics models to be linearized at each numerical integration interval.
        Directly modifies self.A, self.B.

        Default implementation is a no-op.

        Parameters
        ----------
        state : np.ndarray
            Current state vector of the system.
        """

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray):
        self._update_dynamics_matrices(state)
        state_dot = np.matmul(self.A, state) + np.matmul(self.B, control)
        return state_dot
