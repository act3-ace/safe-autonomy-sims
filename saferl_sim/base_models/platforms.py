import abc
import copy

import gym
import numpy as np
import scipy.integrate
import scipy.spatial


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


class BaseActuator(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def space(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default(self):
        raise NotImplementedError


class ContinuousActuator(BaseActuator):
    def __init__(self, name, bounds, default):
        self._name = name
        self._space = "continuous"
        self._bounds = bounds

        if isinstance(default, np.ndarray):
            self._default = default
        else:
            self._default = np.array(default, ndmin=1, dtype=np.float64)

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> str:
        return self._space

    @property
    def bounds(self) -> list:
        return self._bounds.copy()

    @bounds.setter
    def bounds(self, val):
        self._bounds = val.copy()

    @property
    def default(self):
        return copy.deepcopy(self._default)

class BaseActuatorSet:
    def __init__(self, actuators):
        self.actuators = actuators

        self.name_idx_map = {}
        for i, actuator in enumerate(self.actuators):
            self.name_idx_map[actuator.name] = i

    def gen_control(self, actuation=None):
        control_list = []

        if actuation is None:
            actuation = {}

        for actuator in self.actuators:
            actuator_name = actuator.name

            if actuator_name in actuation:
                actuator_control = actuation[actuator_name]
            else:
                actuator_control = actuator.default

            control_list.append(actuator_control)

        control = np.concatenate(control_list)

        return control


class BasePlatform(BaseEnvObj):
    def __init__(self, name, dynamics, actuator_set):

        super().__init__(name)
        self.dependent_objs = []

        self.dynamics = dynamics
        self.actuator_set = actuator_set

        self.reset()

    def reset(self, **kwargs):
        for obj in self.dependent_objs:
            obj.reset(**kwargs)

    def step(self, sim_state, step_size, action=None):

        control = np.array(action)
        self.current_control = control

        # compute new state if dynamics were applied
        self.next_state = self.dynamics.step(step_size, copy.deepcopy(self.state), control)

        for obj in self.dependent_objs:
            obj.step_compute(sim_state, action=action)

        # overwrite platform state with new state from dynamics
        self.state = self.next_state

        for obj in self.dependent_objs:
            obj.step_apply()

    def register_dependent_obj(self, obj):
        self.dependent_objs.append(obj)

    @property
    def x(self):
        return self.state.x

    @property
    def y(self):
        return self.state.y

    @property
    def z(self):
        return self.state.z

    @property
    def position(self):
        return self.state.position

    @property
    def orientation(self):
        return self.state.orientation

    @property
    def velocity(self):
        return self.state.velocity

    @property
    @abc.abstractmethod
    def _state(self) -> np.ndarray:
        raise NotImplemented

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy()


class BasePlatformStateVectorized(BasePlatformState):
    def reset(self, vector=None, vector_deep_copy=True, **kwargs):
        if vector is None:
            self._vector = self.build_vector(**kwargs)
        else:
            assert isinstance(vector, np.ndarray)
            assert vector.shape == self.vector_shape
            if vector_deep_copy:
                self._vector = copy.deepcopy(vector)
            else:
                self._vector = vector

    @abc.abstractmethod
    def build_vector(self):
        raise NotImplementedError

    @property
    def vector_shape(self):
        return self.build_vector().shape

    @property
    def vector(self):
        return copy.deepcopy(self._vector)

    @vector.setter
    def vector(self, value):
        self._vector = copy.deepcopy(value)


class BaseDynamics(abc.ABC):
    @abc.abstractmethod
    def step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):
    def __init__(self, integration_method="Euler"):
        self.integration_method = integration_method
        super().__init__()

    @abc.abstractmethod
    def dx(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, step_size, state, control):

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.dx, (0, step_size), state, args=(control,))

            state = sol.y[:, -1]  # save last timestep of integration solution
        elif self.integration_method == "Euler":
            state_dot = self.dx(0, state, control)
            state = state + step_size * state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return state


class BaseLinearODESolverDynamics(BaseODESolverDynamics):
    def __init__(self, integration_method="Euler"):
        self.A, self.B = self.gen_dynamics_matrices()
        super().__init__(integration_method=integration_method)

    @abc.abstractmethod
    def gen_dynamics_matrices(self):
        raise NotImplementedError

    def update_dynamics_matrices(self, state):
        pass

    def dx(self, t: float, state: np.ndarray, control: np.ndarray):
        self.update_dynamics_matrices(state)
        dx = np.matmul(self.A, state) + np.matmul(self.B, control)
        return dx
