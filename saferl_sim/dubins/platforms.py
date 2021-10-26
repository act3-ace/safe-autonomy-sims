import abc
import math

import numpy as np
from scipy.spatial.transform import Rotation

from saferl_sim.base_models.platforms import (
    BaseODESolverDynamics,
    BasePlatform,
)


class BaseDubinsPlatform(BasePlatform):

    def generate_info(self):
        info = {
            "state": self.state.vector,
            "heading": self.heading,
            "v": self.v,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplemented

    @property
    def yaw(self):
        return self.heading

    @property
    def pitch(self):
        return self.gamma

    @property
    @abc.abstractmethod
    def roll(self):
        return self.state.roll

    @property
    @abc.abstractmethod
    def heading(self):
        return self.state.heading

    @property
    @abc.abstractmethod
    def gamma(self):
        return self.state.gamma

    @property
    def acceleration(self):
        # TODO Fix
        acc = self.current_control
        if self.v <= self.dynamics.v_min and acc < 0:
            acc = 0
        elif self.v >= self.dynamics.v_max and acc > 0:
            acc = 0
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc

    @property
    def velocity(self):
        velocity = np.array(
            [
                self.v * math.cos(self.heading) * math.cos(self.gamma),
                self.v * math.sin(self.heading) * math.cos(self.gamma),
                -1 * self.v * math.sin(self.gamma),
            ],
            dtype=np.float64,
        )
        return velocity

    @property
    def orientation(self):
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])


class Dubins2dPlatform(BaseDubinsPlatform):

    def __init__(self, name, integration_method="RK45"):

        state_min = np.array([-np.inf, -np.inf, -np.inf, 200], dtype=float)
        state_max = np.array([np.inf, np.inf, np.inf, 400], dtype=float)

        control_default = np.zeros((2,))
        control_min = np.array([-np.deg2rad(10), -96.5])
        control_max = np.array([np.deg2rad(10), 96.5])
        control_map = {
            'heading_rate': 0,
            'acceleration': 1,
        }
        
        dynamics = Dubins2dDynamics(state_min=state_min, state_max=state_max, integration_method=integration_method)

        super().__init__(name, dynamics, control_default=control_default, 
            control_min=control_min, control_max=control_max, control_map=control_map)

    def reset(self, state=None, position=None, heading=0, v=200, **kwargs):
        super().reset(state=state, position=position, heading=heading, v=v)

    def build_state(self, position=None, heading=0, v=200):

        if position is None:
            position = [0, 0, 0]

        return np.array([position[0], position[1], heading, v], dtype=np.float64)

    @property
    def x(self):
        return self._state[0]

    @x.setter
    def x(self, value):
        self._state[0] = value

    @property
    def y(self):
        return self._state[1]

    @y.setter
    def y(self, value):
        self._state[1] = value

    @property
    def z(self):
        return 0

    @property
    def heading(self):
        return self._state[2]

    @heading.setter
    def heading(self, value):
        self._state[2] = value

    @property
    def v(self):
        return self._state[3]

    @v.setter
    def v(self, value):
        self._state[3] = value

    @property
    def position(self):
        position = np.zeros((3, ))
        position[0:2] = self._state[0:2]
        return position

    @property
    def gamma(self):
        return 0

    @property
    def roll(self):
        return 0


class Dubins2dDynamics(BaseODESolverDynamics):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dx(self, t, state_vec, control):
        _, _, heading, v = state_vec
        rudder, throttle = control

        x_dot = v * math.cos(heading)
        y_dot = v * math.sin(heading)
        heading_dot = rudder
        v_dot = throttle

        dx_vec = np.array([x_dot, y_dot, heading_dot, v_dot], dtype=np.float64)

        return dx_vec


"""
3D Dubins Implementation
"""


class Dubins3dPlatform(BaseDubinsPlatform):

    def __init__(self, name, controller=None, v_min=10, v_max=100):

        dynamics = Dubins3dDynamics(v_min=v_min, v_max=v_max)
        state = Dubins3dState()

        super().__init__(name, dynamics, state, controller)

    def generate_info(self):
        info = {
            "gamma": self.gamma,
            "roll": self.roll,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret


class Dubins3dState:

    def build_vector(self, x=0, y=0, z=0, heading=0, gamma=0, roll=0, v=100, **kwargs):
        return np.array([x, y, z, heading, gamma, roll, v], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @x.setter
    def x(self, value):
        self._vector[0] = value

    @property
    def y(self):
        return self._vector[1]

    @y.setter
    def y(self, value):
        self._vector[1] = value

    @property
    def z(self):
        return self._vector[2]

    @z.setter
    def z(self, value):
        self._vector[2] = value

    @property
    def heading(self):
        return self._vector[3]

    @heading.setter
    def heading(self, value):
        self._vector[3] = value

    @property
    def gamma(self):
        return self._vector[4]

    @gamma.setter
    def gamma(self, value):
        self._vector[4] = value

    @property
    def roll(self):
        return self._vector[5]

    @roll.setter
    def roll(self, value):
        self._vector[5] = value

    @property
    def v(self):
        return self._vector[6]

    @v.setter
    def v(self, value):
        self._vector[6] = value

    @property
    def position(self):
        position = np.zeros((3, ))
        position[0:3] = self._vector[0:3]
        return position

    @property
    def orientation(self):
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])


class Dubins3dDynamics(BaseODESolverDynamics):

    def __init__(self, v_min=10, v_max=100, roll_min=-math.pi / 2, roll_max=math.pi / 2, g=32.17, *args, **kwargs):
        self.v_min = v_min
        self.v_max = v_max
        self.roll_min = roll_min
        self.roll_max = roll_max
        self.g = g

        super().__init__(*args, **kwargs)

    def step(self, step_size, state, control):
        state = super().step(step_size, state, control)

        # enforce velocity limits
        if state.v < self.v_min or state.v > self.v_max:
            state.v = max(min(state.v, self.v_max), self.v_min)

        # enforce roll limits
        if state.roll < self.roll_min or state.roll > self.roll_max:
            state.roll = max(min(state.roll, self.roll_max), self.roll_min)

        return state

    def dx(self, t, state_vec, control):
        x, y, z, heading, gamma, roll, v = state_vec

        elevator, ailerons, throttle = control

        # enforce velocity limits
        if v <= self.v_min and throttle < 0:
            throttle = 0
        elif v >= self.v_max and throttle > 0:
            throttle = 0

        # enforce roll limits
        if roll <= self.roll_min and ailerons < 0:
            ailerons = 0
        elif roll >= self.roll_max and ailerons > 0:
            ailerons = 0

        x_dot = v * math.cos(heading) * math.cos(gamma)
        y_dot = v * math.sin(heading) * math.cos(gamma)
        z_dot = -1 * v * math.sin(gamma)

        gamma_dot = elevator
        roll_dot = ailerons
        heading_dot = (self.g / v) * math.tan(roll)  # g = 32.17 ft/s^2
        v_dot = throttle

        dx_vec = np.array([x_dot, y_dot, z_dot, heading_dot, gamma_dot, roll_dot, v_dot], dtype=np.float64)

        return dx_vec

if __name__ == "__main__":
    entity = Dubins2dPlatform(name="abc")
    print(entity.state)
    # action = [0.5, 0.75, 1]
    # action = np.array([0.5, 0.75, 1], dtype=float)
    action = {'heading_rate': 0.1, 'acceleration': 0} # after one step, x = 199.667, y = 9.992, v=200, heading=0.1
    # action = {'heading_rate': 0.1, 'acceleration': -20} # after one step, x = 199.667, y = 9.992, v=200, heading=0.1
    # action = {'thrust_x': 0.5, 'thrust_y':0.75, 'thrust_zzzz': 1}
    for i in range(5):
        entity.step(1, action)
        print(entity.state)