import abc
import math
from operator import pos

import numpy as np
from scipy.spatial.transform import Rotation

from saferl_sim.base_models.platforms import (
    BaseODESolverDynamics,
    BasePlatform,
)


class BaseDubinsPlatform(BasePlatform):

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
    @abc.abstractmethod
    def acceleration(self):
        raise NotImplemented

    @property
    def velocity(self):
        velocity = np.array(
            [
                self.v * math.cos(self.heading) * math.cos(self.gamma),
                self.v * math.sin(self.heading) * math.cos(self.gamma),
                -1 * self.v * math.sin(self.gamma),
            ],
            dtype=np.float32,
        )
        return velocity

    @property
    def orientation(self):
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])


class Dubins2dPlatform(BaseDubinsPlatform):

    def __init__(self, name, integration_method="RK45"):

        state_min = np.array([-np.inf, -np.inf, -np.inf, 200], dtype=np.float32)
        state_max = np.array([np.inf, np.inf, np.inf, 400], dtype=np.float32)
        angle_wrap_centers = np.array([None, None, 0, None], dtype=np.float32)

        control_default = np.zeros((2,))
        control_min = np.array([-np.deg2rad(10), -96.5])
        control_max = np.array([np.deg2rad(10), 96.5])
        control_map = {
            'heading_rate': 0,
            'acceleration': 1,
        }
        
        dynamics = Dubins2dDynamics(state_min=state_min, state_max=state_max, angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method)

        super().__init__(name, dynamics, control_default=control_default, 
            control_min=control_min, control_max=control_max, control_map=control_map)

    def reset(self, state=None, position=None, heading=0, v=200):
        super().reset(state=state, position=position, heading=heading, v=v)

    def build_state(self, position=None, heading=0, v=200):

        if position is None:
            position = [0, 0, 0]

        return np.array([position[0], position[1], heading, v], dtype=np.float32)

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

    @property
    def acceleration(self):
        acc = self.state_dot[3]
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc


class Dubins2dDynamics(BaseODESolverDynamics):

    def _compute_state_dot(self, t, state_vec, control):
        _, _, heading, v = state_vec
        rudder, throttle = control

        x_dot = v * math.cos(heading)
        y_dot = v * math.sin(heading)
        heading_dot = rudder
        v_dot = throttle

        state_dot = np.array([x_dot, y_dot, heading_dot, v_dot], dtype=np.float32)

        return state_dot


"""
3D Dubins Implementation
"""


class Dubins3dPlatform(BaseDubinsPlatform):

    def __init__(self, name, integration_method='RK45'):

        state_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi/9, -np.pi/3, 200], dtype=np.float32)
        state_max = np.array([np.inf, np.inf, np.inf, np.inf, np.pi/9, np.pi/3, 400], dtype=np.float32)
        angle_wrap_centers = np.array([None, None, None, 0, 0, 0, None], dtype=np.float32)

        control_default = np.zeros((3,))
        control_min = np.array([-np.deg2rad(10), -np.deg2rad(5), -96.5])
        control_max = np.array([np.deg2rad(10), np.deg2rad(5), 96.5])
        control_map = {
            'gamma_rate': 0,
            'roll_rate': 1,
            'acceleration': 2,
        }
        
        dynamics = Dubins3dDynamics(state_min=state_min, state_max=state_max, angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method)

        super().__init__(name, dynamics, control_default=control_default, 
            control_min=control_min, control_max=control_max, control_map=control_map)

    def reset(self, state=None, position=None, heading=0, gamma=0, roll=0, v=200):
        if position is None:
            position = [0, 0, 0]

        super().reset(state=state, position=position, heading=heading, gamma=gamma, roll=roll, v=v)

    def build_state(self, position, heading, gamma, roll, v):
        assert isinstance(position, list) and len(position) == 3, "position must be a list of length 3"

        return np.array(position + [heading, gamma, roll, v], dtype=np.float32)

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
        return self._state[2]

    @z.setter
    def z(self, value):
        self._state[2] = value

    @property
    def heading(self):
        return self._state[3]

    @heading.setter
    def heading(self, value):
        self._state[3] = value

    @property
    def gamma(self):
        return self._state[4]

    @gamma.setter
    def gamma(self, value):
        self._state[4] = value

    @property
    def roll(self):
        return self._state[5]

    @roll.setter
    def roll(self, value):
        self._state[5] = value

    @property
    def v(self):
        return self._state[6]

    @v.setter
    def v(self, value):
        self._state[6] = value

    @property
    def position(self):
        position = self._state[0:3].copy()
        return position

    @property
    def orientation(self):
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])

    @property
    def acceleration(self):
        acc = self.state_dot[6]
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc


class Dubins3dDynamics(BaseODESolverDynamics):

    def __init__(self, g=32.17, **kwargs):
        self.g = g
        super().__init__(**kwargs)

    def _compute_state_dot(self, t, state_vec, control):
        x, y, z, heading, gamma, roll, v = state_vec

        elevator, ailerons, throttle = control

        x_dot = v * math.cos(heading) * math.cos(gamma)
        y_dot = v * math.sin(heading) * math.cos(gamma)
        z_dot = -1 * v * math.sin(gamma)

        gamma_dot = elevator
        roll_dot = ailerons
        heading_dot = (self.g / v) * math.tan(roll)  # g = 32.17 ft/s^2
        v_dot = throttle

        state_dot = np.array([x_dot, y_dot, z_dot, heading_dot, gamma_dot, roll_dot, v_dot], dtype=np.float32)

        return state_dot

if __name__ == "__main__":
    entity = Dubins2dPlatform(name="abc")
    print(entity.state)
    # action = [0.5, 0.75, 1]
    # action = np.array([0.5, 0.75, 1], dtype=np.float32)
    # action = {'heading_rate': 0.1, 'acceleration': 0} # after one step, x = 199.667, y = 9.992, v=200, heading=0.1
    action = {'heading_rate': 0.1, 'acceleration': 10} 
    # action = {'heading_rate': 0.1, 'acceleration': -20} # after one step, x = 199.667, y = 9.992, v=200, heading=0.1
    # action = {'thrust_x': 0.5, 'thrust_y':0.75, 'thrust_zzzz': 1}
    for i in range(5):
        entity.step(1, action)
        print(f'position={entity.position}, heading={entity.heading}, v={entity.v}, acceleration={entity.acceleration}')

    # entity = Dubins3dPlatform(name="abc")
    # print(entity.state)
    # # action = [0.5, 0.75, 1]
    # # action = np.array([0.5, 0.75, 1], dtype=np.float32)
    # action = {'gamma_rate': 0.1, 'roll_rate': -0.05, 'acceleration': 10}
    # # action = {'gamma_rate': 0, 'roll_rate': 0, 'acceleration': -50} # test derivative state limit, after 1 step, position = [200, 0, 0]
    # # action = {'thrust_x': 0.5, 'thrust_y':0.75, 'thrust_zzzz': 1}
    # for i in range(5):
    #     entity.step(1, action)
    #     print(f'position={entity.position}, heading={entity.heading}, gamma={entity.gamma}, roll={entity.roll}, v={entity.v}, '
    #         f'acceleration={entity.acceleration}')