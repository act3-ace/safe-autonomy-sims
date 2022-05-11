"""
This module implements 2D and 3D Aircraft entities with Dubins physics dynamics models.
"""

import abc
import math

import numpy as np
from scipy.spatial.transform import Rotation

from saferl.backend.base_models.entities import BaseEntity, BaseEntityValidator, BaseODESolverDynamics


class BaseDubinsAircraftValidator(BaseEntityValidator):
    """
    Base validator for Dubins Aircraft implementations.

    Parameters
    ----------
    x : float
        Initial x position
    y : float
        Initial y position
    z : float
        Initial z position
    heading : float
        Initial angle of velocity vector relative to x-axis. Right hand rule sign convention.
    v : float
        Initial velocity magnitude, aka speed, of dubins entity.

    Raises
    ------
    ValueError
        Improper list length for parameter 'position'
    """

    x: float = 0
    y: float = 0
    z: float = 0
    heading: float = 0
    v: float = 200


class BaseDubinsAircraft(BaseEntity):
    """
    Base interface for Dubins Entities.
    """

    def __init__(self, dynamics, control_default, control_min=-np.inf, control_max=np.inf, control_map=None, **kwargs):
        super().__init__(
            dynamics=dynamics,
            control_default=control_default,
            control_min=control_min,
            control_max=control_max,
            control_map=control_map,
            **kwargs
        )
        self.partner = None

    @classmethod
    def _get_config_validator(cls):
        return BaseDubinsAircraftValidator

    def __eq__(self, other):
        if isinstance(other, BaseDubinsAircraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.orientation.as_euler("zyx") == other.orientation.as_euler("zyx")).all()
            eq = eq and self.heading == other.heading
            return eq
        return False

    def register_partner(self, partner: BaseEntity):
        """
        Register another entity as this aircraft's partner. Defines line of communication between entities.

        Parameters
        ----------
        partner: BaseEntity
            Entity with line of communication to this aircraft.

        Returns
        -------
        None
        """
        self.partner = partner

    @property
    @abc.abstractmethod
    def v(self):
        """Get v, the velocity magnitude. aka speed."""
        raise NotImplementedError

    @property
    def yaw(self):
        """Get yaw. Equivalent to heading for Dubins model"""
        return self.heading

    @property
    def pitch(self):
        """Get pitch. Equivalent to gamma for Dubins model"""
        return self.gamma

    @property
    @abc.abstractmethod
    def roll(self):
        """Get roll."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def heading(self):
        """
        Get heading, the angle of velocity relative to the x-axis projected to the xy-plane.
        Right hand rule sign convention.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gamma(self):
        """
        Get gamma, aka flight path angle, the angle of the velocity vector relative to the xy-plane.
        Right hand rule sign convention.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def acceleration(self):
        """Get 3d acceleration vector"""
        raise NotImplementedError

    @property
    def velocity(self):
        """Get 3d velocity vector"""
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
        """
        Get orientation of entity.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation tranformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis (i.e. direction of nose) in the global frame.
            For Dubins, derived from yaw, pitch, roll attributes.
        """
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])


############################
# 2d Dubins Implementation #
############################


class Dubins2dAircraft(BaseDubinsAircraft):
    """
    2D Dubins Aircraft Simulation Entity.

    States
        x
        y
        heading
            range = [-pi, pi] rad
        v
            range = [200, 400] ft/s

    Controls
        heading_rate
            range = [-pi/18, pi/18] rad/s (i.e. +/- 10 deg/s)
        acceleration
            range = [-96.5, 96.5] ft/s^2

    Parameters
    ----------
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics
    kwargs
        Additional keyword args passed to BaseDubinsAircraftValidator
    """

    def __init__(self, integration_method="RK45", **kwargs):

        state_min = np.array([-np.inf, -np.inf, -np.inf, 200], dtype=np.float32)
        state_max = np.array([np.inf, np.inf, np.inf, 400], dtype=np.float32)
        angle_wrap_centers = np.array([None, None, 0, None], dtype=np.float32)

        control_default = np.zeros((2, ))
        control_min = np.array([-np.deg2rad(10), -96.5])
        control_max = np.array([np.deg2rad(10), 96.5])
        control_map = {
            'heading_rate': 0,
            'acceleration': 1,
        }

        dynamics = Dubins2dDynamics(
            state_min=state_min, state_max=state_max, angle_wrap_centers=angle_wrap_centers, integration_method=integration_method
        )

        super().__init__(
            dynamics, control_default=control_default, control_min=control_min, control_max=control_max, control_map=control_map, **kwargs
        )

    def __eq__(self, other):
        if isinstance(other, Dubins2dAircraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.acceleration == other.acceleration).all()
            eq = eq and (self.orientation.as_euler("zyx") == other.orientation.as_euler("zyx")).all()
            eq = eq and self.heading == other.heading
            eq = eq and self.roll == other.roll
            eq = eq and self.gamma == other.gamma
            return eq
        return False

    def _build_state(self):
        return np.array([self.config.x, self.config.y, self.config.heading, self.config.v], dtype=np.float32)

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
        """
        Get gamma, aka flight path angle, the angle of the velocity vector relative to the xy-plane.
        Right hand rule sign convention.
        Always 0 for Dubins 2D.
        """
        return 0

    @property
    def roll(self):
        """
        Get roll. Always 0 for Dubins 2D.
        """
        return 0

    @property
    def acceleration(self):
        acc = self.state_dot[3]
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc


class Dubins2dDynamics(BaseODESolverDynamics):
    """
    State transition implementation of non-linear 2D Dubins dynamics model.
    """

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        _, _, heading, v = state
        rudder, throttle = control

        x_dot = v * math.cos(heading)
        y_dot = v * math.sin(heading)
        heading_dot = rudder
        v_dot = throttle

        state_dot = np.array([x_dot, y_dot, heading_dot, v_dot], dtype=np.float32)

        return state_dot


############################
# 3D Dubins Implementation #
############################


class Dubins3dAircraftValidator(BaseDubinsAircraftValidator):
    """
    Validator for Dubins3dAircraft.

    Parameters
    ----------
    gamma : float
        Initial gamma value of Dubins3dAircraft in radians
    roll : float
        Initial roll value of Dubins3dAircraft in radians
    """
    gamma: float = 0
    roll: float = 0


class Dubins3dAircraft(BaseDubinsAircraft):
    """
    3D Dubins Aircraft Simulation Entity.

    States
        x
        y
        z
        heading
            range = [-pi, pi] rad
        gamma
            range = [-pi/9, pi/9] rad
        roll
            range = [-pi/3, pi/3] rad
        v
            range = [200, 400] ft/s

    Controls
        gamma_rate
            range = [-pi/18, pi/18] rad/s (i.e. +/- 10 deg/s)
        roll_rate
            range = [-pi/36, pi/36] rad/s (i.e. +/- 5 deg/s)
        acceleration
            range = [-96.5, 96.5] ft/s^2

    Parameters
    ----------
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics
    kwargs
        Additional keyword args passed to BaseDubinsAircraftValidator
    """

    def __init__(self, integration_method='RK45', **kwargs):

        state_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi / 9, -np.pi / 3, 200], dtype=np.float32)
        state_max = np.array([np.inf, np.inf, np.inf, np.inf, np.pi / 9, np.pi / 3, 400], dtype=np.float32)
        angle_wrap_centers = np.array([None, None, None, 0, 0, 0, None], dtype=np.float32)

        control_default = np.zeros((3, ))
        control_min = np.array([-np.deg2rad(10), -np.deg2rad(5), -96.5])
        control_max = np.array([np.deg2rad(10), np.deg2rad(5), 96.5])
        control_map = {
            'gamma_rate': 0,
            'roll_rate': 1,
            'acceleration': 2,
        }

        dynamics = Dubins3dDynamics(
            state_min=state_min, state_max=state_max, angle_wrap_centers=angle_wrap_centers, integration_method=integration_method
        )

        super().__init__(
            dynamics, control_default=control_default, control_min=control_min, control_max=control_max, control_map=control_map, **kwargs
        )

    def __eq__(self, other):
        if isinstance(other, Dubins3dAircraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.acceleration == other.acceleration).all()
            eq = eq and (self.orientation.as_euler("zyx") == other.orientation.as_euler("zyx")).all()
            eq = eq and self.heading == other.heading
            eq = eq and self.roll == other.roll
            eq = eq and self.gamma == other.gamma
            eq = eq and self.v == other.v
            return eq
        return False

    @classmethod
    def _get_config_validator(cls):
        return Dubins3dAircraftValidator

    def _build_state(self):
        return np.array(
            [self.config.x, self.config.y, self.config.z, self.config.heading, self.config.gamma, self.config.roll, self.config.v],
            dtype=np.float32
        )

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
    """
    State transition implementation of non-linear 2D Dubins dynamics model.

    Parameters
    ----------
    g : float
        gravitational acceleration constant if ft/s^2
    kwargs
        Additional keyword args passed to parent BaseODESolverDynamics constructor
    """

    def __init__(self, g=32.17, **kwargs):
        self.g = g
        super().__init__(**kwargs)

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        _, _, _, heading, gamma, roll, v = state

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
    # entity = Dubins2dAircraft(name="abc")
    # print(entity.state)
    # # action = [0.5, 0.75, 1]
    # # action = np.array([0.5, 0.75, 1], dtype=np.float32)
    # # action = {'heading_rate': 0.1, 'acceleration': 0} # after one step, x = 199.667, y = 9.992, v=200, heading=0.1
    # # action = {'heading_rate': 0.1, 'acceleration': 10}
    # action = {'heading_rate': 0.1, 'acceleration': -20} # after one step, x = 199.667, y = 9.992, v=200, heading=0.1
    # # action = {'thrust_x': 0.5, 'thrust_y':0.75, 'thrust_zzzz': 1}
    # for i in range(5):
    #     entity.step(1, action)
    #     print(f'position={entity.position}, heading={entity.heading}, v={entity.v}, acceleration={entity.acceleration}')

    entity = Dubins3dAircraft(name="abc")
    print(entity.state)
    # action = [0.5, 0.75, 1]
    # action = np.array([0.5, 0.75, 1], dtype=np.float32)
    action = {'gamma_rate': 0.1, 'roll_rate': -0.05, 'acceleration': 10}
    # action = {'gamma_rate': 0, 'roll_rate': 0, 'acceleration': -50} # tests derivative state limit, after 1 step, position = [200, 0, 0]
    # action = {'thrust_x': 0.5, 'thrust_y':0.75, 'thrust_zzzz': 1}
    for i in range(5):
        entity.step(1, action)
        print(
            f'position={entity.position}, heading={entity.heading}, gamma={entity.gamma}, roll={entity.roll}, v={entity.v}, '
            f'acceleration={entity.acceleration}'
        )
