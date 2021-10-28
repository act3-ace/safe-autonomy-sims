import copy

import numpy as np
from scipy.spatial.transform import Rotation
from pydantic import validator
import typing

from saferl_sim.base_models.entities import BaseEntity, BaseLinearODESolverDynamics, BaseEntityValidator


class CWHSpacecraftValidator(BaseEntityValidator):
    position: typing.List[float] = [0, 0, 0]
    velocity: typing.List[float] = [0, 0, 0]

    @validator("position", "velocity")
    def check_3d_vec_len(cls, v, field):
        """checks 3d vector field for length 3

        Parameters
        ----------
        v : typing.List[float]
            vector quantity to check
        field : string
            name of validator field

        Returns
        -------
        typing.List[float]
            v
        """
        if len(v) != 3:
            raise ValueError(f"{field.name} provided to CWHPlatformValidator is not length 3")
        return v


class CWHSpacecraft(BaseEntity):

    def __init__(self, m=12, n=0.001027, integration_method="RK45", **kwargs):
        dynamics = CWHDynamics(m=m, n=n, integration_method=integration_method)
        self._state = np.array([])

        control_map = {
            'thrust_x': 0,
            'thrust_y': 1,
            'thrust_z': 2,
        }

        super().__init__(dynamics, control_default=np.zeros((3, )), control_min=-1, control_max=1, control_map=control_map, **kwargs)

    @classmethod
    def get_config_validator(cls):
        return CWHSpacecraftValidator

    def _build_state(self):

        state = np.concatenate((self.config.position, self.config.velocity), dtype=np.float32)

        return state

    @property
    def x(self):
        return self._state[0]

    @property
    def y(self):
        return self._state[1]

    @property
    def z(self):
        return self._state[2]

    @property
    def x_dot(self):
        return self._state[3]

    @property
    def y_dot(self):
        return self._state[4]

    @property
    def z_dot(self):
        return self._state[5]

    @property
    def position(self):
        return self._state[0:3].copy()

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])

    @property
    def velocity(self):
        return self._state[3:6].copy()


class CWHDynamics(BaseLinearODESolverDynamics):

    def __init__(self, m=12, n=0.001027, **kwargs):
        self.m = m  # kg
        self.n = n  # rads/s

        super().__init__(**kwargs)

    def gen_dynamics_matrices(self):
        m = self.m
        n = self.n

        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * n**2, 0, 0, 0, 2 * n, 0],
                [0, 0, 0, -2 * n, 0, 0],
                [0, 0, -(n**2), 0, 0, 0],
            ],
            dtype=np.float32,
        )

        B = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1 / m, 0, 0],
                [0, 1 / m, 0],
                [0, 0, 1 / m],
            ],
            dtype=np.float32,
        )

        return A, B


if __name__ == "__main__":
    entity = CWHSpacecraft(name="abc")
    print(entity.state)
    # action = [0.5, 0.75, 1]
    # action = np.array([0.5, 0.75, 1], dtype=np.float32)
    action = {'thrust_x': 0.5, 'thrust_y': 0.75, 'thrust_z': 1}
    # action = {'thrust_x': 0.5, 'thrust_y':0.75, 'thrust_zzzz': 1}
    for i in range(5):
        entity.step(1, action)
        print(entity.state)
