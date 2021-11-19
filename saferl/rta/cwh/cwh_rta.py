from collections import OrderedDict

import numpy as np

from saferl.rta.base_rta import ConstraintModule, ExplicitSimplexModule


class DockingRTA(ExplicitSimplexModule):

    def __init__(
        self, x_vel_limit=10, y_vel_limit=10, T_backup=5, Nskip=1, N_checkall=5, v0=0.2, v1_coef=2, kappa=1, n=0.001027, m=12, **kwargs
    ):
        self.T_backup = T_backup
        self.Nskip = Nskip
        self.N_checkall = N_checkall

        self.v0 = v0
        self.v1_coef = v1_coef

        self.x_vel_limit = x_vel_limit
        self.y_vel_limit = y_vel_limit

        self.kappa = kappa

        self.n = n
        self.v1 = self.v1_coef * self.n

        self.m = m

        self.A, self.B = self._gen_dynamics_matrices()

        super().__init__(**kwargs)

    def _setup_constraints(self):
        # TODO: Automate this?
        self.constraint_rel_vel = Constraint_rel_vel(v0=self.v0, v1=self.v1)
        self.constraint_x_vel = ConstraintStateLimit(limit_val=self.x_vel_limit, state_index=2)
        self.constraint_y_vel = ConstraintStateLimit(limit_val=self.y_vel_limit, state_index=3)
        return ['constraint_rel_vel', 'constraint_x_vel', 'constraint_y_vel']

    def _backup_control(self, action, observation):
        state_vec = self._get_state_vector(observation)

        pred_state = self._pred_state_vector(action, state_vec, self.config.step_size)

        if self.constraint_x_vel.h_x(pred_state) < 0:
            pred_state[2] = self.x_vel_limit * np.sign(pred_state[2])
        if self.constraint_y_vel.h_x(pred_state) < 0:
            pred_state[3] = self.y_vel_limit * np.sign(pred_state[3])
        if self.constraint_rel_vel.h_x(pred_state) < 0:
            rH = np.linalg.norm(pred_state[0:2])
            vH = np.linalg.norm(pred_state[2:4])
            vH_max = self.v1 * rH + self.v0
            pred_state[2:4] = pred_state[2:4] / vH * vH_max
        accel = (pred_state[2:4] - state_vec[2:4]) / self.config.step_size

        backup_action = (accel[0:2] - self.A[2:4] @ pred_state) * self.m

        backup_action = np.clip(backup_action, -1, 1)

        backup_action_dict = self._get_action_dict(backup_action, ["x_thrust", "y_thrust"])

        # TODO: Remove this 2D hack
        backup_action_dict["z_thrust"] = 0

        return backup_action_dict

    def _gen_dynamics_matrices(self):
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

    def _pred_state_vector(self, action, state_vec, step_size):
        action_vec = self._get_action_vector(action)

        next_state = self.A @ state_vec + self.B @ action_vec
        return next_state

    def _get_action_vector(self, action):
        action_vec = []  # TODO: Make more general
        for _, v in action.items():
            action_vec.extend(list(v))
        return np.array(action_vec)

    def _get_action_dict(self, action, keys):
        action_dict = OrderedDict()  # TODO: Find a better way to construct dict
        for i in range(len(action)):
            action_dict[keys[i]] = action[i]
        return action_dict

    def _get_state_vector(self, observation):
        state_vector = []
        for _, v in observation.items():
            state_vector.extend(list(next(iter(v.values()))))
        return np.array(state_vector)


class Constraint_rel_vel(ConstraintModule):

    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1

    def h_x(self, state_vec):
        return self.v0 + self.v1 * (np.linalg.norm(state_vec[0:2]) - np.linalg.norm(state_vec[2:4]))

    def grad(self, state_vec):
        Hs = np.array([[2 * self.v1**2, 0, 0, 0], [0, 2 * self.v1**2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]])

        ghs = Hs @ state_vec
        ghs[0] = ghs[0] + 2 * self.v1 * self.v0 * state_vec[0] / np.linalg.norm(state_vec[0:2])
        ghs[1] = ghs[1] + 2 * self.v1 * self.v0 * state_vec[1] / np.linalg.norm(state_vec[0:2])
        return ghs

    def alpha(self, x):
        return 0.05 * x + 0.1 * x**3


class ConstraintStateLimit(ConstraintModule):

    def __init__(self, limit_val, state_index):
        self.limit_val = limit_val
        self.state_index = state_index

    def h_x(self, state_vec):
        return self.limit_val**2 - state_vec[self.state_index]**2

    def grad(self, state_vec):
        gh = np.zeros((state_vec.size, state_vec.size), dtype=float)
        gh[self.state_index, self.state_index] = -2
        g = gh @ state_vec
        return g

    def alpha(self, x):
        return 0.0005 * x + 0.001 * x**3
