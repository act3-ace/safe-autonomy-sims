from collections import OrderedDict

import numpy as np

from saferl.rta.base_rta import ConstraintModule, ExplicitSimplexModule


class DockingRTA(ExplicitSimplexModule):

    def __init__(self, x_vel_limit=10, y_vel_limit=10, T_backup=5, Nskip=1, N_checkall=5, v0=0.2, v1_coef=2, kappa=1, n=0.001027, **kwargs):
        super().__init__(**kwargs)
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

    def _setup_constraints(self):
        # TODO: Automate this?
        self.constraint_rel_vel = Constraint_rel_vel()
        self.constraint_x_vel = Constraint_x_vel()
        self.constraint_y_vel = Constraint_y_vel()
        return ['constraint_rel_vel', 'constraint_x_vel', 'constraint_y_vel']

    def _backup_control(self, action, observation):
        state_vec = self._get_state_vector(observation)

        pred_state = self._pred_state_vector(sim_state, step_size, control)

        if self.constraint_x_vel.h_x(pred_state) < 0:
            pred_state[2] = self.x_vel_limit * np.sign(pred_state[2])
        if self.constraint_y_vel.h_x(pred_state) < 0:
            pred_state[3] = self.y_vel_limit * np.sign(pred_state[3])
        if self.constraint_rel_vel.h_x(pred_state) < 0:
            rH = np.linalg.norm(pred_state[0:2])
            vH = np.linalg.norm(pred_state[2:4])
            vH_max = self.v1 * rH + self.v0
            pred_state[2:4] = pred_state[2:4] / vH * vH_max
        accel = (pred_state[2:4] - state_vec[2:4]) / step_size

        backup_action = (accel[0:2] - self.A[2:4] @ pred_state) * self.m

        return self.clip(backup_action)

    def _get_action_vector(self, action):
        action_vec = []  # TODO: Make more general
        for k, v in action.items():
            action_vec.extend(list(v))
        return np.ndarray(action_vec)

    def _get_action_dict(self, action, keys):
        action_dict = OrderedDict()  # TODO: Find a better way to construct dict
        for i in range(len(action)):
            action_dict[keys[i]] = action[i]
        return action_dict

    def _get_state_vector(self, observation):
        state_vector = []
        for k, v in observation.items():
            state_vector.extend(list(next(iter(v.values()))))
        return np.ndarray(state_vector)


class Constraint_rel_vel(ConstraintModule):

    def __init__(self):
        self.setup_params()

    def h_x(self, state_vec):
        return self.v0 + self.v1 * np.linalg.norm(state_vec[0:2]) - np.linalg.norm(state_vec[2:4])

    def grad(self, state_vec):
        Hs = np.array([[2 * self.v1**2, 0, 0, 0], [0, 2 * self.v1**2, 0, 0], [0, 0, -2, 0], [0, 0, 0, -2]])

        ghs = Hs @ state_vec
        ghs[0] = ghs[0] + 2 * self.v1 * self.v0 * state_vec[0] / np.linalg.norm(state_vec[0:2])
        ghs[1] = ghs[1] + 2 * self.v1 * self.v0 * state_vec[1] / np.linalg.norm(state_vec[0:2])
        return ghs

    def alpha(self, x):
        return 0.05 * x + 0.1 * x**3


class Constraint_x_vel(ConstraintModule):

    def __init__(self):
        self.setup_params()

    def h_x(self, state_vec):
        return self.x_vel_limit**2 - state_vec[2]**2

    def grad(self, state_vec):
        gh = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -2, 0], [0, 0, 0, 0]])
        g = gh @ state_vec
        return g

    def alpha(self, x):
        return 0.0005 * x + 0.001 * x**3


class Constraint_y_vel(ConstraintModule):

    def __init__(self):
        self.setup_params()

    def h_x(self, state_vec):
        return self.y_vel_limit**2 - state_vec[3]**2

    def grad(self, state_vec):
        gh = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2]])
        g = gh @ state_vec
        return g

    def alpha(self, x):
        return 0.0005 * x + 0.001 * x**3
