"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------
"""

import numpy as np
from scipy.integrate import solve_ivp
import random

M = 12
N = 0.001027

def cwh_derivative(t, state, u):
    x, y, z, x_dot, y_dot, z_dot = state
    fx, fy, fz = u

    x_ddot = 2*N*y_dot + 3*(N**2)*x + fx/M
    y_ddot = -2*N*x_dot + fy/M
    z_ddot = -(N**2)*z + fz/M

    state_dot = np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot], dtype=float)
    return state_dot


def solve_cwh_traj(x0, u, t):
    state0 = np.array(x0, dtype=float)
    sol = solve_ivp(cwh_derivative, (0, t), state0, args=(u,))
    y = sol.y

    position_str = ', '.join([str(x) for x in y[0:3, -1]])
    velocity_str = ', '.join([str(x) for x in y[3:, -1]])

    print(f"- position == `[{position_str}]`\n- velocity == `[{velocity_str}]`")

def cwh_closed_form_solution(position, velocity, t):
    r_0 = np.array(position, dtype=float)
    v_0 = np.array(velocity, dtype=float)
    n = N

    phi_rr = np.array([
        [4-3*np.cos(n*t),       0, 0],
        [6*(np.sin(n*t) - n*t), 1, 0],
        [0,                     0, np.cos(n*t)],
    ], dtype=float)

    phi_rv = np.array([
        [1/n*np.sin(n*t),     2/n*(1-np.cos(n*t)),         0],
        [2/n*(np.cos(n*t)-1), 1/n*(4*np.sin(n*t) - 3*n*t), 0],
        [0,                   0,                            1/n*np.sin(n*t)],
    ], dtype=float)

    phi_vr = np.array([
        [3*n*np.sin(n*t),       0, 0],
        [6*n*(np.cos(n*t) - 1), 0, 0],
        [0,                     0, -n*np.sin(n*t)],
    ], dtype=float)

    phi_vv = np.array([
        [np.cos(n*t),    2*np.sin(n*t),   0],
        [-2*np.sin(n*t), 4*np.cos(n*t)-3, 0],
        [0,              0,               np.cos(n*t)],
    ], dtype=float)

    r_t = phi_rr@r_0 + phi_rv@v_0
    v_t = phi_vr@r_0 + phi_vv@v_0

    position_str = ', '.join([str(x) for x in r_t])
    velocity_str = ', '.join([str(x) for x in v_t])

    print("\n\nclosed form solution:")
    print(f"- position == `[{position_str}]`\n- velocity == `[{velocity_str}]`")

# position = [0, 0, 0]
# velocity = [0, 0, 0]
# u = [-1.1, -1, -1]
# t = 10

position = (np.random.rand(3) * 2000 - 1000).tolist()
velocity = (np.random.rand(3) * 20 - 10).tolist()
u = (np.random.rand(3) * 2 - 1).tolist()
t = random.randint(10, 100)

print("initial conditions")
print(f"- position = `[{', '.join([str(x) for x in position])}]`")
print(f"- velocity = `[{', '.join([str(x) for x in velocity])}]`")
print(f"- control = `[{', '.join([str(x) for x in u])}]`")

print(f"\nSolution @ t = {t}")
solve_cwh_traj(position + velocity, u, t)
if all([x == 0 for x in u]):
    cwh_closed_form_solution(position, velocity, t)