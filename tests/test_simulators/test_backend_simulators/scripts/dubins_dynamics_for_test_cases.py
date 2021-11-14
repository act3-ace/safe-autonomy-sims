import numpy as np
import scipy.integrate as integrate

def dubins_2d_numerical_integration(state_0, u, t):
    theta_dot, v_dot = u
    x0, y0, theta_0, v0 = state_0

    def theta_at_t(t):
        return ((theta_0 + t * theta_dot) + np.pi) % (2*np.pi) - np.pi

    def v_at_t(t):
        return np.clip(v0 + v_dot*t, 200, 400)

    def dx_dt(t):
        return v_at_t(t) * np.cos(theta_at_t(t))

    def dy_dt(t):
        return v_at_t(t) * np.sin(theta_at_t(t))

    xt = integrate.quad(dx_dt, 0, t)[0] + x0
    yt = integrate.quad(dy_dt, 0, t)[0] + y0
    theta_t = theta_at_t(t)
    v_t = v_at_t(t)

    print(f"- state == `[{xt}, {yt}, {theta_t}, {v_t}]`")


position = [0, 0, 0]
heading = 0
v = 200
t = 10
u = [np.pi/18, 10]

state = [position[0], position[1], heading, v]

dubins_2d_numerical_integration(state, u, t)
