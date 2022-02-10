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

def dubins_3d_heading_rate_to_roll(heading_rate, v):
    g = 32.17

    roll = np.arctan2((v * heading_rate), g)

    print(f"- roll = `{roll}`")
    print(f"- roll = `{roll/np.pi}*np.pi`")
    return roll

def dubins_3d_numerical_integration(state_0, u, t):
    g = 32.17

    gamma_rate, roll_rate, v_dot = u
    x0, y0, z0, theta_0, gamma_0, roll_0, v0 = state_0

    def gamma_at_t(t):
        return np.clip(gamma_0 + t*gamma_rate, -np.pi/9, np.pi/9)
    
    def roll_at_t(t):
        return np.clip(roll_0 + t*roll_rate, -np.pi/3, np.pi/3)
    
    def v_at_t(t):
        return np.clip(v0 + v_dot*t, 200, 400)

    def heading_rate_at_t(t):
        heading_rate = g / v_at_t(t) * np.tan(roll_at_t(t))
        return heading_rate

    def theta_at_t(t):
        return integrate.quad(heading_rate_at_t, 0, t)[0] + theta_0

    def dx_dt(t):
        return v_at_t(t) * np.cos(theta_at_t(t)) * np.cos(gamma_at_t(t))

    def dy_dt(t):
        return v_at_t(t) * np.sin(theta_at_t(t)) * np.cos(gamma_at_t(t))
    
    def dz_dt(t):
        return -v_at_t(t) * np.sin(gamma_at_t(t))

    xt = integrate.quad(dx_dt, 0, t)[0] + x0
    yt = integrate.quad(dy_dt, 0, t)[0] + y0
    zt = integrate.quad(dz_dt, 0, t)[0] + z0

    theta_t = theta_at_t(t)
    gamma_t = gamma_at_t(t)
    roll_t = roll_at_t(t)

    vt = v_at_t(t)

    print(f"- position == `[{xt}, {yt}, {zt}]`")
    print(f"- heading == `{theta_t}`")
    print(f"- gamma == `{gamma_t}`")
    print(f"- roll == `{roll_t}`")
    print(f"- v == `{vt}`")
    

# position = [0, 0, 0]
# heading = 0
# v = 200
# t = 10
# u = [np.pi/18, 10]

# state = [position[0], position[1], heading, v]

# dubins_2d_numerical_integration(state, u, t)

# heading_rate = -np.pi/54
# v = 373

# dubins_3d_heading_rate_to_roll(heading_rate, v)


position = [749.324, -653, 2832]
heading = 0
gamma = 0
roll = 0
v = 218
t = 10
u = [-np.pi/54, -np.pi/72, 25]

state = position + [heading, gamma, roll, v]
dubins_3d_numerical_integration(state, u, t)