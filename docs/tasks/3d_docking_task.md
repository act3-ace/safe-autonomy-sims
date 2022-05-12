# 3D CWH Docking

## Description
The 3D CWH Docking task objective is to maneuver a deputy
satellite from a random location into a docking region 
surrounding a chief satellite.

## Training
To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository 
is used. This module must be passed the necessary experiment config file at launch. 
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/docking/docking_3d.yml
```

## State
In the 3D spacecraft docking environment, the state of an active deputy spacecraft is expressed relative to the passive chief spacecraft in Hill's reference frame:
```math
F_H:=(O_H, \hat{i}_H, \hat{j}_H).
```

The origin of Hill's frame $O_H$ is located at the mass center of the chief, the unit vector $\hat{i}_H$ points away from the Earth along a line connecting the center of Earth to $O_H$, and the unit vector $\hat{j}_H$ is aligned with the orbital velocity vector of the chief.

The state of the deputy is defined as:
```math
\boldsymbol{x} = [x, y, z, \dot{x}, \dot{y},\dot{z}]^T \in \mathcal{X} \subset \mathbb{R}^{6}
```
where 
```math
\boldsymbol{r} = x \hat{i}_H + y \hat{j}_H + z
```
is the position vector and 
```math
\boldsymbol{v} = \dot{x}\hat{i}_H + \dot{y} \hat{j}_H + \dot{z}
```
is the velocity vector of the deputy in Hill's Frame. 

The control for the system is defined by
```math
\boldsymbol{u} = [F_x,F_y, F_z]^T=[u_{1},u_{2},u_3]^T \in \mathcal{U} \subset \mathbb{R}^3
```

![Hill's Reference Frame](docs/images/HillsFrame3.png)
*Hill's reference frame centered on a chief spacecraft and used to describe the relative motion of a deputy spacecraft conducting proximity operations (not to scale).*

## Dynamics
A first order approximation of the relative motion dynamics between the deputy and chief spacecraft is given by Clohessy-Wiltshire equations:
```math
\ddot{x} = 2n\dot{y} +3n^2x+ \frac{F_x}{m} \\
\ddot{y} =-2n\dot{x} + \frac{F_y}{m} \\
\ddot{z}  = -n^2z+ \frac{F_z}{m}
```
where $n$ is spacecraft mean motion and $m$ is the mass of the deputy. 


## Success Criteria
The deputy is considered successfully docked when its distance to the chief is less than a desired distance $\rho_d$.  
```math
\varphi_{\text{docking}}: (\Vert {\boldsymbol{r}_{\rm H}}\Vert \leq  \rho_d)
```
where
```math
\Vert {\boldsymbol{r}_{\rm H}}\Vert =(x^2+y^2+z^2)^{1/2}
```

## Safety Constraint
The RL agent must learn to dock while adhering to a dynamic velocity safety constraint that restricts the relative velocity of the deputy to velocity limit that decreases as it approaches the chief. The system is defined to be **safe** if it obeys the following safety constraint for all time:
```math
\varphi_{safety} :=\Vert \boldsymbol{v}_{\rm H} \Vert \leq \nu_0 + \nu_1 \Vert \boldsymbol{r}_{\rm H} \Vert 
```
where, 
```math
\nu_0, \nu_1 \in \mathbb{R}_{\geq 0}
```
and
```math
\Vert {\boldsymbol{r}_{\rm H}}\Vert =(x^2+y^2+z^2)^{1/2}, \quad \Vert {\boldsymbol{v}_{\rm H}} \Vert =(\dot{x}^2+\dot{y}^2+\dot{z}^2)^{1/2}
```
The above safety constraint enacts a distance-dependent speed limit, with $\nu_0$ defining the maximum allowable docking speed and $\nu_1$ defining the rate at which deputy must slow down as it approaches the chief. The values $\nu_0 = 0.2$ m/s, and $\nu_1 = 2n$ ${\rm s^{-1}}$ are selected based on elliptical closed natural motion trajectories (eCNMT).
