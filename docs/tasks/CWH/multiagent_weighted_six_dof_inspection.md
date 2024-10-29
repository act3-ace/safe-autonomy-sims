---
title: Weighted Multiagent Six DOF Inspection
subtitle: Multiagent Six DOF Inspection With Illumination
authors:
date: 2023-10-29
---

# Six DOF Multiagent Spacecraft Inspection With Illumination and Weighted Inspection Points

## Motivation

Autonomous spacecraft inspection is foundational to sustained, complex spacecraft operations and uninterrupted delivery of space-based services. Inspection may enable investigation and characterization of space debris, or be the first step prior to approaching a prematurely defunct satellite to repair or refuel it. Additionally, it may be mission critical to obtain accurate information for characterizing vehicle condition of a cooperative spacecraft, such as in complex in-space assembly missions.

The common thread among all the potential applications is the need to gather information about the resident space object, which can be achieved by inspecting the entire surface of the body. In particular, this environment considers illumination requirements for optical inspection sensors.

## Training

An example training loop for this multiagent six DOF inspection environment can be launched using the `corl.train_rl` training endpoint. This module must be passed the necessary experiment config file at launch.
From the root of this repository, execute the following command:

```commandline
# from safe-autonomy-sims root
python -m corl.train_rl --cfg configs/multiagent-weighted-six-dof-inspection/experiment.yml
```

## Environment

In this inspection environment, the goal is for three deputy spacecraft, controlled by a RL agent, to navigate around and inspect the entire surface of a chief spacecraft.

The chief is covered in 100 inspection points that the agent must observe while they are illuminated by the moving sun. The points are weighted by priority, such that it is more important to inspect some points than others. A unit vector is used to indicate the direction of highest importance, where points are weighted based on their angular distance to this vector. All point weights add up to a value of one. The optimal policy will inspect points whose cumulative weight exceeds 0.95 within 2 revolutions of the sun while using as little fuel as possible. In this six DOF inspection environment, the agent controls its translational and rotational movement, requiring it to orient itself towards the chief for inspection. __Note: the policy selects a new action every 10 seconds__

| Space*         | Details |
|--------------|------|
| Action Space | (3,) |
| Observation Space | (32,) |
| Observation High | [$\infty$, $\infty$, $\infty$, $\infty$, 1, 1, 1, $\infty$, $\infty$, $\infty$, $\infty$, 1, 1, 1, $\infty$, $\infty$, $\infty$, $2\pi$, $2\pi$, $2\pi$, $2\pi$, 1, $2\pi$, 100, 1, 1, 1, 1, 1, 1, 1, 1] |
| Observation Low | [-$\infty$, -$\infty$, -$\infty$, -$\infty$, -1, -1, -1, -$\infty$, -$\infty$, -$\infty$, -$\infty$, -1, -1, -1, -$\infty$, -$\infty$, -$\infty$, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0] |
\* for each agent

### Observation Space

At each timestep, each agent $i$ receives the observation, $o_i = [x, y, z, |pos|, |x|, |y|, |z|, v_x, v_y, v_z, |v|, |v_x|, |v_y|, |v_z|, \omega_{x}, \omega_{y}, \omega_{z}, \theta_{cam}, \theta_{x}, \theta_{y}, \theta_{z}, f, \theta_{sun}, n, x_{ups}, y_{ups}, z_{ups}, x_{pv}, y_{pv}, z_{pv}, w_{points}, p_o]$, for deputy $i$ where:

* $x, y,$ and $z$ represent the deputy's position relative to the chief,
    * Normalized using a Gaussian distribution: $\mu=0m, \sigma=100m$,
* $|pos|, |x|, |y|, |z|$ is the magnorm representation of the deputy's position relative to the chief
    * Normalized using a Gaussian distribution: $\mu=0m, \sigma=[175, 1, 1, 1,]m$
* $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity relative to the chief,
    * Normalized using a Gaussian distribution: $\mu=0m/s, \sigma=0.5m/s$,
* $|v|, |v_x|, |v_y|, |v_z|$ is the magnorm representation of the deputy's velocity relative to the chief
    * Normalized using a Gaussian distribution: $\mu=0m/s, \sigma=[0.866, 1, 1, 1,]m/s$
* $\omega_{x}, \omega_{y}, \omega_{z}$ are the components of the deputy's angular velocity
    * Normalized using a Gaussian distribution: $\mu=0m/s, \sigma=[0.05, 0.05, 0.05]rad/s$
* $\theta_{cam}$ is the camera's orientation in Hill's frame
* $\theta_{x}, \theta_{y}, \theta_{z}$ are the deputy axis coordinates in Hill's frame
* $f$ is the dot-product between the camera orientation vector and the relative position between the deputy and the chief. This value is 1 when the camera is pointing at the chief.  
* $\theta_{sun}$ is the angle of the sun,
* $n$ is the number of points that have been inspected so far and,
    * Normalized using a Gaussian distribution: $\mu=0, \sigma=100$,
* $x_{ups}, y_{ups},$ and $z_{ups}$ are the unit vector elements pointing to the nearest large cluster of unispected points as determined by the *Uninspected Points Sensor*.
* $x_{pv}, y_{pv},$ and $z_{pv}$ are the unit vector elements pointing to the priority vector indicating point priority.
* $w_{points}$ is the cumulative weight of inpsected points
* $p_o$ is the dot-product between the uninspected points cluster given by the Uninspected Points Sensor and the deputy's position. This signals if the uninspected points are occluded for if the camera is facing the points but they are not being inspected, there is occlusion.

__Uninspected Points Sensor:__
This sensor activates every time new points are inspected, scanning for a new cluster of uninspected points. The sensor returns an array for a 3d unit vector, indicating the direction of the nearest cluster of uninspected points. A K-means clustering algorithm is used to identify the clusters of uninspected points. The clusters are initialized from the previously identified clusters and the total number of clusters is never more than $num\_uninspected\_points / 10$. This sensor helps guide each agent towards clusters of uninspected points.

### Action Space

The action space in this environment, which is equivalent to the control space, operates each deputy spacecraft's omni-directional thrusters and a reaction wheel abstraction controlling the deputy's x, y, and z moments with scalar values. These controls are able to move and rotate the spacecraft in any direction.

### Dynamics

The relative translational motion between each deputy and chief are linearized Clohessy-Wiltshire equations [[1]](#1), given by

$$
    \dot{\boldsymbol{x}} = A {\boldsymbol{x}} + B\boldsymbol{u},
$$

where the state $\boldsymbol{x}=[x,y,z,\dot{x},\dot{y},\dot{z}]^T \in \mathcal{X}=\mathbb{R}^6$, the control (same as actions) $\boldsymbol{u}= [F_x,F_y,F_z]^T \in \mathcal{U} = [-1N, 1N]^3$,

$$
    A =
\begin{bmatrix}
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    3n^2 & 0 & 0 & 0 & 2n & 0 \\
    0 & 0 & 0 & -2n & 0 & 0 \\
    0 & 0 & -n^2 & 0 & 0 & 0 \\
\end{bmatrix},
    B =
\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \frac{1}{m} & 0 & 0 \\
    0 & \frac{1}{m} & 0 \\
    0 & 0 & \frac{1}{m} \\
\end{bmatrix},
$$

and $n = 0.001027 rad/s$ is the mean motion constant.

The body frame rotational motion state transition of each spacecraft given its quaternion orientation and angular velocity $[q_1, q_2, q_3, q_4, \omega_x, \omega_y, \omega_z]$ is defined by

$$
\begin{bmatrix}
    \dot{q_1} \\
    \dot{q_2} \\
    \dot{q_3} \\
    \dot{q_4} \\
    \dot{\omega_x} \\
    \dot{\omega_y} \\
    \dot{\omega_z} \\
\end{bmatrix} =

\begin{bmatrix}
    \frac{1}{2}(q_4\omega_x - q_3\omega_y + q_2\omega_z) \\
    \frac{1}{2}(q_3\omega_x + q_4\omega_y - q_1\omega_z) \\
    \frac{1}{2}(-q_2\omega_x + q_1\omega_y + q_4\omega_z) \\
    \frac{1}{2}(-q_1\omega_x - q_2\omega_y - q_3\omega_z) \\
    J_1^{-1}((J_2 - J_3)\omega_y\omega_z) \\
    J_2^{-1}((J_3 - J_1)\omega_x\omega_z) \\
    J_3^{-1}((J_1 - J_2)\omega_x\omega_y) \\
\end{bmatrix}
$$

where

$$
J =
\begin{bmatrix}
    0.0573 & 0.0 & 0.0 \\
    0.0 & 0.0573 & 0.0 \\
    0.0 & 0.0 & 0.0573
\end{bmatrix}
$$

is an inertial matrix.

### Reward Function

We use a mix of sparse and dense rewards to define the desired behavior. These are described in more detail below. Dense rewards are computed at every timestep, while sparse rewards are only applied when the conditions are met.

* `ObservedPointsReward` is a dense reward that rewards each agent +1.0 multiplied by the point weight for every new point inspected in a timestep, $r_t = 1.0(weight\_inspected\_points_t - weight\_inspected\_points_{t-1})$.
* `SafeInspectionSuccessReward` is a sparse reward that rewards each agent for successfully inspecting. A Free Flight Trajectory (FFT) is computed for one orbit following successful inspection to determine if the agent would crash once the episode is over, in which case the agent is punished. $r = 1$ if $weight\_inspected\_points_i \geq 0.95$ and $FFT_radius \geq crash\_region\_radius$, $r = -1$ if $weight\_inspected\_points_i \geq 0.95$ and $FFT_radius < crash\_region\_radius$, else 0.
* `InspectionCrashReward` is a sparse reward that punishes each agent for crashing with the chief spacecraft. $r = -1$ if $radius < crash\_region\_radius$, else 0.
* `MaxDistanceDoneReward` is a sparse reward that punishes each agent for moving too far from the chief spacecraft. $r = -1$ if $||pos_{chief} - pos_{deputy}|| > distance_{max}$, else 0.
* `LiveTimestepReward` is a dense reward that gives a small reward to each agent for each timestep it stays active up until a configurable limit, encouraging the agent to not immediately end the episode. $r_t = 0.001$ if $t < t_{max}$, else 0.
* `FacingChiefReward` is a dense gaussian decaying reward that rewards each agent for facing the chief. $r_t = 0.0005 * e^{-|\delta_t(f, 1)^2 / \epsilon|}$ where
    * $\delta_t(f, 1)$ is the difference between the $f$, dot-product between the camera orientation vector and the relative position between the deputy and the chief, and 1.
    * $\epsilon$ is the length of the reward curve for the exponential decay (configurable)
* `InspectionDeltaVReward` is a dense reward that assigns a cost to using the thrusters that can be thought of similar to a fuel cost, $r = -0.1||\boldsymbol{u}||$

### Initial Conditions

At the start of any episode, the state is randomly initialized with the following conditions:

* chief $(x,y,z)$ = $(0, 0, 0)$
* chief radius = $10 m$
* chief # of points = $100$
* each deputy's position $(x, y, z)$ is converted after randomly selecting the position in polar notation $(r, \phi, \psi)$ using a uniform distribution with
    * $r \in [50, 100] m$
    * $\psi \in [0, 2\pi] rad$
    * $\phi \in [-\pi/2, \pi/2] rad$
    * $x = r \cos{\psi} \cos{\phi}$
    * $y = r \sin{\psi} \cos{\phi}$
    * $z = r \sin{\phi}$
* each deputy's $(v_x, v_y, v_z)$ is converted after randomly selecting the velocity in polar notation $(r, \phi, \psi)$ using a Gaussian distribution with
    * $v \in [0, 0.3]$ m/s
    * $\psi \in [0, 2\pi] rad$
    * $\phi \in [-\pi/2, \pi/2] rad$
    * $v_x = v \cos{\psi} \cos{\phi}$
    * $v_y = v \sin{\psi} \cos{\phi}$
    * $v_z = v \sin{\phi}$
* each deputy's $(\omega_x, \omega_y, \omega_z)$ is sampled from a uniform distribution between $[-0.01, -0.01, -0.01]$ rad/s and $[0.01, 0.01, 0.01]$ rad/s
* Initial sun angle is randomly selected using a uniform distribution
    * $\theta_{sun} \in [0, 2\pi] rad$
    * If the deputy is initialized where it's sensor points within 60 degrees of the sun, its position is negated such that the sensor points away from the sun.

### Done Conditions

An episode will terminate if any of the following conditions are met:

* any agent exceeds a `max_distance = 800` meter radius away from the chief,
* any agent moves within a `crash_region_radius = 15` meter radius around the chief,
* the cumulative weight of inspected points exceeds 0.95, and/or
* the maximum number of timesteps, `max_timesteps = 1224`, is reached.

The episode is considered done and successful if and only if the cumulative weight of inspected points exceeds 0.95 while all deputies remain on a safe trajectory (not on a collision course with the chief).

## Related Works/Environments

There have been many successful attempts to use deep learning techniques for spacecraft control applications in recent years. Dunlap et al. demonstrated the effectiveness of a RL controller for spacecraft docking in tandem with Run-Time-Assurance (RTA) methods to ensure safety [[2]](#2). Gaudet et al. proposed an adaptive guidance system using reinforcement meta-learning for various applications including a Mars landing with random engine failure [[3]](#3). The authors demonstrate the effectiveness of their solution by outperforming a traditional energy-optimal closed-loop guidance algorithm developed by Battin [[4]](#4). Campbell et al. developed a deep learning structure using Convolutional Neural Networks (CNNs) to return the position of an observer based on a digital terrain map, meaning that the pre-trained network can be used for fast and efficient navigation based on image data [[5]](#5). Similarly, Furfaro et al. use a set of Convolutional Neural Networks and Recurrent Neural Networks (RNNs) to relate a sequence of images taken during a landing mission, and the appropriate thrust actions [[6]](#6).

Similarly, previous work has been done to solve the inspection problem using both learning-based and traditional methods. In a recent study by Lei et al., the authors use deep RL to solve the inspection problem using multiple 3-Degree-of-Freedom (DOF) agents, using hierarchical RL [[7]](#7). They split the inspection task into sub-problems: 1) a guidance problem, where the agents are assigned waypoints that will result in optimal coverage, and 2) a navigation problem, in which the agents perform the necessary thrusting maneuvers to visit the points generated in 1). The solutions to the two separate problems are then joined and deployed in unison. Building on this work, Aurand et al. developed a solution for the multi-agent inspection problem of a tumbling spacecraft, but approached this problem by considering collection of range data instead of visiting specific waypoints [[8]](#8). In a very similar application to this paper, Brandonisio et al. using a reinforcement learning based approach to map an uncooperative space object using a free-flying 3 DOF spacecraft [[9]](#9). While the authors consider the role of the sun in generating useful image data, they do so using fixed logic based on incidence angles, rather than an explicit technique such as the ray-tracing technique proposed here.

## Configuration Files

Written out below are the core configuration files necessary for recreating the environment as described above. These are the *Environment Config* found in `configs/multiagent-weighted-six-dof-inspection/environment.yml` and the *Agent Config* found in `configs/multiagent-weighted-six-dof-inspection/agent.yml`.

<details>
<summary>Environment Config</summary>

From `configs/multiagent-weighted-six-dof-inspection/environment.yml`:

```yaml
 "simulator": {
    "type": "InspectionSimulator",
    "config": {
      "inspection_points_map": {
        "chief": {
          "num_points": 100,
          "radius": 10,
          "points_algorithm": "fibonacci",
        },
      },
      "illumination_params":{
          "mean_motion" : 0.001027,
          "avg_rad_Earth2Sun": 150000000000,
          "light_properties" : {'ambient': [1, 1, 1], 'diffuse': [1, 1, 1], 'specular': [1, 1, 1]},
          "chief_properties" : {'ambient': [.1, 0, 0], 'diffuse': [0.7, 0, 0], 'specular': [1, 1, 1], 'shininess': 100, 'reflection': 0.5}, # [.1, 0, 0] = red, [0.753, 0.753, 0.753] = silver
          "resolution" : [200, 200],
          "focal_length" : 9.6e-3,
          "pixel_pitch" : 5.6e-3,
          "bin_ray_flag": True,
          "render_flag_3d": False,
          "render_flag_subplots": False,
      },
      "steps_until_update": 1500,  # Steps between updates to delta-v reward scale
      "delta_v_scale_bounds": [-0.1, -0.001],  # Lower and upper bounds for delta-v reward scale
      "delta_v_scale_step": -0.00005,  # Step value for delta-v reward scale
      "inspected_points_update_bounds": [0.8, 0.9],  # Bounds to increase/decrease delta-v reward scale by step
      "delta_v_updater_criteria": "score",
    },
  },
  "simulator_reset_parameters": {
    "priority_vector_azimuth_angle": {
      "type": "corl.libraries.parameters.UniformParameter",
      "config": {
        "name": "priority_vector_azimuth_angle",
        "units": "radians",
        "low": 0,
        "high": 6.283,
      }
    },
    "priority_vector_elevation_angle": {
      "type": "corl.libraries.parameters.UniformParameter",
      "config": {
        "name": "priority_vector_elevation_angle",
        "units": "radians",
        "low": -1.57,
        "high": 1.57,
      }
    },
    "additional_entities": {
      "chief": { 
        "platform": "cwh",
        "initializer": {
          "functor": "safe_autonomy_sims.simulators.initializers.cwh.PositionVelocityInitializer",
        },
        "config":{
          "x": 0,
          "y": 0,
          "z": 0,
          "x_dot": 0,
          "y_dot": 0,
          "z_dot": 0,
        }
      },
      "sun": { 
        "entity_class": "safe_autonomy_simulation.sims.inspection.sun.Sun",
        "config":{
          "theta": {
            "type": "corl.libraries.parameters.UniformParameter",
            "config": {
              "name": "sun_angle",
              "units": "radians",
              "low": 0,
              "high": 6.283,
            }
          },
        }
      }
    }
  },
  "platforms": "CWHSimulator_Platforms",
  "plugin_paths": ["safe_autonomy_sims.platforms", "safe_autonomy_sims.simulators"],
  "episode_parameter_provider": {
    "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
  },
  "dones": {
    "shared": [
      {
        "functor": "safe_autonomy_sims.dones.common_dones.CollisionDoneFunction",
        "config": { safety_constraint: 10 },
      },
      {
        "functor": "safe_autonomy_sims.dones.common_dones.MultiagentSuccessDoneFunction",
        "config": {
          "success_function_name": "SafeSuccessfulInspectionDoneFunction"
        },
      },
      {
        "functor": "safe_autonomy_sims.dones.common_dones.SetAllDoneFunction",
        "config": {},
      },
    ]
  }
```

</details>

<details>
<summary>Agent Config</summary>

From `configs/multiagent-weighted-six-dof-inspection/agent.yml`:

```yaml
"agent": "corl.agents.base_agent.TrainableBaseAgent"  # agent class
"config": {
    "frame_rate": 0.1,  # Hz
    # Agent platform parts
    "parts": [
        # Platform controllers (thrusters + reaction wheel)
        {"part": "RateController", "config": {"name": "X Thrust", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", "axis": 0, properties: {name: "x_thrust"}}},
        {"part": "RateController", "config": {"name": "Y Thrust", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", "axis": 1, properties: {name: "y_thrust"}}},
        {"part": "RateController", "config": {"name": "Z Thrust", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", "axis": 2, properties: {name: "z_thrust"}}},
        {"part": "RateController", "config": {"name": "X Moment", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.MomentProp", "axis": 3, properties: {name: "x_moment"}}},
        {"part": "RateController", "config": {"name": "Y Moment", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.MomentProp", "axis": 4, properties: {name: "y_moment"}}},
        {"part": "RateController", "config": {"name": "Z Moment", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.MomentProp", "axis": 5, properties: {name: "z_moment"}}},
        # Platform sensors
        {"part": "Sensor_Position"},
        {"part": "Sensor_Velocity"},
        {"part": "Sensor_SunAngle"},
        {"part": "Sensor_InspectedPoints", "config": {"inspector_entity_name": "blue0"}},
        {"part": "Sensor_UninspectedPoints", "config": {"inspector_entity_name": "blue0", "inspection_entity_name": "chief"}},
        {"part": "Sensor_EntityPosition", "config": {"name": "reference_position", "entity_name": "chief"}},
        {"part": "Sensor_EntityVelocity", "config": {"name": "reference_velocity", "entity_name": "chief"}},
        {"part": "Sensor_PriorityVector"},
        {"part": "Sensor_InspectedPointsScore", "config": {"inspector_entity_name": "blue0"}},
        {"part": "Sensor_Quaternion"},
        {"part": "Sensor_AngularVelocity"},
    ],
    "episode_parameter_provider": {
      "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
    },
    "simulator_reset_parameters": {
      "initializer": {
        # Agent initializer which sets agent initial state given a set of initial conditions in polar coordinates
        "functor": "safe_autonomy_sims.simulators.initializers.cwh.CWHSixDOFRadialInitializer",
      },
      "config": {
        # Initial condition parameters expected by the initializer and their sampling distributions

        # Agent platform initial position
        "radius": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "radius",
            "units": "meters",
            "low": 50,
            "high": 100,
          }
        },
        "azimuth_angle": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "azimuth_angle",
            "units": "radians",
            "low": 0,
            "high": 6.283,
          }
        },
        "elevation_angle": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "elevation_angle",
            "units": "radians",
            "low": -1.57,
            "high": 1.57,
          }
        },
        # Agent platform initial velocity
        "vel_mag": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "vel_mag",
            "units": "m/s",
            "low": 0,
            "high": 0.3,
          }
        },
        "vel_azimuth_angle": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "vel_azimuth_angle",
            "units": "radians",
            "low": 0,
            "high": 6.283,
          }
        },
        "vel_elevation_angle": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "vel_elevation_angle",
            "units": "radians",
            "low": -1.57,
            "high": 1.57,
          }
        },
        # Agent platform initial angular velocity
        "wx": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "wx",
            "units": "rad/s",
            "low": -0.01,
            "high": 0.01,
          }
        },
        "wy": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "wx",
            "units": "rad/s",
            "low": -0.01,
            "high": 0.01,
          }
        },
        "wz": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "wx",
            "units": "rad/s",
            "low": -0.01,
            "high": 0.01,
          }
        },
      }
    },
    "glues": [
        # CoRL glue configurations. Glues define the action and observation space
        {
            # X-Axis Thrust Glue (action space)
            "functor": "corl.glues.common.controller_glue.ControllerGlue",
            "config": {
                "controller": "X Thrust",
                "training_export_behavior": "EXCLUDE",
                "normalization": {
                  "enabled": False,
                }
            },
        },
        {
            # Y-Axis Thrust Glue (action space)
            "functor": "corl.glues.common.controller_glue.ControllerGlue",
            "config":{
                "controller": "Y Thrust",
                "training_export_behavior": "EXCLUDE",
                "normalization": {
                  "enabled": False,
                }
            }
        },
        {
            # Z-Axis Thrust Glue (action space)
            "functor": "corl.glues.common.controller_glue.ControllerGlue",
            "config":{
              "controller": "Z Thrust",
              "training_export_behavior": "EXCLUDE",
              "normalization": {
                "enabled": False,
              }
            }
        },
        {
            # X Moment Controller Glue (action space)
            "functor": "corl.glues.common.controller_glue.ControllerGlue",
            "config":{
              "controller": "X Moment",
              "training_export_behavior": "EXCLUDE",
              "normalization": {
                "enabled": False,
              }
            }
        },
        {
            # Y Moment Controller Glue (action space)
            "functor": "corl.glues.common.controller_glue.ControllerGlue",
            "config":{
              "controller": "Y Moment",
              "training_export_behavior": "EXCLUDE",
              "normalization": {
                "enabled": False,
              }
            }
        },
        {
            # Z Moment Controller Glue (action space)
            "functor": "corl.glues.common.controller_glue.ControllerGlue",
            "config":{
              "controller": "Z Moment",
              "training_export_behavior": "EXCLUDE",
              "normalization": {
                "enabled": False,
              }
            }
        },
        {
            # Position Sensor Glue (observation space)
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config": {
              "sensor": "Sensor_Position",
              "output_units": "m",
              "normalization": {
                "normalizer": "corl.libraries.normalization.StandardNormalNormalizer",
                "config": {
                  "mu": 0.0,
                  "sigma": [100, 100, 100],
                }
              }
            },
        },
        {
            # Velocity Sensor Glue (observation space)
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config": {
              "sensor": "Sensor_Velocity",
              "output_units": "m/s",
              "normalization": {
                "normalizer": "corl.libraries.normalization.StandardNormalNormalizer",
                "config": {
                  "mu": 0.0,
                  "sigma": [0.5, 0.5, 0.5],
              }
            },
          },
        },
        {
          # Inspected Points Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_InspectedPoints",
            "normalization": {
              "normalizer": "corl.libraries.normalization.StandardNormalNormalizer",
              "config": {
                "mu": 0.0,
                "sigma": [100.0],
              },
            },
          },
        },
        {
          # Uninspected Points Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_UninspectedPoints",
            "output_units": "m",
            "normalization": {
                "enabled": False
          ``},
          },
        },
        {
          # Sun Angle Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_SunAngle",
            "normalization": {
              "enabled": False
            },
          },
        },
        {
          # Priority Vector Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_PriorityVector",
            "normalization": {
              "enabled": False
            },
          },
        },
        {
          # Inspected Points Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_InspectedPointsScore",
            "normalization": {
                "enabled": False
            },
          },
        },
        {
          # Orientation Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_Quaternion",
            "normalization": {
                "enabled": False
            },
          },
        },
        {
          # Angular Velocity Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_AngularVelocity",
            "normalization": {
              "normalizer": "corl.libraries.normalization.StandardNormalNormalizer",
              "config": {
                "mu": 0.0,
                "sigma": [0.05, 0.05, 0.05],
              }
            }
          },
        },
    ],
    "dones": [
        {
            # Max distance from origin
            "functor": "safe_autonomy_sims.dones.cwh.common.MaxDistanceDoneFunction",
            "references": {
              "max_distance": "max_distance",
              "reference_position_sensor_name": "reference_position_sensor_name"
            },
        },
        {
            # Crash into chief entity
            "functor": "safe_autonomy_sims.dones.cwh.common.CrashDoneFunction",
            "references": {
              "crash_region_radius": "collision_radius",
              "reference_position_sensor_name": "reference_position_sensor_name",
              "reference_velocity_sensor_name": "reference_velocity_sensor_name",
            },
        },
        {
            # Success (inspected all points without crashing)
            "functor": "safe_autonomy_sims.dones.cwh.inspection_dones.SafeSuccessfulInspectionDoneFunction",
            "config":{
              "weight_threshold": 0.95
            },
            "references": {
              "crash_region_radius": "collision_radius",
              "mean_motion": "mean_motion",
            },
        },
        {
            # Crash after inspecting all points (FFT)
            "functor": "safe_autonomy_sims.dones.cwh.inspection_dones.CrashAfterSuccessfulInspectionDoneFunction",
            "config":{
              "weight_threshold": 0.95
            },
            "references": {
              "crash_region_radius": "collision_radius",
              "mean_motion": "mean_motion",
            },
        },
    ],
    "rewards": [
        {
            # reward = number of newly observed points
            "name": "ObservedPointsReward",
            "functor": "safe_autonomy_sims.rewards.cwh.inspection_rewards.ObservedPointsReward",
            "config": {
                "scale": 1.0,
                "weighted_priority": True
            }
        },
        {
            # reward = scale (if all points are inspected)
            "name": "SafeInspectionSuccessReward",
            "functor": "safe_autonomy_sims.rewards.cwh.inspection_rewards.SafeInspectionSuccessReward",
            "config": {
                "scale": 1.0,
                "crash_scale": -1.0,
                "weight_threshold": 0.95,
            },
            "references": {
              "crash_region_radius": "collision_radius",
              "mean_motion": "mean_motion",
            },
        },
        {
            # reward = scale (if crash occurs)
            "name": "InspectionCrashReward",
            "functor": "safe_autonomy_sims.rewards.cwh.inspection_rewards.InspectionCrashReward",
            "config": {
                "scale": -1.0,
            },
            "references": {
                "crash_region_radius": "collision_radius",
                "reference_position_sensor_name": "reference_position_sensor_name",
                "reference_velocity_sensor_name": "reference_velocity_sensor_name",
            },
        },
        {
            "name": "InspectionDeltaVReward",
            # See delta-v reward scale parameters in env config
            "functor": "safe_autonomy_sims.rewards.cwh.inspection_rewards.InspectionDeltaVReward",
            "config": {
                "mode": "scale",
            },
            "references": {
                "step_size": "step_size",
                "mass": "mass",
            },
        },
    ],
    "reference_store": !include configs/multiagent-weighted-six-dof-inspection/parameters.yml
}

```

</details>
</br>

## References

<a id="1">[1]</a>
Clohessy, W., and Wiltshire, R., “Terminal Guidance System for Satellite Rendezvous,” *Journal of the Aerospace Sciences*, Vol. 27, No. 9, 1960, pp. 653–658.

<a id="2">[2]</a>
Dunlap, K., Mote, M., Delsing, K., and Hobbs, K. L., “Run Time Assured Reinforcement Learning for Safe Satellite
Docking,” *Journal of Aerospace Information Systems*, Vol. 20, No. 1, 2023, pp. 25–36. [https://doi.org/10.2514/1.I011126](https://doi.org/10.2514/1.I011126).

<a id="3">[3]</a>
Gaudet, B., Linares, R., and Furfaro, R., “Adaptive Guidance and Integrated Navigation with Reinforcement Meta-Learning,”
*CoRR*, Vol. abs/1904.09865, 2019. URL [http://arxiv.org/abs/1904.09865](http://arxiv.org/abs/1904.09865).

<a id="4">[4]</a>
Battin, R. H., “An introduction to the mathematics and methods of astrodynamics,” 1987.

<a id="5">[5]</a>
Campbell, T., Furfaro, R., Linares, R., and Gaylor, D., “A Deep Learning Approach For Optical Autonomous Planetary Relative
Terrain Navigation,” 2017.

<a id="6">[6]</a>
Furfaro, R., Bloise, I., Orlandelli, M., Di Lizia, P., Topputo, F., and Linares, R., “Deep Learning for Autonomous Lunar
Landing,” 2018.

<a id="7">[7]</a>
Lei, H. H., Shubert, M., Damron, N., Lang, K., and Phillips, S., “Deep reinforcement Learning for Multi-agent Autonomous
Satellite Inspection,” *AAS Guidance Navigation and Control Conference*, 2022.

<a id="8">[8]</a>
Aurand, J., Lei, H., Cutlip, S., Lang, K., and Phillips, S., “Exposure-Based Multi-Agent Inspection of a Tumbling Target Using
Deep Reinforcement Learning,” *AAS Guidance Navigation and Control Conference*, 2023.

<a id="9">[9]</a>
Brandonisio, A., Lavagna, M., and Guzzetti, D., “Reinforcement Learning for Uncooperative Space Objects Smart Imaging
Path-Planning,” The *Journal of the Astronautical Sciences*, Vol. 68, No. 4, 2021, pp. 1145–1169. [https://doi.org/10.1007/s40295-021-00288-7](https://doi.org/10.1007/s40295-021-00288-7).
