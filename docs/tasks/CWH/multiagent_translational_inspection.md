---
title: Multiagent Translational Inspection
subtitle: Multi-Spacecraft Inspection With Illumination
authors:
    - Nate Hamilton
    - David van Wijk
date: 2023-10-29
---

# Translational Multi-Spacecraft Inspection With Illumination

## Motivation

Autonomous spacecraft inspection is foundational to sustained, complex spacecraft operations and uninterrupted delivery of space-based services. Inspection may enable investigation and characterization of space debris, or be the first step prior to approaching a prematurely defunct satellite to repair or refuel it. Additionally, it may be mission critical to obtain accurate information for characterizing vehicle condition of a cooperative spacecraft, such as in complex in-space assembly missions.

The common thread among all the potential applications is the need to gather information about the resident space object, which can be achieved by inspecting the entire surface of the body. As such, the problem addressed in this paper is one of inspecting the entire surface of a chief spacecraft, using simulated imaging sensors on a free-flying deputy spacecraft. In particular, this research considers illumination requirements for optical sensors.

## Training

An example training loop for this translational inspection environment can be launched using the `corl.train_rl` training endpoint. This module must be passed the necessary experiment config file at launch.
From the root of this repository, execute the following command:

```commandline
# from safe-autonomy-sims root
python -m corl.train_rl --cfg configs/multiagent-translational-inspection/experiment.yml
```

## Environment

In this inspection environment, the goal is for three deputy spacecrafts, controlled by a separate RL agent, to navigate around and inspect the entire surface of a single chief spacecraft.

The chief is covered in 100 inspection points that the agent must observe while they are illuminated by the moving sun. The optimal policy will inspect all 100 points within 2 revolutions of the sun while using as little fuel as possible. In this translational inspection environment, the agent only controls its translational motion and is always assumed to be pointing at the chief spacecraft. __Note: the policy selects a new action every 10 seconds__

| Space         | Details |
|--------------|------|
| Action Space | (3,) |
| Observation Space | (11,) |
| Observation High | [$\infty$, $\infty$, $\infty$, $\infty$, $\infty$, $\infty$, $2\pi$, 100, 1, 1, 1] |
| Observation Low | [-$\infty$, -$\infty$, -$\infty$, -$\infty$, -$\infty$, -$\infty$, 0, 0, -1, -1, -1] |

### Observation Space

At each timestep, each agent $i$ receives the observation, $o_i = [x, y, z, v_x, v_y, v_z, \theta_{sun}, n, x_{ups}, y_{ups}, z_{ups}]$ given deputy $i$, where:

* $x, y,$ and $z$ represent the deputy's position in the Hill's frame,
    * Normalized using a Gaussian distribution: $\mu=0m, \sigma=100m$,
* $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity in the Hill's frame,
    * Normalized using a Gaussian distribution: $\mu=0m/s, \sigma=0.5m/s$,
* $\theta_{sun}$ is the angle of the sun,
* $n$ is the number of points that have been inspected so far and,
    * Normalized using a Gaussian distribution: $\mu=0, \sigma=100$,
* $x_{ups}, y_{ups},$ and $z_{ups}$ are the unit vectors pointing to the nearest large cluster of unispected points as determined by the *Uninspected Points Sensor*.

__Uninspected Points Sensor:__
This sensor activates every time new points are inspected, scanning for a new cluster of uninspected points. The sensor returns an array for a 3d unit vector, indicating the direction of the nearest cluster of uninspected points. A K-means clustering algorithm is used to identify the clusters of uninspected points. The clusters are initialized from the previously identified clusters and the total number of clusters is never more than $num\_uninspected\_points / 10$. This sensor helps guide each agent towards clusters of uninspected points.

### Action Space

The action space in this environment, which is equivalent to the control space, operates the deputy spacecraft's omni-directional thrusters with scalar values. These thrusters are able to move the spacecraft in any direction. __In this environment, each inspecting deputy is always assumed to be rotated to point towards the chief spacecraft.__

### Dynamics

The relative motion between each deputy and chief are linearized Clohessy-Wiltshire equations [[1]](#1), given by

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

### Reward Function

We use a mix of sparse and dense rewards to define the desired behavior. These are described in more detail below. Dense rewards are computed at every timestep, while sparse rewards are only applied when the conditions are met.

* `ObservedPointsReward` is a dense reward that rewards each agent +0.01 for every new point inspected in a timestep, $r_t = 0.01(num\_inspected\_points_t - num\_inspected\_points_{t-1})$.
* `InspectionSuccessReward` is a sparse reward that rewards each agent for successfully inspecting, $r = 1$ if $num\_inspected\_points_i == 100$, else 0.
* `InspectionCrashOriginReward` is a sparse reward that punishes each agent for crashing with the chief spacecraft. $r = -1$ if $radius < crash\_region\_radius$, else 0.
* `InspectionRTAReward` is a sparse reward that assigns a punishment to each agent for using the RTA if included in the environment. $r = -0.01$ if $intervene$, else 0.
* `InspectionDeltaVReward` is a dense reward that assigns a cost to each agent for using the thrusters and can be thought of similar to a fuel cost, $r = -0.1||\boldsymbol{u}||$

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
* each deputy's velocity $(v_x, v_y, v_z)$ is converted after randomly selecting the velocity in polar notation $(r, \phi, \psi)$ using a Gaussian distribution with
    * $v \in [0, 0.8] * v_{\rm max}, \quad v_{\rm max} = 2nr$ m/s
    * $\psi \in [0, 2\pi] rad$
    * $\phi \in [-\pi/2, \pi/2] rad$
    * $v_x = v \cos{\psi} \cos{\phi}$
    * $v_y = v \sin{\psi} \cos{\phi}$
    * $v_z = v \sin{\phi}$
* Initial sun angle is randomly selected using a uniform distribution
    * $\theta_{sun} \in [0, 2\pi] rad$

### Done Conditions

An episode will terminate if any of the following conditions are met:

* any agent exceeds a `max_distance = 800` meter radius away from the chief,
* any agent moves within a `crash_region_radius = 10` meter radius around the chief,
* all 100 points around the chief have been inspected, and/or
* the maximum number of timesteps, `max_timesteps = 1224`, is reached.

The episode is considered done and successful if and only if all 100 points have been inspected.

## Related Works/Environments

There have been many successful attempts to use deep learning techniques for spacecraft control applications in recent years. Dunlap et al. demonstrated the effectiveness of a RL controller for spacecraft docking in tandem with Run-Time-Assurance (RTA) methods to ensure safety [[2]](#2). Gaudet et al. proposed an adaptive guidance system using reinforcement meta-learning for various applications including a Mars landing with random engine failure [[3]](#3). The authors demonstrate the effectiveness of their solution by outperforming a traditional energy-optimal closed-loop guidance algorithm developed by Battin [[4]](#4). Campbell et al. developed a deep learning structure using Convolutional Neural Networks (CNNs) to return the position of an observer based on a digital terrain map, meaning that the pre-trained network can be used for fast and efficient navigation based on image data [[5]](#5). Similarly, Furfaro et al. use a set of Convolutional Neural Networks and Recurrent Neural Networks (RNNs) to relate a sequence of images taken during a landing mission, and the appropriate thrust actions [[6]](#6).

Similarly, previous work has been done to solve the inspection problem using both learning-based and traditional methods. In a recent study by Lei et al., the authors use deep RL to solve the inspection problem using multiple 3-Degree-of-Freedom (DOF) agents, using hierarchical RL [[7]](#7). They split the inspection task into sub-problems: 1) a guidance problem, where the agents are assigned waypoints that will result in optimal coverage, and 2) a navigation problem, in which the agents perform the necessary thrusting maneuvers to visit the points generated in 1). The solutions to the two separate problems are then joined and deployed in unison. Building on this work, Aurand et al. developed a solution for the multi-agent inspection problem of a tumbling spacecraft, but approached this problem by considering collection of range data instead of visiting specific waypoints [[8]](#8). In a very similar application to this paper, Brandonisio et al. using a reinforcement learning based approach to map an uncooperative space object using a free-flying 3 DOF spacecraft [[9]](#9). While the authors consider the role of the sun in generating useful image data, they do so using fixed logic based on incidence angles, rather than an explicit technique such as the ray-tracing technique proposed here.

## Configuration Files

Written out below are the core configuration files necessary for recreating the environment as described above. These are the *Environment Config* found in `configs/multiagent-translational-inspection/environment.yml` and the *Agent Config* found in `configs/multiagent-translational-inspection/agent.yml`.

<details>
<summary>Environment Config</summary>

From `configs/multiagent-translational-inspection/environment.yml`:

```yaml
 "simulator": {
    "type": "InspectionSimulator",
    "config": {
      "inspection_points_map": {
        "chief": {
          "num_points": 100,
          "radius": 10,
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
    },
  },
  "simulator_reset_parameters": {
        "init_state": {
      "type": "safe_autonomy_sims.rta.rta_rejection_sampler.RejectionSampler",
      "config": {
        "initializer": {
          "functor": "safe_autonomy_sims.simulators.initializers.cwh.CWH3DRadialWithSunInitializer",
        },
        "rta": {
          "functor": "safe_autonomy_sims.rta.cwh.inspection_rta_1v1.RTAGlueCWHInspection1v1",
          "states": ["position", "velocity", "sun_angle"],
          "args": !include configs/translational-inspection/parameters.yml,
          "arg_map": {
            "step_size": "step_size",
            "collision_radius": "collision_radius",
            "v0": "velocity_threshold",
            "v0_distance": "collision_radius",
            "v1_coef": "vel_limit_slope",
            "n": "mean_motion",
            "r_max": "max_distance",
            "constraints": "constraints",
          }
        }
      },
      "wrapped":{
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
        "sun_angle": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "sun_angle",
            "units": "radians",
            "low": 0,
            "high": 6.28,
          }
        },
      }
    },
    "priority_vector_azimuth_angle": {
      "type": "corl.libraries.parameters.ConstantParameter",
      "config": {
        "name": "priority_vector_azimuth_angle",
        "units": "radians",
        "value": 0.0,
      }
    },
    "priority_vector_elevation_angle": {
      "type": "corl.libraries.parameters.ConstantParameter",
      "config": {
        "name": "priority_vector_elevation_angle",
        "units": "radians",
        "value": 0.0,
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
            "type": "safe_autonomy_sims.simulators.initializers.initializer.SimAttributeAccessor",
            "config": {
              "attribute_name": "init_state.sun_angle",
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
      }
    ]
  }
```

</details>

<details>
<summary>Agent Config</summary>

From `configs/multiagent-translational-inspection/agent.yml`:

```yaml
"agent": "corl.agents.base_agent.TrainableBaseAgent"
"config": {
    "frame_rate": 0.1,  # Hz
    # Agent platform part (controllers + sensors)
    "parts": [
        {"part": "RateController", "config": {"name": "X Thrust", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", "axis": 0, properties: {name: "x_thrust"}}},
        {"part": "RateController", "config": {"name": "Y Thrust", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", "axis": 1, properties: {name: "y_thrust"}}},
        {"part": "RateController", "config": {"name": "Z Thrust", "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", "axis": 2, properties: {name: "z_thrust"}}},
        {"part": "Sensor_Position"},
        {"part": "Sensor_Velocity"},
        {"part": "Sensor_SunAngle"},
        {"part": "Sensor_InspectedPoints", "config": {"inspector_entity_name": "blue0"}},
        {"part": "Sensor_UninspectedPoints", "config": {"inspector_entity_name": "blue0", "inspection_entity_name": "chief"}},
        {"part": "Sensor_EntityPosition", "config": {"name": "reference_position", "entity_name": "chief"}},
        {"part": "Sensor_EntityVelocity", "config": {"name": "reference_velocity", "entity_name": "chief"}},
    ],
    "episode_parameter_provider": {
      "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
    },
    "simulator_reset_parameters": {  # Default agent reset parameters
      "initializer": {
        # Agent initializer which sets agent initial state given a set of initial conditions
        # Rejects sampled initial conditions if they produce unsafe initial states
        "functor": "safe_autonomy_sims.rta.rta_rejection_sampler.RejectionSamplerInitializer",
        "config": {
          "states": ["position", "velocity"],
        }
      },
      "config": {
        "init_state": {
          "type": "safe_autonomy_sims.simulators.initializers.initializer.SimAttributeAccessor",
          "config": {
            "attribute_name": "init_state",
          }
        },
      }
    },
    "glues": [
      {
        # Runtime Assurance glue (action space)
        "functor": "safe_autonomy_sims.rta.cwh.inspection_rta_1v1.RTAGlueCWHInspection1v1",
        "config": {
          "training_export_behavior": "EXCLUDE",  # Exclude from action/obs space during training
          "state_observation_names": ["Obs_Sensor_Position", "Obs_Sensor_Velocity", "Obs_Sensor_SunAngle"],
          "enabled": False, # Set to True to turn RTA on
        },
        "references": {
          "step_size": "step_size",
          "collision_radius": "collision_radius",
          "v0": "velocity_threshold",
          "v0_distance": "collision_radius",
          "v1_coef": "vel_limit_slope",
          "n": "mean_motion",
          "r_max": "max_distance",
          "constraints": "constraints",
        },
        "wrapped": [
          {
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
              "functor": "corl.glues.common.controller_glue.ControllerGlue",
              "config":{
                "controller": "Z Thrust",
                "training_export_behavior": "EXCLUDE",
                "normalization": {
                  "enabled": False,
                }
              }
          }
        ],
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
          },
        },
      },
      {
          # Sun Angle Sensor Glue (observation space)
          "functor": "corl.glues.common.observe_sensor.ObserveSensor",
          "config": {
            "sensor": "Sensor_SunAngle",
            "output_units": "radians",
            "normalization": {
                "enabled": False
          },
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
            # Success (inspected all points)
            "functor": "safe_autonomy_sims.dones.cwh.inspection_dones.SuccessfulInspectionDoneFunction",
            "config":{},
        },
    ],
    "rewards": [
        {
            # reward = number of newly observed points
            "name": "ObservedPointsReward",
            "functor": "safe_autonomy_sims.rewards.cwh.inspection_rewards.ObservedPointsReward",
            "config": {
                "scale": 0.01
            }
        },
        {
            # reward = scale (if all points are inspected)
            "name": "InspectionSuccessReward",
            "functor": "safe_autonomy_sims.rewards.cwh.inspection_rewards.InspectionSuccessReward",
            "config": {
                "scale": 1.0,
            }
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
    "reference_store": !include configs/translational-inspection/parameters.yml
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
