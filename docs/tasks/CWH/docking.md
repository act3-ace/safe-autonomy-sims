---
title: Docking
subtitle: CWH 3D Spacecraft Docking
authors:
date: 2023-10-29
---

# CWH 3D Spacecraft Docking

## Motivation

Autonomous spacecraft control during proximity operations is foundational to sustained, complex spacecraft operations and uninterrupted delivery of space-based services. Spacecraft docking maneuvers require fine operation in high collision risk proximity environments. Limited ground communication opportunities and the increasing scale of spacecraft operations is reducing the feasibility of manual spacecraft proximity operations. Safe autonomous spacecraft proximity control allows space operations to continue without direct oversight under these ever more demanding conditions.

## Training

An example training loop for this docking environment can be launched using the `corl.train_rl` training endpoint. This module must be passed the necessary experiment config file at launch.
From the root of this repository, execute the following command:

```commandline
# from safe-autonomy-sims root
python -m corl.train_rl --cfg configs/docking/experiment.yml
```

## Environment

In this docking environment, the goal is for a single deputy spacecraft, controlled by a RL agent, to navigate towards and dock onto a chief spacecraft.

| Space         | Details |
|--------------|------|
| Action Space | (3,) |
| Observation Space | (8,) |
| Observation High | [$\infty$, $\infty$, $\infty$, $\infty$, $\infty$, $\infty$, $\infty$, $\infty$,] |
| Observation Low | [-$\infty$, -$\infty$, -$\infty$, -$\infty$, -$\infty$, -$\infty$, 0, 0,] |

### Observation Space

At each timestep, the agent receives the observation, $o = [x, y, z, v_x, v_y, v_z, s, v_{limit}]$, where:

* $x, y,$ and $z$ represent the deputy's position in the Hill's frame,
    * Normalized using a Gaussian distribution: $\mu=0m, \sigma=100m$,
* $v_x, v_y,$ and $v_z$ represent the deputy's directional velocity in the Hill's frame,
    * Normalized using a Gaussian distribution: $\mu=0m/s, \sigma=0.5m/s$,
* $s$ is the speed of the deputy,
* $v_{limit}$ is the safe velocity limit given by: $v_{max} + an(d_{chief} - r_{docking})$
    * $v_{max}$ is the maximum allowable velocity of the deputy within the docking region
    * $a$ is the slope of the linear velocity limit as a function of distance from the docking region
    * $n$ is the mean motion constant
    * $d_{chief}$ is the deputy's distance from the chief
    * $r_{docking}$ is the radius of the docking region

### Action Space

The action space in this environment, which is equivalent to the control space, operates the deputy spacecraft's omni-directional thrusters with scalar values. These thrusters are able to move the spacecraft in any direction.

### Dynamics

The relative motion between the deputy and chief are linearized Clohessy-Wiltshire equations [[1]](#1), given by

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

We use a mix of sparse and dense rewards to define the desired behavior. These are described in more detail below. Dense rewards are computed at every timestep, while sparse rewards are only applied when the conditions are met. All rewards can have a scale factor directly applied to them.

* `DockingDistanceExponentialChangeReward` is a dense reward that rewards the agent for approaching the chief, $r_t = c(e^{-ad_t} - e^{-ad_{t-1}})$.
    * $c$ is a scale factor
    * $a$ is the exponential coefficent (can be calculated from a pivot value)
    * $d_t$ is the distance from the chief at time $t$
* `DockingDeltaVReward` is a dense reward that rewards the agent for using the least amount of fuel possible (minimum change in velocity). $r_t = -((\delta{v} / m) + b)$ where
    * $\delta{v}$ is the change in velocity
    * $m$ is the mass of the deputy
    * $b$ is a tunable bia term
* `DockingVelocityConstraintReward` is a dense reward that punishes the agent for violating the velocity constraint. $r_t = v - v_{limit}$ if $v_{limit} < v$, else 0.
    * $v_{limit}$ is the safe velocity limit given by: $v_{max} + an(d_{chief} - r_{docking})$, described above
* `DockingSuccessReward` is a sparse reward that rewards the agent for successfully docking the deputy spacecraft onto the chief in the least amount of time. $r = 1 + (1 - (t/t_{max}))$.
    * $t$ is the current time
    * $t_{max}$ is the maximum episode length before timeout
* `DockingFailureReward` is a sparse reward that punishes the agent for failing to successfully dock the deputy onto the chief. $r = -1.0$ if the agent times out, crashes, or goes out of bounds.

### Initial Conditions

At the start of any episode, the state is randomly initialized with the following conditions:

* chief $(x,y,z)$ = $(0, 0, 0)$
* docking radius = $0.5 m$
* deputy position $(x, y, z)$ is converted after randomly selecting the position in polar notation $(r, \phi, \psi)$ using a uniform distribution with
    * $r \in [100, 150] m$
    * $\psi \in [0, 2\pi] rad$
    * $\phi \in [-\pi/2, \pi/2] rad$
    * $x = r \cos{\psi} \cos{\phi}$
    * $y = r \sin{\psi} \cos{\phi}$
    * $z = r \sin{\phi}$
* deputy $(v_x, v_y, v_z)$ is converted after randomly selecting the velocity in polar notation $(r, \phi, \psi)$ using a Gaussian distribution with
    * $v \in [0, 0.8]$ m/s
    * $\psi \in [0, 2\pi] rad$
    * $\phi \in [-\pi/2, \pi/2] rad$
    * $v_x = v \cos{\psi} \cos{\phi}$
    * $v_y = v \sin{\psi} \cos{\phi}$
    * $v_z = v \sin{\phi}$

### Done Conditions

An episode will terminate if any of the following conditions are met:

* the agent exceeds a `max_distance = 10000` meter radius away from the chief,
* the agent violates the velocity constraint within the docking region (crash),
* the maximum number of timesteps, `max_timesteps = 2000`, is reached
* the velocity limit penalty exceeds -5

The episode is considered done and successful if and only if the agent maneuvers the deputy within the docking region while maintaining a safe velocity.

## Configuration Files

Written out below are the core configuration files necessary for recreating the environment as described above. These are the *Environment Config* found in `configs/docking/environment.yml` and the *Agent Config* found in `configs/docking/agent.yml`.

<details>
<summary>Environment Config</summary>

From `configs/docking/environment.yml`:

```yaml
"simulator": {
    "type": "CWHSimulator",  # Registered CoRL simulator
    "config": {},
},
"simulator_reset_parameters": {
    # Environment reset parameters
    # These will override any default reset parameters defined in other configuration files
    "initializer": {
        # Agent initializer which sets agent initial state given a set of initial conditions in polar coordinates
        "functor": "safe_autonomy_sims.simulators.initializers.docking_initializer.Docking3DRadialInitializer",
        "config": {
            "threshold_distance": 0.5,
            "velocity_threshold": 0.2,
            "mean_motion": 0.001027,
            "slope": 2.0,
        }
    },
    "additional_entities": {
        # Additional simulation entities in the environment not controlled by an agent
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
        }
    }
},
"platforms": "CWHSimulator_Platforms",  # list of registered platform types allowed in the environment
"plugin_paths": ["safe_autonomy_sims.platforms", "safe_autonomy_sims.simulators"],  # python namespaces to search for registered CoRL plugins (platforms and simulators)
"episode_parameter_provider": {
    "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
},

```

</details>

<details>
<summary>Agent Config</summary>

From `configs/docking/agent.yml`:

```yaml
"agent": "corl.agents.base_agent.TrainableBaseAgent"  # agent class
"config": {
    "frame_rate": 1,  # Hz
    # Agent platform parts
    "parts": [
        {
          # X-Axis Thrust
          "part": "RateController", 
          "config": {
            "name": "X Thrust", 
            "axis": 0, 
            "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", 
            properties: {name: "x_thrust"}
          }
        },
        {
          # Y-Axis Thrust
          "part": "RateController", 
          "config": {
            "name": "Y Thrust", 
            "axis": 1, 
            "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", 
            properties: {name: "y_thrust"}
          }
        },
        {
          # Z-Axis Thrust
          "part": "RateController", 
          "config": {
            "name": "Z Thrust", 
            "axis": 2, 
            "property_class": "safe_autonomy_sims.platforms.cwh.cwh_properties.ThrustProp", 
            properties: {name: "z_thrust"}
          }
        },
        {"part": "Sensor_Position"},  # own position sensor
        {"part": "Sensor_Velocity"},  # own velocity sensor
        {"part": "Sensor_EntityPosition", "config": {"name": "reference_position", "entity_name": "chief"}}, # chief position sensor, ensure reference position sensor name consistent with dones and rewards
        {"part": "Sensor_EntityVelocity", "config": {"name": "reference_velocity", "entity_name": "chief"}}, # chief velocity sensor, ensure reference velocity sensor name consistent with dones and rewards
    ],
    "episode_parameter_provider": {
        "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
    },
    "simulator_reset_parameters": {  # Default agent reset parameters
      "initializer": {
        # Agent initializer which sets agent initial state given a set of initial conditions in polar coordinates
        "functor": "safe_autonomy_sims.simulators.initializers.cwh.Docking3DRadialInitializer",
        "config": {
            "threshold_distance": 0.5,
            "velocity_threshold": 0.2,
            "mean_motion": 0.001027,
            "slope": 2.0,
        }
      },
      "config":{
        # Initial condition parameters expected by the initializer and their sampling distributions

        # Agent platform initial position
        "radius": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "radius",
            "units": "meters",
            "low": 100,
            "high": 150,
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
        "vel_max_ratio": {
          "type": "corl.libraries.parameters.UniformParameter",
          "config": {
            "name": "vel_max_ratio",
            "low": 0,
            "high": 0.8,
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
          # Velocity Magnitude Wrapper Glue (observation space)
          # This glue wraps a velocity sensor glue and transforms the output into a magnitude
          "functor": "safe_autonomy_sims.glues.magnitude_glue.SimsMagnitudeGlue",
            "config": {
              "normalization": {
                "normalizer": "corl.libraries.normalization.StandardNormalNormalizer"
              },
            },
            "wrapped": {
                "functor": "corl.glues.common.observe_sensor.ObserveSensor",
                "config":{
                    "sensor": Sensor_Velocity,
                    "output_units": "m/s",
                },
            },
        },
        {
            # Velocity Constraint Glue (observation space)
            # Computes a dynamic velocity safety constraint based on the agent's position
            "functor": "safe_autonomy_sims.glues.vel_limit_glue.VelocityLimitGlue",
            "config":
              {
                "sensor": "Sensor_Position",
                "slope": 2.0,
              },
            "references": {
              "velocity_threshold": "velocity_threshold",
              "threshold_distance": "threshold_distance",
              "mean_motion": "mean_motion",
            },
        },
    ],
    "dones": [
        # CoRL done function configurations. These functions return a boolean value based on the episode state. If any of these functions are true, a done condition has been reach, the agent is removed from the environment, and the episode may end.
        {
            # Timeout done condition
            "functor": "safe_autonomy_sims.dones.common_dones.TimeoutDoneFunction",
            "config":{},
            "references": {
              "max_sim_time": "timeout",
            },
        },
        {
            # Out of bounds/max distance done condition
            "functor": "safe_autonomy_sims.dones.cwh.common.MaxDistanceDoneFunction",
            "config":{},
            "references": {
              "max_distance": "max_distance",
              "reference_position_sensor_name": "reference_position_sensor_name",
            },
        },
        {
            # Crash done condition
            "functor": "safe_autonomy_sims.dones.cwh.common.CrashDoneFunction",
            "config":{
              "velocity_constraint": {
                "velocity_threshold": 0.2,
                "threshold_distance": 0.5,
              }
            },
            "references": {
              "crash_region_radius": "docking_region_radius",
              "reference_position_sensor_name": "reference_position_sensor_name",
              "reference_velocity_sensor_name": "reference_velocity_sensor_name",
            }
        },
        {
            # Success done condition
            # True if the agent successfully docks the deputy spacecraft onto the chief.
            "functor": "safe_autonomy_sims.dones.cwh.docking_dones.SuccessfulDockingDoneFunction",
            "config":{},
            "references": {
              "docking_region_radius": "docking_region_radius",
              "velocity_threshold": "velocity_threshold",
              "threshold_distance": "threshold_distance",
              "mean_motion": "mean_motion",
              "lower_bound": "lower_bound",
              "reference_position_sensor_name": "reference_position_sensor_name",
              "reference_velocity_sensor_name": "reference_velocity_sensor_name",
            }
        },
        {
            # Velocity limit done condition
            # True if the velocity limit has been breached by > 5 m/s cumulatively during an episode.
            "name": DockingVelocityLimitSaturationDone,
            "functor": "safe_autonomy_sims.dones.cwh.common.TerminalRewardSaturationDoneFunction",
            "config":{
              "limit": -5,
              "bound": "lower",
              "reward_functor": safe_autonomy_sims.rewards.cwh.docking_rewards.DockingVelocityConstraintReward,
              "reward_config": {
                "scale": -0.01,
                "bias": -0.01,
                "velocity_threshold": 0.2,
                "threshold_distance": 0.5,
                "mean_motion": 0.001027,
                "lower_bound": False,
              }
            },
        },
    ],
    "rewards": [
        # CoRL reward functions. These functions return a scalar reward value to the agent based on the episode state.
        {
            # Distance change reward
            # Rewards the agent for moving closer to the chief.
            "name": "DockingDistanceExponentialChangeReward",
            "functor": "safe_autonomy_sims.rewards.cwh.docking_rewards.DockingDistanceExponentialChangeReward",
            "config": {
                "pivot": 100
            }
        },
      {
        # Delta V reward
        # Penalizes the agent for using fuel.
        "name": "DockingDeltaVReward",
        "functor": "safe_autonomy_sims.rewards.cwh.docking_rewards.DockingDeltaVReward",
        "config": {
          "scale": -0.01,
          "bias": 0.0,
          "mass": 12.0
        }
      },
      {
        # Velocity constraint reward
        # Penalizes the agent for violating the velocity constraint
        "name": "DockingVelocityConstraintReward",
        "functor": "safe_autonomy_sims.rewards.cwh.docking_rewards.DockingVelocityConstraintReward",
        "config": {
          "scale": -0.01,
          "bias": -0.01,
        },
        "references": {
          "velocity_threshold": "velocity_threshold",
          "threshold_distance": "threshold_distance",
          "mean_motion": "mean_motion",
          "lower_bound": "lower_bound",
          "reference_position_sensor_name": "reference_position_sensor_name",
          "reference_velocity_sensor_name": "reference_velocity_sensor_name",
        }
      },
        {
          # Docking success reward
          # Rewards the agent for successfully completing the docking task
          "name": "DockingSuccessReward",
          "functor": "safe_autonomy_sims.rewards.cwh.docking_rewards.DockingSuccessReward",
          "config": {
            "scale": 1.0,
          },
          "references": {
            "timeout": "timeout",
            "docking_region_radius": "docking_region_radius",
            "velocity_threshold": "velocity_threshold",
            "threshold_distance": "threshold_distance",
            "mean_motion": "mean_motion",
            "lower_bound": "lower_bound",
            "reference_position_sensor_name": "reference_position_sensor_name",
            "reference_velocity_sensor_name": "reference_velocity_sensor_name",
          }
        },
      {
        # Docking failure reward
        # Penalizes the agent for failing the task
        "name": "DockingFailureReward",
        "functor": "safe_autonomy_sims.rewards.cwh.docking_rewards.DockingFailureReward",
        "config": {
          "timeout_reward": -1.0,
          "distance_reward": -1.0,
          "crash_reward": -1.0,
        },
        "references": {
          "timeout": "timeout",
          "max_goal_distance": "max_distance",
          "docking_region_radius": "docking_region_radius",
          "velocity_threshold": "velocity_threshold",
          "threshold_distance": "threshold_distance",
          "mean_motion": "mean_motion",
          "lower_bound": "lower_bound",
          "reference_position_sensor_name": "reference_position_sensor_name",
          "reference_velocity_sensor_name": "reference_velocity_sensor_name",
        }
      },
    ],

    # A set of common parameters referenced by this configuration file
    "reference_store": !include configs/docking/parameters.yml
}
```

</details>
</br>

## References

<a id="1">[1]</a>
Clohessy, W., and Wiltshire, R., “Terminal Guidance System for Satellite Rendezvous,” *Journal of the Aerospace Sciences*, Vol. 27, No. 9, 1960, pp. 653–658.
