# Configuration

The environments and experiments built using CoRL are
designed to be highly configurable. There are several
relevant configuration files used to define a full task
environment and RL experiment. All configuration files
use the YAML file format.

## Agent

The agent configuration file defines the platform parts,
rewards, dones, and glues used by an agent in an RL experiment.
The agent configuration file contains two top level fields:

```yaml
"agent": "corl.agents.base_agent.TrainableBaseAgent"
"config": {
  ...
}
```

- `agent`: agent class of type [BaseAgent]({{corl_docs_url}}/reference/agents/base_agent/#corl.agents.base_agent.BaseAgent)
- `config`: agent configuration parameters (see below)

### Agent Config Parameters

The agent config parameters define the agent platform used
in the RL experiment. Available configuration parameters
include:

- `parts`: list of platform part entries
    - Each part entry includes the part's name (`part`) and optional initialization parameters (`config`)
    - Initialization parameters are unique to the platform part's implementation

    ```yaml
    # Example
    "parts": [
      {"part": "Controller_Thrust", 
        "config": {
          "name": "X Thrust", 
          "axis": 0, 
          "properties": {"name": "x_thrust"}
        }
      },
      {"part": "Sensor_Position"},
      {"part": "Sensor_Velocity"}
    ]
    ```

- `episode_parameter_provider`: object which provides agent initialization parameters during an experiment
    - `type`: class of type [EpisodeParameterProvider]({{corl_docs_url}}/reference/episode_parameter_providers/core/#corl.episode_parameter_providers.core.EpisodeParameterProvider)
    - `config`: implementation specific initialization parameters for the parameter provider

    ```yaml
    # Example
    "episode_parameter_provider": {
      "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
    },
    ```

- `simulator_reset_parameters`: keyword arguments and values passed to the simulator used during the experiment whenever the simulator is reset. These values are simulator-specific, so refer to your simulator documentation to see what values are expected.

    ```yaml
    # Example
    "simulator_reset_parameters": {
      "x": 100,
      "y": 100,
      "z": 100,
      "xdot": 0,
      "ydot": 0,
      "zdot": 0,
    },
    ```

- `glues`: list of glue entries
    - Glues connect an agent platform to a RL training framework by providing an endpoint for observations and actions.
    - A glue may transform and send data from a platform part to a training framework as an observation. The full set of glues which provide observations over all agents in the environment defines the environment's observation space.
    - A glue may transform and send data from a training framework to a platform part as an action. The full set of glues which provide actions over all agents in the environment defines the environment's action space.

    ```yaml
    # Example
    "glues": [
      {
        "functor": "safe_autonomy_sims.core.dones.common_dones.TimeoutDoneFunction",
        "config": { ... },
      },
      {
        "functor": "safe_autonomy_sims.core.dones.docking_dones.MaxDistanceDoneFunction",
        "config": { ... },
      },
      {
        "functor": "safe_autonomy_sims.core.glues.normal.normal_observe_glue.NormalObserveSensorGlue",
        "config": { ... },
      },
    ]
    ```

    - Glue entries have two top level fields:
        - `functor`: a class of type [BaseAgentGlue]({{corl_docs_url}}/reference/glues/base_glue/#corl.glues.base_glue.BaseAgentGlue)
        - `config`: a set of glue-specific configuration arguments

        ```yaml
        # Example
        "functor": "corl.glues.common.controller_glue.ControllerGlue",
        "config": {
          "controller": "X Thrust",
          "training_export_behavior": "EXCLUDE",
          "normalization": {
            "enabled": False,
          }
        }
        ```

- `dones`: list of done functions
    - Done functions provide the terminal conditions for an agent during an episode in an experiment.

    ```yaml
    # Example
    "dones": [
      {
        "functor": "corl.glues.common.controller_glue.ControllerGlue",
        "config": { ... },
      },
      {
        "functor": "corl.glues.common.controller_glue.ControllerGlue",
        "config": { ... },
      },
      {
        "functor": "safe_autonomy_sims.core.rewards.docking_rewards.DockingDeltaVReward",
        "config": { ... },
      },
    ]
    ```

    - Done function entries have two top level fields:
        - `functor`: a class of type [DoneFuncBase]({{corl_docs_url}}/reference/dones/done_func_base/#corl.dones.done_func_base.DoneFuncBase)
        - `config`: a set of function-specific configuration arguments

        ```yaml
        # Example
        "functor": "safe_autonomy_sims.core.dones.docking_dones.CrashDockingDoneFunction",
        "config":{
          "docking_region_radius": 0.5,
          "velocity_threshold": 0.2,
          "threshold_distance": 0.5,
          "mean_motion": 0.001027,
          "lower_bound": False,
        },
        ```

- `rewards`: list of reward functions
    - Reward functions provide the agent rewards at each step during an episode in an experiment.

      ```yaml
      # Example
      "rewards": [
        {
          "name": "DockingDistanceExponentialChangeReward",
          "functor": "safe_autonomy_sims.core.rewards.docking_rewards.DockingDistanceExponentialChangeReward",
          "config": { ... },
        },
        {
          "name": "DockingDeltaVReward",
          "functor": "safe_autonomy_sims.core.rewards.docking_rewards.DockingDeltaVReward",
          "config": { ... },
        },
        {
          "name": "DockingSuccessReward",
          "functor": "safe_autonomy_sims.core.rewards.docking_rewards.DockingSuccessReward",
          "config": { ... },
        },
      ]
      ```

    - Reward function entries have three top level fields:
        - `name`: a name for the reward function
        - `functor`: a class of type [DoneFuncBase]({{corl_docs_url}}/reference/dones/done_func_base/#corl.dones.done_func_base.DoneFuncBase)
        - `config`: a set of function-specific configuration arguments

        ```yaml
        # Example
        "name": "DockingDeltaVReward",
        "functor": "safe_autonomy_sims.core.rewards.docking_rewards.DockingDeltaVReward",
        "config": {
          "scale": -0.01,
          "bias": 0.0,
          "mass": 12.0
        }
        ```

## Environment

The environment configuration file details environment
level configuration options. These include configuring
the environment simulator, defining the available platform
types, specifying paths for various plugins, and defining
and environment-level episode parameter provider.

- `simulator`: the simulator used to process a single step in the environment and update the state of all objects in the environment during an episode.
    - `type`: the registered name of a class of type [BaseSimulator]({{corl_docs_url}}/reference/simulators/base_simulator/#corl.simulators.base_simulator.BaseSimulator)
    - `config`: a simulator-specific set of keyword arguments and values passed to the simulator during initialization

    ```yaml
    # Example
    "simulator": {
      "type": "CWHSimulator",
      "config": {
        "step_size": 1
      },
    },
    ```

- `platforms`: list of registered platform types available in this environment

    ```yaml
    # Example
    "platforms": "CWHSimulator_Platforms",
    ```

- `plugin_paths`: list of module or package paths in which the plugin library should search for CoRL compatible plugins (platforms, platform parts, CoRL simulators)

    ```yaml
    # Example
    "plugin_paths": ["safe_autonomy_sims.core.platforms", "safe_autonomy_sims.core.simulators"],
    ```

- `episode_parameter_provider`: object which provides environment initialization parameters during an experiment
    - `type`: class of type [EpisodeParameterProvider]({{corl_docs_url}}/reference/episode_parameter_providers/core/#corl.episode_parameter_providers.core.EpisodeParameterProvider)
    - `config`: implementation specific initialization parameters for the parameter provider

    ```yaml
    # Example
    "episode_parameter_provider": {
      "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
    },
    ```

## Platforms

The platforms configuration file allows you to specify which
registered platform types you are using in your experiment.
Incompatible or unspecified platform types will not be allowed,
preventing use of inappropriate platforms with inappropriate simulators, parts, etc.

- `name`: name of allowed platform type

  ```yaml
  # Example
  {
    name: CWH
  }
  ```

## Policy

The policy configuration file allows you to define a custom
policy or override parameters for a provided policy in your
training framework. This file contains a YAML dictionary of
policy specific configuration options. This dictionary can
be left blank if the policy default parameters are used.

## Task

The task configuration file defines the experiment class you
wish to use and overrides any default training parameters
of your chosen training framework.

- `experiment_class`: a class of type [BaseExperiment]({{corl_docs_url}}/reference/experiments/base_experiment/#corl.experiments.base_experiment.BaseExperiment)
    - The experiment class interfaces with your chosen training framework and defines how training is handled.
- `config`: experiment class specific configuration parameters

    ```yaml
    # Example using Ray RLLib and Tune, assumes framework specific configuration files `ray.yml`, `tune.yml`, and `rllib.yml` exist in the context:
    experiment_class: corl.experiments.rllib_experiment.RllibExperiment
    config:
      rllib_config_updates: &rllib_config_updates
      
        # No overrides for ray as there are no changes
        ray_config_updates: &ray_config_updates
          local_mode: False
      
        # Change the default path for saving out the data
        env_config_updates: &env_config_updates
          TrialName: DOCKING
          output_path: /tmp/safe_autonomy_sims/
      
        # Change the default path for saving out the data
        tune_config_updates: &tune_config_updates
          local_dir: /tmp/safe_autonomy_sims/ray_results/
      
        ####################################################################
        # Setup the actual keys used by the code
        # Note that items are patched from the update section
        ###################################################################
        rllib_configs:
          default: [!include rllib.yml, *rllib_config_updates]
          local: [!include rllib.yml,  *rllib_config_updates]
      
        ray_config: [!include ray.yml, *ray_config_updates]
        env_config: [!include ../../environments/docking.yml, *env_config_updates]  # Environment configuration file
        tune_config: [!include tune.yml, *tune_config_updates]
    ```
