"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines an experiment class which runs training episodes via ray rllib's API.
This is useful for debugging and validating new environments and can easily be modified to log state data.
"""
# pylint: disable=C0301,R0914,R1732,W0612,R0915

import argparse
import pickle
import socket
import typing
from datetime import datetime

import ray
import tqdm
from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.experiments.rllib_experiment import RllibExperiment, RllibExperimentValidator
from corl.libraries.factory import Factory

# import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.ppo import ppo


class RllibAPIExperimentValidator(RllibExperimentValidator):
    """
    The validator for the RllibAPIExperiment class.

    Attributes
    ----------
    serialized_ray_config_path: str
        path to params.pkl file
    trained_models_checkpoint_path: str
        path to checkpoint directory
    """
    serialized_ray_config_path: str
    trained_models_checkpoint_path: str


class RllibAPIExperiment(RllibExperiment):
    """
    This class loads an environment from given checkpoint folders, then runs 10 rollout
    training episdoes using rllib's API directly. This allows for debugging with access to state, done,
    and reward data and logic for environment development and validation.
    """

    @staticmethod
    def get_validator() -> typing.Type[RllibAPIExperimentValidator]:
        return RllibAPIExperimentValidator

    def run_experiment(self, args: argparse.Namespace):

        # setup
        rllib_config = self._select_rllib_config(args.compute_platform)

        if args.compute_platform in ['ray']:
            self._update_ray_config_for_ray_platform()

        self._add_trial_creator()

        # This needs to be before the ray cluster is initialized
        if args.debug:
            self.config.ray_config['local_mode'] = True

        ray.init(**self.config.ray_config)

        ray_resources = ray.available_resources()

        self.config.env_config["agents"], self.config.env_config["agent_platforms"] = self.create_agents(
            args.platform_config, args.agent_config
        )

        self.config.env_config["horizon"] = rllib_config["horizon"]

        if args.other_platform:
            self.config.env_config["other_platforms"] = self.create_other_platforms(args.other_platform)

        if not self.config.ray_config['local_mode']:
            self.config.env_config['episode_parameter_provider'] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                Factory(**self.config.env_config['episode_parameter_provider']),
                actor_name=ACT3MultiAgentEnv.episode_parameter_provider_name
            )

            for agent_name, agent_configs in self.config.env_config['agents'].items():
                agent_configs.class_config.config['episode_parameter_provider'] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                    Factory(**agent_configs.class_config.config['episode_parameter_provider']), agent_name
                )

        self.config.env_config['epp_registry'] = ACT3MultiAgentEnvValidator(**self.config.env_config).epp_registry

        tmp = ACT3MultiAgentEnv(self.config.env_config)
        tmp_as = tmp.action_space
        tmp_os = tmp.observation_space
        tmp_ac = self.config.env_config['agents']

        policies = {
            policy_name: (
                tmp_ac[policy_name].policy_config["policy_class"],
                policy_obs,
                tmp_as[policy_name],
                tmp_ac[policy_name].policy_config["config"]
            )
            for policy_name,
            policy_obs in tmp_os.spaces.items()
            if tmp_ac[policy_name]
        }

        train_policies = [policy_name for policy_name in policies.keys() if tmp_ac[policy_name].policy_config["train"]]

        rllib_config["multiagent"] = {
            "policies": policies, "policy_mapping_fn": lambda agent_id: agent_id, "policies_to_train": train_policies
        }

        rllib_config["env"] = ACT3MultiAgentEnv
        rllib_config["callbacks"] = self.get_callbacks()
        rllib_config["env_config"] = self.config.env_config
        now = datetime.now()
        rllib_config["env_config"]["output_date_string"] = f"{now.strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
        rllib_config["create_env_on_driver"] = True
        rllib_config["batch_mode"] = "complete_episodes"

        if args.debug:
            rllib_config['num_workers'] = 0

        self._enable_episode_parameter_provider_checkpointing()

        if args.profile:
            if "stop" not in self.config.tune_config:
                self.config.tune_config["stop"] = {}
            self.config.tune_config["stop"]["training_iteration"] = args.profile_iterations

        search_class = None
        if self.config.hparam_search_class is not None:
            if self.config.hparam_search_config is not None:
                search_class = self.config.hparam_search_class(**self.config.hparam_search_config)
            else:
                search_class = self.config.hparam_search_class()
            search_class.add_algorithm_hparams(rllib_config, self.config.tune_config)

        ################################# Manual rllib API Episodes ####################################

        # instantiate env
        env = ACT3MultiAgentEnv(self.config.env_config)

        # load config from checkpoint
        ray_config_file = open(self.config.serialized_ray_config_path, "rb")
        ray_config = pickle.load(ray_config_file)

        # override serialized environment config
        ray_config["env_config"] = self.config.env_config

        # instantiate agents and restore models from checkpoint
        agents = ppo.PPOTrainer(config=ray_config, env=rllib_config['env'])
        agents.restore(self.config.trained_models_checkpoint_path)

        # step for loop
        num_rollouts = 10
        for _ in tqdm.tqdm(range(num_rollouts)):
            # run until episode ends
            dones = {"placeholder": False}
            obs = env.reset()
            step_num = 0

            while not all(dones.values()):
                # progress environment state
                action_dict = {}
                for name in obs.keys():
                    action_dict[name] = agents.compute_single_action(obs[name], policy_id=name)

                obs, rewards, dones, info = env.step(action_dict)
                step_num += 1
