import argparse
import socket
import typing
from datetime import datetime
import jsonargparse

import ray
from pydantic import PyObject, validator
from ray import tune

from act3_rl_core.agents.base_agent import BaseAgentParser
from act3_rl_core.environment.default_env_rllib_callbacks import EnvironmentDefaultCallbacks
from act3_rl_core.environment.multi_agent_env import ACT3MultiAgentEnv
from act3_rl_core.experiments.base_experiment import BaseExperiment, BaseExperimentValidator
from act3_rl_core.parsers.yaml_loader import apply_patches
from act3_rl_core.policies.base_policy import BasePolicyValidator

from act3_rl_core.experiments.base_experiment import ExperimentParse
from act3_rl_core.parsers.yaml_loader import load_file

import pathlib
import numpy as np
from collections import OrderedDict



class RllibExperimentValidator(BaseExperimentValidator):
    """
    ray_config: dictionary to be fed into ray init, validated by ray init call
    env_config: environment configuration, validated by environment class
    rllib_configs: a dictionary
    Arguments:
        BaseModel {[type]} -- [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type] -- [description]
    """
    ray_config: typing.Dict[str, typing.Any]
    env_config: typing.Dict[str, typing.Any]
    rllib_configs: typing.Dict[str, typing.Dict[str, typing.Any]]
    tune_config: typing.Dict[str, typing.Any]
    trainable_config: typing.Optional[typing.Dict[str, typing.Any]]

    @validator('rllib_configs', pre=True)
    def apply_patches_rllib_configs(cls, v):  # pylint: disable=E0213, R0201
        """
        The dictionary of rllib configs may come in as a dictionary of
        lists of dictionaries, this function is responsible for collapsing
        the list down to a typing.Dict[str, typing.Dict[str, typing.Any]]
        instead of
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]]

        Raises:
            RuntimeError: [description]

        Returns:
            [type] -- [description]
        """
        if not isinstance(v, dict):
            raise RuntimeError("rllib_configs are expected to be a dict of keys to different compute configs")
        rllib_configs = {}
        for key, value in v.items():
            if isinstance(value, list):
                rllib_configs[key] = apply_patches(value)
            elif isinstance(value, dict):
                rllib_configs[key] = value
        return rllib_configs

    @validator('ray_config', 'tune_config', 'trainable_config', 'env_config', pre=True)
    def apply_patches_configs(cls, v):  # pylint: disable=E0213, R0201
        """
        reduces a field from
        typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]]
        to
        typing.Dict[str, typing.Any]

        by patching the first dictionary in the list with each patch afterwards

        Returns:
            [type] -- [description]
        """
        if isinstance(v, list):
            v = apply_patches(v)
        return v


class RllibPolicyValidator(BasePolicyValidator):
    """
    policy_class: callable policy class None will use default from trainer
    train: should this policy be trained
    Arguments:
        BaseModel {[type]} -- [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type] -- [description]
    """
    policy_class: typing.Union[PyObject, None] = None
    train: bool = True


def make_action(agent_name,x_thrust,y_thrust,z_thrust):
    action = OrderedDict()
    action[agent_name] = OrderedDict()
    action[agent_name]['X_thrust'] = OrderedDict()
    action[agent_name]['X_thrust']['x_thrust'] = np.array([x_thrust],dtype=np.float32)
    action[agent_name]['Y_thrust'] = OrderedDict()
    action[agent_name]['Y_thrust']['y_thrust'] = np.array([y_thrust],dtype=np.float32)
    action[agent_name]['Z_thrust'] = OrderedDict()
    action[agent_name]['Z_thrust']['z_thrust'] = np.array([z_thrust],dtype=np.float32)
    return action




class RllibExperiment(BaseExperiment):
    """
    Now is modified
    The Rllib Experiment is an experiment for running
    multiagent configurable environments with patchable settings
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = typing.cast(RllibExperimentValidator, self.config)

    @classmethod
    def get_validator(cls):
        return RllibExperimentValidator

    @classmethod
    def get_policy_validator(cls):
        return RllibPolicyValidator

    def run_experiment(self, args: argparse.Namespace) -> None:

        rllib_config = self._select_rllib_config(args.compute_platform)
        if args.compute_platform in ['ray']:
            self._update_ray_config_for_ray_platform()

        self.config.env_config["agents"] = self.create_agents(args.agent_config)

        if args.other_platform:
            self.config.env_config["other_platforms"] = self.create_other_platforms(args.other_platform)

        # Create episode parameter providers
        ACT3MultiAgentEnv.create_episode_parameter_provider(self.config.env_config, remote=(not self.config.ray_config['local_mode']))
        for agent_name, agent_configs in self.config.env_config['agents'].items():
            agent_configs.class_config.agent.create_episode_parameter_provider(
                BaseAgentParser(
                    **agent_configs.class_config.config, agent_name=agent_name, multiple_workers=(not self.config.ray_config['local_mode'])
                )
            )

        tmp = ACT3MultiAgentEnv(self.config.env_config)
        tmp_as = tmp.action_space
        tmp_os = tmp.observation_space
        tmp_ac = self.config.env_config['agents']

        policies = {
            policy_name:
            (tmp_ac[policy_name].policy_config["policy_class"], policy_obs, tmp_as[policy_name], tmp_ac[policy_name].policy_config)
            for policy_name,
            policy_obs in tmp_os.spaces.items()
        }

        train_policies = [policy_name for policy_name in policies.keys() if tmp_ac[policy_name].policy_config["train"]]

        rllib_config["multiagent"] = {
            "policies": policies, "policy_mapping_fn": lambda agent_id: agent_id, "policies_to_train": train_policies
        }

        rllib_config["env"] = ACT3MultiAgentEnv
        rllib_config["callbacks"] = EnvironmentDefaultCallbacks
        rllib_config["env_config"] = self.config.env_config
        now = datetime.now()
        rllib_config["env_config"]["output_date_string"] = f"{now.strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
        rllib_config["horizon"] = self.config.env_config["horizon"]
        rllib_config["create_env_on_driver"] = True

        # .add_default_plugin_paths()
        # pl.plugin_test()

        env_class = rllib_config["env"]
        env = env_class(self.config.env_config)

        # call reset on env
        obs = env.reset()
        # setup action

        # setup for loop for specified number of steps
        num_steps = 10

        # setup a list of actions
        #actions = [make_action('blue0',-1.0,1.0,2.0),make_action('blue0',-3.0,2.0,2.0),make_action('blue0',-1.0,1.0,4.0)]
        # make sure you have a list of actions i.e. ordered dicts

        actions = []
        for _ in range(num_steps):
            actions.append(make_action("blue0",1,-1,-2))


        for i in range(num_steps):
            data = env.step(actions[i])
            print(data)
            print('completed step=',i)

        # loop thru list and call step
        # test with normal docking config

    def _select_rllib_config(self, platform: typing.Optional[str]) -> typing.Dict[str, typing.Any]:
        """Extract the rllib config for the proper computational platform

        Parameters
        ----------
        platform : typing.Optional[str]
            Specification of the computational platform to use, such as "local", "hpc", etc.  This must be present in the rllib_configs.
            If None, the rllib_configs must only have a single entry.

        Returns
        -------
        typing.Dict[str, typing.Any]
            Rllib configuration for the desired computational platform.

        Raises
        ------
        RuntimeError
            The requested computational platform does not exist or None was used when multiple platforms were defined.
        """
        if platform is not None:
            return self.config.rllib_configs[platform]

        if len(self.config.rllib_configs) == 1:
            return self.config.rllib_configs[next(iter(self.config.rllib_configs))]

        raise RuntimeError(f'Invalid rllib_config for platform "{platform}"')

    def _update_ray_config_for_ray_platform(self) -> None:
        """Update the ray configuration for ray platforms
        """
        self.config.ray_config['address'] = 'auto'
        self.config.ray_config['log_to_driver'] = False


class MainUtilACT3Core:
    """
    Contains all the procedures that allow for argument parsing to setup an experiment
    """

    DEFAULT_CONFIG_PATH = str(pathlib.Path(__file__).parent.absolute() / 'config' / 'tasks' / 'single_lear_capture.yml')

    @staticmethod
    def parse_args(alternate_argv: typing.Optional[typing.Sequence[str]] = None):
        """
        Processes the arguments as main entry point for ACT3/ deep reinforcement training code


        Parameters
        ----------
        alternate_argv : Sequence[str], optional
            Arguments that should be parsed rather than sys.argv.  The default of None parses sys.argv.
            See https://docs.python.org/3/library/argparse.html#beyond-sys-argv.

        Returns
        -------
        namespace
            The arguments from the parser
        """

        parser = jsonargparse.ArgumentParser()
        parser.add_argument(
            "--cfg",
            help="an alternative way to provide arguments, the path to a json/yml file containing the running arguments",
            action=jsonargparse.ActionConfigFile
        )
        parser.add_argument(
            "--config",
            type=str,
            default=MainUtilACT3Core.DEFAULT_CONFIG_PATH,
            help=f"Path to config.yml file used to setup the training environment Default={MainUtilACT3Core.DEFAULT_CONFIG_PATH}",
        )

        parser.add_argument(
            "--compute-platform",
            type=str,
            help="Compute platform [ace, hpc, local] of experiment. Used to select rllib_config",
        )
        parser.add_argument(
            "-ac",
            "--agent-config",
            action="append",
            nargs=4,
            metavar=("agent-name", "configuration-file", "platform-file", "policy-file"),
            help="the specification for an agent in the environment"
        )
        parser.add_argument("-op", "--other-platform", action="append", nargs=2, metavar=("agent-name", "platform-file"), help="help:")
        parser.add_argument('--verbose', type=int, default=1)

        return parser.parse_args(args=alternate_argv)


def main():
    """
    Main method of the module that allows for arguments parsing  for experiment setup.
    """
    args = MainUtilACT3Core.parse_args()
    config = load_file(config_filename=args.config)

    # print(config)
    experiment_parse = ExperimentParse(**config)
    experiment_class = experiment_parse.experiment_class(**experiment_parse.config)
    print(experiment_class.config.dict())
    experiment_class.run_experiment(args)


if __name__ == "__main__":
    main()
