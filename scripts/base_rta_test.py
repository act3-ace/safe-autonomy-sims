from pydantic import PyObject, validator
import socket
from datetime import datetime
import argparse
import typing
import jsonargparse
import pathlib

from corl.experiments.base_experiment import BaseExperiment, BaseExperimentValidator
from corl.parsers.yaml_loader import apply_patches
from corl.policies.base_policy import BasePolicyValidator
from corl.environment.default_env_rllib_callbacks import EnvironmentDefaultCallbacks
from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.libraries.factory import Factory
from corl.experiments.base_experiment import BaseExperiment, BaseExperimentValidator
from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file

from ray.tune.utils.log import Verbosity


class RTAExperimentValidator(BaseExperimentValidator):
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


class RTAPolicyValidator(BasePolicyValidator):
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


class RTAExperiment(BaseExperiment):
    """
    Now is modified
    The Rllib Experiment is an experiment for running
    multiagent configurable environments with patchable settings
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config: RTAExperimentValidator = typing.cast(RTAExperimentValidator, self.config)

    @property
    def get_validator(self) -> typing.Type[RTAExperimentValidator]:
        return RTAExperimentValidator

    @property
    def get_policy_validator(self) -> typing.Type[RTAPolicyValidator]:
        """Return validator"""
        return RTAPolicyValidator

    def run_experiment(self, args: argparse.Namespace) -> None:
        raise NotImplementedError

    def setup_experiment(self, args: argparse.Namespace):
        rllib_config = self._select_rllib_config(args.compute_platform)
        if args.compute_platform in ['ray']:
            self._update_ray_config_for_ray_platform()

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

        rllib_config["env"] = ACT3MultiAgentEnv
        rllib_config["callbacks"] = EnvironmentDefaultCallbacks
        rllib_config["env_config"] = self.config.env_config
        now = datetime.now()
        rllib_config["env_config"]["output_date_string"] = f"{now.strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
        rllib_config["create_env_on_driver"] = True


        env_class = rllib_config["env"]
        env = env_class(self.config.env_config)
        return args, env

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
            platform = 'default'
            return self.config.rllib_configs[platform]

        if len(self.config.rllib_configs) == 1:
            return self.config.rllib_configs[next(iter(self.config.rllib_configs))]

        raise RuntimeError(f'Invalid rllib_config for platform "{platform}"')

    def _update_ray_config_for_ray_platform(self) -> None:
        """Update the ray configuration for ray platforms
        """
        self.config.ray_config['address'] = 'auto'
        self.config.ray_config['log_to_driver'] = False


class MainUtilRTAExpr:
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
            default=MainUtilRTAExpr.DEFAULT_CONFIG_PATH,
            help=f"Path to config.yml file used to setup the training environment Default={MainUtilRTAExpr.DEFAULT_CONFIG_PATH}",
        )

        parser.add_argument(
            "--compute-platform",
            type=str,
            default="auto",
            help="Compute platform [ace, hpc, local, auto] of experiment. Used to select rllib_config",
        )
        parser.add_argument(
            "-pc",
            "--platform-config",
            action="append",
            nargs=2,
            metavar=("platform-name", "platform-file"),
            help="the specification for a platform in the environment"
        )
        parser.add_argument(
            "-ac",
            "--agent-config",
            action="append",
            nargs=4,
            metavar=("agent-name", "platform-name", "configuration-file", "policy-file"),
            help="the specification for an agent in the environment"
        )
        parser.add_argument("-op", "--other-platform", action="append", nargs=2, metavar=("agent-name", "platform-file"), help="help:")
        parser.add_argument(
            '--verbose',
            type=Verbosity,
            choices=[Verbosity.V0_MINIMAL, Verbosity.V1_EXPERIMENT, Verbosity.V2_TRIAL_NORM, Verbosity.V3_TRIAL_DETAILS],
            default=Verbosity.V3_TRIAL_DETAILS
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help="Tells your specified experiment to switch configurations to debug mode, experiments may ignore this flag"
        )
        parser.add_argument('--profile', action='store_true', help="Tells experiment to switch configuration to profile mode")
        parser.add_argument('--profile-iterations', type=int, default=10)
        return parser.parse_args(args=alternate_argv)


def main():
    """
    Main method of the module that allows for arguments parsing  for experiment setup.
    """
    args = MainUtilRTAExpr.parse_args()
    config = load_file(config_filename=args.config)

    experiment_parse = ExperimentParse(**config)
    experiment_class = experiment_parse.experiment_class(**experiment_parse.config)
    experiment_class.run_experiment(args)