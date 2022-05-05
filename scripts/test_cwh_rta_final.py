import argparse
import socket
import typing
from datetime import datetime
import jsonargparse
import scipy

import ray
from pydantic import PyObject, validator
from ray import tune

from act3_rl_core.agents.base_agent import BaseAgentParser
from act3_rl_core.environment.default_env_rllib_callbacks import EnvironmentDefaultCallbacks
from act3_rl_core.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from act3_rl_core.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from act3_rl_core.experiments.base_experiment import BaseExperiment, BaseExperimentValidator
from act3_rl_core.libraries.factory import Factory
from act3_rl_core.parsers.yaml_loader import apply_patches
from act3_rl_core.policies.base_policy import BasePolicyValidator

from act3_rl_core.experiments.base_experiment import ExperimentParse
from act3_rl_core.parsers.yaml_loader import load_file

import pathlib
import numpy as np
from collections import OrderedDict



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


def make_action(agent_name, x_thrust, y_thrust, z_thrust):
    action = {
        agent_name: {
            "RTAModule":
                (
                    {'x_thrust': np.array([x_thrust], dtype=np.float32)},
                    {'y_thrust': np.array([y_thrust], dtype=np.float32)},
                    {'z_thrust': np.array([z_thrust], dtype=np.float32)},
                )
        }
    }
    return action


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

        rllib_config = self._select_rllib_config(args.compute_platform)
        if args.compute_platform in ['ray']:
            self._update_ray_config_for_ray_platform()

        self.config.env_config["agents"] = self.create_agents(args.agent_config)

        if args.other_platform:
            self.config.env_config["other_platforms"] = self.create_other_platforms(args.other_platform)

        # Create episode parameter providers
        # ACT3MultiAgentEnv.create_episode_parameter_provider(self.config.env_config, remote=(not self.config.ray_config['local_mode']))
        # for agent_name, agent_configs in self.config.env_config['agents'].items():
        #     agent_configs.class_config.agent.create_episode_parameter_provider(
        #         BaseAgentParser(
        #             **agent_configs.class_config.config, agent_name=agent_name, multiple_workers=(not self.config.ray_config['local_mode'])
        #         )
        #     )

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
            policy_name:(
                tmp_ac[policy_name].policy_config["policy_class"],
                policy_obs,
                tmp_as[policy_name],
                tmp_ac[policy_name].policy_config
            )
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
        rllib_config["create_env_on_driver"] = True

        # .add_default_plugin_paths()
        # pl.plugin_test()

        env_class = rllib_config["env"]
        env = env_class(self.config.env_config)

        # call reset on env
        obs = env.reset()
        # setup action

        # setup for loop for specified number of steps
        num_steps = 100

        # setup a list of actions
        #actions = [make_action('blue0',-1.0,1.0,2.0),make_action('blue0',-3.0,2.0,2.0),make_action('blue0',-1.0,1.0,4.0)]
        # make sure you have a list of actions i.e. ordered dicts

        agent = args.agent_config[0][0]
        data = []
        controller = TestController()
        done = False
        i = 0

        intervening_list = []

        while not done:
            
            if i == 999:
                print(1)
            
            position = obs[agent]["ObserveSensor_Sensor_Position"]["direct_observation"]
            velocity = obs[agent]["ObserveSensor_Sensor_Velocity"]["direct_observation"]
            state = np.concatenate((position, velocity))

            u_des = controller.compute_feedback_control(state)

            action = make_action(agent, *u_des)
            obs, rewards, dones, info = env.step(action)

            intervening = info['blue0']['RTAModule']['intervening']
            control_actual = info['blue0']['RTAModule']['actual control']
            control_desired = info['blue0']['RTAModule']['desired control']

            done = dones['__all__']

            step_data = {
                'state': state,
                'intervening': intervening,
                'control': control_actual,
            }
            print('completed step=', i)
            data.append(step_data)
            i += 1

        # loop thru list and call step
        # test with normal docking config

        self.plot_pos_vel(data)

    def plot_pos_vel(self, data):
        from matplotlib import pyplot as plt

        states = np.empty([len(data), 6])
        control = np.empty([len(data), 3])
        intervening = np.empty([len(data)])
        for i in range(len(data)):
            states[i, :] = data[i]['state']
            control[i, :] = data[i]['control']
            intervening[i] = data[i]['intervening']

        fig = plt.figure()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        ax1 = fig.add_subplot(231, projection='3d')
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        max = np.max(np.abs(states[:, 0:3]))*1.1
        RTAon = np.ma.masked_where(intervening != 1, states[:, 1])
        ax1.plot(0, 0, 'kx')
        ax1.plot(states[0, 0], states[0, 1], states[0, 2], 'r+')
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'b')
        ax1.plot(states[:, 0], RTAon, states[:, 2], 'c')
        ax1.set_xlim([-max, max])
        ax1.set_ylim([-max, max])
        ax1.set_zlim([-max, max])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('y')
        ax1.set_title('Trajectory')
        ax1.grid(True)

        v = np.empty([len(states), 2])
        for i in range(len(states)):
            v[i, :] = [np.linalg.norm(states[i, 0:3]), np.linalg.norm(states[i, 3:6])]
        RTAon = np.ma.masked_where(intervening != 1, v[:, 1])
        ax2.plot([0, 15000], [0.2, 0.2+4*0.001027*15000], 'g--')
        ax2.plot(v[:, 0], v[:, 1], 'b')
        ax2.plot(v[:, 0], RTAon, 'c')
        ax2.set_xlim([0, np.max(v[:, 0])*1.1])
        ax2.set_ylim([0, np.max(v[:, 1])*1.1])
        ax2.set_xlabel('Relative Position')
        ax2.set_ylabel('Relative Velocity')
        ax2.set_title('Distance Dependent Speed Limit')
        ax2.grid(True)

        ax3.plot(0, 0, 'kx')
        ax3.plot(0, 0, 'r+')
        ax3.plot(0, 0, 'g')
        ax3.plot(0, 0, 'b')
        ax3.plot(0, 0, 'c')
        ax3.legend(['Chief', 'Deputy Initial Position', 'Constraint', 'RTA Not Intervening', 'RTA Intervening'])
        ax3.axis('off')
        ax3.set_xlim([1, 2])
        ax3.set_ylim([1, 2])

        RTAonx = np.ma.masked_where(intervening != 1, states[:, 3])
        RTAony = np.ma.masked_where(intervening != 1, states[:, 4])
        RTAonz = np.ma.masked_where(intervening != 1, states[:, 5])
        ax4.plot([0, len(states)*1.1], [10, 10], 'g--')
        ax4.plot([0, len(states)*1.1], [-10, -10], 'g--')
        ax4.plot(range(len(states)), states[:, 3], 'b')
        ax4.plot(range(len(states)), RTAonx, 'c')
        ax4.plot(range(len(states)), states[:, 4], 'r')
        ax4.plot(range(len(states)), RTAony, 'tab:orange')
        ax4.plot(range(len(states)), states[:, 5], 'tab:brown')
        ax4.plot(range(len(states)), RTAonz, 'tab:green')
        ax4.set_xlim([0, len(states)*1.1])
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Velocity')
        ax4.set_title('Max Velocity Constraint')
        ax4.grid(True)

        RTAonx = np.ma.masked_where(intervening != 1, control[:, 0])
        RTAony = np.ma.masked_where(intervening != 1, control[:, 1])
        RTAonz = np.ma.masked_where(intervening != 1, control[:, 2])
        ax5.plot([0, len(control)*1.1], [1, 1], 'g--')
        ax5.plot([0, len(control)*1.1], [-1, -1], 'g--')
        ax5.plot(range(len(control)), control[:, 0], 'b')
        ax5.plot(range(len(control)), RTAonx, 'c')
        ax5.plot(range(len(control)), control[:, 1], 'r')
        ax5.plot(range(len(control)), RTAony, 'tab:orange')
        ax5.plot(range(len(control)), control[:, 2], 'tab:brown')
        ax5.plot(range(len(control)), RTAonz, 'tab:green')
        ax5.set_xlim([0, len(control)*1.1])
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Force')
        ax5.set_title('Actions')
        ax5.grid(True)

        ax6.plot(0, 0, 'g--')
        ax6.plot(0, 0, 'b')
        ax6.plot(0, 0, 'c')
        ax6.plot(0, 0, 'r')
        ax6.plot(0, 0, 'tab:orange')
        ax6.plot(0, 0, 'tab:brown')
        ax6.plot(0, 0, 'tab:green')
        ax6.legend(['Constraint', 'vx/Fx: RTA Not Intervening', 'vx/Fx: RTA Intervening', 'vy/Fy: RTA Not Intervening',
                    'vy/Fy: RTA Intervening', 'vz/Fz: RTA Not Intervening', 'vz/Fz: RTA Intervening'])
        ax6.axis('off')
        ax6.set_xlim([1, 2])
        ax6.set_ylim([1, 2])

        # plt.show()
        plt.savefig('rta_test.png')

    def plot_constraint_line(self, ax, x, y, border_linewidth=10, color='r', border_alpha=0.25):
        ax.plot(x, y, color, linewidth=border_linewidth, alpha=border_alpha)
        ax.plot(x, y, color)

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


class TestController():
    def __init__(self, m = 12, n = 0.001027):
        # Define in-plane CWH Dynamics
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [3*n**2, 0, 0, 0, 2*n, 0],
                      [0, 0, 0, -2*n, 0, 0],
                      [0, 0, -n**2, 0, 0, 0]])
        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1/m, 0, 0],
                      [0, 1/m, 0],
                      [0, 0, 1/m]])

        # Specify LQR gains
        Q = np.multiply(.050, np.eye(6))   # State cost
        R = np.multiply(1000, np.eye(3))   # Control cost

        # Solve ARE
        Xare = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        # Compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R)*(B.T*Xare))
        # Change to Array
        self.Klqr = -np.squeeze(np.asarray(K))

    def compute_feedback_control(self, x0):
        u = np.matmul(self.Klqr, x0)
        return np.clip(u, -1, 1)


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
