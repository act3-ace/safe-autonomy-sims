"""
This module defines a python API for running CoRL's Evaluation Framework. It also defines helper functions to support the reuse
of training configs, evaluation of multiple trained policies, and creation of comparative plots of chosen Metrics. These functions
streamline visualization and analysis of comparative RL test assays.

Author: John McCarroll
"""

import os
import pickle
import sys
import typing
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from corl.evaluation.default_config_updates import DoNothingConfigUpdate
from corl.evaluation.evaluation_artifacts import (
    EvaluationArtifact_EvaluationOutcome,
    EvaluationArtifact_Metrics,
    EvaluationArtifact_Visualization,
)
from corl.evaluation.launchers import launch_evaluate, launch_generate_metrics, launch_visualize
from corl.evaluation.loader.check_point_file import CheckpointFile
from corl.evaluation.recording.folder import Folder, FolderRecord
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibTrainer
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Task
from corl.evaluation.runners.section_factories.teams import Agent, Platform, Teams
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseManager
from corl.evaluation.visualization.print import Print
from corl.parsers.yaml_loader import load_file


def evaluate(
    task_config_path: str,
    checkpoint_path: str,
    output_path: str,
    experiment_config_path: str,
    platform_serializer_class: PlatformSerializer,
    test_case_manager_config: dict = None
):
    """
    This function is responsible for instantiating necessary arguments and then launching the first stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    task_config_path: str
        The absolute path to the task_config used in training
    checkpoint_path: str
        The absolute path to the checkpoint from which the policy under evaluation will be loaded
    output_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiment_config_path: str
        The absolute path to the experiment config used in training
    platform_serializer_class: PlatformSerializer
        The PlatformSerializer subclass capable of saving the state of the Platforms used in the evaluation environment
    test_case_manager_config: dict
        An optional map of TestCaseManager constructor arguments
    """

    # handle default test_case_manager
    if test_case_manager_config is None:
        test_case_manager_config = {
            "test_case_strategy_class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
            "config": {
                "num_test_cases": 3
            }
        }

    # construct teams map and test_cases for evaluation

    team_participant_map = construct_teams_map_from_experiment_config(experiment_config_path, checkpoint_path)

    # plugins
    platform_serialization_obj = platform_serializer_class()
    plugins_args = {"platform_serialization": platform_serialization_obj}
    eval_config_updates = [DoNothingConfigUpdate()]  # default creates list of string(s), isntead of objects

    # engine
    rllib_engine_args = {"callbacks": [], "workers": 0}

    # recorders
    recorder_args = {"dir": output_path, "append_timestamp": False}

    # instantiate eval objects
    teams = Teams(team_participant_map=team_participant_map)
    task = Task(config_yaml_file=task_config_path)
    test_case_manager = TestCaseManager(**test_case_manager_config)
    plugins = Plugins(**plugins_args)
    plugins.eval_config_update = eval_config_updates
    engine = RllibTrainer(**rllib_engine_args)
    recorder = Folder(**recorder_args)

    # construct namespace dict
    namespace = {
        "teams": teams,
        "task": task,
        "test_cases": {
            "test_case_manager": test_case_manager
        },
        "plugins": plugins,
        "engine": {
            "rllib": engine
        },
        "recorders": [recorder]
    }

    # call main
    launch_evaluate.main(namespace)


def generate_metrics(evaluate_output_path: str, metrics_config: dict):
    """
    This function is responsible for instantiating necessary arguments and then launching the second stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    evaluate_output_path: str
        The absolute path to the directory in which evaluation episodes' data was saved (from the initial 'evaluate' step of the
        Evaluation Framework)
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    """

    # define constructor args
    location = FolderRecord(absolute_path=evaluate_output_path)

    # TODO: enable evaluation without alerts
    alerts_config = {
        "world": [
            {
                "name": "Short Episodes",
                "metric": "rate_of_runs_lt_5steps",
                "scope": "evaluation",
                "thresholds": [{
                    "type": "warning", "condition": {
                        "operator": ">", "lhs": 0
                    }
                }]
            }
        ]
    }

    raise_error_on_alert = True

    # instantiate eval objects
    outcome = EvaluationArtifact_EvaluationOutcome(location=location)
    metrics = EvaluationArtifact_Metrics(location=evaluate_output_path)

    # construct namespace dict
    namespace = {
        "artifact_evaluation_outcome": outcome,
        "artifact_metrics": metrics,
        "metrics_config": metrics_config,
        "alerts_config": alerts_config,
        "raise_on_error_alert": raise_error_on_alert
    }

    launch_generate_metrics.main(namespace)


def visualize(output_path: str):
    """
    This function is responsible for instantiating necessary arguments and then launching the third stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    evaluate_output_path: str
        The absolute path to the directory in which evaluation episodes' data was saved (from the initial 'evaluate' step
        of the Evaluation Framework)
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    """

    artifact_metrics = EvaluationArtifact_Metrics(location=output_path)
    artifact_visualization = EvaluationArtifact_Visualization(location=output_path)
    visualizations = [Print(event_table_print=True)]

    namespace = {"artifact_metrics": artifact_metrics, "artifact_visualization": artifact_visualization, "visualizations": visualizations}

    launch_visualize.main(namespace)


def construct_teams_map_from_experiment_config(experiment_config_path: str, checkpoint_path: str):
    """
    This function is responsible for creating the team_participant_map required by the Evaluation Framework. It uses the experiment
    config file from training to get agent and platform info required by the Teams class. Use of this function assumes the user wishes
    to replicate the training environment for evaluation episodes.

    Parameters
    ----------
    experiment_config_path: str
        The absolute path to the experiment config used in training
    checkpoint_path: str
        The absolute path to the checkpoint from which each agent's policy will be loaded

    Returns
    -------
    team_participation_map: dict
        A map of every platform, policy_id, agent_config, and name for each entity present in the training environment
    """

    # parse experiment config
    experiment_config = load_file(experiment_config_path)

    assert 'agent_config' in experiment_config
    assert 'platform_config' in experiment_config
    agents_config = experiment_config['agent_config']
    platforms_config = experiment_config['platform_config']

    # populate teams based on experiment config
    blue_team = []  # TODO: fix one team assumption!!!
    for index, agent_info in enumerate(agents_config):
        agent_name, _, agent_config_path, policy_config_path = agent_info  # assumes policy_id == agent_name!!!
        _, platform_config_path = platforms_config[index]

        # handle relative paths from experiment config
        ## identify if experiment config from corl or sas

        is_corl_experiment = 'corl' in experiment_config_path.split('/')
        is_sas_experiment = 'safe-autonomy-sims' in experiment_config_path.split('/')
        if is_corl_experiment:
            # assumes corl root is cwd
            path_to_assumed_root = experiment_config_path.split('corl')[0] + 'corl/'
        elif is_sas_experiment:
            # assumes safe-autonomy-sims root is cwd
            path_to_assumed_root = experiment_config_path.split('safe-autonomy-sims')[0] + 'safe-autonomy-sims/'
        else:
            raise ValueError(
                "Experiment config {} does not reside in corl or safe-autonomy-sims repositories".format(experiment_config_path)
            )

        agent_config_path = os.path.join(path_to_assumed_root, agent_config_path)
        platform_config_path = os.path.join(path_to_assumed_root, platform_config_path)
        policy_config_path = os.path.join(path_to_assumed_root, policy_config_path)

        agent_loader = CheckpointFile(checkpoint_filename=checkpoint_path, policy_id=agent_name)

        agent_config = {
            "name": agent_name, "agent_config": agent_config_path, "policy_config": policy_config_path, "agent_loader": agent_loader
        }

        agents = [Agent(**agent_config)]

        platform_config = {"platform_config": platform_config_path, "agents": agents}
        blue_team.append(Platform(**platform_config))

    team_participant_map = {"blue": blue_team}

    return team_participant_map


def checkpoints_list_from_training_output(training_output_path: str, output_dir: str, experiment_name: str):
    """
    This function is responsible for compiling a list of paths to each checkpoint in a training output directory.
    This acts as a helper function for when users want to evaluate a series of checkpoints from a single training job.

    Parameters
    ----------
    training_output_path: str
        The absolute path to the directory containing saved checkpoints and results from a training job
    output_dir: str
        The absolute path to the directory that will hold the results from Evaluation Episodes
    experiment_name: str
        The name of the experiment under evaluation

    Returns
    -------
    checkpoint_paths: list
        A list of absolute paths to each checkpoint file fouund in the provided training_output_path
    output_dir_paths: list
        A list of absolute paths used to determine where to save Evaluation Episode data for each checkpoint
    """

    # assume training_output path absolute
    ckpt_dirs = sorted(glob(training_output_path + "/checkpoint_*"), key=lambda path: int(path.split("/")[-1].split("_")[-1]))
    output_dir_paths = []
    checkpoint_paths = []
    # add checkpoint files
    for ckpt_dir in ckpt_dirs:
        ckpt_num = str(int(ckpt_dir.split("/")[-1].split("_")[-1]))
        checkpoint_paths.append(ckpt_dir + "/checkpoint-" + ckpt_num)

        # add checkpoint to list of outputs
        output_path = output_dir + "/" + experiment_name + "/" + "ckpt_" + ckpt_num
        output_dir_paths.append(output_path)

    return checkpoint_paths, output_dir_paths


def add_required_metrics(metrics_config: dict):
    """
    This helper function is responsible for adding in a few Metrics that are required by the Evaluation Framework. This is to
    simplify the configuration process for the user.

    Parameters
    ----------
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    metrics_config: dict
        The mutated metrics_config
    """

    # add required metrics to metrics_config
    episode_length_metric = {
        "name": "EpisodeLength(Steps)",
        "functor": "corl.evaluation.metrics.generators.meta.episode_length.EpisodeLength_Steps",
        "config": {
            "description": "episode length of test case rollout in number of steps"
        }
    }
    episode_length_alert_metric = {
        "name": "rate_of_runs_lt_5steps",
        "functor": "corl.evaluation.metrics.aggregators.criteria_rate.CriteriaRate",
        "config": {
            "description": "alert metric to see if any episode length is less than 5 steps",
            "metrics_to_use": "EpisodeLength(Steps)",
            "scope": {
                "type": "corl.evaluation.metrics.scopes.from_string", "config": {
                    "name": "evaluation"
                }
            },
            "condition": {
                "operator": '<', "lhs": 5
            }
        }
    }

    if 'world' in metrics_config:
        assert isinstance(metrics_config['world'], list), "'world' metrics in metrics_config must be list of dicts"
        metrics_config['world'].append(episode_length_metric)
        metrics_config['world'].append(episode_length_alert_metric)
    else:
        metrics_config['world'] = [episode_length_metric, episode_length_alert_metric]

    return metrics_config


def run_ablation_study(
    experiment_training_output_paths: dict,
    task_config_path: str,
    experiemnt_config_path: str,
    metrics_config: dict,
    platfrom_serializer_class: PlatformSerializer,
    plot_output_path: str,
    plot_config: dict = None,
    create_plot: bool = True,
    test_case_manager_config: dict = None
):
    """
    This function serves as a high level interface for users conducting ablative studies using CoRL. It is
    responsible for processing a collection of CoRL training outputs. This includes iteratively loading policies
    from checkpoints, running those policies in evaluation episodes, collecting data, analyzing data with Metrics,
    and visualizing calculated Metrics in a plot.

    Parameters
    ----------
    experiment_training_output_paths: dict[str, str]
        A map of experiment name string keys to training output directory path string values
    task_config_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiemnt_config_path: str
        The absolute path to the experiment config used in training
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    platfrom_serializer_class: PlatformSerializer
        The class object capable of storing data specific to the Platform type used in training
    plot_output_path: str
        The absolute path to the location where the resulting plot will be saved
    plot_config: dict
        The kwargs that will be passed to seaborn's lineplot method
    create_plot: bool
        A boolean to determine whether or not to create the seaborn plot. If True, the plot is created and saved
    test_case_manager_config: dict
        The kwargs passed to the TestCaseManager constructor. This must define the TestCaseStrategy class and its config

    Returns
    -------
    data: pandas.DataFrame
        The DataFrame containing Metric data from all provided experiments collected during evaluation
    """

    # handle default TestCaseManager config
    if test_case_manager_config is None:
        test_case_manager_config = {
            "test_case_strategy_class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
            "config": {
                "num_test_cases": 3
            }
        }
    # handle default plot config
    if plot_config is None:
        plot_config = {}

    # add required metrics
    metrics_config = add_required_metrics(metrics_config)

    evaluation_ouput_dir = '/tmp/ablation_results/'

    # evaluate provided trained policies' checkpoints
    experiment_to_eval_results_map = {}
    for experiment_name, experiment_dir in experiment_training_output_paths.items():
        checkpoint_paths, output_paths = checkpoints_list_from_training_output(experiment_dir, evaluation_ouput_dir, experiment_name)
        metadata = extract_metadata(checkpoint_paths)
        experiment_to_eval_results_map[experiment_name] = {"output_paths": output_paths, "metadata": metadata}
        run_evaluations(
            task_config_path,
            experiemnt_config_path,
            metrics_config,
            checkpoint_paths,
            output_paths,
            platfrom_serializer_class,
            test_case_manager_config=test_case_manager_config,
            visualize_metrics=False
        )

    # create sample complexity plot
    data = construct_dataframe(experiment_to_eval_results_map, metrics_config)

    # TODO: for metric in metric_config: (plot every given metric?)
    if create_plot:
        create_sample_complexity_plot(data, plot_output_path, **plot_config)

    return data


def run_evaluations(
    task_config_path: str,
    experiemnt_config_path: str,
    metrics_config: dict,
    checkpoint_paths: list,
    output_paths: list,
    platfrom_serializer_class: PlatformSerializer,
    test_case_manager_config: dict = None,
    visualize_metrics: bool = False
):
    """
    This function is responsible for taking a list of checkpoint paths and iteratively running them through
    each stage of the Evaluation Framework (running Evaluation Episodes, processing results to generate Metrics,
    and optionally visualizing those Metrics).

    Parameters
    ----------
    task_config_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiemnt_config_path: str
        The absolute path to the experiment config used in training
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    checkpoint_paths: list
        A list of path strings to each checkpoint, typically checkpoints are from a single training job
    output_paths: list
        A list of unique path strings at which each checkpoint's Evaluation Episodes data and Metrics will be stored.
        Must match the length of the checkpoint_paths list
    platfrom_serializer_class: PlatformSerializer
        The class object capable of storing data specific to the Platform type used in training
    test_case_manager_config: dict
        The kwargs passed to the TestCaseManager constructor. This must define the TestCaseStrategy class and its config
    visualize_metrics: bool
        A boolean to determine whether or not to run the Evaluation Framework's Visualize stage.
        If True, visualize is called
    """

    kwargs = {}
    if test_case_manager_config is not None:
        kwargs['test_case_manager_config'] = test_case_manager_config

    # run sequence of evaluation
    for index, ckpt_path in enumerate(checkpoint_paths):
        # run evaluation episodes
        try:
            evaluate(task_config_path, ckpt_path, output_paths[index], experiemnt_config_path, platfrom_serializer_class, **kwargs)
        except SystemExit:
            print(sys.exc_info()[0])

        # generate evaluation metrics
        generate_metrics(output_paths[index], metrics_config)

        if visualize_metrics:
            # generate visualizations
            visualize(output_paths[index])


def extract_metadata(checkpoint_paths):
    """
    This function is responsible for collecting training duration information from the tune_metadata
    file stored in each checkpoint directory. This function currently collects the number of training
    iterations, Episodes, environment interactions and walltime seconds that had occured at the time
    the checkpoint was created.

    Parameters
    ----------
    checkpoint_paths: list
        A list of path strings to each checkpoint, typically checkpoints are from a single training job

    Returns
    -------
    training_meta_data: list[dict]
        An ordered collection of training duration information for each provided checkpoint
    """

    # iterate through checkpoint dirs for given experiment, extract data on training duration
    training_meta_data = []
    for ckpt_path in checkpoint_paths:
        metadata_path = ckpt_path + ".tune_metadata"
        metadata = pickle.load(open(metadata_path, 'rb'))
        num_env_interactions = metadata["timesteps_total"] if metadata["timesteps_total"] else metadata["last_result"]["counters"][
            "num_env_steps_trained"]
        checkpoint_data = {
            'num_env_interactions': num_env_interactions,
            'num_episodes': metadata["episodes_total"],
            'walltime': metadata["time_total"],
            'iterations': metadata["iteration"],
        }
        training_meta_data.append(checkpoint_data)

    return training_meta_data


def parse_metrics_config(metrics_config: dict) -> typing.Dict[str, str]:
    """
    This function is responsible for walking the metrics config and creating a dictionary of Metric
    names present in the metrics config.

    Parameters
    ----------
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    metrics_names: dict
        A dictionary resembling the metrics_config, but containing only Metric names
    """
    # collect metric names
    metrics_names = {}  # type: ignore
    if 'world' in metrics_config:
        metrics_names['world'] = []
        for world_metric in metrics_config["world"]:
            metrics_names['world'].append(world_metric['name'])

    if 'agent' in metrics_config:
        # TODO: support agent-specific metrics
        metrics_names['agent'] = {}
        if '__default__' in metrics_config['agent']:
            metrics_names['agent']['__default__'] = []
            for agent_metric in metrics_config["agent"]["__default__"]:
                metrics_names['agent']['__default__'].append(agent_metric['name'])

    return metrics_names


def construct_dataframe(results: dict, metrics_config: dict):
    """
    This function is responsible for parsing Metric data from the results of Evaluation Episodes.
    It collects values from all Metrics included in the metrics_config into a single pandas.DataFrame,
    which is returned by the function.

    Parameters
    ----------
    results: dict[str, dict]
        A collection mapping experiment names to Evaluation result locations and training duration metadata
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    dataframe: pandas.DataFrame
        A DataFrame containing Metric values collected for each checkpoint of each experiment by row
    """
    metric_names = parse_metrics_config(metrics_config)
    columns = ['experiment', 'iterations', 'num_episodes', 'num_interactions', 'episode ID']

    # need to parse each metrics.pkl file + construct DataFrame
    data = []
    for experiment_name in results.keys():
        output_paths = results[experiment_name]['output_paths']
        training_metadata = results[experiment_name]['metadata']

        # collect metric values per experiment
        for index, output_path in enumerate(output_paths):

            # extract amouunt of training policy had before eval
            num_env_interactions = training_metadata[index]["num_env_interactions"]
            num_episodes = training_metadata[index]["num_episodes"]
            num_iterations = training_metadata[index]["iterations"]

            # collect agent metrics per ckpt
            with open(output_path + "/metrics.pkl", 'rb') as metrics_file:
                metrics = pickle.load(metrics_file)
            episode_events = list(metrics.participants.values())[0].events  # TODO: remove single agent env assumption

            index = 0
            for event in episode_events:
                # aggregate trial data (single dataframe entry)
                row = [experiment_name, num_iterations, num_episodes, num_env_interactions]

                # track trial number / ID
                episode_id = index
                index += 1
                row.append(episode_id)

                # collect agent's metric values on each trial in eval
                for metric_name in metric_names['agent']['__default__']:  # type: ignore
                    assert metric_name in event.metrics, "{} not an available metric!".format(metric_name)
                    # add metric value to row
                    row.append(event.metrics[metric_name].value)
                    # add metric name to columns
                    if metric_name not in columns:
                        columns.append(metric_name)

                # add row to dataset (experiment name, iteration, num_episodes, num_interactions, episode ID/trial, **custom_metrics)
                data.append(row)

            # collect world metrics per ckpt
            ...

    # convert data into DataFrame
    dataframe = pd.DataFrame(data, columns=columns)

    return dataframe


def create_sample_complexity_plot(
    data: pd.DataFrame,
    plot_output_file: str,
    xaxis='num_interactions',
    yaxis='TotalReward',
    hue='experiment',
    ci='sd',
    xmax=None,
    ylim=None,
    **kwargs
):
    """
    This function is responsible for creating a comparative sample complexity plot with seaborn. The function takes in
    a pandas.DataFrame of Evaluation results and specifications of kwargs to determine x and y axis and confidence
    intervals and saves the resulting plot to the desired location.

    Parameters
    ----------
    data: pd.DataFrame
        A DataFrame containing Metric values collected for each checkpoint of each experiment by row
    plot_output_file: str
        The location at which the plot will be saved
    xaxis: str
        The x axis of the plot. Must be a column in the 'data' DataFrame. Currently 'iterations',
        'num_episodes', and 'num_interactions' are supported
    yaxis: str
        The y axis of the plot. Must be a column in the 'data' DataFrame, typically the name of a Metric that was
        included in the metrics_config during Evaluation.
    hue: str
        The column of the 'data' DataFrame on which to group data
    ci: int or str
        The confidence interval to be displayed on the plot
    xmax: int
        The maximum value displayed on the x axis
    ylim: int
        The maximum value displayed on the y axis
    kwargs
        Additional kwargs to pass to seaborn's lineplot method

    Returns
    -------
    plot: matplotlib.axis.Axis
        An axis containing the resulting plot
    """
    # create seaborn plot
    # TODO: upgrade seaborn to 0.12.0 and swap depricated 'ci' for 'errorbar' kwarg: errorbar=('sd', 1) or (''ci', 95)
    # TODO: add smooth kwarg

    sns.set(style="darkgrid", font_scale=1.5)

    # feed Dataframe to seaborn to create labelled plot
    plot = sns.lineplot(data=data, x=xaxis, y=yaxis, hue=hue, ci=ci, **kwargs)

    # format plot

    plt.legend(loc='best').set_draggable(True)

    if xaxis == 'num_interactions':
        plt.xlabel('Timesteps')
    if xaxis == 'num_episodes':
        plt.xlabel('Episodes')
    if xaxis == 'iterations':
        plt.xlabel('Iterations')
    if yaxis == 'TotalReward':
        plt.ylabel('Average Reward')

    if xmax is None:
        xmax = np.max(np.asarray(data[xaxis]))
    plt.xlim(right=xmax)

    if ylim is not None:
        plt.ylim(ylim)

    xscale = xmax > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)

    # save plot

    plot.get_figure().savefig(plot_output_file)

    return plot
