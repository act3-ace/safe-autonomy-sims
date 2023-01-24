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
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis


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
            "class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
            "config": {
                "num_test_cases": 3
            }
        }

    # construct teams map and test_cases for evaluation

    teams = construct_teams(experiment_config_path, checkpoint_path)

    # plugins
    platform_serialization_obj = platform_serializer_class()
    plugins_args = {"platform_serialization": platform_serialization_obj}
    eval_config_updates = [DoNothingConfigUpdate()]  # default creates list of string(s), isntead of objects

    # engine
    rllib_engine_args = {"callbacks": [], "workers": 0}

    # recorders
    recorder_args = {"dir": output_path, "append_timestamp": False}

    # instantiate eval objects
    # teams = Teams(team_participant_map=team_participant_map)
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


def construct_teams(experiment_config_path: str, checkpoint_path: str):
    """
    This function is responsible for creating the Teams object required by the Evaluation Framework. It uses the experiment
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
    Teams: corl.evaluation.runners.section_factories.teams.Teams
        Maintains a map of every platform, policy_id, agent_config, and name for each entity present in the training environment
    """

    # parse experiment config
    experiment_config = load_file(experiment_config_path)

    assert 'agent_config' in experiment_config
    assert 'platform_config' in experiment_config
    agents_config = experiment_config['agent_config']
    platforms_config = experiment_config['platform_config']

    # populate teams based on experiment config
    teams_platforms_config = []
    teams_agents_config = []
    for index, agent_info in enumerate(agents_config):
        agent_name = agent_info["name"]
        platforms = agent_info["platforms"]
        agent_config_path = agent_info["config"]
        policy_config_path = agent_info["policy"]
        platform_name = platforms_config[index]["name"]
        platform_config_path = platforms_config[index]["config"]

        # TODO: handle relative paths better
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

        agent_loader = CheckpointFile(checkpoint_filename=checkpoint_path)

        teams_platforms_config.append(Platform(name=platform_name, config=platform_config_path))
        teams_agents_config.append(
            Agent(name=agent_name, config=agent_config_path, platforms=platforms, policy=policy_config_path, agent_loader=agent_loader)
        )

    return Teams(agent_config=teams_agents_config, platform_config=teams_platforms_config)


def checkpoints_list_from_training_output(training_output_path: str, output_dir: str, experiment_name: str):
    """
    DEPRICATED: This function was developed for use with ray1.x

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
    experiment_state_path_map: dict,
    task_config_path: str,
    experiemnt_config_path: str,
    metrics_config: dict,
    platfrom_serializer_class: PlatformSerializer,
    plot_output_path: str = None,
    plot_config: dict = None,
    create_plot: bool = False,
    test_case_manager_config: dict = None
):
    """
    This function serves as a high level interface for users conducting ablative studies using CoRL. It is
    responsible for processing a collection of CoRL training outputs (experiment_state files). This includes iteratively
    loading policies from checkpoints, running those policies in evaluation episodes, collecting data, analyzing data
    with Metrics, and optionally visualizing calculated Metrics in a plot.

    Parameters
    ----------
    experiment_state_path_map: dict[str, str]
        A map of experiment name string keys to experiment_state file path string values
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

    evaluation_ouput_dir = '/tmp/ablation_results'

    # evaluate provided trained policies' checkpoints
    experiment_to_eval_results_map = {}
    for experiment_name, experiment_state in experiment_state_path_map.items():

        # if single training output provided, wrap in list
        experiment_states = [experiment_state] if not isinstance(experiment_state, list) else experiment_state
        for index, experiment_state_path in enumerate(experiment_states):

            # append index to evaluation_output path
            experiment_indexed_name = experiment_name + f"__{index}"  # TODO: add info from experiment_state
            experiment_analysis = ExperimentAnalysis(experiment_state_path)
            assert len(experiment_analysis.trials) == 1, f"more than one Trial in experiment_state {experiment_state_path}"  # type: ignore

            checkpoint_paths, output_paths = checkpoints_list_from_experiment_analysis(
                experiment_analysis,
                evaluation_ouput_dir,
                experiment_indexed_name
            )
            metadata = extract_metadata(experiment_analysis)

            # temporarily index experiemnts listed under same name
            experiment_to_eval_results_map[experiment_name + f'__{index}'] = {"output_paths": output_paths, "metadata": metadata}
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
        assert plot_output_path is not None, "'plot_output_path' must be defined in oder to save created plot"
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
    columns = ['experiment', 'experiment_index', 'evaluation_trial_index', "training_iteration"]

    # need to parse each metrics.pkl file + construct DataFrame
    dataframes = []
    for experiment_name in results.keys():
        data = []
        output_paths = results[experiment_name]['output_paths']
        training_metadata_df = results[experiment_name]['metadata']

        # collect metric values per experiment
        for output_path in output_paths:

            # collect agent metrics per ckpt
            with open(output_path + "/metrics.pkl", 'rb') as metrics_file:
                metrics = pickle.load(metrics_file)
            episode_events = list(metrics.participants.values())[0].events  # TODO: remove single agent env assumption

            expr_name, expr_index = experiment_name.split("__")  # separate appended indexing from experiment name
            checkpoint_num = int(output_path.split("_")[-1])  # TODO: want better way to get checkpoint num data here

            for eval_trial_index, event in enumerate(episode_events):
                # aggregate trial data (single dataframe entry)
                row = [expr_name, expr_index, eval_trial_index, checkpoint_num]

                # collect agent's metric values on each trial in eval
                for metric_name in metric_names['agent']['__default__']:  # type: ignore
                    assert metric_name in event.metrics, "{} not an available metric!".format(metric_name)
                    assert hasattr(event.metrics[metric_name], "value"), "metric {} has no 'value' attribute".format(metric_name)
                    # add metric value to row
                    row.append(event.metrics[metric_name].value)
                    # add metric name to columns
                    if metric_name not in columns:
                        columns.append(metric_name)

                # add row to dataset (experiment name, iteration, num_episodes, num_interactions, episode ID/trial, **custom_metrics)
                data.append(row)

            # collect world metrics per ckpt
            ...

        # create experiment dataframe
        expr_dataframe = pd.DataFrame(data, columns=columns)
        # join metadata on training iteration / checkpoint_num
        expr_dataframe = pd.merge(expr_dataframe, training_metadata_df, how="inner", on='training_iteration')
        # add to experiment dataframes collection
        dataframes.append(expr_dataframe)

    # collect experiment dataframes into single DataFrame
    full_dataframe = pd.concat(dataframes).reset_index()

    return full_dataframe


def create_sample_complexity_plot(
    data: pd.DataFrame,
    plot_output_file: str,
    xaxis='timesteps_total',
    yaxis='TotalReward',
    hue='experiment',
    ci=95,
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

    if xaxis == 'timesteps_total':
        plt.xlabel('Timesteps')
    if xaxis == 'episodes_total':
        plt.xlabel('Episodes')
    if xaxis == 'training_iteration':
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


def checkpoints_list_from_experiment_analysis(
    experiment_analysis: ExperimentAnalysis, output_dir: str, experiment_name: str, agent_name: str = "blue0_ctrl"
):
    """
    This function is responsible for compiling a list of paths to each checkpoint in an experiment.
    This acts as a helper function for when users want to evaluate a series of checkpoints from a single training job.

    Parameters
    ----------
    experiment_analysis: ExperimentAnalysis
        The ExperimentAnalysis object containing checkpoints and results of a single training job
    output_dir: str
        The absolute path to the directory that will hold the results from Evaluation Episodes
    experiment_name: str
        The name of the experiment under evaluation
    agent_name: str
        The policy_id used to retrieve the checkpoint for a specific agent

    Returns
    -------
    checkpoint_paths: list
        A list of absolute paths to each checkpoint file fouund in the provided training_output_path
    output_dir_paths: list
        A list of absolute paths used to determine where to save Evaluation Episode data for each checkpoint
    """

    # create ExperimentAnalysis object to handle Trial Checkpoints
    trial = experiment_analysis.trials[0]  # type: ignore
    ckpt_paths = experiment_analysis.get_trial_checkpoints_paths(trial, "training_iteration")

    output_dir_paths = []
    checkpoint_paths = []
    for path, trainig_iteration in ckpt_paths:
        # TODO: look for a way to eliminate manual parse of subdirs -_-
        # TODO: verify policy_state desired over algorithm_state!
        if os.path.isdir(path):
            path += "/policies/" + agent_name + "/policy_state.pkl"

        checkpoint_paths.append(path)
        output_path = output_dir + "/" + experiment_name + "/" + "checkpoint_" + str(trainig_iteration)
        output_dir_paths.append(output_path)

    return checkpoint_paths, output_dir_paths


def extract_metadata(experiment_analysis: ExperimentAnalysis) -> pd.DataFrame:
    """
    This function is responsible for collecting training duration information from the ExperimentAnalysis object.
    This function currently collects the number of training iterations, Episodes, environment interactions, and
    walltime seconds that had occured at the time the checkpoint was created.

    Parameters
    ----------
    checkpoint_paths: list
        A list of path strings to each checkpoint, typically checkpoints are from a single training job

    Returns
    -------
    training_meta_data: pandas.DataFrame
        A collection of training duration information for each provided checkpoint
    """

    # assumes one trial per training job
    trial = experiment_analysis.trials[0]  # type: ignore
    df = experiment_analysis.trial_dataframes[trial.logdir]
    metadata_df = df[['training_iteration', 'timesteps_total', 'episodes_total', 'time_total_s']]

    return metadata_df
