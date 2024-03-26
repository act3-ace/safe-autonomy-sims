"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines a python API for running CoRL's Evaluation Framework.

It also defines helper functions to support the reuse
of training configs, evaluation of multiple trained policies,
and creation of comparative plots of chosen Metrics. These functions
streamline visualization and analysis of comparative RL test assays.
"""

# pylint: disable=E0401

import os
import pickle
import re
import sys
import typing
from glob import glob

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from corl.evaluation.default_config_updates import DoNothingConfigUpdate
from corl.evaluation.evaluation_artifacts import (
    EvaluationArtifact_EvaluationOutcome,
    EvaluationArtifact_Metrics,
    EvaluationArtifact_Visualization,
)
from corl.evaluation.launchers import launch_evaluate, launch_generate_metrics, launch_visualize
from corl.evaluation.loader.policy_checkpoint import PolicyCheckpoint
from corl.evaluation.recording.folder import Folder, FolderRecord
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibTrainer
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Task
from corl.evaluation.runners.section_factories.teams import LoadableCorlAgent, Teams
from corl.evaluation.visualization.print import Print
from corl.parsers.agent_and_platform import CorlPlatformConfigArgs
from corl.parsers.yaml_loader import load_file
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis


def evaluate(
    task_config_path: str,
    checkpoint_path: str,
    output_path: str,
    experiment_config_path: str,
    launch_dir_of_experiment: str,
    platform_serializer_class: PlatformSerializer,
    test_case_manager_config: dict | None = None,
    rl_algorithm_name: str | None = None,
    num_workers: int = 1
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
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    platform_serializer_class: PlatformSerializer
        The PlatformSerializer subclass capable of saving the state of the Platforms used in the evaluation environment
    test_case_manager_config: dict
        An optional map of TestCaseManager constructor arguments
    """

    # handle default test_case_manager
    if test_case_manager_config is None:
        test_case_manager_config = {
            "type": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
            "config": {
                "num_test_cases": 3
            }
        }

    # construct teams map and test_cases for evaluation

    teams = construct_teams(experiment_config_path, launch_dir_of_experiment, checkpoint_path)

    # plugins
    platform_serialization_obj = platform_serializer_class()
    eval_config_updates = [DoNothingConfigUpdate()]  # default creates list of string(s), instead of objects
    plugins_args = {"platform_serialization": platform_serialization_obj, 'eval_config_update': eval_config_updates}

    # recorders
    recorder_args = {"dir": output_path, "append_timestamp": False}

    # instantiate eval objects
    task = Task(config_yaml_file=task_config_path)
    plugins = Plugins(**plugins_args)
    plugins.eval_config_update = eval_config_updates

    # get rllib_engine config from task config
    if rl_algorithm_name:
        trainer_cls = rl_algorithm_name
    else:
        trainer_cls = task.experiment_parse.config['tune_config'][1].get('run_or_experiment',
                                                                         None) if task.experiment_parse.config['tune_config'][1] else None
        trainer_cls = task.experiment_parse.config['tune_config'][0].get('run_or_experiment', 'PPO') if trainer_cls is None else trainer_cls

    # handle multiprocessing
    rllib_engine_args = {"callbacks": [], "workers": num_workers, "trainer_cls": trainer_cls}
    task.experiment_parse.config['rllib_configs']['default'][1]["rollout_fragment_length"] = "auto"

    # handle grid search seeds
    if isinstance(task.experiment_parse.config['rllib_configs']['local'][0]['seed'], dict):
        task.experiment_parse.config['rllib_configs']['local'][0]['seed'] = 1

    print("seed: " + str(task.experiment_parse.config['rllib_configs']['local'][0]['seed']) + '\n')

    engine = RllibTrainer(**rllib_engine_args)
    recorder = Folder(**recorder_args)

    # construct namespace dict
    namespace = {
        "teams": teams,
        "task": task,
        "test_case_manager": test_case_manager_config,
        "plugins": plugins,
        "engine": {
            "rllib": engine
        },
        "recorders": [recorder],
        "tmpdir_base": '/tmp/'
    }

    args = jsonargparse.dict_to_namespace({"include_dashboard": False, "tmpdir_base": "/tmp", "cfg": "./generated_config.yml"})

    # call main
    launch_evaluate.main(instantiated_args=args, config=namespace)


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


def visualize(evaluate_output_path: str):
    """
    This function is responsible for instantiating necessary arguments and then launching the third stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    evaluate_output_path: str
        The absolute path to the directory in which evaluation episodes' data was saved (from the initial 'evaluate' step
        of the Evaluation Framework)
    """

    artifact_metrics = EvaluationArtifact_Metrics(location=evaluate_output_path)
    artifact_visualization = EvaluationArtifact_Visualization(location=evaluate_output_path)
    visualizations = [Print(event_table_print=True)]

    namespace = {"artifact_metrics": artifact_metrics, "artifact_visualization": artifact_visualization, "visualizations": visualizations}

    launch_visualize.main(namespace)


def construct_teams(experiment_config_path: str, launch_dir_of_experiment: str, checkpoint_path: str):
    """
    This function is responsible for creating the Teams object required by the Evaluation Framework. It uses the experiment
    config file from training to get agent and platform info required by the Teams class. Use of this function assumes the user wishes
    to replicate the training environment for evaluation episodes.

    Parameters
    ----------
    experiment_config_path: str
        The absolute path to the experiment config used in training
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    checkpoint_path: str
        The absolute path to the checkpoint from which each agent's policy will be loaded.
        The directory titled the agent's name will be filled in programmatically during this function.
        An example of the format of this string is: '/path/to/experiment/output/checkpoint_00005/policies/{}/policy_state.pkl'.

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

        # handle relative paths from experiment config
        agent_config_path = os.path.join(launch_dir_of_experiment, agent_config_path)
        platform_config_path = os.path.join(launch_dir_of_experiment, platform_config_path)
        policy_config_path = os.path.join(launch_dir_of_experiment, policy_config_path)

        agent_loader = PolicyCheckpoint(checkpoint_filename=checkpoint_path, trained_agent_id=agent_name)

        teams_platforms_config.append(CorlPlatformConfigArgs(name=platform_name, config=platform_config_path))
        teams_agents_config.append(
            LoadableCorlAgent(
                name=agent_name, config=agent_config_path, platforms=platforms, policy=policy_config_path, agent_loader=agent_loader
            )
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
    launch_dir_of_experiment: str,
    metrics_config: dict,
    platfrom_serializer_class: PlatformSerializer,
    plot_output_path: str | None = None,
    plot_config: dict | None = None,
    create_plot: bool = False,
    test_case_manager_config: dict | None = None,
    trial_indices: list | None = None,
    rl_algorithm_name: str | None = None
):  # pylint:disable=too-many-locals
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
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
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

    # construct keys by appending experiment group name, experiment index, and trial index together
    experiment_to_eval_results_map = {}

    for experiment_name, experiment_state in experiment_state_path_map.items():

        # if single training output provided, wrap in list
        experiment_states = [experiment_state] if not isinstance(experiment_state, list) else experiment_state
        for experiment_index, experiment_state_path in enumerate(experiment_states):

            # append index to evaluation_output path
            ray.init()
            experiment_analysis = ExperimentAnalysis(experiment_state_path)

            if trial_indices is None:
                trial_indices = list(range(0, len(experiment_analysis.trials)))

            for trial_index in trial_indices:  # type: ignore

                # create trial key
                trial_key = f"{experiment_name}__{experiment_index}__{trial_index}"

                checkpoint_paths, output_paths = checkpoints_list_from_experiment_analysis(
                    experiment_analysis,
                    evaluation_ouput_dir,
                    trial_key,
                    trial_index=trial_index
                )
                metadata = extract_metadata(experiment_analysis, trial_index=trial_index)

                # temporarily index experiemnts listed under same name
                experiment_to_eval_results_map[trial_key] = {"output_paths": output_paths, "metadata": metadata}
                run_evaluations(
                    task_config_path,
                    experiemnt_config_path,
                    launch_dir_of_experiment,
                    metrics_config,
                    checkpoint_paths,
                    output_paths,
                    platfrom_serializer_class,
                    test_case_manager_config=test_case_manager_config,
                    visualize_metrics=False,
                    rl_algorithm_name=rl_algorithm_name,
                    experiment_name=experiment_name
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
    launch_dir_of_experiment: str,
    metrics_config: dict,
    checkpoint_paths: list,
    output_paths: list,
    platfrom_serializer_class: PlatformSerializer,
    test_case_manager_config: dict | None = None,
    visualize_metrics: bool = False,
    rl_algorithm_name: str | None = None,
    experiment_name: str = "",
    num_workers: int = 1
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
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
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
        If True, visualize is called.
    experiment_name: str
        The name of the experiment. Used for standard output progress updates.
    """

    kwargs: dict[str, typing.Any] = {}
    if test_case_manager_config:
        kwargs['test_case_manager_config'] = test_case_manager_config

    # rl algorithm
    if rl_algorithm_name is not None:
        kwargs['rl_algorithm_name'] = rl_algorithm_name

    kwargs["num_workers"] = num_workers

    # run sequence of evaluation
    for index, ckpt_path in enumerate(checkpoint_paths):
        # print progress
        ckpt_num_regex = re.search(r'(checkpoint_)\w+', ckpt_path)

        if ckpt_num_regex is not None:
            ckpt_num = ckpt_path[ckpt_num_regex.start():ckpt_num_regex.end()]
            if experiment_name:
                print("\nExperiment: " + experiment_name)
            print("Evaluating " + ckpt_num + ". " + str(index + 1) + " of " + str(len(checkpoint_paths)) + " checkpoints.")

            # run evaluation episodes
            try:
                evaluate(
                    task_config_path,
                    ckpt_path,
                    output_paths[index],
                    experiemnt_config_path,
                    launch_dir_of_experiment,
                    platfrom_serializer_class,
                    **kwargs
                )
            except SystemExit:
                print(sys.exc_info()[0])

            # generate evaluation metrics
            generate_metrics(output_paths[index], metrics_config)

            if visualize_metrics:
                # generate visualizations
                visualize(output_paths[index])


def run_one_evaluation(
    task_config_path: str,
    experiment_config_path: str,
    launch_dir_of_experiment: str,
    metrics_config: dict,
    checkpoint_path: str,
    platfrom_serializer_class: PlatformSerializer,
    test_case_manager_config: dict | None = None,
    rl_algorithm_name: str | None = None,
    num_workers: int = 1
):
    """
    This function is responsible for taking a single checkpoint path and running it through
    each stage of the Evaluation Framework (running Evaluation Episodes, processing results
    to generate Metrics, and optionally visualizing those Metrics).

    Parameters
    ----------
    task_config_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiment_config_path: str
        The absolute path to the experiment config used in training
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    checkpoint_path: str
        Path strings to the checkpoint
    platfrom_serializer_class: PlatformSerializer
        The class object capable of storing data specific to the Platform type used in training
    test_case_manager_config: dict
        The kwargs passed to the TestCaseManager constructor. This must define the TestCaseStrategy class and its config
    """

    kwargs: dict[str, typing.Any] = {}
    if test_case_manager_config:
        kwargs['test_case_manager_config'] = test_case_manager_config

    # rl algorithm
    if rl_algorithm_name:
        kwargs['rl_algorithm_name'] = rl_algorithm_name

    kwargs["num_workers"] = num_workers

    metrics_config = add_required_metrics(metrics_config)

    exp_name = 'single_episode__0__0'
    output_path = '/tmp/eval_results/' + exp_name

    # run evaluation episodes
    try:
        evaluate(
            task_config_path,
            checkpoint_path,
            output_path,
            experiment_config_path,
            launch_dir_of_experiment,
            platfrom_serializer_class,
            **kwargs
        )
    except SystemExit:
        print(sys.exc_info()[0])

    # generate evaluation metrics
    generate_metrics(output_path, metrics_config)

    experiment_to_eval_results_map = {}
    experiment_to_eval_results_map[exp_name] = {"output_paths": [output_path], "metadata": pd.DataFrame({'training_iteration': [0.]})}
    data = construct_dataframe(experiment_to_eval_results_map, metrics_config)
    return data


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


def construct_dataframe(results: dict, metrics_config: dict):  # pylint: disable=R0914
    """
    This function is responsible for parsing Metric data from the results of Evaluation Episodes.
    It collects values from all Metrics included in the metrics_config into a single pandas.DataFrame,
    which is returned by the function.

    Parameters
    ----------
    results: dict[str, dict]
        A map of experiment names to Evaluation result locations and training duration metadata
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    dataframe: pandas.DataFrame
        A DataFrame containing Metric values collected for each checkpoint of each experiment by row
    """
    metric_names = parse_metrics_config(metrics_config)
    columns = ['experiment', 'experiment_index', 'evaluation_episode_index', "training_iteration", "agent_name", "agent"]

    # need to parse each metrics.pkl file + construct DataFrame
    dataframes = []
    for trial_id in results.keys():  # pylint: disable=R1702
        data = []
        output_paths = results[trial_id]['output_paths']
        training_metadata_df = results[trial_id]['metadata']

        # collect metric values per experiment
        for output_path in output_paths:

            # collect agent metrics per ckpt
            with open(output_path + "/metrics.pkl", 'rb') as metrics_file:
                metrics = pickle.load(metrics_file)

            for agent_name, participant in metrics.participants.items():

                episode_events = list(participant.events)

                experiment_name, experiment_index, _ = trial_id.split("__")
                checkpoint_num = int(output_path.split("_")[-1])  # TODO: want better way to get checkpoint num data here

                experiment_agent = experiment_name + '_' + agent_name

                for evaluation_episode_index, event in enumerate(episode_events):

                    # aggregate trial data (single dataframe entry)
                    row = [experiment_name, experiment_index, evaluation_episode_index, checkpoint_num, agent_name, experiment_agent]

                    # collect agent's metric values on each trial in eval
                    for metric_name in metric_names['agent']['__default__']:  # type: ignore
                        assert metric_name in event.metrics, f"{metric_name} not an available metric!"
                        # add metric value to row
                        if hasattr(event.metrics[metric_name], "value"):
                            row.append(event.metrics[metric_name].value)
                        elif hasattr(event.metrics[metric_name], "arr"):
                            row.append(event.metrics[metric_name].arr)
                        else:
                            raise ValueError("Metric must have attribute 'value' or 'arr'")
                        # add metric name to columns
                        if metric_name not in columns:
                            columns.append(metric_name)

                    # add row to dataset [experiment name, iteration, num_episodes, num_interactions, episode ID/trial, **custom_metrics]
                    data.append(row)

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
    hue='agent',
    errorbar=('ci', 95),
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
    errorbar: tuple
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

    plt.clf()

    sns.set(style="darkgrid", font_scale=1.5)

    # feed Dataframe to seaborn to create labelled plot
    plot = sns.lineplot(data=data, x=xaxis, y=yaxis, hue=hue, errorbar=errorbar, **kwargs)

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
    experiment_analysis: ExperimentAnalysis, output_dir: str, experiment_name: str, trial_index: int = 0
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

    Returns
    -------
    checkpoint_paths: list
        A list of absolute paths to each checkpoint file fouund in the provided training_output_path
    output_dir_paths: list
        A list of absolute paths used to determine where to save Evaluation Episode data for each checkpoint
    """

    # create ExperimentAnalysis object to handle Trial Checkpoints
    trial = experiment_analysis.trials[trial_index]  # type: ignore
    ckpt_paths = experiment_analysis.get_trial_checkpoints_paths(trial, "training_iteration")

    output_dir_paths = []
    checkpoint_paths = []
    for path, trainig_iteration in ckpt_paths:
        # TODO: verify policy_state desired over algorithm_state!
        if os.path.isdir(path):
            path += "/policies/{}/policy_state.pkl"

        checkpoint_paths.append(path)
        output_path = output_dir + "/" + experiment_name + "/" + "checkpoint_" + str(trainig_iteration)
        output_dir_paths.append(output_path)

    return checkpoint_paths, output_dir_paths


def extract_metadata(experiment_analysis: ExperimentAnalysis, trial_index: int = 0) -> pd.DataFrame:
    """
    This function is responsible for collecting training duration information from the ExperimentAnalysis object.
    This function currently collects the number of training iterations, Episodes, environment interactions, and
    walltime seconds that had occured at the time the checkpoint was created.

    Parameters
    ----------
    experiment_analysis : ExperimentAnalysis
        Ray Tune experiment analysis object
    trial_index : int
        the index of the trial of interest

    Returns
    -------
    training_meta_data: pandas.DataFrame
        A collection of training duration information for each provided checkpoint
    """

    # assumes one trial per training job
    trial = experiment_analysis.trials[trial_index]  # type: ignore
    df = experiment_analysis.trial_dataframes[trial.logdir]
    training_meta_data = df[['training_iteration', 'timesteps_total', 'episodes_total', 'time_total_s']]
    # add trial index
    training_meta_data['trial_index'] = [trial_index] * training_meta_data.shape[0]

    return training_meta_data
