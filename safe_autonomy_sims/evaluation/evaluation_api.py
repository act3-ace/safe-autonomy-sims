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

import re
import sys
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from corl.evaluation.api import evaluate
from corl.evaluation.api_utils import add_required_metrics, construct_dataframe, extract_metadata_from_result_file, get_checkpoints_paths
from corl.evaluation.evaluation_artifacts import (
    EvaluationArtifact_EvaluationOutcome,
    EvaluationArtifact_Metrics,
    EvaluationArtifact_Visualization,
)
from corl.evaluation.launchers import launch_generate_metrics, launch_visualize
from corl.evaluation.recording.folder import FolderRecord
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.evaluation.runners.section_factories.test_cases.default_strategy import DefaultStrategy
from corl.evaluation.visualization.print import Print


def run_ablation_study(
    experiment_output_paths_map: dict[str, str],
    task_config_path: str,
    experiment_config_path: str,
    launch_dir_of_experiment: str,
    metrics_config: dict[str, dict | list],
    platfrom_serializer_class: PlatformSerializer,
    evaluation_ouput_dir: str,
    plot_output_path: str | None = None,
    plot_config: dict | None = None,
    create_plot: bool = False,
    test_case_manager_config: dict | None = None,
    rl_algorithm_name: str | None = None,
    num_workers: int = 1
) -> pd.DataFrame:  # pylint:disable=too-many-locals
    """
    This function serves as a high level interface for users conducting ablation studies using CoRL. It is
    responsible for processing a collection of CoRL training outputs (experiment output directories).
    This includes iteratively loading policies from checkpoints, running those policies in evaluation episodes,
    collecting data, analyzing data with Metrics, and optionally visualizing calculated Metrics in a plot.

    Parameters
    ----------
    experiment_output_paths_map: dict[str, str]
        A map of experiment name string keys to experiment output directory path strings
    task_config_path: str
        The absolute path to the task config yaml file used in training
    experiment_config_path: str
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
        test_case_manager_config = {"type": f"{DefaultStrategy.__module__}.{DefaultStrategy.__name__}", "config": {"num_test_cases": 3}}
    # handle default plot config
    if plot_config is None:
        plot_config = {}

    # add required metrics
    metrics_config = add_required_metrics(metrics_config)

    # evaluate provided trained policies' checkpoints

    # construct keys by appending experiment name and trial index together
    trial_to_eval_results_map = {}

    for experiment_name, trial_dir_path in experiment_output_paths_map.items():

        # if single training output provided, wrap in list
        trial_dirs = [trial_dir_path] if not isinstance(trial_dir_path, list) else trial_dir_path
        for trial_index, trial_dir_path in enumerate(trial_dirs):

            # create trial key
            trial_key = f"{experiment_name}__{trial_index}"

            checkpoint_paths, output_paths = get_checkpoints_paths(trial_dir_path, trial_key, evaluation_ouput_dir)
            metadata = extract_metadata_from_result_file(trial_dir_path)

            # temporarily index experiments listed under same name
            trial_to_eval_results_map[trial_key] = {"output_paths": output_paths, "metadata": metadata}
            run_evaluations(
                task_config_path,
                experiment_config_path,
                launch_dir_of_experiment,
                metrics_config,
                checkpoint_paths,
                output_paths,
                platfrom_serializer_class,
                test_case_manager_config=test_case_manager_config,
                visualize_metrics=False,
                rl_algorithm_name=rl_algorithm_name,
                experiment_name=experiment_name,
                num_workers=num_workers,
            )

    # create sample complexity plot
    data = construct_dataframe(trial_to_eval_results_map, metrics_config)

    # TODO: for metric in metric_config: (plot every given metric?)
    if create_plot:
        assert plot_output_path is not None, "'plot_output_path' must be defined in oder to save created plot"
        create_sample_complexity_plot(data, plot_output_path, **plot_config)

    return data


def generate_metrics(evaluate_output_path: str, metrics_config: dict[str, dict | list]):
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


def run_evaluations(
    task_config_path: str,
    experiment_config_path: str,
    launch_dir_of_experiment: str,
    metrics_config: dict[str, dict | list],
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
    experiment_config_path: str
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
    rl_algorithm_name: str
        The name of the RL algorithm to use in Evaluation Episodes
    experiment_name: str
        The name of the experiment. Used for standard output progress updates.
    num_workers: int
        The number of ray workers to use when running evaluation episodes in parallel.
    """

    kwargs: dict[str, typing.Any] = {}
    if test_case_manager_config:
        kwargs['test_case_manager_config'] = test_case_manager_config

    # rl algorithm
    if rl_algorithm_name:
        kwargs['rl_algorithm_name'] = rl_algorithm_name

    kwargs["num_workers"] = num_workers

    # run sequence of evaluation
    for index, ckpt_path in enumerate(checkpoint_paths):
        # print progress
        ckpt_num_regex = re.search(r'(checkpoint_)\w+', ckpt_path)
        ckpt_num = ckpt_path[ckpt_num_regex.start():ckpt_num_regex.end()]  # type: ignore
        if experiment_name:
            print("\nExperiment: " + experiment_name)
        print("Evaluating " + ckpt_num + ". " + str(index + 1) + " of " + str(len(checkpoint_paths)) + " checkpoints.")

        # run evaluation episodes
        try:
            evaluate(
                task_config_path,
                ckpt_path,
                output_paths[index],
                experiment_config_path,
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
    metrics_config: dict[str, dict | list],
    checkpoint_path: str,
    platfrom_serializer_class: PlatformSerializer,
    test_case_manager_config: dict | None = None,
    rl_algorithm_name: str | None = None,
    num_workers: int = 1
) -> pd.DataFrame:
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
    rl_algorithm_name: str
        The name of the RL algorithm to use in Evaluation Episodes
    num_workers: int
        The number of ray workers to use when running evaluation episodes in parallel.
    """

    kwargs: dict[str, typing.Any] = {}
    if test_case_manager_config:
        kwargs['test_case_manager_config'] = test_case_manager_config

    # rl algorithm
    if rl_algorithm_name:
        kwargs['rl_algorithm_name'] = rl_algorithm_name

    kwargs["num_workers"] = num_workers

    metrics_config = add_required_metrics(metrics_config)

    checkpoint_path += "/policies/{}/policy_state.pkl"
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


def create_sample_complexity_plot(
    data: pd.DataFrame,
    plot_output_file: str,
    xaxis="timesteps_total",
    yaxis="TotalReward",
    hue="agent",
    errorbar=("ci", 95),
    xmax=None,
    ylim=None,
    **kwargs,
) -> matplotlib.axis.Axis:
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
    # TODO: upgrade seaborn to 0.12.0 and swap deprecated 'ci' for 'errorbar' kwarg: errorbar=('sd', 1) or (''ci', 95)
    # TODO: add smooth kwarg

    plt.clf()

    sns.set(style="darkgrid", font_scale=1.5)

    # feed Dataframe to seaborn to create labelled plot
    plot = sns.lineplot(data=data, x=xaxis, y=yaxis, hue=hue, errorbar=errorbar, **kwargs)

    # format plot

    plt.legend(loc="best").set_draggable(True)

    if xaxis == "timesteps_total":
        plt.xlabel("Timesteps")
    if xaxis == "episodes_total":
        plt.xlabel("Episodes")
    if xaxis == "training_iteration":
        plt.xlabel("Iterations")
    if yaxis == "TotalReward":
        plt.ylabel("Average Reward")

    if xmax is None:
        xmax = np.max(np.asarray(data[xaxis]))
    plt.xlim(right=xmax)

    if ylim is not None:
        plt.ylim(ylim)

    xscale = xmax > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout(pad=0.5)

    # save plot

    plot.get_figure().savefig(plot_output_file)

    return plot
