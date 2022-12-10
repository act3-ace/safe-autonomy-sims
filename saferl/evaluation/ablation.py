"""
This module leverages CoRL's evaluation framework's python API to streamline visualization and analysis of comparative RL test assays.

Author: John McCarroll
"""

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer

from saferl.evaluation.evaluation_api import (
    add_required_metrics,
    checkpoints_list_from_training_output,
    evaluate,
    generate_metrics,
    visualize,
)

default_test_case_manager_config = {
    "test_case_strategy_class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
    "config": {
        "num_test_cases": 3
    }
}


def run_ablation_study(
    experiment_training_output_paths: dict,
    task_config_path: str,
    experiemnt_config_path: str,
    metrics_config: dict,
    platfrom_serializer_class: PlatformSerializer,
    plot_output_path: str,
    plot_config: dict = {},
    create_plot=True,
    test_case_manager_config: dict = default_test_case_manager_config
):

    # add required metrics
    metrics_config = add_required_metrics(metrics_config)

    # evaluate provided trained policies' checkpoints
    experiment_to_eval_results_map = {}
    for experiment_name, experiment_dir in experiment_training_output_paths.items():
        checkpoint_paths, output_paths = checkpoints_list_from_training_output(experiment_dir, experiment_name=experiment_name)
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
            visualize=False
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
    test_case_manager_config: dict = default_test_case_manager_config,
    visualize: bool = False
):

    # run sequence of evaluation
    for i in range(0, len(checkpoint_paths)):
        # run evaluation episodes
        try:
            evaluate(
                task_config_path,
                checkpoint_paths[i],
                output_paths[i],
                experiemnt_config_path,
                platfrom_serializer_class,
                test_case_manager_config=test_case_manager_config
            )
        except SystemExit:
            print(sys.exc_info()[0])

        # generate evaluation metrics
        generate_metrics(output_paths[i], metrics_config)

        if visualize:
            # generate visualizations
            visualize(output_paths[i])


def extract_metadata(checkpoint_paths):
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


def parse_metrics_config(metrics_config: dict):
    # collect metric names
    metrics_names = {}
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
    function to parse reward data from evaluation results. returns a pandas.DataFrame
    """
    metric_names = parse_metrics_config(metrics_config)
    columns = ['experiment', 'iterations', 'num_episodes', 'num_interactions', 'episode ID']

    # need to parse each metrics.pkl file + construct DataFrame
    data = []
    for experiment_name in results.keys():
        output_paths = results[experiment_name]['output_paths']
        training_metadata = results[experiment_name]['metadata']

        # collect metric values per experiment
        for i in range(0, len(output_paths)):

            # extract amouunt of training policy had before eval
            num_env_interactions = training_metadata[i]["num_env_interactions"]
            num_episodes = training_metadata[i]["num_episodes"]
            num_iterations = training_metadata[i]["iterations"]

            # collect agent metrics per ckpt
            metrics_file = open(output_paths[i] + "/metrics.pkl", 'rb')
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
                for metric_name in metric_names['agent']['__default__']:
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
    # create seaborn plot
    # TODO: upgrade seaborn to 0.12.0 and swap depricated 'ci' for 'errorbar' kwarg: errorbar=('sd', 1) or (''ci', 95)
    # TODO: add smooth kwarg

    sns.set(style="darkgrid", font_scale=1.5)

    # feed Dataframe to seaborn to create labelled plot
    plot = sns.lineplot(data=data, x=xaxis, y=yaxis, hue=hue, ci=ci, **kwargs)

    # format plot

    plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    if xaxis == 'num_interactions':
        plt.xlabel('Timesteps')
    if xaxis == 'num_episodes':
        plt.xlabel('Episodes')
    if xaxis == 'iterations':
        plt.xlabel('Iterations')
    if yaxis == 'TotalReward':
        plt.ylabel('Average Reward')
    # if yaxis == 'AverageTestEpRet' or yaxis == 'AverageAltTestEpRet':
    #     plt.ylabel('Average Return')
    # if yaxis == 'TestEpLen' or yaxis == 'AltTestEpLen':
    #     plt.ylabel('Average Episode Length')
    # if yaxis == 'Success' or yaxis == 'AltSuccess':
    #     plt.ylabel('Average Success')

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


# # ### Example Useage

# inspection = True

# if not inspection:
#     from corl.evaluation.serialize_platforms import serialize_Docking_1d

#     # define vars
#     expr_config = "/home/john/AFRL/ACT3/corl/config/experiments/docking_1d.yml"
#     task_config_path = "/home/john/AFRL/ACT3/corl/config/tasks/docking_1d/docking1d_task.yml"
#     training_output_dirs = {
#         'a': '/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34',
#         'b': '/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_41448_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-11-29_12-21-26'
#     }
#     plot_ouput = '/media/john/HDD/AFRL/test_sample_complexity_plot.png'
#     plot_config = {
#         "y_axis": "CustomMetricName",
#         "x_axis": "num_episodes"
#     }

#     metrics_config = {
#         "agent": {
#             "__default__": [
#                 {
#                     "name": "TotalReward",
#                     "functor": "corl.evaluation.metrics.generators.rewards.TotalReward",
#                     "config": {
#                         "description": "total reward calculated from test case rollout"
#                     }
#                 },
#             ]
#         }
#     }

#     # begin ablation study pipeline (training outputs -> sample complexity plot)
#     run_ablation_study(training_output_dirs, task_config_path, expr_config, metrics_config, serialize_Docking_1d, plot_output_path=plot_ouput)

# else:

#     from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
#     from saferl.evaluation.evaluation_api import evaluate, generate_metrics, visualize, checkpoints_list_from_training_output, add_required_metrics
#     from saferl.evaluation.ablation import run_ablation_study, run_evaluations, create_sample_complexity_plot
#     # import our serializer for CWH platforms
#     from saferl.evaluation.launch.serialize_cwh3d import SerializeCWH3D

#     # Define variables

#     training_output_dirs = {
#         'seed11': '/media/john/HDD/AFRL/inspection/CWH-3D-INSPECTION-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_4bbe6_00000_0_num_gpus=0,num_workers=6,rollout_fragment__2022-12-07_11-14-59',
#         'seed843': '/media/john/HDD/AFRL/inspection/CWH-3D-INSPECTION-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_a36f9_00000_0_num_gpus=0,num_workers=6,rollout_fragment__2022-12-07_12-00-23'
#         # 'seed11': '/path/to/training/output/folder',
#         # 'seed843': '/path/to/training/output/folder'
#     }
#     expr_config = "/home/john/AFRL/ACT3/safe-autonomy-sims/configs/experiments/inspection/inspection_3d.yml"
#     task_config_path = "/home/john/AFRL/ACT3/safe-autonomy-sims/configs/tasks/cwh3d_inspection/cwh3d_task.yml"
#     plot_ouput = '/media/john/HDD/AFRL/test_sample_complexity_plot.png'
#     # plot_ouput = '/path/to/desired/plot/save/location.png'

#     plot_config = {
#         "y_axis": "TotalReward",
#         "x_axis": "num_iteration"
#     }

#     metrics_config = {
#         "agent": {
#             "__default__": [
#                 {
#                     "name": "TotalReward",
#                     "functor": "corl.evaluation.metrics.generators.rewards.TotalReward",
#                     "config": {
#                         "description": "total reward calculated from test case rollout"
#                     }
#                 },
#             ]
#         }
#     }

#     test_case_manager_config = {
#         "test_case_strategy_class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
#         "config": {
#             "num_test_cases": 1
#         }
#     }

#     # begin ablation study pipeline (training outputs -> sample complexity plot)
#     dataframe = run_ablation_study(training_output_dirs, task_config_path, expr_config, metrics_config, SerializeCWH3D, test_case_manager_config=test_case_manager_config, plot_output_path=plot_ouput)
