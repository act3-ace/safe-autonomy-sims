"""
Test script for running eval framework programmatically (python api).

Author: John McCarroll
"""

import sys
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from saferl.evaluation.evaluation_api import evaluate, generate_metrics, visualize, checkpoints_list_from_training_output


def run_ablation_study(experiment_training_outputs: dict, task_config_path: str, experiemnt_config_path: str, plot_output_path: str):
    # evaluate provided trained policies' checkpoints
    experiment_to_eval_results_map = {}
    for experiment_name, experiment_dir in experiment_training_outputs.items():
        checkpoint_paths, output_paths = checkpoints_list_from_training_output(experiment_dir, experiment_name=experiment_name)
        metadata = extract_metadata(checkpoint_paths)
        experiment_to_eval_results_map[experiment_name] = {
            "output_paths": output_paths,
            "metadata": metadata
        }
        run_evaluations(task_config_path, experiemnt_config_path, checkpoint_paths, output_paths)
    
    # create sample complexity plot
    data = construct_reward_dataframe(experiment_to_eval_results_map)
    create_sample_complexity_plot(data, plot_output_path)

    
def run_evaluations(task_config_path: str, experiemnt_config_path: str, checkpoint_paths: list, output_paths: list):
    # run sequence of evaluation
    for i in range(0, len(checkpoint_paths)):
        # run evaluation episodes
        try:
            evaluate(task_config_path, checkpoint_paths[i], output_paths[i], experiemnt_config_path)
        except SystemExit:
            print(sys.exc_info()[0])
        # generate evaluation metrics
        generate_metrics(output_paths[i])
        # generate visualizations
        visualize(output_paths[i])


def extract_metadata(checkpoint_paths):
    # iterate through checkpoint dirs for given experiment, extract data on training duration
    training_meta_data = []
    for ckpt_path in checkpoint_paths:
        metadata_path = ckpt_path + ".tune_metadata"
        metadata = pickle.load(open(metadata_path, 'rb'))
        num_env_interactions = metadata["timesteps_total"] if metadata["timesteps_total"] else metadata["last_result"]["counters"]["num_env_steps_trained"]
        checkpoint_data = {
            'num_env_interactions': num_env_interactions,
            'num_episodes': metadata["episodes_total"],
            'walltime': metadata["time_total"],
            'iterations': metadata["iteration"],
        }
        training_meta_data.append(checkpoint_data)

    return training_meta_data


def construct_reward_dataframe(results: dict):
    """
    function to parse reward data from evaluation results. returns a pandas.DataFrame
    """
    # need to parse each metrics.pkl file + construct DataFrame
    data = []
    rewards = {}
    for experiment_name in results.keys():
        output_paths = results[experiment_name]['output_paths']
        training_metadata = results[experiment_name]['metadata']

        # collect reward info per experiment
        for i in range(0, len(output_paths)):
        
            # extract amouunt of training policy had before eval
            num_env_interactions = training_metadata[i]["num_env_interactions"]
            num_episodes = training_metadata[i]["num_episodes"]
            num_iterations = training_metadata[i]["iterations"]

            # collect reward info per ckpt
            metrics_file = open(output_paths[i] + "/metrics.pkl", 'rb')
            metrics = pickle.load(metrics_file)
            episode_events = list(metrics.participants.values())[0].events  # assumes single agent environment

            # episode_artifacts = []
            # episode_rewards = []
            # episode_initial_parameters = []

            index = 0
            for event in episode_events:
                # collect reward on each trial in eval
                episode_artifact = event.data
                parameters = episode_artifact.parameter_values
                reward = event.metrics['TotalReward'].value

                # episode_artifacts.append(episode_artifact)
                # episode_rewards.append(parameters)
                # episode_initial_parameters.append(reward)
                
                episode_id = index
                index += 1
                
                # add row to dataset (experiment name, iteration, num_episodes, num_interactions, episode ID/trial, param values, reward)
                row = [experiment_name, num_iterations, num_episodes, num_env_interactions, episode_id, reward]
                data.append(row)

    # convert data into DataFrame
    dataframe = pd.DataFrame(data, columns=['experiment', 'iterations', 'num_episodes', 'num_interactions', 'episode ID', 'reward'])

    file = open('/media/john/HDD/AFRL/test_dataframe.df', 'wb')
    pickle.dump(dataframe, file)

    return dataframe


def create_sample_complexity_plot(data: pd.DataFrame, plot_output_file: str, xaxis='num_interactions', yaxis='reward', hue='experiment', ci='sd', xmax=None, ylim=None, **kwargs):
    # format plot
    # TODO: upgrade seaborn to 0.12.0 and swap depricated 'ci' for 'errorbar' kwarg: errorbar=('sd', 1) or (''ci', 95)
    # TODO: add smooth kwarg

    sns.set(style="darkgrid", font_scale=1.5)

    # feed Dataframe to seaborn to create labelled plot
    plot = sns.lineplot(data=data, x=xaxis, y=yaxis, hue=hue, ci=ci, **kwargs)

    # more format plot

    plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    if xaxis == 'num_interactions':
        plt.xlabel('Timesteps')
    if xaxis == 'num_episodes':
        plt.xlabel('Episodes')
    if xaxis == 'iterations':
        plt.xlabel('Iterations')
    if yaxis == 'reward':
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


# sample script:

# define list of experiments + their paths
# result_path = run_ablation_study(**args)
# create_sample_complexity_plot(result_path)

# IDEA: put all eval results in one dir, then walk that dir for DataFrame creation**
#       experiment names must be coupled w their output paths -> expr names become sub dirs of ablation 'result' output dir



### Example Useage

# define vars
expr_config = "../corl/config/experiments/docking_1d.yml"
task_config_path = "../corl/config/tasks/docking_1d/docking1d_task.yml"
training_output_dirs = {
    'a': '/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34',
    'b': '/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_41448_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-11-29_12-21-26'
}
plot_ouput = '/media/john/HDD/AFRL/test_sample_complexity_plot.png'

# begin ablation study pipeline (training outputs -> sample complexity plot)
run_ablation_study(training_output_dirs, task_config_path, expr_config, plot_output_path=plot_ouput)


