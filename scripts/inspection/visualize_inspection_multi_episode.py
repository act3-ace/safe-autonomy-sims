from safe_autonomy_sims.evaluation.evaluation_api import run_one_evaluation
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D
from inspection_multi_episode_6dof import MultiEpisodeAnimationFromCheckpoint


if __name__ == '__main__':
    # ****Replace with your paths****
    # checkpoint_path = 'example_checkpoint' # Path to checkpoint directory
    checkpoint_path = '/path/to/checkpoint/checkpoint_######'
    expr_config_path = '/path/to/experiment/config/experiment.yml' # Path to experiment config
    launch_dir_of_experiment = '/path/to/safe-autonomy-sims'
    task_config_path = '/path/to/task/config/task.yml' # Path to task config
    save_dir = '/path/to/save/directory/safe-autonomy-sims/evaluation/' # Path to directory to save png/mp4
    training_data_dir = '/path/to/training/data/safe-autonomy-sims/output/tune/ACT3-RLLIB-AGENTS'
    mode = 'operator' # Plotting mode: 'operator', 'obs_act', or '3d_pos'
    last_step = True # True for png of last step, False for mp4
    NUM_TEST_CASES =  200 # Number of test cases to run

    metrics_config = {
        "agent": {
            "__default__": [
                {
                    "name": "ObservationVector",
                    "functor": "safe_autonomy_sims.evaluation.general_metrics.ObservationVector",
                    "config": {
                        "description": "Observations"
                    }
                },
                {
                    "name": "ControlVector",
                    "functor": "safe_autonomy_sims.evaluation.general_metrics.ControlVector",
                    "config": {
                        "description": "Actions"
                    }
                },
                {
                    "name": "PositionVelocityVector",
                    "functor": "safe_autonomy_sims.evaluation.general_metrics.PositionVelocityVector",
                    "config": {
                        "descriptions": "Position_and_Velocity_from_state"
                    }
                }
            ]
        }
    }

    default_test_case_manager_config = {
        "class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
        "config": {
            "num_test_cases": NUM_TEST_CASES
        }
    }

    dataframe = run_one_evaluation(task_config_path, expr_config_path, launch_dir_of_experiment, metrics_config, checkpoint_path, SerializeCWH3D, test_case_manager_config=default_test_case_manager_config)

    animation = MultiEpisodeAnimationFromCheckpoint()
    data_dict = animation.get_data(dataframe)
    data_dict['delta_vs'] = animation.calc_delta_v(data_dict)
    animation.make_overall_plots(data_dict, save_dir=save_dir, mode=mode, last_step=last_step)

    import os
    import matplotlib.pyplot as plt
    import glob
    import pandas as pd
    import numpy as np

    metric_list = [
        'custom_metrics/rewards_cumulative/blue0_ctrl/ObservedPointsReward_mean',
        'episode_len_mean',
        'episode_reward_mean',
        'custom_metrics/done_results/blue0/SafeSuccessfulInspectionDoneFunction_mean',
    ]
    x_axis: str = 'timesteps_total'

    # listdir = os.listdir(training_data_dir)
    listdir = [os.path.split(checkpoint_path)[0]]
    # breakpoint()
    for met in metric_list:
        plt.figure()
        label = met.split('/')[-1]
        plt.xlabel(x_axis)
        plt.ylabel(label)
        plt.grid(True)
        for my_file in listdir:
            t_dir = os.path.join(str(training_data_dir), my_file)
            if os.path.isdir(t_dir):
                progress = glob.glob(t_dir + "/*.csv")[0]
                d_f = pd.read_csv(progress)
                metric_data = np.array(d_f[met])
                plt.plot(np.array(d_f[x_axis]), metric_data)
        plt.savefig(os.path.join(save_dir, label + '.png'))
