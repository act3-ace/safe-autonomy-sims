"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Run an evaluation episode and make animation for the inspection environment
"""
from safe_autonomy_sims.evaluation.animation.inspection_animation import InspectionAnimation


if __name__ == '__main__':
    # ****Replace with your paths****
    checkpoint_path = 'example_checkpoint' # Path to checkpoint directory
    expr_config_path = '/root/repos/safe-autonomy-sims/configs/weighted-six-dof-inspection-v2/experiment.yml' # Path to experiment config
    launch_dir_of_experiment = 'safe-autonomy-sims'
    task_config_path = '/root/repos/safe-autonomy-sims/configs/weighted-six-dof-inspection-v2/task.yml' # Path to task config
    save_dir = 'safe-autonomy-sims/safe_autonomy_sims/evaluation/' # Path to directory to save png/mp4
    mode = 'operator' # Plotting mode: 'operator', 'obs_act', or '3d_pos'
    last_step = True # True for png of last step, False for mp4

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
            ]
        }
    }

    default_test_case_manager_config = {
        "class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
        "config": {
            "num_test_cases": 1
        }
    }

    dataframe = run_one_evaluation(task_config_path, expr_config_path, launch_dir_of_experiment, metrics_config, checkpoint_path, SerializeCWH3D, test_case_manager_config=default_test_case_manager_config)

    animation = AnimationFromCheckpoint()
    animation.make_animation(dataframe, save_dir, mode, last_step)

    import os
    import matplotlib.pyplot as plt
    import glob
    import pandas as pd
    import numpy as np

    metric_list = [
        'custom_metrics/rewards_cumulative/blue0_ctrl/ObservedPointsReward_mean',
        'custom_metrics/blue0_ctrl/ObserveSensor_Sensor_Position_DeltaV/delta_v_scale_mean',
        'episode_len_mean',
        'episode_reward_mean',
        'custom_metrics/done_results/blue0/SafeSuccessfulInspectionDoneFunction_mean',
    ]
    x_axis: str = 'timesteps_total'
    training_data_dir = '/root/repos/safe-autonomy-sims/output/tune/ACT3-RLLIB-AGENTS'

    listdir = os.listdir(training_data_dir)
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
