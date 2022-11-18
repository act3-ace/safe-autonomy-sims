"""
Test script for running eval framework programmatically (python api).

Author: John McCarroll
"""

import sys
from saferl.evaluation.evaluation_api import construct_teams_map_from_task_config, evaluate, generate_metrics, visualize


# define variables
output_paths = [
    "/tmp/output_1",
    "/tmp/output_2"
]
expr_config = "../corl/config/experiments/docking_1d.yml"
task_config_path = "../corl/config/tasks/docking_1d/docking1d_task.yml"
checkpoint_paths = [
    "/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34/checkpoint_000150/checkpoint-150",
    "/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34/checkpoint_000125/checkpoint-125"
]

# run sequence of evaluation
for i in range(0, len(checkpoint_paths)):
    # run evaluation episodes
    try:
        evaluate(task_config_path, checkpoint_paths[i], output_paths[i], expr_config)
    except SystemExit:
        print(sys.exc_info()[0])
    # generate evaluation metrics
    generate_metrics(output_paths[i])
    # generate visualizations
    visualize(output_paths[i])
