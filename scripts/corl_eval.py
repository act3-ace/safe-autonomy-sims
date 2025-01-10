"""
This module quickly sets up and runs CoRL evaluation. This module is used to generate
policies used for gymnasium and pettingzoo environment validation tests.

Author: John McCarroll
"""

# use corl eval framework api to collect data for env validation tests
from corl.evaluation.api import evaluate
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D
import numpy as np


# Set up
task_config_path = "/absolute/path/to/task.yml"
checkpoint_path = "/absolute/path/to/experiment_output_dir/checkpoint_000000"
output_path = "/absolute/path/to/desired_output_dir"
experiment_config_path = "/absolute/path/to/experiment.yml"
launch_dir_of_experiment = "/absolute/path/to/safe-autonomy-sims/"
platform_serializer_class = SerializeCWH3D
test_case_manager_config = None
rl_algorithm_name = None
num_workers= 1
np.random.seed(3)

# Execute eval episode
evaluate(
    task_config_path,
    checkpoint_path,
    output_path,
    experiment_config_path,
    launch_dir_of_experiment,
    platform_serializer_class,
    test_case_manager_config=test_case_manager_config,
    rl_algorithm_name=rl_algorithm_name,
    num_workers=num_workers
)
