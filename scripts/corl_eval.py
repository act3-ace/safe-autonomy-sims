# use corl eval framework api to collect data for env validation tests
from corl.evaluation.api import evaluate
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D
from pathlib import Path

# set up
task_config_path = "/home/john/AFRL/test_initiative/safe-autonomy-sims/configs/multiagent-weighted-translational-inspection/task.yml"
checkpoint_path = "/tmp/safe-autonomy-sims/output/tune/MULTIAGENT-WEIGHTED-TRANSLATIONAL-INSPECTION/MULTIAGENT-WEIGHTED-TRANSLATIONAL-INSPECTION-test-PPO_CorlMultiAgentEnv_8cd7b_00000_0_2024-11-14_17-34-03/checkpoint_000000"
output_path = "/tmp/safe-autonomy-sims/multiagent_weighted_translational_inspection_v0_validation_testing"
# experiment_config_path = "/home/john/AFRL/test_initiative/safe-autonomy-sims/configs/docking/experiment.yml"
experiment_config_path = Path("/home/john/AFRL/test_initiative/safe-autonomy-sims/configs/multiagent-weighted-translational-inspection/experiment.yml")
# experiment_config_path = "configs/docking/experiment.yml"
launch_dir_of_experiment = "/home/john/AFRL/test_initiative/safe-autonomy-sims"
platform_serializer_class = SerializeCWH3D
test_case_manager_config = None
rl_algorithm_name = None
num_workers= 1


# execute eval episode
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
