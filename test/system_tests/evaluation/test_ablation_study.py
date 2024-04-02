from safe_autonomy_sims.evaluation.evaluation_api import run_ablation_study
import pytest
import os
import ray
from pathlib import Path

from corl.test_utils import full_training
from corl.train_rl import build_experiment, parse_corl_args
import corl
from safe_autonomy_sims.evaluation.launch.serialize_cwh3d import SerializeCWH3D

def test_evaluation_api_ablation_study_tabular(tmp_path):
    # train translational inspection
    experiment_config = "configs/test/translational-inspection/experiment.yml"
    
    ppo_rllib_config = {
        "rollout_fragment_length": 10,
        "train_batch_size": 10,
        "sgd_minibatch_size": 10,
        "num_workers": 1,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_gpus_per_worker": 0,
        "num_gpus": 0,
        "num_sgd_iter": 30,
        "seed": 1,
    } # TODO: stop criteria? ckpt freq?

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, ppo_rllib_config, tmp_path)
    experiment_class.config.tune_config["stop"]["training_iteration"] = 2
    experiment_class.config.tune_config["keep_checkpoints_num"] = 1
    experiment_class.run_experiment(experiment_file_validated)
    ray.shutdown()

    # Determine filename of the checkpoint
    trial_glob = list(tmp_path.glob("training/**/checkpoint_*/../"))
    trial = trial_glob[-1]

    # setup api variables
    # TODO: add tests for multiple experiments and multiple seeds of the same experiment
    experiment_output_paths = {
        # experiment_name: tune trial output dirs (that stores ckpt dirs) [str|list]
        'experiment1': str(trial)
    }

    launch_dir_of_experiment = os.getcwd()
    task_config_path = "configs/test/translational-inspection/task.yml"
    platform_serializer_class = SerializeCWH3D

    metrics_config = {
        "agent": {
            "__default__": [
                {
                    "name": "TotalReward",
                    "functor": "corl.evaluation.metrics.generators.rewards.TotalReward",
                    "config": {
                        "description": "total reward calculated from test case rollout"
                    }
                },
            ]
        }
    }

    # tabular_test_case_manager_config = {
    #     "type": "corl.evaluation.runners.section_factories.test_cases.tabular_strategy.TabularStrategy",
    #     "config": {
    #         'config': {
    #             "data": f"{corl.__path__[0]}/../config/tasks/docking_1d/evaluation/test_cases_config/docking1d_tests.yml",
    #             "source_form": 'FILE_YAML_CONFIGURATION',
    #             "randomize": False,
    #         }
    #     }
    # }
    tabular_test_case_manager_config = {
        "type": "corl.evaluation.runners.section_factories.test_cases.tabular_strategy.TabularStrategy",
        "config": {
            "config": {
                "data": "configs/test/translational-inspection/eval_test_cases.csv",
                "source_form": 'FILE_CSV',
                "randomize": False,
            }
        }
    }

    # eval output
    tabular_evaluation_ouput_dir = str(tmp_path) + "/eval_output_tabular"
    os.mkdir(str(tmp_path) + "/eval_output_tabular")
    assert os.path.exists(tabular_evaluation_ouput_dir)

    # TODO: add parameterized vars to test SAC + PPO
    # rl_alg_insert
    rl_alg = "SAC"

    # launch ablation study
    tabular_dataframe = run_ablation_study(
        experiment_output_paths,
        task_config_path,
        experiment_config,
        launch_dir_of_experiment,
        metrics_config,
        platform_serializer_class,
        tabular_evaluation_ouput_dir,
        test_case_manager_config=tabular_test_case_manager_config,
        # rl_algorithm_name=rl_alg
    )

    # validate output (check if test case output saved / dataframe output correct)
    assert not tabular_dataframe.empty
    expected_columns = ['index', 'experiment', 'experiment_index', 'evaluation_episode_index', 'training_iteration', 'agent_name', 'agent', 'TotalReward', 'timesteps_total', 'episodes_total', 'time_total_s']
    assert list(tabular_dataframe.columns) == expected_columns
    expected_num_test_cases = 10
    assert tabular_dataframe.index.stop == expected_num_test_cases


def test_evaluation_api_ablation_study_default(tmp_path):
    # train translational inspection
    experiment_config = "configs/test/translational-inspection/experiment.yml"

    ppo_rllib_config = {
        "rollout_fragment_length": 10,
        "train_batch_size": 10,
        "sgd_minibatch_size": 10,
        "num_workers": 1,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_gpus_per_worker": 0,
        "num_gpus": 0,
        "num_sgd_iter": 30,
        "seed": 1,
    } # TODO: stop criteria? ckpt freq?

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, ppo_rllib_config, tmp_path)
    experiment_class.config.tune_config["stop"]["training_iteration"] = 2
    experiment_class.config.tune_config["keep_checkpoints_num"] = 1
    experiment_class.run_experiment(experiment_file_validated)
    ray.shutdown()

    # Determine filename of the checkpoint
    trial_glob = list(tmp_path.glob("training/**/checkpoint_*/../"))
    trial = trial_glob[-1]

    # setup api variables
    # TODO: add tests for multiple experiments and multiple seeds of the same experiment
    experiment_output_paths = {
        # experiment_name: tune trial output dirs (that stores ckpt dirs) [str|list]
        'experiment1': str(trial)
    }
    launch_dir_of_experiment = os.getcwd()
    task_config_path = "configs/test/translational-inspection/task.yml"
    platform_serializer_class = SerializeCWH3D

    metrics_config = {
        "agent": {
            "__default__": [
                {
                    "name": "TotalReward",
                    "functor": "corl.evaluation.metrics.generators.rewards.TotalReward",
                    "config": {
                        "description": "total reward calculated from test case rollout"
                    }
                },
            ]
        }
    }

    expected_num_test_cases = 4
    default_test_case_manager_config = {
        "type": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
        "config": {
            "num_test_cases": expected_num_test_cases
        }
    }

    # eval output
    default_evaluation_ouput_dir = str(tmp_path) + "/eval_output_default"
    os.mkdir(str(tmp_path) + "/eval_output_default")
    assert os.path.exists(default_evaluation_ouput_dir)

    # TODO: add parameterized vars to test SAC + PPO
    # rl_alg_insert
    rl_alg = "SAC"

    # launch ablation study
    default_dataframe = run_ablation_study(
        experiment_output_paths,
        task_config_path,
        experiment_config,
        launch_dir_of_experiment,
        metrics_config,
        platform_serializer_class,
        default_evaluation_ouput_dir,
        test_case_manager_config=default_test_case_manager_config,
        # rl_algorithm_name=rl_alg
    )

    # validate output (check if test case output saved / dataframe output correct)
    assert not default_dataframe.empty
    expected_columns = ['index', 'experiment', 'experiment_index', 'evaluation_episode_index', 'training_iteration', 'agent_name', 'agent', 'TotalReward', 'timesteps_total', 'episodes_total', 'time_total_s']
    assert list(default_dataframe.columns) == expected_columns
    assert default_dataframe.index.stop == expected_num_test_cases
