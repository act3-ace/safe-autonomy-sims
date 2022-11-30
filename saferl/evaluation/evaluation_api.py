"""
Test script for running eval framework programmatically (python api).

Author: John McCarroll
"""
import os
from glob import glob

from corl.evaluation.default_config_updates import DoNothingConfigUpdate
from corl.evaluation.evaluation_artifacts import (
    EvaluationArtifact_EvaluationOutcome,
    EvaluationArtifact_Metrics,
    EvaluationArtifact_Visualization,
)
from corl.evaluation.launchers import launch_evaluate, launch_generate_metrics, launch_visualize
from corl.evaluation.loader.check_point_file import CheckpointFile
from corl.evaluation.recording.folder import Folder, FolderRecord
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibTrainer
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Task
from corl.evaluation.runners.section_factories.teams import Agent, Platform, Teams
from corl.evaluation.runners.section_factories.test_cases.pandas import Pandas
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseManager
from corl.evaluation.serialize_platforms import serialize_Docking_1d
from corl.evaluation.visualization.print import Print
from corl.parsers.yaml_loader import load_file


def evaluate(
    task_config_path: str,
    checkpoint_path: str,
    output_path: str,
    experiment_config_path: str,
    serialize_platforms_class: PlatformSerializer = serialize_Docking_1d
):
    # TODO: remove PlatformSerializer default
    # construct constructor args

    team_participant_map = construct_teams_map_from_experiment_config(experiment_config_path, checkpoint_path)

    test_case_strategy_config = {
        "data": "../corl/config/evaluation/test_cases_config/docking1d_tests.yml",
        "source_form": Pandas.SourceForm.FILE_YAML_CONFIGURATION,
        "randomize": False,
        "separator": '.'
    }
    test_case_manager_config = {
        "test_case_strategy_class_path": "corl.evaluation.runners.section_factories.test_cases.tabular_strategy.TabularStrategy",
        "config": test_case_strategy_config
    }
    # test_case_strategy_config = {
    #     "num_test_cases": 5
    # }
    # test_case_manager_config = {
    #     "test_case_strategy_class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
    #     "config": test_case_strategy_config
    # }

    ## plugins
    platform_serialization_obj = serialize_platforms_class()
    plugins_args = {"platform_serialization": platform_serialization_obj}
    eval_config_updates = [DoNothingConfigUpdate()]  # default creates list of string(s), isntead of objects

    ## engine
    rllib_engine_args = {"callbacks": [], "workers": 0}

    # recorders
    recorder_args = {"dir": output_path, "append_timestamp": False}

    # instantiate eval objects
    teams = Teams(team_participant_map=team_participant_map)
    task = Task(config_yaml_file=task_config_path)
    test_case_manager = TestCaseManager(**test_case_manager_config)
    plugins = Plugins(**plugins_args)
    plugins.eval_config_update = eval_config_updates
    engine = RllibTrainer(**rllib_engine_args)
    recorder = Folder(**recorder_args)

    # construct namespace dict
    namespace = {
        "teams": teams,
        "task": task,
        "test_cases": {
            "test_case_manager": test_case_manager
        },
        "plugins": plugins,
        "engine": {
            "rllib": engine
        },
        "recorders": [recorder]
    }

    # call main
    launch_evaluate.main(namespace)


def generate_metrics(evaluate_output_path: str):
    # construct constructor args

    location = FolderRecord(absolute_path=evaluate_output_path)

    # alerts_config = None    # TODO: enable evaluation without alertsssssss
    alerts_config_path = "../corl/config/evaluation/alerts/base_alerts.yml"

    # Note: could also point to metrics config
    metrics_config = {
        "world": [
            {
                "name": "WallTime(Sec)",
                "functor": "corl.evaluation.metrics.generators.meta.runtime.Runtime",
                "config": {
                    "description": "calculated runtime of test case rollout"
                }
            },
            {
                "name": "AverageWallTime",
                "functor": "corl.evaluation.metrics.aggregators.average.Average",
                "config": {
                    "description": "calculated average wall time over all test case rollouts",
                    "metrics_to_use": "WallTime(Sec)",
                    "scope": None  #null
                }
            },
            {
                "name": "EpisodeLength(Steps)",
                "functor": "corl.evaluation.metrics.generators.meta.episode_length.EpisodeLength_Steps",
                "config": {
                    "description": "episode length of test case rollout in number of steps"
                }
            },
            {
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
            },
        ],
        "agent": {
            "__default__": [
                {
                    "name": "Result",
                    "functor": "corl.evaluation.metrics.generators.dones.StatusCode",
                    "config": {
                        "description": "was docking performed successfully or not", "done_condition": "DockingDoneFunction"
                    }
                },
                {
                    "name": "Dones",
                    "functor": "corl.evaluation.metrics.generators.dones.DonesVec",
                    "config": {
                        "description": "dones triggered at end of each rollout"
                    }
                },
                {
                    "name": "TotalReward",
                    "functor": "corl.evaluation.metrics.generators.rewards.TotalReward",
                    "config": {
                        "description": "total reward calculated from test case rollout"
                    }
                },
                {
                    "name": "CompletionRate",
                    "functor": "corl.evaluation.metrics.aggregators.criteria_rate.CriteriaRate",
                    "config": {
                        "description": "out of the number of test case rollouts how many resulted in successful docking",
                        "metrics_to_use": "Result",
                        "scope": None,  #null
                        "condition": {
                            "operator": "==",
                            "lhs": {
                                "functor": "corl.dones.done_func_base.DoneStatusCodes",
                                "config": {
                                    "value": 1
                                }  # 1 is win
                            }
                        }
                    }
                },
            ]
        }
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
        "alerts_config": alerts_config_path,
        "raise_on_error_alert": raise_error_on_alert
    }

    launch_generate_metrics.main(namespace)


def visualize(output_path: str):

    artifact_metrics = EvaluationArtifact_Metrics(location=output_path)
    artifact_visualization = EvaluationArtifact_Visualization(location=output_path)
    visualizations = [Print(event_table_print=True)]

    namespace = {"artifact_metrics": artifact_metrics, "artifact_visualization": artifact_visualization, "visualizations": visualizations}

    launch_visualize.main(namespace)


# assumes cwd appropriate to experiment_config paths
def construct_teams_map_from_experiment_config(experiment_config_path: str, checkpoint_path: str):

    # parse experiment config
    experiment_config = load_file(experiment_config_path)

    assert 'agent_config' in experiment_config
    assert 'platform_config' in experiment_config
    agents_config = experiment_config['agent_config']
    platforms_config = experiment_config['platform_config']

    # populate teams based on experiment config
    blue_team = []  # TODO: fix one team assumption!!!
    for i in range(0, len(agents_config)):
        agent_name, platform_name, agent_config_path, policy_config_path = agents_config[i]  # assumes policy_id == agent_name!!!
        platform_name, platform_config_path = platforms_config[i]

        # handle relative paths (join w cwd)
        agent_config_path = os.path.join(os.getcwd(), agent_config_path)
        platform_config_path = os.path.join(os.getcwd(), platform_config_path)
        policy_config_path = os.path.join(os.getcwd(), policy_config_path)

        agent_loader = CheckpointFile(checkpoint_filename=checkpoint_path, policy_id=agent_name)

        agent_config = {
            "name": agent_name, "agent_config": agent_config_path, "policy_config": policy_config_path, "agent_loader": agent_loader
        }

        agents = [Agent(**agent_config)]

        platform_config = {"platform_config": platform_config_path, "agents": agents}
        blue_team.append(Platform(**platform_config))

    team_participant_map = {"blue": blue_team}

    return team_participant_map


def checkpoints_list_from_training_output(
    training_output_path: str, output_dir: str = "/tmp/ablation_results", experiment_name: str = "test"
):
    # assume training_output path absolute
    ckpt_dirs = sorted(glob(training_output_path + "/checkpoint_*"), key=lambda path: int(path.split("/")[-1].split("_")[-1]))
    output_dir_paths = []
    # add checkpoint files
    for i in range(0, len(ckpt_dirs)):
        ckpt_num = str(int(ckpt_dirs[i].split("/")[-1].split("_")[-1]))
        ckpt_dirs[i] += "/checkpoint-" + ckpt_num

        # add checkpoint to list of outputs
        output_path = output_dir + "/" + experiment_name + "/" + "ckpt_" + ckpt_num
        output_dir_paths.append(output_path)

    return ckpt_dirs, output_dir_paths  # TODO: need ckpt nums for tracking iterations / training interactions?


# # define variables
output_path = "/tmp/output_1"
expr_config = "../corl/config/experiments/docking_1d.yml"
task_config_path = "../corl/config/tasks/docking_1d/docking1d_task.yml"
checkpoint_path = "/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34/checkpoint_000150/checkpoint-150"

evaluate(task_config_path, checkpoint_path, output_path, expr_config)

# stage  3
# visualize(output_path)

# # test ckpt getter
# checkpoints_list_from_training_output("/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34/")
