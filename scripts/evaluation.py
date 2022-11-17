"""
Test script for running eval framework programmatically.

Author: John McCarroll
"""

#### python api
from corl.evaluation.launchers import launch_evaluate
from corl.evaluation.runners.section_factories.teams import Teams, Platform, Agent
from corl.evaluation.runners.section_factories.task import Task
from corl.evaluation.runners.section_factories.test_cases.pandas import Pandas
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.default_config_updates import DoNothingConfigUpdate
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibTrainer
from corl.evaluation.recording.folder import Folder
from corl.evaluation.loader.check_point_file import CheckpointFile
# TODO: flexible import of platform serialization class
from corl.evaluation.serialize_platforms import serialize_Docking_1d


# define variables
task_path = "../corl/config/tasks/docking_1d/docking1d_task.yml"
output_path = "/tmp/omg_save_me"
# define per agent
checkpoint_filename = "/media/john/HDD/AFRL/Docking-1D-EpisodeParameterProviderSavingTrainer_ACT3MultiAgentEnv_cbccc_00000_0_num_gpus=0,num_workers=4,rollout_fragment_length=_2022-06-22_11-41-34/checkpoint_000150/checkpoint-150"
policy_id = "blue0"

# construct constructor args
## teams participent map -> eval Platform and Agent objects

agent_loader = CheckpointFile(checkpoint_filename=checkpoint_filename, policy_id=policy_id)

agent_config = {
    "name": "blue0",
    "agent_config": "../corl/config/tasks/docking_1d/docking1d_agent.yml",
    "policy_config": "../corl/config/policy/ppo/default_config.yml",
    "agent_loader": agent_loader
}

agents = [
    Agent(**agent_config)
]

platform_config = {
    "platform_config": "../corl/config/tasks/docking_1d/docking1d_platform.yml",
    "agents": agents
}
blue_team = [Platform(**platform_config)]

team_participant_map = {
    "blue": blue_team
}

## Panda panda panda panda
panda_args = {
    "data": "../corl/config/evaluation/test_cases_config/docking1d_tests.yml",
    "source_form": Pandas.SourceForm.FILE_YAML_CONFIGURATION,
    "randomize": False
}

## plugins
platform_serialization_obj = serialize_Docking_1d()
plugins_args = {
    "platform_serialization": platform_serialization_obj
}
eval_config_updates = [DoNothingConfigUpdate()]     # default creates list of string(s), isntead of objects

## engine (choo choo thomas)
rllib_engine_args = {
    "callbacks": [],
    "workers": 0
}

## recorders
## ASSUME: just needs list of Folders?
recorder_args = {
    "dir": output_path,
    "append_timestamp": False
}

# instantiate eval objects
teams = Teams(team_participant_map=team_participant_map)
task = Task(config_yaml_file=task_path)
panda_panda_panda = Pandas(**panda_args)
plugins = Plugins(**plugins_args)
plugins.eval_config_update = eval_config_updates
engine = RllibTrainer(**rllib_engine_args)
recorder = Folder(**recorder_args)


# construct namespace dict
namespace = {
    "teams": teams,
    "task": task,
    "test_cases": {
        "pandas": panda_panda_panda
    },
    "plugins": plugins,
    "engine": {
        "rllib": engine
    },
    "recorders": [recorder]
}

# call main [which would need to be flexible to dict (not namespace obj)]
launch_evaluate.main(namespace)

