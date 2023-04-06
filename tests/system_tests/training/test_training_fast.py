"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------

This module holds fixtures common to the safe_autonomy_sims package tests.

Author: John McCarroll
"""
import os

import pytest

from tests.conftest import read_test_cases
from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file
from corl.train_rl import parse_corl_args

from safe_autonomy_sims.experiments.rllib_api_experiment import RllibAPIExperiment


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/system_tests/training_fast.yml")
parameterized_fixture_keywords = ["experiment_config"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name="run_training")
def fixture_run_training(experiment_config, tmp_path, self_managed_ray):
    """
    Launches an Experiment from a given config file.
    """

    # change cwd to repo root
    # assumes test launch from repo root or 'tests' dir
    cwd = os.getcwd()
    parent_dir, current_dir = os.path.split(cwd)

    if current_dir == "tests":
        repo_root = parent_dir
    else:
        repo_root = cwd

    os.chdir(repo_root)

    try:
        print(self_managed_ray)
        args = parse_corl_args(["--cfg", experiment_config])

        config = load_file(config_filename=args.config)

        experiment_parse = ExperimentParse(**config)

        # RllibAPIExperiment is used for debuging not training
        if experiment_parse.experiment_class is RllibAPIExperiment:
            return

        experiment_class = experiment_parse.experiment_class(**experiment_parse.config)

        experiment_class.config.rllib_configs["local"] = {
            'horizon': 10,
            'rollout_fragment_length': 10,
            'train_batch_size': 10,
            'sgd_minibatch_size': 10,
            'batch_mode': 'complete_episodes',
            'num_workers': 1,
            'num_cpus_per_worker': 1,
            'num_envs_per_worker': 1,
            'num_cpus_for_driver': 1,
            'num_gpus_per_worker': 0,
            'num_gpus': 0,
            'num_sgd_iter': 30,
            'seed': 1
        }

        if "model" in experiment_class.config.rllib_configs["local"]:
            experiment_class.config.rllib_configs["local"]["model"].reset()

        experiment_class.config.ray_config['ignore_reinit_error'] = True
        if "_temp_dir" in experiment_class.config.ray_config:
            del experiment_class.config.ray_config["_temp_dir"]

        experiment_class.config.env_config["output_path"] = str(tmp_path / "training")

        experiment_class.config.tune_config['stop']['training_iteration'] = 1
        experiment_class.config.tune_config['local_dir'] = str(tmp_path / "training")
        experiment_class.config.tune_config['checkpoint_freq'] = 1
        experiment_class.config.tune_config['max_failures'] = 1
        args.compute_platform = "local"
        experiment_class.run_experiment(args)

    except Exception as e:
        print(e)

    finally:
        # change back to previous cwd
        os.chdir(cwd)


@pytest.mark.system_test
@pytest.mark.parametrize(parameterized_fixture_keywords, test_configs, ids=IDs)
def test_training_fast(
    run_training,
    tmp_path,
):
    """Test a training for a single iteration
    """
    # Determine filename of the checkpoint
    checkpoint_glob = list(tmp_path.glob('training/**/checkpoint*1'))
    assert len(checkpoint_glob) == 1
