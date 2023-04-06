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
import jsonlines
import pytest

from tests.conftest import read_test_cases, delimiter
from tests.system_tests.training.constants import CUSTOM_METRICS
from tests.system_tests.training.success_criteria import SuccessCriteria
from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file
from corl.train_rl import parse_corl_args

from safe_autonomy_sims.experiments.rllib_api_experiment import RllibAPIExperiment


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/system_tests/training_performance.yml")
parameterized_fixture_keywords = ["experiment_config", "seed", "max_training_iteration", "success_mean_metric", "expected_success_rate"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name="max_training_iteration")
def fixture_max_training_iteration(request):
    return request.param


@pytest.fixture(name="seed")
def fixture_seed(request):
    return request.param


@pytest.fixture(name="success_mean_metric")
def fixture_success_mean_metric(request):
    return request.param


@pytest.fixture(name="success_rate")
def fixture_run_training(
    experiment_config,
    tmp_path,
    self_managed_ray,
    max_training_iteration,
    seed,
    expected_success_rate,
    success_mean_metric
):
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

        experiment_class.config.rllib_configs["local"]["seed"] = seed

        # if "model" in experiment_class.config.rllib_configs["local"]:
        #     experiment_class.config.rllib_configs["local"]["model"].reset()

        experiment_class.config.ray_config['ignore_reinit_error'] = True
        if "_temp_dir" in experiment_class.config.ray_config:
            del experiment_class.config.ray_config["_temp_dir"]

        experiment_class.config.env_config["output_path"] = str(tmp_path / "training")

        experiment_class.config.tune_config['stop'] = SuccessCriteria(
            success_mean_metric=success_mean_metric,
            success_threshold=expected_success_rate, 
            max_iterations=max_training_iteration
        )

        experiment_class.config.tune_config['local_dir'] = str(tmp_path / "training")
        experiment_class.config.tune_config['checkpoint_freq'] = 0
        experiment_class.config.tune_config['max_failures'] = 1
        args.compute_platform = "local"
        experiment_class.run_experiment(args)

        # get success rate
        with jsonlines.open(str(list(tmp_path.glob('training/*/*/result.json'))[0])) as results:
            results = [line for line in results]
            success_rate = results[-1][CUSTOM_METRICS][success_mean_metric]

        # return success rate
        yield success_rate

    except Exception as e:
        print(e)
        return 0.0
    
    finally:
        # change back to previous cwd
        os.chdir(cwd)


@pytest.mark.performance_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs)
def test_training_performance(
    success_rate,
    expected_success_rate,
    tmp_path,
):
    """Test a training for a single iteration
    """
    # compare success rates
    assert expected_success_rate <= success_rate
