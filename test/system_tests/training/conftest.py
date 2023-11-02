"""
# -------------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Runtime Assurance Extensions

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

This module defines fixtures, functions, and constants common to the entire test suite.
Python packages used in test case configs must be imported to this module for error free value loading.

Author: John McCarroll
"""

import pytest
import ray
import tempfile
import os
from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file
from corl.train_rl import parse_corl_args, merge_cfg_and_args, ExperimentFileParse


@pytest.fixture(name="_ray_session_temp_dir", scope="session", autouse=True)
def ray_session_temp_dir():
    """Create temp dir
    """
    with tempfile.TemporaryDirectory() as ray_temp_dir:
        return ray_temp_dir


@pytest.fixture(scope="session", autouse=True)
def ray_session(_ray_session_temp_dir):
    """Create ray session
    """
    os.putenv("CUDA_VISIBLE_DEVICES", "-1")
    ray_config = {
        "address": None,
        "include_dashboard": False,
        "num_gpus": 0,
        "_temp_dir": _ray_session_temp_dir,
        "_redis_password": None,
        "ignore_reinit_error": False
    }
    ray.init(**ray_config)
    yield
    ray.shutdown()


@pytest.fixture(name="self_managed_ray")
def create_self_managed_ray(_ray_session_temp_dir):
    """Enable a test to manage their own ray initialization.

    The `ray_session` fixture above ensures that all tests have a properly initialized ray
    environment.  However, some tests need more control over the ray configuration for the duration
    of the test.  The most common example is tests that need to specify `local_mode=True` within
    the evaluation.  The trivial implementation of these tests is to put `ray.shutdown` at the
    beginning of the test and then to configure ray for that particular test.  The problem with this
    approach is that it does not restore ray to a properly initialized state for any other unit test
    that assumes that the `ray_session` fixture had properly initialized ray.

    Therefore, the recommended approach for any test that needs to manage their own ray
    configuration is to use this fixture.  It automatically ensures that ray is not active at the
    beginning of the test and ensures that ray is restored to the expected configuration afterwards.
    """

    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()
    ray_config = {"include_dashboard": False, "num_gpus": 0, "_temp_dir": _ray_session_temp_dir}
    ray.init(**ray_config)
    return ray_config


@pytest.fixture(name="experiment_config_path")
def fixture_experiment_config(request):
    """
    parameterized fixture for experiment config path.
    """
    return request.param


@pytest.fixture(name="cwd")
def fixture_change_cwd():
    # change cwd to repo root
    # assumes test launch from repo root or 'tests' dir
    cwd = os.getcwd()
    parent_dir, current_dir = os.path.split(cwd)

    if current_dir == "tests":
        repo_root = parent_dir
    else:
        repo_root = cwd

    os.chdir(repo_root)

    return cwd


@pytest.fixture(name="training_config")
def fixture_training_config(experiment_config_path):
    cfg=None
    try:
        args = parse_corl_args(["--cfg", experiment_config_path])
        cfg = load_file(config_filename=args.cfg)
        cfg = merge_cfg_and_args(cfg, args)

    except Exception as e:
        print(e)

    return cfg

@pytest.fixture(name="run_training")
def fixture_run_training(cwd, training_config, tmp_path, self_managed_ray):
    """
    Launches an Experiment from a given config file.
    """

    try:
        print(self_managed_ray)
        
        experiment_file_validated = ExperimentFileParse(**training_config)
        config = load_file(config_filename=str(experiment_file_validated.config))
        experiment_parse = ExperimentParse(**config)
        experiment_parse.experiment_class.process_cli_args(experiment_parse.config, experiment_file_validated)

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
            'seed': 1,
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
        experiment_class.run_experiment(experiment_file_validated)

    except Exception as e:
        print(e)

    finally:
        # change back to previous cwd
        os.chdir(cwd)
