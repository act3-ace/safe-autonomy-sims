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

This module holds fixtures common to the saferl package tests.

Author: John McCarroll
"""
import os
import tempfile

import pytest
import ray
from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file
from corl.train_rl import parse_corl_args

from saferl.experiments.rllib_api_experiment import RllibAPIExperiment
from scripts.base_rta_test import RTAExperiment


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


experiment_files = []
whitelist_experiments = [
    'docking_3d.yml', 
    'docking_3d_rta.yml', 
    # 'inspection_3d.yml', 
    'rejoin_2d.yml', 
    'rejoin_3d.yml', 
    # 'cwh_3d_inspection_multiagent.yml', 
    'cwh_3d_multiagent.yml'
]
for dirpath, dirs, files in os.walk('configs/experiments'):
    for filename in files:
        fname = os.path.join(dirpath, filename)
        if fname.endswith('.yml') and fname.split('/')[-1] in whitelist_experiments:
            experiment_files.append(fname)



@pytest.mark.system_test
@pytest.mark.parametrize("experiment_config", experiment_files, ids=experiment_files)
def test_experiment(
    experiment_config,
    tmp_path,
    self_managed_ray,
):
    """Test a configuration file
    """
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

    if not isinstance(experiment_class, RTAExperiment):
        # Determine filename of the checkpoint
        checkpoint_glob = list(tmp_path.glob('training/**/checkpoint-1'))
        assert len(checkpoint_glob) == 1
