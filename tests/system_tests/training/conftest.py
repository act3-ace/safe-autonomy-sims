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


@pytest.fixture(name="experiment_config")
def fixture_experiment_config(request):
    """
    parameterized fixture for experiment config path.
    """
    return request.param
