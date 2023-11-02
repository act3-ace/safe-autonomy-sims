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

from test.conftest import read_test_cases


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/system_tests/training_fast.yml")
parameterized_fixture_keywords = ["experiment_config_path"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.system_test
@pytest.mark.parametrize(parameterized_fixture_keywords, test_configs, ids=IDs)
def test_training_fast(
    run_training,
    tmp_path,
):
    """Test a training for a single iteration
    """
    relative_path_to_checkpoints = 'training/ACT3-RLLIB-AGENTS/*/checkpoint*'
    # Determine filename of the checkpoint
    checkpoint_glob = list(tmp_path.glob(relative_path_to_checkpoints))
    assert len(checkpoint_glob) == 1
