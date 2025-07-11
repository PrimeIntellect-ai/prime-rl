import os

import pytest

from prime_rl.trainer import envs


def test_training_env_defaults():
    """Test default values for training environment variables"""
    assert envs.SHARDCAST_OUTPUT_DIR is None


def test_training_env_custom_values():
    """Test custom values for training environment variables"""
    os.environ.update({"SHARDCAST_OUTPUT_DIR": "path/to/dir"})

    assert envs.SHARDCAST_OUTPUT_DIR == "path/to/dir"


def test_invalid_env_vars():
    """Test that accessing invalid environment variables raises AttributeError"""
    with pytest.raises(AttributeError):
        envs.INVALID_VAR
