import os

import pytest

from zeroband.training import envs as training_env


def test_training_env_defaults():
    """Test default values for training environment variables"""
    assert training_env.SHARDCAST_OUTPUT_DIR is None


def test_training_env_custom_values():
    """Test custom values for training environment variables"""
    os.environ.update({"SHARDCAST_OUTPUT_DIR": "path/to/dir"})

    assert training_env.SHARDCAST_OUTPUT_DIR == "path/to/dir"


def test_invalid_env_vars():
    """Test that accessing invalid environment variables raises AttributeError"""
    with pytest.raises(AttributeError):
        training_env.INVALID_VAR


def test_no_env_mixing():
    """Test that inference env doesn't have training-specific variables"""
    with pytest.raises(AttributeError):
        training_env.SHARDCAST_SERVERS
