import os

import pytest

from prime_rl.trainer import envs


def test_training_env_defaults():
    """Test default values for training environment variables"""
    assert envs.RANK == 0


def test_training_env_custom_values():
    """Test custom values for training environment variables"""
    os.environ.update({"RANK": "1"})

    assert envs.RANK == 1


def test_invalid_env_vars():
    """Test that accessing invalid environment variables raises AttributeError"""
    with pytest.raises(AttributeError):
        envs.INVALID_VAR
