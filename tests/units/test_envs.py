import os

import pytest

from zeroband.inference import envs as inference_env
from zeroband.training import envs as training_env


@pytest.fixture
def set_env():
    original_env = dict(os.environ)

    def _set_env(new_env):
        os.environ.update(new_env)

    yield _set_env

    # Restore original environment after test
    os.environ.clear()
    os.environ.update(original_env)


def test_inference_env_defaults():
    """Test default values for inference environment variables"""
    assert inference_env.PRIME_LOG_LEVEL == "INFO"  # shared env
    assert inference_env.SHARDCAST_SERVERS is None  # inference specific


def test_training_env_defaults():
    """Test default values for training environment variables"""
    assert training_env.PRIME_LOG_LEVEL == "INFO"  # shared env
    assert training_env.SHARDCAST_OUTPUT_DIR is None


def test_inference_env_custom_values(set_env):
    """Test custom values for inference environment variables"""
    set_env({"PRIME_LOG_LEVEL": "DEBUG", "SHARDCAST_SERVERS": "server1,server2"})

    assert inference_env.PRIME_LOG_LEVEL == "DEBUG"
    assert inference_env.SHARDCAST_SERVERS == ["server1", "server2"]


def test_training_env_custom_values(set_env):
    """Test custom values for training environment variables"""
    set_env({"PRIME_LOG_LEVEL": "DEBUG", "SHARDCAST_OUTPUT_DIR": "path/to/dir"})

    assert training_env.PRIME_LOG_LEVEL == "DEBUG"
    assert training_env.SHARDCAST_OUTPUT_DIR == "path/to/dir"


def test_invalid_env_vars():
    """Test that accessing invalid environment variables raises AttributeError"""
    with pytest.raises(AttributeError):
        inference_env.INVALID_VAR

    with pytest.raises(AttributeError):
        training_env.INVALID_VAR


def test_no_env_mixing():
    """Test that inference env doesn't have training-specific variables"""
    with pytest.raises(AttributeError):
        training_env.SHARDCAST_SERVERS
    with pytest.raises(AttributeError):
        inference_env.SHARDCAST_OUTPUT_DIR
