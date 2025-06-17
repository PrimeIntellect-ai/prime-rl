import os

import pytest

from zeroband.inference import envs as inference_env


def test_inference_env_defaults():
    """Test default values for inference environment variables"""
    assert inference_env.SHARDCAST_SERVERS is None  # inference specific


def test_inference_env_custom_values():
    """Test custom values for inference environment variables"""
    os.environ.update({"PRIME_LOG_LEVEL": "DEBUG", "SHARDCAST_SERVERS": "server1,server2"})

    assert inference_env.PRIME_LOG_LEVEL == "DEBUG"
    assert inference_env.SHARDCAST_SERVERS == ["server1", "server2"]


def test_invalid_env_vars():
    """Test that accessing invalid environment variables raises AttributeError"""
    with pytest.raises(AttributeError):
        inference_env.INVALID_VAR


def test_no_env_mixing():
    """Test that inference env doesn't have training-specific variables"""
    with pytest.raises(AttributeError):
        inference_env.SHARDCAST_OUTPUT_DIR
