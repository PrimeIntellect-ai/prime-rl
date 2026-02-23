import pytest
from pydantic import ValidationError

from prime_rl.orchestrator.config import OrchestratorConfig


def test_defaults_batching_when_unset():
    config = OrchestratorConfig()

    assert config.batch_size == 128
    assert config.token_batch_size is None
    assert config.max_inflight_rollouts == 128


def test_rollout_batch_defaults_max_inflight_rollouts():
    config = OrchestratorConfig(batch_size=64)

    assert config.batch_size == 64
    assert config.token_batch_size is None
    assert config.max_inflight_rollouts == 64


def test_token_batch_requires_max_inflight_rollouts():
    with pytest.raises(ValidationError, match="max_inflight_rollouts must be set when token_batch_size is set"):
        OrchestratorConfig(token_batch_size=2048)


def test_batch_fields_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="Set exactly one of batch_size or token_batch_size"):
        OrchestratorConfig(batch_size=64, token_batch_size=2048, max_inflight_rollouts=64)


def test_rollout_batch_size_must_be_divisible():
    with pytest.raises(ValidationError, match="Batch size must be divisible by the number of samples per problem"):
        OrchestratorConfig(batch_size=10, rollouts_per_example=3)


def test_token_batching_mode_keeps_batch_size_unset():
    config = OrchestratorConfig(token_batch_size=1024, max_inflight_rollouts=12, rollouts_per_example=3)

    assert config.batch_size is None
    assert config.token_batch_size == 1024
    assert config.max_inflight_rollouts == 12
