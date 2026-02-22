import pytest
from pydantic import ValidationError

from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl import RLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig


def test_batch_size_defaults_max_inflight_rollouts() -> None:
    config = RLConfig(
        trainer=RLTrainerConfig(),
        orchestrator=OrchestratorConfig(
            batch_size=64,
            rollouts_per_example=8,
        ),
    )

    assert config.orchestrator.batch_size == 64
    assert config.orchestrator.max_inflight_rollouts == 64


def test_token_batch_size_requires_max_inflight_rollouts() -> None:
    with pytest.raises(ValidationError, match="max_inflight_rollouts must be set when token_batch_size is set"):
        RLConfig(
            trainer=RLTrainerConfig(),
            orchestrator=OrchestratorConfig(token_batch_size=2048),
        )


def test_token_batch_size_allows_explicit_max_inflight_rollouts() -> None:
    config = RLConfig(
        trainer=RLTrainerConfig(),
        orchestrator=OrchestratorConfig(
            token_batch_size=2048,
            max_inflight_rollouts=77,
        ),
    )
    assert config.orchestrator.token_batch_size == 2048
    assert config.orchestrator.max_inflight_rollouts == 77


def test_batch_and_token_batch_sizes_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="Set either batch_size or token_batch_size, not both"):
        RLConfig(
            trainer=RLTrainerConfig(),
            orchestrator=OrchestratorConfig(
                batch_size=64,
                token_batch_size=2048,
                max_inflight_rollouts=80,
            ),
        )


def test_batching_defaults_to_rollout_mode() -> None:
    config = RLConfig(trainer=RLTrainerConfig(), orchestrator=OrchestratorConfig())
    assert config.orchestrator.batch_size == 128
    assert config.orchestrator.max_inflight_rollouts == 128


def test_filesystem_weight_broadcast_defaults() -> None:
    config = RLTrainerConfig()
    assert config.weight_broadcast.type == "filesystem"
    assert config.weight_broadcast.keep_last == 1
    assert config.weight_broadcast.min_broadcast_interval == 0.0
