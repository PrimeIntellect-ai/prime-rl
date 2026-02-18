from prime_rl.trainer.rl.config import RLTrainerConfig


def test_rl_trainer_config_has_no_batch_fields() -> None:
    assert "token_batch_size" not in RLTrainerConfig.model_fields
    assert "rollout_batch_size" not in RLTrainerConfig.model_fields


def test_rl_trainer_config_still_instantiates_with_defaults() -> None:
    config = RLTrainerConfig()
    assert config.max_steps is None
