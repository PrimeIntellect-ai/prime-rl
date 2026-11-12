import pytest
from transformers import Qwen3Config

from prime_rl.configs.trainer import InitConfig, ModelConfig, TokenizerConfig
from prime_rl.trainer.model import check_ckpt_weights_dir, setup_tokenizer


def test_a_ckpt_weights_step_dir_needs_a_config_json(tmp_path):
    step_dir = tmp_path / "step_5"
    step_dir.mkdir()
    config = ModelConfig(name=str(step_dir), init=InitConfig(type="ckpt_weights"))
    with pytest.raises(FileNotFoundError, match="has no config.json"):
        check_ckpt_weights_dir(config)
    Qwen3Config().save_pretrained(step_dir)
    check_ckpt_weights_dir(config)


def test_a_model_name_without_a_tokenizer_asks_for_tokenizer_name(tmp_path):
    Qwen3Config().save_pretrained(tmp_path)
    with pytest.raises(FileNotFoundError, match="set tokenizer.name"):
        setup_tokenizer(TokenizerConfig(name=str(tmp_path)), model_name=str(tmp_path))
