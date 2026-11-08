import pytest
from transformers import LlamaConfig, Qwen3Config

from prime_rl.configs.pretrain import PretrainModelConfig
from prime_rl.trainer.ckpt import check_model_config, write_model_config
from prime_rl.trainer.model import get_model_config

FIELDS = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}


def test_model_type_defines_the_architecture():
    """Without a name, model.config's model_type picks the config class and its other fields override the defaults."""
    config = get_model_config(PretrainModelConfig(config={"model_type": "qwen3", **FIELDS}))
    assert isinstance(config, Qwen3Config)
    assert {key: getattr(config, key) for key in FIELDS} == FIELDS
    assert config.vocab_size == Qwen3Config().vocab_size


def test_model_config_overrides_a_named_model(tmp_path):
    LlamaConfig(**FIELDS, vocab_size=1000).save_pretrained(tmp_path)
    config = get_model_config(PretrainModelConfig(name=str(tmp_path), config={"num_hidden_layers": 3}))
    assert isinstance(config, LlamaConfig)
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (64, 3, 1000)
    with pytest.raises(ValueError, match="model_type"):
        get_model_config(PretrainModelConfig(name=str(tmp_path), config={"model_type": "qwen3"}))


def test_model_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="hiden_size"):
        get_model_config(PretrainModelConfig(config={"model_type": "qwen3", "hiden_size": 64}))


def test_a_checkpoint_with_another_architecture_does_not_load(tmp_path):
    config = Qwen3Config(**FIELDS)
    write_model_config(config, tmp_path)
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match="num_hidden_layers"):
        check_model_config(Qwen3Config(**{**FIELDS, "num_hidden_layers": 3}), tmp_path)
