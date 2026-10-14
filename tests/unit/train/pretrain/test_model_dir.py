import pytest
from transformers import AutoConfig, AutoTokenizer

from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.entrypoints.pretrain import write_model_dir

TOKENIZER = "PrimeIntellect/Qwen3-0.6B"


def make_config(tmp_path, **arch) -> PretrainConfig:
    return PretrainConfig(
        model={"arch": {"model_type": "glm4_moe", "hidden_size": 64, "num_hidden_layers": 2, **arch}},
        tokenizer={"name": TOKENIZER},
        data={"type": "fake", "seq_len": 8, "batch_tokens": 8, "micro_batch_tokens": 8},
        optim={"type": "muon", "lr": 1e-3},
        max_steps=1,
        output_dir=tmp_path,
    )


def test_model_dir_takes_vocabulary_and_special_tokens_from_tokenizer(tmp_path):
    config = make_config(tmp_path)
    write_model_dir(config)
    hf_config = AutoConfig.from_pretrained(config.model.name)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    assert hf_config.model_type == "glm4_moe"
    assert hf_config.hidden_size == 64
    assert hf_config.vocab_size == -(-len(tokenizer) // 128) * 128
    assert hf_config.eos_token_id == tokenizer.eos_token_id


def test_model_dir_rejects_unknown_arch_fields(tmp_path):
    with pytest.raises(ValueError, match="hiden_size"):
        write_model_dir(make_config(tmp_path, hiden_size=64))
