import json

import pytest
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3VLConfig

from prime_rl.trainer.ckpt import ModelFiles, check_model_config

FIELDS = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def test_a_checkpoint_step_dir_loads_as_the_model(tmp_path, tokenizer):
    ModelFiles(Qwen3Config(**FIELDS), tokenizer, None, GenerationConfig(do_sample=True, temperature=0.6)).save(tmp_path)

    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, Qwen3Config) and config.num_hidden_layers == 2
    check_model_config(config, tmp_path)
    assert GenerationConfig.from_pretrained(tmp_path).temperature == 0.6

    loaded = AutoTokenizer.from_pretrained(tmp_path)
    assert loaded.chat_template == CHAT_TEMPLATE
    assert loaded.encode("Hello world") == tokenizer.encode("Hello world")


def test_a_vlm_checkpoint_step_dir_keeps_the_configured_tokenizer(tmp_path):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    tokenizer.chat_template = CHAT_TEMPLATE
    ModelFiles(Qwen3Config(**FIELDS), tokenizer, processor, None).save(tmp_path)

    assert type(AutoProcessor.from_pretrained(tmp_path)) is type(processor)
    assert AutoTokenizer.from_pretrained(tmp_path).chat_template == CHAT_TEMPLATE


def test_a_checkpoint_with_another_architecture_does_not_load(tmp_path, tokenizer):
    config = Qwen3Config(**FIELDS)
    check_model_config(config, tmp_path)  # without config.json
    ModelFiles(config, tokenizer, None, None).save(tmp_path)
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match="num_hidden_layers 2 -> 3"):
        check_model_config(Qwen3Config(**{**FIELDS, "num_hidden_layers": 3}), tmp_path)

    # A field the checkpoint's config.json lacks has the class default there.
    saved = json.loads((tmp_path / "config.json").read_text())
    del saved["attention_bias"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match=r"attention_bias \(missing\) -> True"):
        check_model_config(Qwen3Config(**FIELDS, attention_bias=True), tmp_path)

    saved["layer_types"][1] = "sliding_attention"
    (tmp_path / "config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=r"has another architecture: layer_types \(differs from index 1\)$"):
        check_model_config(config, tmp_path)


def test_a_vlm_checkpoint_compares_its_sub_configs_by_architecture(tmp_path):
    config = Qwen3VLConfig()
    saved = json.loads(config.to_json_string(use_diff=False))
    saved["text_config"]["transformers_version"] = "0.0.0"
    saved["vision_config"]["dtype"] = "float16"
    (tmp_path / "config.json").write_text(json.dumps(saved))
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match=r"text_config\.hidden_size 4096 -> 1024"):
        check_model_config(Qwen3VLConfig(text_config={"hidden_size": 1024}), tmp_path)
