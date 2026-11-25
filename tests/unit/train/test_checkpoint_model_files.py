import json

import pytest
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3VLConfig

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.ckpt import ModelFiles, check_model_config, setup_model_files
from prime_rl.trainer.model import get_model
from tests.unit.train.models.extra_config_stub import (
    EXTRA_CONFIG,
    EXTRA_CONFIG_ATTR,
    EXTRA_CONFIG_FILE,
    EXTRA_CONFIG_KEY,
    KEY_VALUE,
    HeadStub,
    KeyStub,
    register,
    write_config_key,
    write_stub,
)

FIELDS = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
MODELING_OVERRIDES = {"attention": {"full_attention": "block_bidirectional"}, "readout": "slot_lm"}


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

    # A field the checkpoint's config.json lacks takes the value its other fields give: layer_types has 2 entries,
    # not the 28 of the class default.
    saved = json.loads((tmp_path / "config.json").read_text())
    del saved["attention_bias"], saved["layer_types"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match=r"attention_bias \(missing\) -> True"):
        check_model_config(Qwen3Config(**FIELDS, attention_bias=True), tmp_path)

    saved["layer_types"] = ["full_attention", "sliding_attention"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=r"has another architecture: layer_types \(differs from index 1\)$"):
        check_model_config(config, tmp_path)


def test_a_step_dir_keeps_a_config_key_the_config_class_does_not_define(tmp_path, tokenizer):
    """A config.json key the config class does not define survives a step directory and counts as architecture: the
    inference server may read it. vLLM serves a Qwen3.5 checkpoint with ``modeling_overrides`` as another class."""
    release, step_dir = tmp_path / "release", tmp_path / "run" / "checkpoints" / "step_5"
    Qwen3Config(**FIELDS).save_pretrained(release)
    saved = json.loads((release / "config.json").read_text())
    (release / "config.json").write_text(json.dumps({**saved, "modeling_overrides": MODELING_OVERRIDES}))
    setup_model_files(ModelConfig(name=str(release), attn="flash_attention_2"), tokenizer, None).save(step_dir)

    assert json.loads((step_dir / "config.json").read_text())["modeling_overrides"] == MODELING_OVERRIDES
    config = AutoConfig.from_pretrained(step_dir)
    assert config.modeling_overrides == MODELING_OVERRIDES
    check_model_config(config, step_dir)
    config.modeling_overrides = {"readout": "slot_lm"}
    with pytest.raises(ValueError, match=r"has another architecture: modeling_overrides$"):
        check_model_config(config, step_dir)
    del config.modeling_overrides
    with pytest.raises(ValueError, match=r"has another architecture: modeling_overrides$"):
        check_model_config(config, step_dir)


def test_a_vlm_checkpoint_compares_its_sub_configs_by_architecture(tmp_path):
    config = Qwen3VLConfig()
    saved = json.loads(config.to_json_string(use_diff=False))
    saved["text_config"]["transformers_version"] = "0.0.0"
    saved["vision_config"]["dtype"] = "float16"
    (tmp_path / "config.json").write_text(json.dumps(saved))
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match=r"text_config\.hidden_size 4096 -> 1024"):
        check_model_config(Qwen3VLConfig(text_config={"hidden_size": 1024}), tmp_path)


def test_a_step_dir_holds_the_extra_config_file(tmp_path, monkeypatch):
    """A step directory keeps the extra config file next to the stock config.json, so it selects the class again."""
    register(monkeypatch, HeadStub)
    release, step_dir = tmp_path / "release", tmp_path / "run" / "checkpoints" / "step_5"
    write_stub(release)
    model_config = ModelConfig(name=str(release), attn="flash_attention_2")
    setup_model_files(model_config, AutoTokenizer.from_pretrained(release), None).save(step_dir)

    assert json.loads((step_dir / EXTRA_CONFIG_FILE).read_text()) == EXTRA_CONFIG
    saved = json.loads((step_dir / "config.json").read_text())
    assert EXTRA_CONFIG_ATTR not in saved and saved["architectures"] == ["Qwen3ForCausalLM"]
    model = get_model(ModelConfig(name=str(step_dir), attn="flash_attention_2"))
    assert type(model) is HeadStub and getattr(model.config, EXTRA_CONFIG_ATTR) == EXTRA_CONFIG

    stock = tmp_path / "stock"
    write_stub(stock, extra_config=None)
    model_config = ModelConfig(name=str(stock), attn="flash_attention_2")
    setup_model_files(model_config, AutoTokenizer.from_pretrained(stock), None).save(tmp_path / "stock_step")
    assert not (tmp_path / "stock_step" / EXTRA_CONFIG_FILE).exists()


def test_a_step_dir_selects_the_class_of_a_config_key_again(tmp_path, monkeypatch):
    """A key that selects a class is part of config.json, which a step directory keeps: it selects the class again."""
    register(monkeypatch, KeyStub)
    release, step_dir = tmp_path / "release", tmp_path / "run" / "checkpoints" / "step_5"
    write_stub(release, extra_config=None)
    write_config_key(release)
    model_config = ModelConfig(name=str(release), attn="flash_attention_2")
    files = setup_model_files(model_config, AutoTokenizer.from_pretrained(release), None)
    assert files.selection == (KeyStub, KEY_VALUE)
    files.save(step_dir)

    saved = json.loads((step_dir / "config.json").read_text())
    assert saved[EXTRA_CONFIG_KEY] == KEY_VALUE and saved["architectures"] == ["Qwen3ForCausalLM"]
    model = get_model(ModelConfig(name=str(step_dir), attn="flash_attention_2"))
    assert type(model) is KeyStub and getattr(model.config, EXTRA_CONFIG_KEY) == KEY_VALUE
