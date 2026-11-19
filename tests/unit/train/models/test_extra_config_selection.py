"""Selecting a model class by an extra config file next to a stock config.json, on stubs over prime-rl's Qwen3."""

import json

import pytest
import torch
from huggingface_hub import constants
from transformers import AutoTokenizer

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.ckpt import setup_model_files
from prime_rl.trainer.model import get_model, read_extra_config
from prime_rl.trainer.models import check_selected_by_extra_config
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.utils.logger import get_logger
from tests.unit.train.models.extra_config_stub import (
    EXTRA_CONFIG,
    EXTRA_CONFIG_ATTR,
    EXTRA_CONFIG_FILE,
    ConvertingStub,
    HeadStub,
    load_like_the_trainer,
    register,
    write_release_layout,
    write_stub,
)

META = torch.device("meta")


def model_config(name, **kwargs) -> ModelConfig:
    return ModelConfig(name=str(name), attn="flash_attention_2", **kwargs)


def test_a_listed_class_sets_what_selects_it():
    check_selected_by_extra_config((HeadStub,))
    with pytest.raises(TypeError, match="Qwen3ForCausalLM must set extra_config_file, extra_config_attr, stock_model"):
        check_selected_by_extra_config((Qwen3ForCausalLM,))
    renamed = type("HeadStub", (HeadStub,), {"extra_config_file": "other.json"})
    with pytest.raises(ValueError, match="Two classes selected by an extra config file have the same name"):
        check_selected_by_extra_config((HeadStub, renamed))
    with pytest.raises(ValueError, match="Two classes selected by an extra config file extend the same stock config"):
        check_selected_by_extra_config((HeadStub, ConvertingStub))
    two = type("TwoStub", (HeadStub,), {"export_files": {"head.": "a.safetensors", "lm_head.": "b.safetensors"}})
    with pytest.raises(ValueError, match="TwoStub.export_files may name one file"):
        check_selected_by_extra_config((two,))
    stock_named = type("Qwen3ForCausalLM", (HeadStub,), {})
    with pytest.raises(ValueError, match="A class selected by an extra config file has the name of a stock archit"):
        check_selected_by_extra_config((stock_named,))


def test_an_ordinary_model_is_unaffected(tmp_path):
    write_stub(tmp_path)
    model = get_model(model_config(tmp_path), device=META)
    assert type(model) is Qwen3ForCausalLM and not hasattr(model.config, EXTRA_CONFIG_ATTR)
    files = setup_model_files(model_config(tmp_path), AutoTokenizer.from_pretrained(tmp_path), None)
    assert files.selection is None


def test_the_extra_config_file_selects_the_class(tmp_path, monkeypatch):
    register(monkeypatch, HeadStub)
    write_stub(tmp_path / "stub")
    model = get_model(model_config(tmp_path / "stub"), device=META)
    assert type(model) is HeadStub and model.head.weight.shape == (3, 16)
    assert getattr(model.config, EXTRA_CONFIG_ATTR) == EXTRA_CONFIG
    # No rename: the config keeps the stock architecture
    assert model.config.architectures == ["Qwen3ForCausalLM"]

    files = setup_model_files(model_config(tmp_path / "stub"), AutoTokenizer.from_pretrained(tmp_path / "stub"), None)
    assert files.selection == (HeadStub, EXTRA_CONFIG)
    assert files.config.architectures == ["Qwen3ForCausalLM"] and not hasattr(files.config, EXTRA_CONFIG_ATTR)

    write_stub(tmp_path / "stock", extra_config=None)
    assert type(get_model(model_config(tmp_path / "stock"), device=META)) is Qwen3ForCausalLM

    # Another architecture next to the file is not the stub's
    write_stub(tmp_path / "other")
    config = json.loads((tmp_path / "other" / "config.json").read_text())
    (tmp_path / "other" / "config.json").write_text(json.dumps(config | {"architectures": ["Qwen3Model"]}))
    assert type(get_model(model_config(tmp_path / "other"), device=META)) is Qwen3ForCausalLM


def test_a_config_that_names_a_listed_class_is_refused(tmp_path, monkeypatch):
    register(monkeypatch, HeadStub)
    write_stub(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**config, "architectures": ["HeadStub"]}))
    with pytest.raises(ValueError, match="names HeadStub, which is selected by stub_head_config.json next to a"):
        get_model(model_config(tmp_path), device=META)


@pytest.mark.parametrize(
    "content, error",
    [("{", f"{EXTRA_CONFIG_FILE} is not valid JSON"), ("[3]", "must hold a JSON object, got list")],
)
def test_a_malformed_extra_config_file_names_its_path(tmp_path, monkeypatch, content, error):
    register(monkeypatch, HeadStub)
    write_stub(tmp_path)
    (tmp_path / EXTRA_CONFIG_FILE).write_text(content)
    with pytest.raises(ValueError, match=error):
        get_model(model_config(tmp_path), device=META)


@pytest.mark.parametrize("tied", [False, True])
def test_the_stub_loads_on_cpu(tmp_path, monkeypatch, tied):
    register(monkeypatch, HeadStub)
    saved = write_stub(tmp_path, tied=tied, weights=True)
    model = get_model(model_config(tmp_path), device=torch.device("cpu"))
    assert type(model) is HeadStub and torch.equal(model.head.weight, saved["head.weight"].bfloat16())


def test_the_class_converts_a_release_lacking_a_tensor(tmp_path, monkeypatch):
    register(monkeypatch, ConvertingStub)
    saved = write_release_layout(tmp_path)
    model = get_model(model_config(tmp_path), device=META)
    assert type(model) is ConvertingStub
    load_like_the_trainer(model, tmp_path, monkeypatch)
    assert torch.equal(model.head.weight, saved["head.weight"].bfloat16())
    assert torch.equal(model.lm_head.weight, saved["model.embed_tokens.weight"].bfloat16())


def test_a_cached_snapshot_answers_without_the_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    warnings = []
    sink = get_logger().add(lambda message: warnings.append(str(message)), level="WARNING", format="{message}")
    repo = tmp_path / "models--org--stub"
    snapshot = repo / "snapshots" / ("0" * 40)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(snapshot.name)
    try:
        write_stub(snapshot)
        assert read_extra_config("org/stub", EXTRA_CONFIG_FILE) == EXTRA_CONFIG
        # A config.json alone, as fetching the config leaves it, doesn't answer: offline, the file counts as absent
        (snapshot / EXTRA_CONFIG_FILE).unlink()
        assert read_extra_config("org/stub", EXTRA_CONFIG_FILE) is None and len(warnings) == 1
        # Cached weights do answer
        (snapshot / "model.safetensors.index.json").write_text("{}")
        assert read_extra_config("org/stub", EXTRA_CONFIG_FILE) is None and len(warnings) == 1
    finally:
        get_logger().remove(sink)
