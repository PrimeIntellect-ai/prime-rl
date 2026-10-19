import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerFast

from prime_rl.utils.export_assets import prepare_export_config, save_model_assets


@pytest.fixture
def local_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    source = tmp_path / "base-model"
    source.mkdir()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(source)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 1,
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 3,
        "eos_token_id": 1,
        "pad_token_id": None,
        "use_cache": False,
        "dtype": "float32",
        "torch_dtype": "float16",
        "custom_export_attribute": {"preserve": True},
    }
    (source / "config.json").write_text(json.dumps(config))
    return source, config


@pytest.mark.parametrize(("depth", "expected"), [(2, 2), (12, 8)])
def test_prepare_export_config_applies_debug_depth_and_pad_without_mutating(depth, expected):
    config = {"num_hidden_layers": 8, "eos_token_id": [3, 4], "pad_token_id": None, "use_cache": True}
    original = deepcopy(config)
    result = prepare_export_config(config, {"debug": {"num_layers": depth}})
    assert result["num_hidden_layers"] == expected
    assert result["pad_token_id"] == 3
    assert result["use_cache"] is False
    assert config == original
    assert prepare_export_config({**config, "pad_token_id": 9}, {})["pad_token_id"] == 9


@pytest.mark.parametrize(
    ("config_override", "model_override", "error"),
    [
        ({}, {"lora": {"rank": 4}}, "LoRA"),
        ({}, {"vlm": {}}, "text-only"),
        ({"vision_config": {}}, {}, "text-only"),
        ({"text_config": {}}, {}, "text-only"),
        ({"quantization_config": {"quant_method": "fp8"}}, {}, "quantization_config"),
    ],
)
def test_prepare_export_config_rejects_unsupported_checkpoints(config_override, model_override, error):
    config = {"num_hidden_layers": 1, "eos_token_id": 1, **config_override}
    with pytest.raises(ValueError, match=error):
        prepare_export_config(config, model_override)


@pytest.mark.parametrize("template_from_file", [False, True])
def test_save_model_assets_preserves_json_and_local_tokenizer(local_assets, tmp_path, template_from_file):
    source, config = local_assets
    model = {"name": str(source)}
    prepared = prepare_export_config(config, model)
    template = "{% for message in messages %}{{ message['content'] }}<eos>{% endfor %}"
    template_setting = template
    if template_from_file:
        template_path = tmp_path / "chat.jinja"
        template_path.write_text(template)
        template_setting = str(template_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "assets"
    save_model_assets(prepared, model, {"chat_template": template_setting}, output)

    exported_config = json.loads((output / "config.json").read_text())
    assert exported_config == prepared
    assert exported_config["dtype"] == "bfloat16"
    assert "torch_dtype" not in exported_config
    assert GenerationConfig.from_pretrained(output).use_cache is True
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1
    assert tokenizer.chat_template == template
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "hello<eos>"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert not list(output.glob("*.safetensors"))


@pytest.fixture
def tiny_run(tmp_path, local_assets, monkeypatch):
    source, config = local_assets
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    run = tmp_path / "run"
    step = run / "checkpoints" / "step_1"
    checkpoint = step / "trainer"
    configs = run / "configs" / "latest" / "resolved"
    configs.mkdir(parents=True)
    (configs / "trainer.json").write_text(
        json.dumps({"model": {"name": str(source)}, "tokenizer": {"chat_template": "{{ messages[0]['content'] }}"}})
    )
    state = {
        "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "model.norm.weight": torch.ones(2),
        "lm_head.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }
    prefix = "model.layers.0"
    for name in ("input_layernorm", "post_attention_layernorm"):
        state[f"{prefix}.{name}.weight"] = torch.ones(2)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[f"{prefix}.self_attn.{name}.weight"] = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    for name in ("gate_proj", "up_proj", "down_proj"):
        state[f"{prefix}.mlp.{name}.weight"] = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dcp.save(
        {"app": {"model": state, "optimizer": {"exp_avg": torch.full((2, 2), 99.0)}, "step": 1}},
        checkpoint_id=checkpoint,
        no_dist=True,
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    cli = importlib.import_module("convert_dcp_to_bf16")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    return cli, step, state, before


@pytest.mark.parametrize("fail_writer", [False, True], ids=["publish", "writer-failure"])
def test_convert_publishes_complete_assets_or_no_output(tmp_path, tiny_run, monkeypatch, fail_writer):
    cli, step, state, before = tiny_run
    output = step / "weights"
    if fail_writer:

        def fail_after_assets(groups, staging):
            assert (staging / "config.json").exists()
            assert (staging / "tokenizer.json").exists()
            (staging / "partial.safetensors").write_bytes(b"incomplete")
            raise OSError("injected writer failure")

        monkeypatch.setattr(cli, "write_hf_shards", fail_after_assets)
        with pytest.raises(OSError, match="injected writer failure"):
            cli.convert(step)
        assert not output.exists()
    else:
        assert cli.convert(step) == output
        index = json.loads((output / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == set(state)
        actual = {}
        for filename in set(index["weight_map"].values()):
            actual.update(load_file(output / filename))
        assert actual.keys() == state.keys()
        for name, tensor in state.items():
            assert actual[name].dtype == torch.bfloat16
            torch.testing.assert_close(actual[name], tensor.bfloat16(), rtol=0, atol=0)
        exported_config = json.loads((output / "config.json").read_text())
        assert exported_config["dtype"] == "bfloat16"
        assert "torch_dtype" not in exported_config
        assert GenerationConfig.from_pretrained(output).use_cache is True
        tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
        assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1
        assert tokenizer.chat_template == "{{ messages[0]['content'] }}"
    assert not list(step.glob(".weights-*"))
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(output)
    }
    assert after == before
