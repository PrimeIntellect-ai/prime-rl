import json
from copy import deepcopy

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups
from prime_rl.utils.export_assets import prepare_export_config, save_model_assets
from prime_rl.utils.export_conversion import conversion_chain
from prime_rl.utils.utils import default_dtype


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


def test_prepare_export_config_matches_deepseek_kv_precision_override():
    config = {"model_type": "deepseek_v4", "num_hidden_layers": 4, "kv_precision": "bf16"}
    result = prepare_export_config(config, {"dsv4_kv_precision": "fp8"})
    assert result["kv_precision"] == "fp8"
    assert config["kv_precision"] == "bf16"


def test_prepare_export_config_rejects_lora():
    with pytest.raises(ValueError, match="LoRA"):
        prepare_export_config({"num_hidden_layers": 1}, {"lora": {"rank": 4}})


def test_prepare_export_config_uses_native_layer_schedule_defaults():
    config = {"model_type": "nemotron_h", "hybrid_override_pattern": "M*E*", "hidden_size": 8}
    result = prepare_export_config(config, {"debug": {"num_layers": 2}})
    assert result["num_hidden_layers"] == 2
    assert len(result["layers_block_type"]) == 2
    assert result["dtype"] == "bfloat16"


def test_prepare_export_config_preserves_vlm_and_normalizes_quantization():
    config = {
        "model_type": "qwen3_5",
        "quantization_config": {"quant_method": "fp8"},
        "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
        "text_config": {
            "num_hidden_layers": 8,
            "dtype": "float32",
            "quantization_config": {"quant_method": "fp8"},
            "indexer_types": ["full", "shared"],
            "eos_token_id": 1,
        },
        "vision_config": {"depth": 4, "dtype": "float32"},
    }
    original = deepcopy(config)
    result = prepare_export_config(
        config, {"vlm": {}, "trust_remote_code": True, "debug": {"num_layers": 2}, "dsa_backend": "test"}
    )
    assert config == original
    assert result["auto_map"] == config["auto_map"]
    assert result["text_config"]["num_hidden_layers"] == 2
    assert result["text_config"]["use_index_cache"] is True
    assert result["dsa_backend"] == "test"
    assert "dsa_backend" not in result["text_config"]
    assert result["vision_config"]["depth"] == 4
    for value in (result, result["text_config"], result["vision_config"]):
        assert value["dtype"] == "bfloat16"
        assert value["use_cache"] is False
        assert "quantization_config" not in value


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
    assert exported_config["dtype"] == "bfloat16"
    assert "torch_dtype" not in exported_config
    assert GenerationConfig.from_pretrained(output).use_cache is True
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1
    assert tokenizer.chat_template == template
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "hello<eos>"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert not list(output.glob("*.safetensors"))


def test_save_model_assets_handles_nested_text_generation(local_assets, tmp_path):
    source, text = local_assets
    config = {
        "model_type": "qwen3_5",
        "text_config": text,
        "vision_config": {"depth": 2},
    }
    prepared = prepare_export_config(config, {"vlm": {}})
    output = tmp_path / "nested-assets"
    save_model_assets(prepared, {"name": str(source)}, {}, output)
    generation = GenerationConfig.from_pretrained(output)
    assert generation.use_cache is True
    assert generation.eos_token_id == 1


def test_saved_config_matches_native_bf16_meta_model_serialization(local_assets, tmp_path):
    from transformers import AutoConfig, AutoModelForCausalLM

    source, config = local_assets
    native = AutoConfig.for_model("llama", **{key: value for key, value in config.items() if key != "model_type"})
    native.use_cache = False
    native.pad_token_id = 1
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(native, dtype=torch.bfloat16, attn_implementation="eager")
    reference = tmp_path / "reference"
    model.config.save_pretrained(reference)
    output = tmp_path / "serialized"
    prepared = prepare_export_config(config, {"optimization_dtype": "float32"})
    save_model_assets(prepared, {"name": str(source)}, {}, output)
    expected = json.loads((reference / "config.json").read_text())
    actual = json.loads((output / "config.json").read_text())
    assert actual["dtype"] == "bfloat16"
    assert actual == expected


@pytest.mark.parametrize("remote_source", [False, True], ids=["local", "cached-remote"])
def test_save_model_assets_localizes_trusted_code(local_assets, tmp_path, monkeypatch, remote_source):
    from transformers import dynamic_module_utils

    from prime_rl.utils import export_assets

    source, config = local_assets
    code_dir = tmp_path / "custom-code"
    code_dir.mkdir()
    (code_dir / "helper.py").write_text("ANSWER = 42\n")
    (code_dir / "configuration_custom.py").write_text("from .helper import ANSWER\n")
    (code_dir / "modeling_custom.py").write_text("from .configuration_custom import ANSWER\n")
    prefix = "fake/custom--" if remote_source else ""
    config["auto_map"] = {
        "AutoConfig": prefix + "configuration_custom.CustomConfig",
        "AutoModelForCausalLM": prefix + "modeling_custom.CustomModel",
    }
    calls = []
    if remote_source:

        def cached_file(repo, module, **kwargs):
            calls.append((repo, module, kwargs))
            assert repo == "fake/custom"
            return str(code_dir / module)

        monkeypatch.setattr(dynamic_module_utils, "get_cached_module_file", cached_file)
    else:
        for path in code_dir.glob("*.py"):
            (source / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(export_assets, "try_load_processor", lambda *a, **kw: None)
    original = deepcopy(config)
    output = tmp_path / "trusted-export"
    save_model_assets(config, {"name": str(source), "trust_remote_code": True}, {}, output)
    assert config == original
    for path in code_dir.glob("*.py"):
        assert (output / path.name).read_bytes() == path.read_bytes()
    exported = json.loads((output / "config.json").read_text())
    assert exported["auto_map"] == {
        "AutoConfig": "configuration_custom.CustomConfig",
        "AutoModelForCausalLM": "modeling_custom.CustomModel",
    }
    if remote_source:
        assert {module for _, module, _ in calls} == {"configuration_custom.py", "modeling_custom.py"}


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


def test_qwen3_5_shortened_real_model_export_matches_debug_schedule(tmp_path):
    from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    text = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
    )
    prepared = prepare_export_config(text.to_dict(), {"debug": {"num_layers": 2}})
    assert prepared["num_hidden_layers"] == 2
    assert len(prepared["layer_types"]) == 2

    shortened = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
    )
    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = Qwen3_5ForCausalLM._from_config(shortened)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    expected = apply_prime_to_hf(dict(state), conversion_chain(prepared))
    output = {name: tensor for group in iter_cpu_hf_groups(checkpoint, prepared) for name, tensor in group.items()}
    assert output.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(output[name], tensor, rtol=0, atol=0)
