import importlib
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_conversion import conversion_chain
from prime_rl.utils.utils import default_dtype


@pytest.mark.parametrize("format", ["bf16", "fp8"])
@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_export_device_dispatch(tiny_run, monkeypatch, format, device):
    _, step, _, _ = tiny_run
    cli = importlib.import_module(f"convert_dcp_to_{format}")
    output = step / f"export-{format}"
    calls = []

    def convert_cuda(*args):
        calls.append(args)
        return output

    monkeypatch.setitem(
        sys.modules,
        "convert_dcp_to_bf16_gpu",
        SimpleNamespace(convert=convert_cuda, convert_fp8=convert_cuda),
    )
    argv = ["export", str(step), str(output)]
    if device is not None:
        argv += ["--device", device]
    if format == "fp8":
        argv += ["--block-size", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    if device == "cuda":
        expected = (step, output, 2) if format == "fp8" else (step, output)
        assert calls == [expected]
        assert not output.exists()
    else:
        assert calls == []
        assert (output / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("format", ["bf16", "fp8"])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_multi_rank_export_uses_cuda(tmp_path, monkeypatch, format, device):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    cli = importlib.import_module(f"convert_dcp_to_{format}")
    calls = []
    output = tmp_path / "export"

    def convert_cuda(*args):
        calls.append(args)
        return output

    monkeypatch.setitem(
        sys.modules, "convert_dcp_to_bf16_gpu", SimpleNamespace(convert=convert_cuda, convert_fp8=convert_cuda)
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    checkpoint = tmp_path / "missing-checkpoint"
    kwargs = {} if device is None else {"device": device}
    assert cli.convert(checkpoint, output, **kwargs) == output
    expected = (checkpoint, output, 128) if format == "fp8" else (checkpoint, output)
    assert calls == [expected]


@pytest.mark.parametrize("format", ["bf16", "fp8"])
def test_cpu_export_overwrites_existing_output(tiny_run, format):
    _, step, _, _ = tiny_run
    cli = importlib.import_module(f"convert_dcp_to_{format}")
    output = step / "existing-output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unrelated file")
    (output / "config.json").write_text("old config")
    assert cli.convert(step, output) == output
    assert json.loads((output / "config.json").read_text())["model_type"] == "llama"
    assert (output / "model.safetensors.index.json").exists()
    assert sentinel.read_text() == "unrelated file"
    assert not list(step.glob(".existing-output-*"))


@pytest.mark.parametrize("num_groups", [1, 2])
def test_hf_shard_names(tmp_path, num_groups):
    groups = [{f"weight_{number}": torch.full((2,), float(number))} for number in range(num_groups)]
    output = tmp_path / "weights"
    write_hf_shards(iter(groups), output)
    index_path = output / "model.safetensors.index.json"
    if num_groups == 1:
        assert {path.name for path in output.iterdir()} == {"model.safetensors"}
        torch.testing.assert_close(load_file(output / "model.safetensors")["weight_0"], torch.zeros(2))
    else:
        expected = {f"model-{number:05d}-of-00002.safetensors" for number in (1, 2)}
        assert {path.name for path in output.iterdir()} == expected | {index_path.name}
        index = json.loads(index_path.read_text())
        assert index["metadata"]["total_size"] == 16
        for number in range(num_groups):
            name = f"weight_{number}"
            filename = f"model-{number + 1:05d}-of-00002.safetensors"
            assert index["weight_map"][name] == filename
            torch.testing.assert_close(load_file(output / filename)[name], torch.full((2,), float(number)))


@pytest.mark.parametrize("num_groups", [1, 2])
def test_publish_export_overwrites_existing_weight_layout(tmp_path, num_groups):
    from prime_rl.utils.export_assets import publish_export

    output = tmp_path / "weights"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    (output / "model.safetensors.index.json").write_text("stale index")
    (output / "model.safetensors").write_bytes(b"stale weights")
    (output / "model-00001.safetensors").write_bytes(b"old CPU shard")
    (output / "model-00003-of-00003.safetensors").write_bytes(b"old numbered shard")
    (output / "model-00001-of-00002.safetensors").write_bytes(b"matching shard to overwrite")
    staging = tmp_path / "staging"
    groups = [{f"weight_{number}": torch.ones(2)} for number in range(num_groups)]
    write_hf_shards(iter(groups), staging)
    publish_export(staging, output)
    assert not staging.exists()
    assert sentinel.read_text() == "keep"
    expected_shards = (
        {"model.safetensors"}
        if num_groups == 1
        else {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    )
    assert {path.name for path in output.glob("*.safetensors")} == expected_shards
    index = output / "model.safetensors.index.json"
    if num_groups == 1:
        assert not index.exists()
        assert (output / "model.safetensors").exists()
    else:
        assert not (output / "model.safetensors").exists()
        assert len(json.loads(index.read_text())["weight_map"]) == 2


def _model_state(layout):
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.models.llama.configuration_llama import LlamaConfig

    from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
    from prime_rl.trainer.models.layers.moe import GroupedExperts
    from prime_rl.trainer.models.llama import LlamaForCausalLM
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    kwargs = dict(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    if layout == "dense":
        config, model_cls = LlamaConfig(**kwargs), LlamaForCausalLM
    elif layout == "moe":
        config = Qwen3MoeConfig(**kwargs, moe_intermediate_size=4, num_experts=2, num_experts_per_tok=1)
        model_cls = Qwen3MoeForCausalLM
    elif layout == "custom":
        config = Glm4MoeConfig(
            **kwargs,
            moe_intermediate_size=4,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
            first_k_dense_replace=0,
            norm_topk_prob=True,
            partial_rotary_factor=0.5,
        )
        model_cls = Glm4MoeForCausalLM
    else:
        config = GPT2Config(vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2, tie_word_embeddings=False)
        model_cls = GPT2LMHeadModel
    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = model_cls._from_config(config)
    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.init_weights(config.initializer_range)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    gathered = {}
    for name, tensor in state.items():
        if tensor.is_floating_point() and layout != "hf-native":
            dtype = torch.float32 if model.keep_in_fp32_for_weight_transfer(name) else torch.bfloat16
            tensor = tensor.to(dtype)
        gathered[name] = tensor
    return config, state, apply_prime_to_hf(gathered, conversion_chain(config.to_dict()))


@pytest.mark.parametrize("layout", ["dense", "moe", "custom", "hf-native"])
def test_export_model_layouts(tmp_path, layout):
    import torch.distributed.checkpoint as dcp

    config, state, expected = _model_state(layout)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    output = tmp_path / "hf"
    write_hf_shards(iter_cpu_hf_groups(checkpoint, config.to_dict()), output, max_shard_size=256)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    exported = {
        name: tensor
        for filename in set(index["weight_map"].values())
        for name, tensor in load_file(output / filename).items()
    }
    assert exported.keys() == expected.keys() == index["weight_map"].keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0)
    config.save_pretrained(output)
    if layout == "hf-native":
        from transformers import AutoModelForCausalLM

        restored = AutoModelForCausalLM.from_pretrained(output)
        assert torch.isfinite(restored(torch.tensor([[0, 1]])).logits).all()


@pytest.mark.parametrize("broken", ["missing-attention", "missing-experts", "wrong-shape"])
def test_rejects_broken_checkpoint(tmp_path, broken):
    import torch.distributed.checkpoint as dcp

    config, state, _ = _model_state("moe")
    if broken == "missing-attention":
        del state["model.layers.0.self_attn.q_proj.weight"]
    elif broken == "missing-experts":
        del state["model.layers.0.mlp.experts.gate_proj"]
    else:
        state["model.layers.0.mlp.experts.gate_proj"] = torch.ones(1)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, config.to_dict()))


@pytest.mark.parametrize("broken", ["gap", "overlap", "out-of-bounds"])
def test_rejects_bad_checkpoint_chunks(tiny_run, broken):
    _, step, _, _ = tiny_run
    checkpoint = step / "trainer"
    path = checkpoint / ".metadata"
    metadata = pickle.loads(path.read_bytes())
    entry = metadata.state_dict_metadata["app.model.model.embed_tokens.weight"]
    chunk = entry.chunks[0]
    if broken == "gap":
        chunk.sizes = torch.Size([chunk.sizes[0] - 1, chunk.sizes[1]])
    elif broken == "overlap":
        entry.chunks.append(deepcopy(chunk))
    else:
        chunk.offsets = torch.Size([1, 0])
    path.write_bytes(pickle.dumps(metadata))
    config = json.loads((step.parents[1] / "configs/latest/resolved/trainer.json").read_text())
    config = json.loads((Path(config["model"]["name"]) / "config.json").read_text())
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, config))


@pytest.mark.parametrize("attn", ["flash_attention_2", "auto"])
def test_custom_vlm_schema_uses_trainer_attention(attn):
    from prime_rl.trainer.models.qwen3_5 import Qwen3_5Config
    from prime_rl.utils.export_conversion import prime_schema_details

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["full_attention"],
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            num_position_embeddings=16,
        ),
    )
    schema, _, dtypes = prime_schema_details(config.to_dict(), attn_implementation=attn)
    assert schema["model.visual.blocks.0.attn.qkv.weight"] == (48, 16)
    assert dtypes["model.visual.blocks.0.attn.qkv.weight"].is_floating_point
