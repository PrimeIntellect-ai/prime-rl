import importlib
import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_conversion import conversion_chain
from prime_rl.utils.utils import default_dtype


@pytest.mark.parametrize("format", ["bf16", "fp8"])
def test_cpu_export_workers_match_single_worker(tiny_run, format):
    _, step, _, _ = tiny_run
    cli = importlib.import_module(f"convert_dcp_to_{format}")
    single = step / "export-single"
    threaded = step / "export-threaded"
    cli.convert(step, single)
    assert cli.convert(step, threaded, workers=2) == threaded
    names = sorted(path.name for path in single.iterdir())
    assert sorted(path.name for path in threaded.iterdir()) == names
    for name in names:
        assert (threaded / name).read_bytes() == (single / name).read_bytes()
    assert not list(step.glob(".export-threaded-*"))


def _moe_state():
    from prime_rl.trainer.models.layers.moe import GroupedExperts
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=False,
        moe_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
    )
    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM._from_config(config)
    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.init_weights(config.initializer_range)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    gathered = {}
    for name, tensor in state.items():
        if tensor.is_floating_point():
            dtype = torch.float32 if model.keep_in_fp32_for_weight_transfer(name) else torch.bfloat16
            tensor = tensor.to(dtype)
        gathered[name] = tensor
    return config, state, apply_prime_to_hf(gathered, conversion_chain(config.to_dict()))


def test_cpu_export_matches_conversion_chain(tmp_path):
    import torch.distributed.checkpoint as dcp

    config, state, expected = _moe_state()
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


@pytest.mark.parametrize("broken", ["missing-attention", "missing-experts", "wrong-shape"])
def test_rejects_broken_checkpoint(tmp_path, broken):
    import torch.distributed.checkpoint as dcp

    config, state, _ = _moe_state()
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


def test_tied_qwen3_5_moe_export_reties_headless_weights(tmp_path):
    """Real Qwen 3.5 MoE tie: the stale head must not ship; native HF reties the actual embed."""
    from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        tie_word_embeddings=True,
    )
    with torch.device("cpu"):
        model = Qwen3_5ForCausalLM._from_config(config)
    state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
    state["model.embed_tokens.weight"].fill_(1)
    state["lm_head.weight"].fill_(99)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)

    from convert_bf16_to_fp8 import quantize_state_dict

    for group in iter_cpu_hf_groups(checkpoint, config.to_dict()):
        quantized, _ = quantize_state_dict(group, 2, "cpu")
        assert "lm_head.weight" not in quantized
    exported = {
        name: tensor for group in iter_cpu_hf_groups(checkpoint, config.to_dict()) for name, tensor in group.items()
    }
    assert "lm_head.weight" not in exported
    assert exported["model.embed_tokens.weight"].flatten()[0].item() == 1

    export = tmp_path / "hf"
    write_hf_shards(iter([exported]), export)
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig as NativeConfig

    native = NativeConfig(**config.to_dict())
    native.save_pretrained(export)
    restored, info = AutoModelForCausalLM.from_pretrained(export, config=native, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert restored.get_input_embeddings().weight.flatten()[0].item() == 1
    assert restored.get_output_embeddings().weight.flatten()[0].item() == 1
