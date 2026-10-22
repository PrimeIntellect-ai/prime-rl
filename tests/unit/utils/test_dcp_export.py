import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import convert_state_dict_to_hf


@pytest.mark.parametrize("format", ["bf16", "fp8"])
def test_cpu_export_workers_match_single_worker(tiny_run, format):
    _, step, _, _ = tiny_run
    cli = _cli(format)
    single = step / "export-single"
    threaded = step / "export-threaded"
    cli.convert(step, single)
    assert cli.convert(step, threaded, workers=2) == threaded
    names = sorted(path.name for path in single.iterdir())
    assert sorted(path.name for path in threaded.iterdir()) == names
    for name in names:
        assert (threaded / name).read_bytes() == (single / name).read_bytes()
    assert not list(step.glob(".export-threaded-*"))


def test_cpu_export_rejects_world_size_above_one(tiny_run, monkeypatch):
    cli, step, _, _ = tiny_run
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError):
        cli.convert(step, step / "weights")
    assert not (step / "weights").exists()


def test_cpu_export_rejects_non_positive_workers(tiny_run):
    cli, step, _, _ = tiny_run
    with pytest.raises(ValueError):
        cli.convert(step, step / "weights", workers=0)
    assert not (step / "weights").exists()


def _cli(format: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module(f"convert_dcp_to_{format}")


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
    return model, state, convert_state_dict_to_hf(model, gathered)


def test_cpu_export_matches_conversion_chain(tmp_path):
    model, state, expected = _moe_state()
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    cli = _cli("bf16")
    opened = cli.Checkpoint(checkpoint, FileSystemReader(checkpoint).read_metadata(), model, None, None)
    exported = {name: tensor for keys in cli.get_layers(model) for name, tensor in cli.load_layer(opened, keys).items()}
    assert exported.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0)


def test_rejects_broken_checkpoint(tmp_path):
    model, state, _ = _moe_state()
    del state["model.layers.0.self_attn.q_proj.weight"]
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    cli = _cli("bf16")
    opened = cli.Checkpoint(checkpoint, FileSystemReader(checkpoint).read_metadata(), model, None, None)
    with pytest.raises(BaseException):
        for keys in cli.get_layers(model):
            cli.load_layer(opened, keys)


def test_tied_qwen3_5_moe_export_drops_headless_weights(tmp_path):
    """Real Qwen 3.5 MoE tie: the stale head must not ship; the export reties the actual embed."""
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
    cli = _cli("bf16")
    opened = cli.Checkpoint(checkpoint, FileSystemReader(checkpoint).read_metadata(), model, None, None)
    exported = {name: tensor for keys in cli.get_layers(model) for name, tensor in cli.load_layer(opened, keys).items()}
    assert "lm_head.weight" not in exported
    assert exported["model.embed_tokens.weight"].flatten()[0].item() == 1
