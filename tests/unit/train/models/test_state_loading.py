import copy

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM
from prime_rl.utils.weights import load_state_dict_keys


def laguna_config(num_hidden_layers: int = 2) -> LagunaConfig:
    return LagunaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        sliding_window=None,
        layer_types=["full_attention"] * num_hidden_layers,
    )


@pytest.fixture
def model() -> LagunaForCausalLM:
    with torch.device("meta"):
        return LagunaForCausalLM(laguna_config())


@pytest.mark.gpu
def test_load_dcp_from_hf_keeps_checkpoint_selection_bias(model, tmp_path, monkeypatch):
    """Checkpoint values for the persistent selection bias must survive loading."""
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def fake_dcp_load(state_dict, storage_reader=None, process_group=None):
        buffer = state_dict["model.layers.1.mlp.router.selection_bias"]
        buffer.copy_(expected.to(device=buffer.device, dtype=buffer.dtype))

    monkeypatch.setattr("prime_rl.trainer.model.dcp_load", fake_dcp_load)
    monkeypatch.setattr("prime_rl.trainer.model.load_state_dict_keys", lambda path: model.state_dict().keys())
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)

    load_dcp_from_hf(model, ModelConfig(name=str(tmp_path)))

    selection_bias = model.model.layers[1].mlp.router.selection_bias
    torch.testing.assert_close(selection_bias.cpu(), expected.to(selection_bias.dtype))


@pytest.mark.parametrize("rank", [0, 1])
def test_load_dcp_from_hf_refuses_block_fp8_checkpoints(model, tmp_path, monkeypatch, rank):
    """Every rank refuses FP8 weights with `*_scale_inv` scales before the master converts them or the others wait."""
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    weight = "model.layers.1.mlp.experts.0.gate_proj.weight"
    checkpoint = {weight: torch.ones(16, 32).to(torch.float8_e4m3fn), f"{weight}_scale_inv": torch.ones(1, 1)}
    save_file(checkpoint, tmp_path / "model.safetensors")

    monkeypatch.setattr(model, "to_empty", lambda device: nn.Module.to_empty(model, device="cpu"))
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.load_state_dict", lambda *args, **kwargs: pytest.fail("converted"))
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: pytest.fail("waited"))

    with pytest.raises(ValueError, match="block-FP8 checkpoint.*convert_fp8_to_bf16.py"):
        load_dcp_from_hf(model, ModelConfig(name=str(tmp_path)))


@pytest.mark.parametrize(
    ("runs", "incomplete", "caches"),
    [
        ((2, None, 2), [], ["prime", "prime-L2"]),
        ((None, 2), [], ["prime"]),
        ((3, None), [], ["prime"]),
        ((2,), ["prime"], ["prime", "prime-L2"]),
        ((2,), ["prime-L2"], ["prime-L2"]),
    ],
    ids=[
        "cut-first",
        "full-first",
        "num-layers-at-full-depth",
        "cut-beside-incomplete-full-cache",
        "incomplete-cut-cache-rebuilt",
    ],
)
def test_debug_cut_and_full_runs_share_a_snapshot(runs, incomplete, caches, tmp_path, monkeypatch):
    """`model.debug.num_layers` runs and full-depth runs load the exact checkpoint from one HF snapshot."""
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)

    # Layer 0 is dense, so a two-layer cut still converts a MoE layer.
    config = laguna_config(num_hidden_layers=3)
    torch.manual_seed(0)
    reference = LagunaForCausalLM(config)
    with torch.no_grad():
        for tensor in [*reference.parameters(), *reference.buffers()]:
            if tensor.is_floating_point():
                tensor.copy_(torch.randn_like(tensor))
    expected = {name: value.clone() for name, value in reference.state_dict().items()}
    snapshot = tmp_path / "snapshot"
    config.save_pretrained(snapshot)
    checkpoint = reference.convert_to_hf({name: value.clone() for name, value in expected.items()})
    save_file({name: value.contiguous() for name, value in checkpoint.items()}, snapshot / "model.safetensors")
    for cache in incomplete:
        # Left over from an interrupted conversion: no .prime-v1 marker.
        (snapshot / cache).mkdir()
        (snapshot / cache / "partial").touch()

    for num_layers in runs:
        run_config = copy.deepcopy(config)
        if num_layers is not None:
            run_config.num_hidden_layers = min(num_layers, config.num_hidden_layers)  # what get_model does
        with torch.device("meta"):
            model = LagunaForCausalLM(run_config)
        model.to_empty = lambda device, model=model: nn.Module.to_empty(model, device="cpu")
        load_dcp_from_hf(model, ModelConfig(name=str(snapshot), debug={"num_layers": num_layers}))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    assert sorted(path.name for path in snapshot.iterdir() if path.is_dir()) == caches
    if "prime" in incomplete:
        # A cut run leaves the full cache alone, complete or not.
        assert [path.name for path in (snapshot / "prime").iterdir()] == ["partial"]
    if "prime-L2" in incomplete:
        assert (snapshot / "prime-L2" / ".prime-v1").is_file() and not (snapshot / "prime-L2" / "partial").exists()
    if "prime-L2" in caches:
        # The cut cache holds only the layers the cut model loads.
        assert not any(".layers.2." in name for name in load_state_dict_keys(snapshot / "prime-L2"))
