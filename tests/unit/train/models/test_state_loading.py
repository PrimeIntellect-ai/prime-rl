import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM


@pytest.fixture
def model() -> LagunaForCausalLM:
    config = LagunaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
    )
    with torch.device("meta"):
        return LagunaForCausalLM(config)


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
