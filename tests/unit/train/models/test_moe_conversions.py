from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.models.afmoe.converting_afmoe import conversion_chain as afmoe_conversion_chain
from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoePreTrainedModel
from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.laguna.converting_laguna import conversion_chain as laguna_conversion_chain
from prime_rl.trainer.models.laguna.modeling_laguna import LagunaPreTrainedModel
from prime_rl.trainer.models.minimax_m2.converting_minimax_m2 import conversion_chain as minimax_conversion_chain
from prime_rl.trainer.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2PreTrainedModel


def _afmoe_state_dict() -> dict[str, torch.Tensor]:
    prefix = "model.layers.0.mlp"
    state_dict = {
        f"{prefix}.router.gate.weight": torch.randn(2, 4),
        f"{prefix}.expert_bias": torch.randn(2),
        f"{prefix}.shared_experts.gate_proj.weight": torch.randn(3, 4),
        f"{prefix}.shared_experts.up_proj.weight": torch.randn(3, 4),
        f"{prefix}.shared_experts.down_proj.weight": torch.randn(4, 3),
    }
    for expert in range(2):
        state_dict[f"{prefix}.experts.{expert}.gate_proj.weight"] = torch.randn(3, 4)
        state_dict[f"{prefix}.experts.{expert}.up_proj.weight"] = torch.randn(3, 4)
        state_dict[f"{prefix}.experts.{expert}.down_proj.weight"] = torch.randn(4, 3)
    return state_dict


def _laguna_state_dict() -> dict[str, torch.Tensor]:
    prefix = "model.layers.0.mlp"
    state_dict = {
        f"{prefix}.gate.weight": torch.randn(2, 4),
        f"{prefix}.experts.e_score_correction_bias": torch.randn(2),
        f"{prefix}.shared_expert.gate_proj.weight": torch.randn(3, 4),
        f"{prefix}.shared_expert.up_proj.weight": torch.randn(3, 4),
        f"{prefix}.shared_expert.down_proj.weight": torch.randn(4, 3),
    }
    for expert in range(2):
        state_dict[f"{prefix}.experts.{expert}.gate_proj.weight"] = torch.randn(3, 4)
        state_dict[f"{prefix}.experts.{expert}.up_proj.weight"] = torch.randn(3, 4)
        state_dict[f"{prefix}.experts.{expert}.down_proj.weight"] = torch.randn(4, 3)
    return state_dict


def _minimax_state_dict() -> dict[str, torch.Tensor]:
    prefix = "model.layers.0.block_sparse_moe"
    state_dict = {
        f"{prefix}.gate.weight": torch.randn(2, 4),
        f"{prefix}.e_score_correction_bias": torch.randn(2),
    }
    for expert in range(2):
        state_dict[f"{prefix}.experts.{expert}.w1.weight"] = torch.randn(3, 4)
        state_dict[f"{prefix}.experts.{expert}.w3.weight"] = torch.randn(3, 4)
        state_dict[f"{prefix}.experts.{expert}.w2.weight"] = torch.randn(4, 3)
    return state_dict


@pytest.mark.parametrize(
    ("model_cls", "operations", "hf_state_dict"),
    [
        (AfmoePreTrainedModel, afmoe_conversion_chain(SimpleNamespace(num_hidden_layers=1)), _afmoe_state_dict()),
        (LagunaPreTrainedModel, laguna_conversion_chain(SimpleNamespace(num_hidden_layers=1)), _laguna_state_dict()),
        (
            MiniMaxM2PreTrainedModel,
            minimax_conversion_chain(SimpleNamespace(num_hidden_layers=1)),
            _minimax_state_dict(),
        ),
    ],
)
def test_current_hf_moe_conversion_roundtrip(model_cls, operations, hf_state_dict):
    original = {name: tensor.clone() for name, tensor in hf_state_dict.items()}

    assert model_cls.is_hf_state_dict(hf_state_dict)
    assert not model_cls.is_prime_state_dict(hf_state_dict)

    apply_hf_to_prime(hf_state_dict, operations)

    assert model_cls.is_prime_state_dict(hf_state_dict)
    assert not model_cls.is_hf_state_dict(hf_state_dict)
    assert {
        "model.layers.0.mlp.router.gate.weight",
        "model.layers.0.mlp.experts.gate_proj",
        "model.layers.0.mlp.experts.up_proj",
        "model.layers.0.mlp.experts.down_proj",
    } <= hf_state_dict.keys()

    apply_prime_to_hf(hf_state_dict, operations)

    assert hf_state_dict.keys() == original.keys()
    for name, tensor in original.items():
        torch.testing.assert_close(hf_state_dict[name], tensor, rtol=0, atol=0)
