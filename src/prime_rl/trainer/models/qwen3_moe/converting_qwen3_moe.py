"""HF<->PrimeRL weight conversion for Qwen3-MoE.

The declarative chain is the source of truth. The eager functions remain as
compatibility wrappers for existing checkpoint and layerwise broadcast call
sites while the same operations can also be replayed over lazy tensors.
"""

from __future__ import annotations

from torch import Tensor

from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.models.conversion_ops import (
    ConvOp,
    Rename,
    apply_hf_to_tt,
    apply_tt_to_hf,
    routed_experts_op,
)


def _layer_conversion_chain(layer_idx: int) -> list[ConvOp]:
    prefix = f"model.layers.{layer_idx}"
    return [
        Rename(f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.router.gate.weight"),
        routed_experts_op(prefix, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True),
    ]


def conversion_chain(config) -> list[ConvOp]:
    """Build the complete, present-guarded conversion chain for ``config``."""
    return [op for layer_idx in range(config.num_hidden_layers) for op in _layer_conversion_chain(layer_idx)]


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert one layer from HF to PrimeRL format in place."""
    apply_hf_to_tt(state_dict, _layer_conversion_chain(layer_idx))


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert one layer from PrimeRL to HF format in place."""
    apply_tt_to_hf(state_dict, _layer_conversion_chain(layer_index))


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert Qwen3-MoE weights from HF to PrimeRL format in place."""
    for layer_idx in range(get_max_layer_num(state_dict)):
        convert_hf_layer_to_tt(state_dict, layer_idx)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert Qwen3-MoE weights from PrimeRL to HF format in place."""
    for layer_idx in range(get_max_layer_num(state_dict)):
        convert_tt_layer_to_hf(state_dict, layer_idx)
