"""Per-model declarative conversion chains.

Each ``build_*_chain(num_layers, ...)`` returns the flat list of
:class:`~prime_rl.trainer.models.conversion_ops.ConvOp` that defines a model's
HF<->prime conversion, with concrete (fully templated) keys. The model classes
expose these via ``conversion_ops`` and the base ``convert_to_*`` methods play
them forward/backward. Equivalence to the legacy imperative converters is
checked in tests/unit/train/models/conversions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import (
    ConvOp,
    Drop,
    FusedGateUp,
    MoEExperts,
    Rename,
    SqueezeLeading,
    Synthetic,
)

# Per-layer routed-expert proj order shared by the Llama-style MoE models:
# prime w1=gate, w2=down, w3=up.
_GATE_DOWN_UP = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))


def _routed_experts_op(
    prefix: str,
    *,
    hf_experts: str,
    tt_experts: str,
    proj_order=_GATE_DOWN_UP,
    hf_proj_suffix: str = ".weight",
    fused: bool = False,
) -> MoEExperts:
    """Build the MoEExperts op for one layer.

    ``hf_experts``/``tt_experts`` are the (relative) expert container names,
    e.g. ``mlp.experts`` / ``block_sparse_moe.experts``. ``proj_order`` maps
    prime ``wN`` to the HF per-expert proj name."""
    projs = {
        f"{prefix}.{tt_experts}.{wn}": f"{prefix}.{hf_experts}.{{e}}.{hf_proj}{hf_proj_suffix}"
        for wn, hf_proj in proj_order
    }
    fused_spec = None
    if fused:
        fused_spec = FusedGateUp(
            gate_up=f"{prefix}.{hf_experts}.gate_up_proj",
            down=f"{prefix}.{hf_experts}.down_proj",
            w_gate=f"{prefix}.{tt_experts}.w1",
            w_down=f"{prefix}.{tt_experts}.w2",
            w_up=f"{prefix}.{tt_experts}.w3",
            split_dim=1,
        )
    return MoEExperts(projs=projs, fused=fused_spec)


def build_qwen3_moe_chain(num_layers: int) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(num_layers):
        p = f"model.layers.{i}"
        ops.append(Rename(f"{p}.mlp.gate.weight", f"{p}.mlp.router.gate.weight"))
        ops.append(_routed_experts_op(p, hf_experts="mlp.experts", tt_experts="mlp.experts", fused=True))
    return ops
