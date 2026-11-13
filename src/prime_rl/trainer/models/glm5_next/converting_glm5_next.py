"""HF<->PrimeRL weight conversion for the GLM-5.3 Flash text model.

* MoE layers convert exactly like GLM-4 MoE (stacked experts, router renames).
* The checkpoint stores each mHC's three tensors flat on the layer
  (``hc_attn_fn``); PrimeRL keeps them on the hyper-connection module (``attn_hc.fn``).
* The vision tower and the MTP layer have no PrimeRL counterpart and are dropped.
"""

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, Rename
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    layers_prefix = "model.language_model.layers" if hasattr(config, "text_config") else "model.layers"
    ops: list[ConvOp] = [Drop("model.visual.", is_prefix=True)]
    for layer_idx in range(text_config.num_hidden_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"
        for sublayer in ("attn", "ffn"):
            for tensor in ("fn", "base", "scale"):
                ops.append(Rename(f"{layer_prefix}.hc_{sublayer}_{tensor}", f"{layer_prefix}.{sublayer}_hc.{tensor}"))
        if text_config.mlp_layer_types[layer_idx] == "sparse":
            ops.extend(glm_moe_layer_ops(layer_idx, layers_prefix))
    for mtp_idx in range(text_config.num_nextn_predict_layers):
        ops.append(Drop(f"{layers_prefix}.{text_config.num_hidden_layers + mtp_idx}.", is_prefix=True))
    return ops


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(".hc_attn_fn" in name or name.endswith("mlp.gate.weight") for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(".attn_hc.fn" in name or name.endswith("mlp.router.gate.weight") for name in state_dict)


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
