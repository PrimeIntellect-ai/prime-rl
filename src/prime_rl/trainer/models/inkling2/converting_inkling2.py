"""HF <-> PrimeRL weight conversion for Inkling 2.

The trainer reads BF16 checkpoints in the release layout; `tools/convert_inkling2_to_bf16.py` dequantizes
the released NVFP4 / MXFP8 routed experts into it.

* Language model: HF `model.llm.*` -> prime `model.language_model.*`. The vision tower (`model.visual.*`)
  and the speculator (`mtp.safetensors`, unprefixed keys) are not trained and are dropped.
* Block attention residuals: HF keeps every layer's queries in `attn_res`; prime moves them into the layer
  that reads them (`attn_queries.{i - 1}` -> `layers.{i}.attn_query`, `mlp_queries.{i}` ->
  `layers.{i}.mlp_query`), so each layer's weights travel together.
* Attention: HF `attn.{wq_du,wk_dv,wv_dv,wr_du,gate_attn,attn_sink_proj,wo_ud}` -> prime
  `self_attn.{q,k,v,rel,gate,sink,o}_proj`.
* Router: HF `mlp.gate.weight` -> prime `mlp.router.gate.weight`, and the selection-only bias
  `mlp.gate.bias` -> prime `mlp.router.selection_bias`.
* Routed experts: HF `w13_weight [E, 2I, H]` interleaves gate (even) and up (odd) rows -> prime
  `gate_proj` / `up_proj`; HF `w2_weight` -> prime `down_proj`.
* Unembedding: HF `unembed.weight` covers the padded vocabulary. Serving masks the padding rows out of the
  softmax, so prime's `lm_head.weight` holds the real vocabulary only and the padding rows ride along in the
  `unembed_padding` buffer.
"""

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Drop, PrefixRename, Rename

HF_PREFIX = "model.llm."
PRIME_PREFIX = "model.language_model."

ATTENTION_RENAMES = (
    ("wq_du.weight", "q_proj.weight"),
    ("wk_dv.weight", "k_proj.weight"),
    ("wv_dv.weight", "v_proj.weight"),
    ("wr_du.weight", "rel_proj.weight"),
    ("gate_attn.weight", "gate_proj.weight"),
    ("attn_sink_proj.weight", "sink_proj.weight"),
    ("wo_ud.weight", "o_proj.weight"),
    ("q_l2_clip.scale", "q_scale"),
    ("rel_l2_clip.scale", "rel_scale"),
    ("rel_logits_proj.proj", "rel_logits_proj"),
)


class Inkling2Experts(ConvOp):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def hf_to_prime(self, state_dict: dict[str, Tensor]) -> None:
        gate_up_name = f"{self.prefix}.w13_weight"
        if gate_up_name not in state_dict:
            return

        gate_up = state_dict.pop(gate_up_name)
        if not gate_up.is_floating_point() or gate_up.element_size() == 1:
            raise ValueError(
                f"{gate_up_name} holds quantized experts ({gate_up.dtype}). "
                "Dequantize the checkpoint with tools/convert_inkling2_to_bf16.py first."
            )
        state_dict[f"{self.prefix}.gate_proj"] = gate_up[:, 0::2]
        state_dict[f"{self.prefix}.up_proj"] = gate_up[:, 1::2]
        state_dict[f"{self.prefix}.down_proj"] = state_dict.pop(f"{self.prefix}.w2_weight")

    def prime_to_hf(self, state_dict: dict[str, Tensor]) -> None:
        gate_name = f"{self.prefix}.gate_proj"
        if gate_name not in state_dict:
            return

        gate = state_dict.pop(gate_name)
        up = state_dict.pop(f"{self.prefix}.up_proj")
        gate_up = gate.new_empty((gate.shape[0], 2 * gate.shape[1], gate.shape[2]))
        gate_up[:, 0::2] = gate
        gate_up[:, 1::2] = up
        state_dict[f"{self.prefix}.w13_weight"] = gate_up
        state_dict[f"{self.prefix}.w2_weight"] = state_dict.pop(f"{self.prefix}.down_proj")


class UnpaddedUnembed(ConvOp):
    def __init__(self, unembed: str, lm_head: str, padding: str, unpadded_vocab_size: int) -> None:
        self.unembed = unembed
        self.lm_head = lm_head
        self.padding = padding
        self.unpadded_vocab_size = unpadded_vocab_size

    def hf_to_prime(self, state_dict: dict[str, Tensor]) -> None:
        if self.unembed not in state_dict:
            return
        unembed = state_dict.pop(self.unembed)
        state_dict[self.lm_head] = unembed[: self.unpadded_vocab_size]
        state_dict[self.padding] = unembed[self.unpadded_vocab_size :]

    def prime_to_hf(self, state_dict: dict[str, Tensor]) -> None:
        if self.lm_head not in state_dict:
            return
        lm_head = state_dict.pop(self.lm_head)
        # The buffer keeps the model's dtype; the weight is already in its wire dtype.
        padding = state_dict.pop(self.padding).to(dtype=lm_head.dtype, device=lm_head.device)
        state_dict[self.unembed] = torch.cat((lm_head, padding))


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.startswith(HF_PREFIX) for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.startswith((PRIME_PREFIX, "lm_head.")) for name in state_dict)


def layer_ops(layer_idx: int) -> list[ConvOp]:
    layer = f"{PRIME_PREFIX}layers.{layer_idx}"
    attn_res = f"{PRIME_PREFIX}attn_res"
    ops: list[ConvOp] = [Rename(f"{attn_res}.mlp_queries.{layer_idx}", f"{layer}.mlp_query")]
    if layer_idx > 0:
        ops.append(Rename(f"{attn_res}.attn_queries.{layer_idx - 1}", f"{layer}.attn_query"))
    ops.append(PrefixRename(f"{layer}.attn.", f"{layer}.self_attn."))
    ops.extend(Rename(f"{layer}.self_attn.{hf}", f"{layer}.self_attn.{prime}") for hf, prime in ATTENTION_RENAMES)
    ops.extend(
        [
            Rename(f"{layer}.mlp.gate.weight", f"{layer}.mlp.router.gate.weight"),
            Rename(f"{layer}.mlp.gate.bias", f"{layer}.mlp.router.selection_bias"),
            Inkling2Experts(f"{layer}.mlp.experts"),
        ]
    )
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = config.text_config
    unembed = f"{PRIME_PREFIX}unembed.weight"
    ops: list[ConvOp] = [
        Drop("model.visual.", is_prefix=True),
        # The speculator's weights, which a checkpoint directory carries in `mtp.safetensors`.
        Drop("layers.", is_prefix=True),
        Drop("markov_head.", is_prefix=True),
        Drop("fuse_weight"),
        Drop("mask_embed"),
        Drop("norm.weight"),
        PrefixRename(HF_PREFIX, PRIME_PREFIX),
        Rename(f"{PRIME_PREFIX}embed.weight", f"{PRIME_PREFIX}embed_tokens.weight"),
        Rename(f"{PRIME_PREFIX}attn_res.output_query", f"{PRIME_PREFIX}output_query"),
    ]
    if text_config.unpadded_vocab_size < text_config.vocab_size:
        ops.append(
            UnpaddedUnembed(
                unembed, "lm_head.weight", f"{PRIME_PREFIX}unembed_padding", text_config.unpadded_vocab_size
            )
        )
    else:
        ops.append(Rename(unembed, "lm_head.weight"))
    for layer_idx in range(text_config.num_hidden_layers):
        ops.extend(layer_ops(layer_idx))
    return ops
