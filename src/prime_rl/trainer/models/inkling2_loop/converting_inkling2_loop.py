"""Checkpoint <-> PrimeRL weight conversion for the looped Inkling 2 models.

"Checkpoint" means the names Thinking Machines publish (``model.llm.layers.N.attn.wq_du.weight``, ...), which is
also what the serving stack loads. The chain runs forward for ``load_dcp_from_hf`` and reversed for the weight
broadcast. Packed experts are dequantized before the chain (``dequantize.py``) and are never re-packed: the
reverse direction always emits BF16 experts.
"""

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp, Rename, StateDict
from prime_rl.trainer.models.inkling2.converting_inkling2 import InterleavedGateUp

CHECKPOINT_PREFIX = "model.llm."
# Loop message gates, as (checkpoint controller name, PrimeRL gate name).
MESSAGE_STREAMS = (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp"))


class KeepLanguageModel(ConvOp):
    """Drop what the text trainer does not own: the vision tower and the draft head shipped next to the model."""

    def hf_to_prime(self, sd: StateDict) -> None:
        for key in [key for key in sd if not key.startswith(CHECKPOINT_PREFIX)]:
            del sd[key]

    def prime_to_hf(self, sd: StateDict) -> None:
        return None


class MessageQueries(ConvOp):
    """The checkpoint stores the loop's message queries as two ``[2 * core_layers, H]`` tensors: the (query, mlp)
    pair of every core layer, and its (key, value) pair. PrimeRL holds ``[core_layers, 4, H]`` in (query, key,
    value, mlp) order. A model that loops holds every checkpoint row, also when it is truncated inside the core, so
    its conversion serves every shorter truncation; a model without a core holds none."""

    def __init__(self, queries: str, kv_queries: str, merged: str, has_core: bool) -> None:
        self.queries, self.kv_queries, self.merged = queries, kv_queries, merged
        self.has_core = has_core

    def hf_to_prime(self, sd: StateDict) -> None:
        if self.queries not in sd:
            return
        queries = sd.pop(self.queries).unflatten(0, (-1, 2))
        kv_queries = sd.pop(self.kv_queries).unflatten(0, (-1, 2))
        if self.has_core:
            sd[self.merged] = torch.cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1)

    def prime_to_hf(self, sd: StateDict) -> None:
        if self.merged not in sd:
            return
        merged = sd.pop(self.merged)
        sd[self.queries] = merged[:, [0, 3]].flatten(0, 1)
        sd[self.kv_queries] = merged[:, 1:3].flatten(0, 1)


class PaddedUnembedding(ConvOp):
    """The checkpoint's unembedding has one row per embedding row, and only the first ``unpadded_vocab_size`` are
    logits. The rows past them are unused (and not zero in the release): they are dropped on load and written back
    as zeros."""

    def __init__(self, padded: str, unpadded: str, vocab_size: int, unpadded_vocab_size: int) -> None:
        self.padded, self.unpadded = padded, unpadded
        self.vocab_size, self.unpadded_vocab_size = vocab_size, unpadded_vocab_size

    def hf_to_prime(self, sd: StateDict) -> None:
        if self.padded in sd:
            sd[self.unpadded] = sd.pop(self.padded)[: self.unpadded_vocab_size]

    def prime_to_hf(self, sd: StateDict) -> None:
        if self.unpadded not in sd:
            return
        weight = sd.pop(self.unpadded)
        padding = weight.new_zeros((self.vocab_size - weight.shape[0], weight.shape[1]))
        sd[self.padded] = torch.cat([weight, padding])


def _layer_ops(layer_idx: int) -> list[ConvOp]:
    checkpoint = f"{CHECKPOINT_PREFIX}layers.{layer_idx}"
    prime = f"model.layers.{layer_idx}"
    attn, self_attn = f"{checkpoint}.attn", f"{prime}.self_attn"
    ops: list[ConvOp] = [
        Rename(f"{attn}.wq_du.weight", f"{self_attn}.q_proj.weight"),
        Rename(f"{attn}.wk_dv.weight", f"{self_attn}.k_proj.weight"),
        Rename(f"{attn}.wv_dv.weight", f"{self_attn}.v_proj.weight"),
        Rename(f"{attn}.wr_du.weight", f"{self_attn}.rel_proj.weight"),
        Rename(f"{attn}.gate_attn.weight", f"{self_attn}.gate_proj.weight"),
        Rename(f"{attn}.attn_sink_proj.weight", f"{self_attn}.sink_proj.weight"),
        Rename(f"{attn}.wo_ud.weight", f"{self_attn}.o_proj.weight"),
        Rename(f"{attn}.q_l2_clip.scale", f"{self_attn}.q_scale"),
        Rename(f"{attn}.rel_l2_clip.scale", f"{self_attn}.rel_scale"),
        Rename(f"{attn}.rel_logits_proj.proj", f"{self_attn}.rel_table"),
        Rename(f"{checkpoint}.mlp_norm.weight", f"{prime}.post_attention_layernorm.weight"),
        Rename(f"{checkpoint}.attn_sconv.weight", f"{prime}.attn_conv.weight"),
        Rename(f"{checkpoint}.mlp_sconv.weight", f"{prime}.mlp_conv.weight"),
        Rename(f"{checkpoint}.mlp.gate.weight", f"{prime}.mlp.router.gate.weight"),
        Rename(f"{checkpoint}.mlp.gate.bias", f"{prime}.mlp.router.selection_bias"),
        InterleavedGateUp(
            f"{checkpoint}.mlp.experts.w13_weight",
            f"{prime}.mlp.experts.gate_proj",
            f"{prime}.mlp.experts.up_proj",
        ),
        Rename(f"{checkpoint}.mlp.experts.w2_weight", f"{prime}.mlp.experts.down_proj"),
        Rename(f"{CHECKPOINT_PREFIX}attn_res.mlp_queries.{layer_idx}", f"{prime}.mlp_input.query"),
        # Sliding layers: one input norm and static K/V convolutions.
        Rename(f"{checkpoint}.attn_norm.weight", f"{prime}.input_layernorm.weight"),
        # Full layers: one input norm per stream and K/V convolutions with per-token taps.
        *(Rename(f"{checkpoint}.{s}_input_norm.weight", f"{prime}.{s}_input_layernorm.weight") for s in "qkv"),
    ]
    for stream in "kv":
        conv, prime_conv = f"{attn}.{stream}_sconv", f"{self_attn}.{stream}_conv"
        ops += [
            Rename(f"{conv}.weight", f"{prime_conv}.weight"),
            Rename(f"{conv}.static_w", f"{prime_conv}.static_weight"),
            Rename(f"{conv}.weight_proj", f"{prime_conv}.weight_proj.weight"),
        ]
    # Layer 0 reads the embedding directly, so the checkpoint's attention queries start at layer 1.
    if layer_idx > 0:
        ops.append(
            Rename(f"{CHECKPOINT_PREFIX}attn_res.attn_queries.{layer_idx - 1}", f"{prime}.attention_input.query")
        )
    for controller, gate in MESSAGE_STREAMS:
        source, target = f"{checkpoint}.controllers.{controller}", f"{prime}.message_gates.{gate}"
        ops += [
            Rename(f"{source}.down_weight", f"{target}.down_proj.weight"),
            Rename(f"{source}.gain_weight", f"{target}.message_proj.weight"),
            Rename(f"{source}.forget_weight", f"{target}.forget_proj.weight"),
        ]
    return ops


def conversion_chain(config) -> list[ConvOp]:
    text_config = getattr(config, "text_config", config)
    ops: list[ConvOp] = [KeepLanguageModel()]
    for layer_idx in range(text_config.num_hidden_layers):
        ops.extend(_layer_ops(layer_idx))
    ops += [
        Rename(f"{CHECKPOINT_PREFIX}embed.weight", "model.embed_tokens.weight"),
        Rename(f"{CHECKPOINT_PREFIX}embed_norm.weight", "model.embed_norm.weight"),
        Rename(f"{CHECKPOINT_PREFIX}attn_res.output_query", "model.norm.query"),
        Rename(f"{CHECKPOINT_PREFIX}norm.weight", "model.norm.weight"),
        MessageQueries(
            f"{CHECKPOINT_PREFIX}middle_cycle_cross_pass_queries",
            f"{CHECKPOINT_PREFIX}middle_cycle_cross_pass_kv_queries",
            "model.message_queries",
            has_core=len(text_config.loop_core) > 0,
        ),
        PaddedUnembedding(
            f"{CHECKPOINT_PREFIX}unembed.weight",
            "lm_head.weight",
            text_config.vocab_size,
            text_config.num_logits,
        ),
    ]
    return ops


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.startswith(CHECKPOINT_PREFIX) for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    # Weight transports ask this of one decoder layer at a time and of everything outside the layers as its own
    # group. That last group holds no MoE key, so the embedding norm marks it.
    return any(name.endswith("mlp.router.gate.weight") or name == "model.embed_norm.weight" for name in state_dict)


__all__ = ["conversion_chain", "is_hf_state_dict", "is_prime_state_dict"]
