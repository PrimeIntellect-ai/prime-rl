"""HF<->prime weight conversion for Inkling 2 (text tower).

The release checkpoint is a Megatron-style export: ``model.llm.*`` prefixes,
``attn.*`` projection names, per-layer ``attn_res`` queries outside the layer
prefix, and routed experts stored as two fused tensors with **interleaved
gate/up rows** (``w13_weight[E, 2I, H]``: row 2i is gate output i, row 2i+1 is
up output i).

PrimeRL stores the experts de-interleaved as ``experts.gate_proj`` /
``experts.up_proj`` (``[E, I, H]``) because the shared ``MoE`` layer consumes
them separately; everything else keeps the checkpoint's names modulo the
``model.llm.`` -> ``model.`` prefix swap. The vision tower and the DSpark/MTP
head are dropped (text-only training).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prime_rl.trainer.models.conversion_ops import (
    ConvOp,
    Drop,
    MapValue,
    PrefixRename,
    Rename,
    StateDict,
)


@dataclass
class InterleavedGateUp(ConvOp):
    """Fused interleaved ``w13`` <-> separate prime ``gate_proj`` / ``up_proj``.

    ``w13[E, 2I, H]`` holds gate output ``i`` at row ``2i`` and up output ``i`` at
    row ``2i+1``; prime stores the two halves as separate stacked parameters.
    """

    hf: str
    gate: str
    up: str
    dim: int = 1

    def hf_to_prime(self, sd: StateDict) -> None:
        if self.hf not in sd:
            return
        w13 = sd.pop(self.hf)
        gate = w13[self._sl(0)].contiguous()
        up = w13[self._sl(1)].contiguous()
        sd[self.gate] = gate
        sd[self.up] = up

    def prime_to_hf(self, sd: StateDict) -> None:
        if self.gate not in sd or self.up not in sd:
            return
        gate = sd.pop(self.gate)
        up = sd.pop(self.up)
        shape = list(gate.shape)
        shape[self.dim] *= 2
        w13 = gate.new_empty(shape)
        w13[self._sl(0)] = gate
        w13[self._sl(1)] = up
        sd[self.hf] = w13

    def _sl(self, offset: int):
        return (slice(None),) * self.dim + (slice(offset, None, 2),)


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = [
        # Text backbone prefix. `model.visual.*` (vision tower) and `mtp.*` have no
        # prime counterpart and are dropped.
        PrefixRename("model.llm.", "model."),
        PrefixRename("model.visual.", "__dropped_visual__."),
        Drop("__dropped_visual__.", is_prefix=True),
        Drop("mtp.", is_prefix=True),
    ]

    ops += [
        Rename("model.embed.weight", "model.embed_tokens.weight"),
        Rename("model.embed_norm.weight", "model.embed_norm.weight"),
        Rename("model.norm.weight", "model.norm.weight"),
        Rename("model.unembed.weight", "lm_head.weight"),
    ]

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        for hf, prime in (
            ("attn.wq_du.weight", "self_attn.q_proj.weight"),
            ("attn.wk_dv.weight", "self_attn.k_proj.weight"),
            ("attn.wv_dv.weight", "self_attn.v_proj.weight"),
            ("attn.wo_ud.weight", "self_attn.o_proj.weight"),
            ("attn.wr_du.weight", "self_attn.r_proj.weight"),
            ("attn.gate_attn.weight", "self_attn.g_proj.weight"),
            ("attn.attn_sink_proj.weight", "self_attn.sink_proj.weight"),
            ("attn.q_l2_clip.scale", "self_attn.q_l2_clip.scale"),
            ("attn.rel_l2_clip.scale", "self_attn.rel_l2_clip.scale"),
            ("attn.rel_logits_proj.proj", "self_attn.rel_logits_proj"),
            ("attn.k_sconv.weight", "self_attn.k_sconv.conv1d.weight"),
            ("attn.k_sconv.static_w", "self_attn.k_sconv.static_weight"),
            ("attn.k_sconv.weight_proj", "self_attn.k_sconv.weight_proj.weight"),
            ("attn.v_sconv.weight", "self_attn.v_sconv.conv1d.weight"),
            ("attn.v_sconv.static_w", "self_attn.v_sconv.static_weight"),
            ("attn.v_sconv.weight_proj", "self_attn.v_sconv.weight_proj.weight"),
            ("attn_norm.weight", "input_layernorm.weight"),
            ("q_input_norm.weight", "q_input_layernorm.weight"),
            ("k_input_norm.weight", "k_input_layernorm.weight"),
            ("v_input_norm.weight", "v_input_layernorm.weight"),
            ("mlp_norm.weight", "post_attention_layernorm.weight"),
            ("attn_sconv.weight", "post_attention_convolution.conv1d.weight"),
            ("mlp_sconv.weight", "post_mlp_convolution.conv1d.weight"),
            ("mlp.gate.weight", "mlp.router.gate.weight"),
            ("mlp.gate.bias", "mlp.router.selection_bias"),
        ):
            ops.append(Rename(f"{p}.{hf}", f"{p}.{prime}"))

        ops.append(
            InterleavedGateUp(
                hf=f"{p}.mlp.experts.w13_weight",
                gate=f"{p}.mlp.experts.gate_proj",
                up=f"{p}.mlp.experts.up_proj",
            )
        )
        ops.append(Rename(f"{p}.mlp.experts.w2_weight", f"{p}.mlp.experts.down_proj"))

    # `model.attn_res.*` keeps its checkpoint name via the prefix rename. The release stores
    # these tensors in FP32; training keeps them in the model dtype (FSDP requires one storage
    # dtype) and the export upcasts them back.
    for name in (
        *(f"model.attn_res.attn_queries.{i}" for i in range(config.num_hidden_layers - 1)),
        *(f"model.attn_res.mlp_queries.{i}" for i in range(config.num_hidden_layers)),
        "model.attn_res.output_query",
    ):
        ops.append(
            MapValue(
                prime=name,
                forward=lambda t: t.to(torch.bfloat16),
                backward=lambda t: t.to(torch.float32),
            )
        )
    return ops
