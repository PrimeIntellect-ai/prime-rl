"""DeepSeek V4 decoder stack, model and causal-LM head.

This is where the pieces built in `attention.py`, `moe.py`, `hyperconnections.py` and
`rotary.py` come together. The one structural surprise is the residual: it is not a single
stream but `hc_mult` parallel ones, carried as `mhc_states` of shape
`(batch, seq, hc_mult, hidden)` from the embedding all the way to `hc_head`, which collapses
them back before the final norm.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from prime_rl.trainer.models.base import CPSupport, PrimeModel
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.converting_deepseek_v4 import conversion_chain
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_state_dict_
from prime_rl.trainer.models.deepseek_v4.hyperconnections import (
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.cp import CPContext


class DeepseekV4DecoderLayer(nn.Module):
    """One hyper-connected block: mHC, attention, mHC, MoE.

    Both sublayers read the single sequence their `DeepseekV4HyperConnection` collapsed the
    streams into, and write back through two gates. `post` broadcasts the sublayer output
    over the streams; `comb` remixes the streams among themselves and is consumed
    transposed, i.e. summing over the *source* stream axis. Sinkhorn leaves `comb` doubly
    stochastic but not symmetric, so that direction is not a free choice.

    Both gates come out of the hyper-connection in fp32 and are cast back to the residual's
    dtype before mixing, so the residual keeps the dtype it entered with.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int, rotary_emb: DeepseekV4RotaryEmbedding):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx, rotary_emb)
        self.mlp = DeepseekV4MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        mhc_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
        *,
        packed: PackedContext,
    ) -> torch.Tensor:
        post, comb, collapsed = self.attn_hc(mhc_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), packed=packed)
        mhc_states = self.attn_hc.update_states(post, comb, attn_output, mhc_states)

        post, comb, collapsed = self.ffn_hc(mhc_states)
        mlp_output = self.mlp(
            self.post_attention_layernorm(collapsed), input_ids=input_ids, routed_experts=routed_experts
        )
        return self.ffn_hc.update_states(post, comb, mlp_output, mhc_states)


# Mirrors HF's `_keep_in_fp32_modules_strict`, with `e_score_correction_bias` renamed to
# the `selection_bias` prime-rl's router keeps it under. The bare `norm` entry subsumes the
# named norms; both are kept so the list stays a one-to-one image of HF's.
KEEP_IN_FP32_MODULES = (
    "attn_hc",
    "ffn_hc",
    "hc_head",
    "sinks",
    "position_bias",
    "selection_bias",
    "q_a_norm",
    "kv_norm",
    "input_layernorm",
    "post_attention_layernorm",
    "norm",
)


class DeepseekV4Model(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Shared by every attention layer instead of passing RoPE tensors through forward, which FSDP casts to bf16.
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(config, layer_idx, self.rotary_emb)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.hc_head = DeepseekV4HyperHead(config)
        self.cp_context = CPContext()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        """
        input_ids (`(batch_size, sequence_length)`):
            Token ids. Threaded down to every decoder layer, not just the embedding: the
            bootstrap layers route on `tid2eid[input_ids]` and cannot run without them.
        position_ids (`(batch_size, sequence_length)`):
            Each token's position within its own document. Only checked, never consulted: the
            positions every rotation and every causal threshold reads are derived from
            `seq_lens`, so the two cannot disagree.
        seq_lens (`(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract). Clips the
            sliding window at document boundaries and lays out the compressors' entries per
            document, so a packed row gives every document what running it alone would.
        seq_lens_are_pre_shard:
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries. V4 shards the
            queries alone and keeps every key, entry and index value global, so it needs the
            whole row's boundaries: this must be set exactly when context parallelism is on.
        routed_experts (`(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        cp_rank, cp_world_size = self.cp_context.cp_rank, self.cp_context.cp_world_size
        assert seq_lens_are_pre_shard == (cp_world_size > 1), (
            f"seq_lens_are_pre_shard={seq_lens_are_pre_shard} disagrees with cp_world_size={cp_world_size}"
        )

        inputs_embeds = self.embed_tokens(input_ids)

        # `seq_lens` describes the whole row and `inputs_embeds` carries this rank's shard of it.
        total_tokens = int(seq_lens.sum())
        assert total_tokens == inputs_embeds.shape[1] * cp_world_size, (
            f"seq_lens covers {total_tokens} tokens, but {cp_world_size} CP rank(s) holding "
            f"{inputs_embeds.shape[1]} tokens each"
        )

        # Every layer type attends over the same local window; the compressed variants add their
        # own out-of-window entries and the per-query bias that gates them. One layout per distinct
        # compress rate, shared by every layer that pools at that rate; sliding layers own no
        # compressor and contribute no rate.
        packed = PackedContext.build(
            rotary_emb=self.rotary_emb,
            seq_lens=seq_lens,
            device=inputs_embeds.device,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        )
        packed.check_position_ids(position_ids)

        mhc_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer_idx, decoder_layer in enumerate(self.layers):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            mhc_states = decoder_layer(
                mhc_states,
                input_ids=input_ids,
                routed_experts=routed_experts_layer,
                packed=packed,
            )

        return self.norm(self.hc_head(mhc_states))


class DeepseekV4ForCausalLM(PrimeModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def cp_support(cls, config: DeepseekV4Config) -> CPSupport:
        return CPSupport(
            frozenset({"ring"}),
            "currently only supporting the minimal ring strategy where all keys, or tensors "
            "required to form the keys are all-gathered in the CP region.",
        )

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return any(module_name in name for module_name in KEEP_IN_FP32_MODULES)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # The published checkpoint's `ffn.*` naming, the only unconverted naming that exists --
        # see `converting_deepseek_v4.py`'s module docstring.
        return any(name.endswith("ffn.gate.weight") or "ffn.shared_experts." in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # The NCCL transport broadcasts one decoder layer at a time, plus everything outside
        # `model.layers.` as its own group, and asks this of each group separately. That last group
        # (embeddings, final norm, `hc_head`, `lm_head`) holds no MoE key, so `model.hc_head.` is
        # what stops it claiming to be unconverted and skipping the chain.
        return any(
            name.endswith("mlp.router.gate.weight") or "mlp.shared_expert." in name or "model.hc_head." in name
            for name in state_dict
        )

    @classmethod
    def conversion_chain(cls, config: DeepseekV4Config):
        return conversion_chain(config)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a HuggingFace state dict to PrimeRL format in-place.

        Dequantizes the real checkpoint's fp8/MXFP4 weights to `bfloat16` first, on the raw
        on-disk key names: a no-op for any snapshot that doesn't carry `.scale` siblings
        (e.g. the plain-`bfloat16` mini test checkpoint).
        """
        dequantize_state_dict_(state_dict)
        return super().convert_to_prime(state_dict)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> PrimeLmOutput:
        hidden_states = self.model(input_ids, position_ids, seq_lens, seq_lens_are_pre_shard, routed_experts)
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)

    def init_buffers_post_meta(self) -> None:
        # `to_empty()` leaves every buffer uninitialized and this runs before `dcp_load`, so a
        # buffer is restored either here or by the checkpoint. Rebuilt here are the ones no
        # checkpoint carries: the rotary tables (non-persistent, and the one rotary every attention
        # layer shares) and `tokens_per_expert`. The router's persistent buffers, `selection_bias` and a
        # hash layer's `tid2eid`, are in the checkpoint that `dcp_load` applies next, so they are
        # left to it.
        for module in self.modules():
            if isinstance(module, DeepseekV4RotaryEmbedding):
                module.init_buffers_post_meta()
            elif isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()


__all__ = [
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
]
