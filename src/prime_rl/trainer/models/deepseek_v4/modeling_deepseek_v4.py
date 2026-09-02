"""DeepSeek V4 decoder stack, model and causal-LM head.

This is where the pieces built in `attention.py`, `moe.py`, `hyperconnections.py` and
`rotary.py` come together. The one structural surprise is the residual: it is not a single
stream but `hc_mult` parallel ones, carried as `(batch, seq, hc_mult, hidden)` from the
embedding all the way to `hc_head`, which collapses them back before the final norm.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast

from prime_rl.trainer.models.base import CPSupport, PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.converting_deepseek_v4 import conversion_chain
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_state_dict_
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection, DeepseekV4HyperHead
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    """One hyper-connected block: mHC, attention, mHC, MoE.

    Both sublayers read the single sequence their `DeepseekV4HyperConnection` collapsed the
    streams into, and write back through two gates. `post` broadcasts the sublayer output
    over the streams; `comb` remixes the streams among themselves and is consumed
    transposed, i.e. summing over the *source* stream axis. Sinkhorn leaves `comb` doubly
    stochastic but not symmetric, so that direction is not a free choice.

    Both gates come out of the hyper-connection in fp32 and are cast back to the residual's
    dtype before mixing, so the residual keeps the dtype it entered with.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4MoE(config, layer_idx)
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
        *,
        packed: PackedContext,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), packed=packed)
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(
            self.post_attention_layernorm(collapsed), input_ids=input_ids, routed_experts=routed_experts
        )
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


# Mirrors HF's `_keep_in_fp32_modules_strict`, with `e_score_correction_bias` renamed to
# the `selection_bias` prime-rl's router keeps it under. The bare `norm` entry subsumes the
# named norms; both are kept so the list stays a one-to-one image of HF's.
_KEEP_IN_FP32_MODULES = (
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


class DeepseekV4PreTrainedModel(PreTrainedModelPrimeRL):
    config: DeepseekV4Config
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # V4 attention is eager-only, as in HF: FlashAttention caps the head dim at 256 while
    # V4 uses 512, SDPA carries no per-head sink logit, and FlexAttention's BlockMask
    # cannot grow to cover the compressed entries the block concatenates onto the KV axis.
    # `DeepseekV4Attention` reads no dispatch table, so `config._attn_implementation` is
    # inert here; these flags only keep transformers from advertising a backend we lack.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": DeepseekV4DecoderLayer}
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]

    @classmethod
    def cp_support(cls, config) -> CPSupport:
        return CPSupport(
            frozenset(),
            "its sliding window is a dense local mask built from post-shard document boundaries, "
            "which CP's global (pre-shard) boundaries cannot address",
        )

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        init_std = self.config.initializer_range
        if isinstance(module, (DeepseekV4Attention, DeepseekV4HyperConnection, DeepseekV4HyperHead)):
            module.init_weights(init_std)
        elif isinstance(module, DeepseekV4MoE):
            module.init_weights(init_std, module.tokens_per_expert.device)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            module.init_buffers_post_meta()

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return any(module_name in name for module_name in _KEEP_IN_FP32_MODULES)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Both the real on-disk checkpoint naming (`ffn.*`) and the `transformers`-native
        # in-memory naming (`mlp.*`) count as "HF, not yet converted" -- see
        # `converting_deepseek_v4.py`'s module docstring for why the two differ.
        return any(
            name.endswith("mlp.gate.weight")
            or name.endswith("ffn.gate.weight")
            or "mlp.shared_experts." in name
            or "ffn.shared_experts." in name
            for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(name.endswith("mlp.router.gate.weight") or "mlp.shared_expert." in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a HuggingFace state dict to PrimeRL format in-place.

        Dequantizes the real checkpoint's fp8/MXFP4 weights to `bfloat16` first, on the raw
        on-disk key names: a no-op for any snapshot that doesn't carry `.scale` siblings
        (e.g. the plain-`bfloat16` mini test checkpoint).
        """
        dequantize_state_dict_(state_dict)
        return super().convert_to_prime(state_dict)

    def init_buffers_post_meta(self) -> None:
        # `to_empty()` leaves every buffer uninitialized and this runs before `dcp_load`, so a
        # buffer is restored either here or by the checkpoint. Rebuilt here are the ones no
        # checkpoint carries: the rotary tables (non-persistent, and there is one rotary per
        # compressor and per indexer on top of the model-level one, so every instance has to be
        # walked) and `tokens_per_expert`. The router's persistent buffers, `selection_bias` and a
        # hash layer's `tid2eid`, are in the checkpoint that `dcp_load` applies next, so they are
        # left to it.
        for module in self.modules():
            if isinstance(module, DeepseekV4RotaryEmbedding):
                module.init_buffers_post_meta()
            elif isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> MoeModelOutputWithPast:
        """
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token ids. Threaded down to every decoder layer, not just the embedding: the
            bootstrap layers route on `tid2eid[input_ids]` and cannot run without them.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Each token's position within its own document. Only checked, never consulted: the
            positions every rotation and every causal threshold reads are derived from
            `seq_lens`, so the two cannot disagree.

            NOTE: omitting it is not the same as having no positions. `inject_prime_lm_head`
            rebinds this model's forward and substitutes a 1-based `arange(1, N + 1)`, which is
            wrong for V4 twice over, since entries are numbered and rotated from 0: it shifts
            every query-entry RoPE distance by one and admits an entry whose last source token
            is one past the query. `check_position_ids` rejects it, so a caller that passes
            nothing gets a `ValueError` rather than a working default.

            TODO: make that substitution 0-based, matching HF's convention and every other
            model's own fallback, after which an omitted `position_ids` would just be accepted.
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract). Clips the
            sliding window at document boundaries and lays out the compressors' entries per
            document, so a packed row gives every document what running it alone would.
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries. Rejected: the
            window mask is dense and local, so global boundaries cannot address it.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if seq_lens_are_pre_shard:
            raise NotImplementedError(
                "DeepSeek V4 does not support context parallelism: pre-shard document boundaries "
                "do not address the local sliding-window mask."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Every layer type attends over the same local window; the compressed variants add their
        # own out-of-window entries and the per-query bias that gates them. One layout per distinct
        # compress rate, shared by every layer that pools at that rate; sliding layers own no
        # compressor and contribute no rate.
        packed = PackedContext.build(
            rotary_emb=self.rotary_emb,
            seq_lens=seq_lens,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        if position_ids is not None:
            packed.check_position_ids(position_ids)

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer_idx, decoder_layer in enumerate(self.layers):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                input_ids=input_ids,
                routed_experts=routed_experts_layer,
                packed=packed,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token
            logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are given.
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row. Required rather than defaulting to one
            document, so a caller that forgets it fails instead of silently getting a window
            that spans neighbouring rollouts.
        """
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )


__all__ = [
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4PreTrainedModel",
]
