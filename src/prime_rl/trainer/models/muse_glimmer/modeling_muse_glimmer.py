import torch
from torch import Tensor, nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention, head_partitions
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.norms import Norm
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig, apply_rotary_pos_emb
from prime_rl.trainer.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig, MuseGlimmerTextConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class MuseGlimmerRMSNorm(Norm):
    """RMSNorm computed in fp32 and cast back to the input dtype.

    Without `dim` the norm has no weight (QK-norm and the token-embedding norm). Decoder-layer norms
    store `weight` as an offset from one (`weight_offset=1`); the final norm stores it directly.
    """

    def __init__(
        self, dim: int | None = None, eps: float = 1e-6, weight_offset: float = 0.0, output_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight_offset = weight_offset
        self.output_scale = output_scale
        self.weight = nn.Parameter(torch.zeros(dim) if weight_offset else torch.ones(dim)) if dim else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        out = hidden_states.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * (self.weight.float() + self.weight_offset)
        if self.output_scale != 1.0:
            out = out * self.output_scale
        return out.type_as(hidden_states)


class MuseGlimmerAttention(FlashAttention):
    """Attention with weightless fp32 QK-norm before RoPE, a query multiplier, and a per-channel sigmoid
    output gate read from the attention input. Sliding-window layers use RoPE; full-attention layers use none."""

    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int) -> None:
        flash_attn_version = {
            "flash_attention_2": 2,
            "flash_attention_3": 3,
            "flash_attention_4": 4,
        }[config._attn_implementation]
        super().__init__(
            AttentionConfig(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                is_causal=True,
                attention_bias=config.attention_bias,
                output_bias=config.attention_bias,
                use_qk_norm=False,
                rms_norm_eps=config.rms_norm_eps,
            ),
            flash_attn_version=flash_attn_version,
        )
        self.use_rope = config.layer_types[layer_idx] == "sliding_attention"
        if self.use_rope:
            self.sliding_window = config.sliding_window
        self.qk_norm = MuseGlimmerRMSNorm(eps=config.rms_norm_eps)
        self.query_scale = config.query_scale
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)

    def muon_head_partitions(
        self, heads_per_block: int, split_values: bool
    ) -> dict[nn.Parameter, tuple[int, ...] | None]:
        partitions = super().muon_head_partitions(heads_per_block, split_values)
        partitions[self.gate_proj.weight] = head_partitions(self.num_heads, (self.head_dim,), heads_per_block)
        return partitions

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> tuple[Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states = self.qk_norm(query_states.view(hidden_shape)) * self.query_scale
        key_states = self.qk_norm(key_states.view(hidden_shape))
        value_states = value_states.view(hidden_shape)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states.transpose(1, 2), key_states.transpose(1, 2), cos, sin
            )
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        attn_output = self._compute_attention(
            query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen
        ).reshape(*input_shape, -1)
        attn_output = torch.sigmoid(self.gate_proj(hidden_states)) * attn_output
        return self.o_proj(attn_output), None


class MuseGlimmerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int) -> None:
        super().__init__()
        if config.hidden_activation != "silu":
            raise ValueError(f"MuseGlimmer uses a silu MLP, got {config.hidden_activation!r}")
        self.self_attn = MuseGlimmerAttention(config, layer_idx)
        self.mlp = FeedForward(dim=config.hidden_size, hidden_dim=config.intermediate_size, activation="silu")
        self.input_layernorm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps, weight_offset=1.0)
        self.post_attention_layernorm = MuseGlimmerRMSNorm(config.hidden_size, config.post_norm_eps, weight_offset=1.0)
        self.pre_feedforward_layernorm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps, weight_offset=1.0)
        self.post_feedforward_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, config.post_norm_eps, weight_offset=1.0
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class MuseGlimmerPreTrainedModel(PreTrainedModelPrimeRL):
    config_class = MuseGlimmerConfig
    base_model_prefix = "model"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Text weights use the checkpoint's names, so there is no separate PrimeRL format.
        return False


class MuseGlimmerTextModel(MuseGlimmerPreTrainedModel):
    config_class = MuseGlimmerTextConfig

    def __init__(self, config: MuseGlimmerTextConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Token embeddings are RMS-normalized (no weight) instead of scaled by sqrt(hidden_size).
        self.embed_norm = MuseGlimmerRMSNorm(eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            MuseGlimmerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        # The logits are scaled by `output_multiplier` before the softcap. The LM head is linear, so the
        # scale is folded into the final norm and the shared LM head only applies the softcap.
        self.norm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps, output_scale=config.output_multiplier)
        self.rotary_emb = RotaryEmbedding(
            RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings,
                rope_type=config.rope_parameters.get("rope_type", "default"),
                model_config=config,
            )
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class MuseGlimmerModel(nn.Module):
    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.language_model = MuseGlimmerTextModel(config.text_config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(embeddings)

    def forward(self, *args, **kwargs) -> BaseModelOutputWithPast:
        return self.language_model(*args, **kwargs)


class MuseGlimmerForCausalLM(MuseGlimmerPreTrainedModel):
    """Text-only MuseGlimmer. Built from the composite checkpoint config, its weights keep the
    checkpoint's `model.language_model.*` names; the vision tower is not loaded."""

    def __init__(self, config: MuseGlimmerConfig | MuseGlimmerTextConfig) -> None:
        super().__init__(config)
        text_config = getattr(config, "text_config", config)
        self.model = MuseGlimmerModel(config) if hasattr(config, "text_config") else MuseGlimmerTextModel(config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = VanillaOutputLinear(
            text_config.hidden_size, text_config.vocab_size, softcap=text_config.final_logit_softcapping
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | Tensor = 0,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        return self.lm_head(
            outputs.last_hidden_state[:, slice_indices],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
            sampling_mask=sampling_mask[:, slice_indices] if sampling_mask is not None else None,
        )

    def init_buffers_post_meta(self) -> None:
        rotary_emb = getattr(self.model, "language_model", self.model).rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)


__all__ = [
    "MuseGlimmerDecoderLayer",
    "MuseGlimmerForCausalLM",
    "MuseGlimmerModel",
    "MuseGlimmerPreTrainedModel",
    "MuseGlimmerTextModel",
]
