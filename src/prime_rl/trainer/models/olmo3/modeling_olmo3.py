from copy import copy
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig, SDPAAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.utils.sequence_packing import infer_cu_seqlens_from_position_ids


def _prime_attn_impl(config: Olmo3Config) -> str:
    attn_impl = config._attn_implementation
    return "fa4" if attn_impl == "flash_attention_4" else attn_impl


def _rope_type(rope_config: dict) -> str:
    return rope_config.get("rope_type", rope_config.get("type", "default"))


def _flat_rope_config(config: Olmo3Config) -> dict:
    for attr in ("rope_parameters", "rope_scaling"):
        rope_config = getattr(config, attr, None)
        if isinstance(rope_config, dict) and _rope_type(rope_config) is not None:
            return rope_config
    return {"rope_type": "default", "rope_theta": getattr(config, "rope_theta", 10000.0)}


def _default_rope_config(config: Olmo3Config, rope_config: dict | None = None) -> dict:
    rope_config = rope_config or _flat_rope_config(config)
    rope_theta = rope_config.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    return {"rope_type": "default", "rope_theta": rope_theta}


def _layer_rope_config(config: Olmo3Config, layer_type: str) -> dict:
    for attr in ("rope_parameters", "rope_scaling"):
        rope_config = getattr(config, attr, None)
        if isinstance(rope_config, dict) and isinstance(rope_config.get(layer_type), dict):
            layer_rope_config = rope_config[layer_type]
            if layer_type == "sliding_attention" and _rope_type(layer_rope_config) != "default":
                raise ValueError("RoPE scaling is not applied to sliding window attention layers in OLMo3")
            return layer_rope_config

    rope_config = _flat_rope_config(config)
    if layer_type == "sliding_attention":
        return _default_rope_config(config, rope_config)
    return rope_config


def _rotary_embedding_config(config: Olmo3Config, layer_type: str) -> RotaryEmbeddingConfig:
    rope_config = _layer_rope_config(config, layer_type)
    rotary_model_config = copy(config)
    rotary_model_config.rope_parameters = rope_config
    rotary_model_config.rope_scaling = rope_config
    return RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type=_rope_type(rope_config),
        model_config=rotary_model_config,
    )


def _sliding_window_mask(q_len: int, k_len: int, window: int, device: torch.device) -> torch.Tensor:
    query_positions = torch.arange(q_len, device=device).unsqueeze(1)
    key_positions = torch.arange(k_len, device=device).unsqueeze(0)
    allowed = (key_positions <= query_positions) & (key_positions >= query_positions - window + 1)
    mask = torch.full((q_len, k_len), torch.finfo(torch.float32).min, device=device)
    return mask.masked_fill(allowed, 0.0)


class Olmo3SDPAAttention(SDPAAttention):
    def _attention_core_single_sequence(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is None:
            out = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
        else:
            mask = _sliding_window_mask(
                query_states.shape[-2], key_states.shape[-2], sliding_window, query_states.device
            )
            out = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return self._attention_core_single_sequence(query_states, key_states, value_states)

        if query_states.shape[0] != 1:
            raise ValueError("Packed SDPA reference path expects batch size 1")

        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            outputs.append(
                self._attention_core_single_sequence(
                    query_states[:, :, start:end, :],
                    key_states[:, :, start:end, :],
                    value_states[:, :, start:end, :],
                )
            )
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_output = self.output_proj(attn_output)
        return attn_output, None


class Olmo3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            use_qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            qk_norm_type="per_layer",
        )
        attn_impl = _prime_attn_impl(config)
        if attn_impl == "sdpa":
            self.self_attn = Olmo3SDPAAttention(attn_config)
        else:
            self.self_attn = ATTN_IMPL2CLASS[attn_impl](attn_config)

        self.self_attn.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )
        self.mlp = MLP(
            MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_act,
                bias=False,
            )
        )
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_feedforward_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Olmo3PreTrainedModel(PreTrainedModelPrimeRL):
    config: Olmo3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Olmo3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Olmo3DecoderLayer,
    }

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


class Olmo3Model(Olmo3PreTrainedModel):
    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        self.rotary_embs = nn.ModuleDict(
            {
                layer_type: RotaryEmbedding(_rotary_embedding_config(config, layer_type))
                for layer_type in dict.fromkeys(config.layer_types)
            }
        )
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if _prime_attn_impl(self.config) in ("flash_attention_2", "flash_attention_3", "fa4"):
            if cu_seqlens is None or max_seqlen is None:
                cu_seqlens, max_seqlen = infer_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)

        hidden_states = inputs_embeds
        position_embeddings_by_type = {
            layer_type: rotary_emb(hidden_states, position_ids) for layer_type, rotary_emb in self.rotary_embs.items()
        }

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings_by_type[self.config.layer_types[layer_idx]],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Olmo3ForCausalLM(Olmo3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Olmo3Config):
        super().__init__(config)

        self.model = Olmo3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
            If not provided, the wrapped LM head returns logits only.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        cu_seqlens (`torch.LongTensor`, *optional*):
            Explicit packed-sequence cumulative lengths for FlashAttention varlen kernels.
        max_seqlen (`int`, *optional*):
            Maximum packed subsequence length corresponding to `cu_seqlens`.
        """
        assert use_cache is None or use_cache is False, "use_cache is not supported for custom olmo3"
        assert past_key_values is None, "past_key_values is not supported for custom olmo3"

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["Olmo3ForCausalLM", "Olmo3Model", "Olmo3PreTrainedModel"]
