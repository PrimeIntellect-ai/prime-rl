# coding=utf-8
# Copyright 2026 Xiaomi MiMo Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import maybe_autocast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.trainer.models.mimo_v2.configuration_mimo_v2 import MiMoV2Config
from prime_rl.trainer.models.mimo_v2.converting_mimo_v2 import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt,
    convert_tt_layer_to_hf,
    convert_tt_to_hf,
)
from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids


def _windowed_sink_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    sliding_window: int,
    sink_bias: torch.Tensor,
    scaling: float,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Exact sliding-window attention with a per-head sink bias over packed sequences.

    Flash attention cannot express the attention sink (it changes the softmax denominator) and
    its returned LSE is not differentiable, so SWA layers use this chunked implementation: each
    query block only materializes logits against its (window + chunk)-sized key slice, keeping
    memory at O(total * window) instead of O(total^2).

    Args:
        query_states: ``(total, num_heads, head_dim)`` after GQA expansion.
        key_states: ``(total, num_heads, head_dim)``.
        value_states: ``(total, num_heads, v_head_dim)``.
        cu_seqlens: Cumulative sequence lengths delimiting packed samples.
        sliding_window: Window size; key ``j`` is visible to query ``i`` iff ``i - window < j <= i``.
        sink_bias: ``(num_heads,)`` bias appended as an extra softmax column.
        scaling: Attention logit scale.
    """
    total, num_heads, _ = query_states.shape
    positions = torch.arange(total, device=query_states.device)
    sample_ids = torch.searchsorted(cu_seqlens, positions, right=True)
    sink = sink_bias.float().view(num_heads, 1, 1)

    outputs = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        key_start = max(0, start - (sliding_window - 1))

        logits = (
            torch.einsum("qhd,khd->hqk", query_states[start:end].float(), key_states[key_start:end].float()) * scaling
        )

        query_pos = positions[start:end, None]
        key_pos = positions[None, key_start:end]
        visible = (key_pos <= query_pos) & (key_pos > query_pos - sliding_window)
        visible &= sample_ids[start:end, None] == sample_ids[None, key_start:end]
        logits = logits.masked_fill(~visible.unsqueeze(0), float("-inf"))

        logits = torch.cat([logits, sink.expand(num_heads, end - start, 1)], dim=-1)
        probs = logits.softmax(dim=-1)[..., :-1]
        outputs.append(torch.einsum("hqk,khd->qhd", probs, value_states[key_start:end].float()))

    return torch.cat(outputs).to(query_states.dtype)


class MiMoV2AttentionBase(nn.Module):
    """Shared projection / RoPE / sink plumbing for the MiMo-V2 attention variants.

    Mirrors the hub checkpoint layout: a fused ``qkv_proj``, asymmetric query/key vs value head
    dimensions, partial RoPE on the leading dims, an optional value scale, and an optional frozen
    per-head attention sink bias on sliding-window layers.
    """

    def __init__(self, config: MiMoV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = config.hybrid_layer_pattern[layer_idx] == 1

        if self.is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.num_attention_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window if self.is_swa else None
        self.v_scale = config.attention_value_scale
        self.q_size = self.num_attention_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim

        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + self.k_size + self.v_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_attention_heads * self.v_head_dim, config.hidden_size, bias=False)

        use_sink = config.add_swa_attention_sink_bias if self.is_swa else config.add_full_attention_sink_bias
        if use_sink:
            # Frozen in the reference implementation as well
            self.attention_sink_bias = nn.Parameter(torch.zeros(self.num_attention_heads), requires_grad=False)
        else:
            self.attention_sink_bias = None

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to q/k/v in ``(batch, seq, heads, dim)`` layout with RoPE applied."""
        input_shape = hidden_states.shape[:-1]

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.split([self.q_size, self.k_size, self.v_size], dim=-1)
        query_states = query_states.view(*input_shape, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(*input_shape, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(*input_shape, self.num_key_value_heads, self.v_head_dim)

        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        # apply_rotary_pos_emb rotates only the leading cos.shape[-1] dims (partial RoPE)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        return query_states, key_states, value_states


class MiMoV2FlashAttention(MiMoV2AttentionBase):
    def __init__(self, config: MiMoV2Config, layer_idx: int, flash_attn_version: int = 2):
        super().__init__(config, layer_idx)
        self._flash_attn_version = flash_attn_version
        self.func = FlashAttention._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, max_seqlen):
        """Run flash attention with the value padded up to the query/key head dim.

        Flash attention requires q/k/v to share a head dim; padding V with zero columns yields
        an output whose leading ``v_head_dim`` columns match exactly.
        """
        if self.v_head_dim != self.head_dim:
            v = F.pad(v, (0, self.head_dim - self.v_head_dim))
        kwargs: dict = {"causal": True}
        if self.sliding_window is not None:
            kwargs["window_size"] = (self.sliding_window - 1, 0)
        if self._flash_attn_version == 4:
            kwargs["cu_seqlens_q"] = cu_seqlens
            kwargs["cu_seqlens_k"] = cu_seqlens
            out = self._flash_attn_call(q, k, v, **kwargs)
        else:
            out = self._flash_attn_call(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out[..., : self.v_head_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)

        if self.attention_sink_bias is not None:
            key_expanded = key_states[0].repeat_interleave(self.num_key_value_groups, dim=1)
            value_expanded = value_states[0].repeat_interleave(self.num_key_value_groups, dim=1)
            out = _windowed_sink_attention(
                query_states[0],
                key_expanded,
                value_expanded,
                cu_seqlens,
                self.sliding_window,
                self.attention_sink_bias,
                self.scaling,
            )
        else:
            out = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)

        attn_output = out.reshape(1, out.shape[0], -1)
        return self.o_proj(attn_output), None


class MiMoV2SDPAAttention(MiMoV2AttentionBase):
    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        # (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2).transpose(1, 2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2).transpose(1, 2)
        seq_len = query_states.shape[2]
        positions = torch.arange(seq_len, device=query_states.device)
        visible = positions[None, :] <= positions[:, None]
        if self.sliding_window is not None:
            visible &= positions[None, :] > positions[:, None] - self.sliding_window

        if self.attention_sink_bias is not None:
            logits = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) * self.scaling
            logits = logits.masked_fill(~visible, float("-inf"))
            sink = self.attention_sink_bias.float().view(1, -1, 1, 1).expand(logits.shape[0], -1, seq_len, 1)
            logits = torch.cat([logits, sink], dim=-1)
            probs = logits.softmax(dim=-1)[..., :-1]
            out = torch.matmul(probs, value_states.float()).to(query_states.dtype)
        else:
            attn_mask = None if self.sliding_window is None else visible
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                is_causal=attn_mask is None,
            )
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(query_states, key_states, value_states)
        return self.o_proj(attn_output), None


ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(MiMoV2FlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(MiMoV2FlashAttention, flash_attn_version=3),
    "fa4": functools.partial(MiMoV2FlashAttention, flash_attn_version=4),
    "sdpa": MiMoV2SDPAAttention,
    # The SDPA variant computes sink layers eagerly, so it is exact for eager as well
    "eager": MiMoV2SDPAAttention,
}


class MiMoV2RotaryEmbedding(nn.Module):
    """Default RoPE with per-attention-type base and partial rotary dims."""

    inv_freq: torch.Tensor

    def __init__(self, config: MiMoV2Config, is_swa: bool, device=None):
        super().__init__()
        head_dim = config.swa_head_dim if is_swa else config.head_dim
        self.base = config.swa_rope_theta if is_swa else config.rope_theta
        self.rope_dim = int(head_dim * config.partial_rotary_factor)
        if self.rope_dim % 2 != 0:
            raise ValueError(f"MiMoV2 rotary dimension must be even, got {self.rope_dim}")
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)

    def _compute_inv_freq(self, device=None) -> torch.Tensor:
        exponents = torch.arange(0, self.rope_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        return 1.0 / (self.base ** (exponents / self.rope_dim))

    def init_inv_freq_post_meta(self) -> None:
        self.inv_freq.copy_(self._compute_inv_freq(self.inv_freq.device))

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiMoV2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiMoV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_swa = config.hybrid_layer_pattern[layer_idx] == 1
        self.self_attn = ATTN_IMPL2CLASS[config._attn_implementation](config, layer_idx)

        if config.moe_layer_freq[layer_idx]:
            moe_args = MoEArgs(
                num_experts=config.n_routed_experts,
                num_shared_experts=0,
                score_func="sigmoid",
                route_norm=config.norm_topk_prob,
                route_scale=config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                load_balance_coeff=1e-3,
                use_grouped_mm=config.use_grouped_mm,
                fp8=getattr(config, "fp8", False),
            )
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
            mlp_config = MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_act,
                bias=False,
            )
            self.mlp = MLP(mlp_config)

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layernorm_epsilon))
        self.post_attention_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        hidden_states = residual + hidden_states
        return hidden_states


class MiMoV2PreTrainedModel(PreTrainedModelPrimeRL):
    config: MiMoV2Config
    config_class = MiMoV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiMoV2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MiMoV2DecoderLayer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict contains MoE layers in HuggingFace format."""
        return any("mlp.experts.1.up_proj" in name or "mlp.gate.e_score_correction_bias" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict contains MoE layers in PrimeRL training format."""
        return any("mlp.experts.w1" in name for name in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert weights from PrimeRL training format to HuggingFace format in-place."""
        convert_tt_to_hf(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert weights from HuggingFace format to PrimeRL training format in-place."""
        convert_hf_to_tt(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's weights from PrimeRL format to HuggingFace format in-place."""
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's weights from HuggingFace format to PrimeRL format in-place."""
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict


class MiMoV2Model(MiMoV2PreTrainedModel):
    # MTP layers (speculative decoding only) and multimodal towers are not trained
    _keys_to_ignore_on_load_unexpected = [
        r"model\.mtp\..*",
        r"visual\..*",
        r"audio_encoder\..*",
        r"speech_embeddings\..*",
    ]

    def __init__(self, config: MiMoV2Config):
        super().__init__(config)
        if config.scoring_func != "sigmoid":
            raise ValueError(f"MiMoV2 only supports sigmoid routing, got {config.scoring_func}")
        if config.topk_method != "noaux_tc" or config.n_group != 1:
            raise ValueError("MiMoV2 only supports noaux_tc routing with n_group=1")

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiMoV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layernorm_epsilon))
        self.rotary_emb = MiMoV2RotaryEmbedding(config, is_swa=False)
        self.swa_rotary_emb = MiMoV2RotaryEmbedding(config, is_swa=True)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        """
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4"):
            cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            max_seqlen = None
            cu_seqlens = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = self.swa_rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=swa_position_embeddings if decoder_layer.is_swa else position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                routed_experts=routed_experts_layer,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class MiMoV2ForCausalLM(MiMoV2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MiMoV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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
        routed_experts: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices of input tokens in the KV cache. Accepted only for HuggingFace API
            compatibility — prime-rl asserts `use_cache is None` since training does not
            perform autoregressive decoding, so this argument is unused.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
            If not provided, the wrapped LM head returns logits only.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        assert use_cache is None, "use_cache is not supported for custom mimo_v2 for now"
        assert past_key_values is None, "past_key_values is not supported for custom mimo_v2 for now"

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        for rotary_emb in (self.model.rotary_emb, self.model.swa_rotary_emb):
            if rotary_emb.inv_freq.device.type != "meta":
                rotary_emb.init_inv_freq_post_meta()


__all__ = ["MiMoV2Config", "MiMoV2PreTrainedModel", "MiMoV2Model", "MiMoV2ForCausalLM"]
