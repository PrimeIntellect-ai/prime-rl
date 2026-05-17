# coding=utf-8
# Copyright 2025 Zyphra and the HuggingFace Inc. team. All rights reserved.
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
import copy
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import (
    FlashAttention,
    SDPAAttention,
    flash_attn_3_varlen_func,
    flash_attn_4_varlen_func,
    flash_attn_varlen_func,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import ZayaMoE
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig, apply_rotary_pos_emb
from prime_rl.trainer.models.layers.ulysses_attn import ULYSSES_PARAMS, _all_to_all_head_to_seq, _all_to_all_seq_to_head
from prime_rl.trainer.models.zaya.configuration_zaya import ZayaConfig
from prime_rl.trainer.models.zaya.converting_zaya import (
    convert_hf_layer_to_prime,
    convert_hf_to_prime,
    convert_prime_layer_to_hf,
    convert_prime_to_hf,
)
from prime_rl.trainer.models.zaya.converting_zaya import is_hf_state_dict as _is_hf_state_dict
from prime_rl.trainer.models.zaya.converting_zaya import is_prime_state_dict as _is_prime_state_dict
from prime_rl.trainer.models.zaya.vllm_postprocessing import convert_prime_to_vllm
from prime_rl.utils.cp import gather_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids


def _all_to_all_seq_to_head_batched(t: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup) -> torch.Tensor:
    assert t.shape[0] == 1, f"Zaya CP currently expects batch size 1, got {t.shape[0]}"
    return _all_to_all_seq_to_head(t.squeeze(0), cp_size, cp_group).unsqueeze(0)


def _all_to_all_head_to_seq_batched(t: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup) -> torch.Tensor:
    assert t.shape[0] == 1, f"Zaya CP currently expects batch size 1, got {t.shape[0]}"
    return _all_to_all_head_to_seq(t.squeeze(0), cp_size, cp_group).unsqueeze(0)


class ZayaResidualScaling(nn.Module):
    def __init__(self, config: ZayaConfig):
        super().__init__()
        self.hidden_states_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.residual_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.residual_bias = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        residual = (residual + self.residual_bias) * self.residual_scale
        return hidden_states + residual

class ZayaCCAProjection(nn.Module):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size

        self.depthwise_kernel_size = config.cca_time0
        self.grouped_kernel_size = config.cca_time1
        self.conv_kernel_size = (self.depthwise_kernel_size - 1) + (self.grouped_kernel_size - 1)

        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        query_hidden_size = self.num_attention_heads * self.head_dim
        key_value_hidden_size = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, query_hidden_size, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, key_value_hidden_size, bias=self.config.attention_bias)
        self.v_proj_current = nn.Linear(self.hidden_size, key_value_hidden_size // 2, bias=self.config.attention_bias)
        self.v_proj_delayed = nn.Linear(self.hidden_size, key_value_hidden_size // 2, bias=self.config.attention_bias)

        conv_channels = key_value_hidden_size + query_hidden_size
        self.conv_qk_depthwise = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=self.depthwise_kernel_size,
            groups=conv_channels,
            padding=0,
            stride=1,
        )
        self.conv_qk_grouped = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=self.grouped_kernel_size,
            groups=(self.num_key_value_heads + self.num_attention_heads),
            padding=0,
            stride=1,
        )
        self._cp_group = None
        self._cp_rank = 0
        self._cp_world_size = 1

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def _local_head_channel_indices(self) -> torch.Tensor:
        local_q = self.num_attention_heads // self._cp_world_size
        local_kv = self.num_key_value_heads // self._cp_world_size
        q_start = self._cp_rank * local_q * self.head_dim
        q_end = q_start + local_q * self.head_dim
        k_start = self.num_attention_heads * self.head_dim + self._cp_rank * local_kv * self.head_dim
        k_end = k_start + local_kv * self.head_dim
        return torch.cat(
            [
                torch.arange(q_start, q_end, device=self.conv_qk_depthwise.weight.device),
                torch.arange(k_start, k_end, device=self.conv_qk_depthwise.weight.device),
            ]
        )

    def _forward_context_parallel(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None):
         # TODO: support packed sequences in Context Parallel
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states.dtype)

        input_shape = hidden_states.shape[:-1]
        projected_queries = self.q_proj(hidden_states).view(*input_shape, self.num_attention_heads, self.head_dim)
        projected_keys = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        value_current = self.v_proj_current(hidden_states)
        delayed_v_state = self.v_proj_delayed(hidden_states)

        projected_queries = _all_to_all_seq_to_head_batched(projected_queries, self._cp_world_size, self._cp_group)
        projected_keys = _all_to_all_seq_to_head_batched(projected_keys, self._cp_world_size, self._cp_group)

        local_q_heads = projected_queries.shape[-2]
        local_kv_heads = projected_keys.shape[-2]
        local_groups = local_q_heads // local_kv_heads
        query_residual = projected_queries
        key_residual = _repeat_kv(projected_keys.transpose(1, 2), local_groups).transpose(1, 2)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(*query_residual.shape[:2], local_kv_heads, local_groups, self.head_dim).mean(
            dim=-2
        )

        qk_states = torch.cat([projected_queries.flatten(-2), projected_keys.flatten(-2)], dim=-1).transpose(1, 2)
        qk_states = F.pad(qk_states, (self.conv_kernel_size, 0))
        channel_idx = self._local_head_channel_indices()
        depthwise_weight = self.conv_qk_depthwise.weight.index_select(0, channel_idx)
        depthwise_bias = (
            self.conv_qk_depthwise.bias.index_select(0, channel_idx)
            if self.conv_qk_depthwise.bias is not None
            else None
        )
        qk_states = F.conv1d(qk_states, depthwise_weight, depthwise_bias, groups=channel_idx.numel())
        grouped_weight = self.conv_qk_grouped.weight.index_select(0, channel_idx)
        grouped_bias = (
            self.conv_qk_grouped.bias.index_select(0, channel_idx) if self.conv_qk_grouped.bias is not None else None
        )
        qk_states = F.conv1d(qk_states, grouped_weight, grouped_bias, groups=local_q_heads + local_kv_heads).transpose(
            1, 2
        )

        q_size = local_q_heads * self.head_dim
        query = qk_states[..., :q_size].view(*qk_states.shape[:2], local_q_heads, self.head_dim) + query_residual
        key = qk_states[..., q_size:].view(*qk_states.shape[:2], local_kv_heads, self.head_dim) + key_residual

        recurrent_v_state = self.v_proj_delayed(hidden_states.new_zeros(input_shape[0], 1, self.hidden_size))
        delayed_v_state_full = gather_for_cp(delayed_v_state.contiguous(), self._cp_group)
        value_delayed_full = torch.cat([recurrent_v_state, delayed_v_state_full[:, :-1]], dim=1)
        seq_start = self._cp_rank * input_shape[1]
        seq_end = seq_start + input_shape[1]
        value_delayed = value_delayed_full[:, seq_start:seq_end]
        value = torch.cat([value_current, value_delayed], dim=-1).view(
            *input_shape, self.num_key_value_heads, self.head_dim
        )
        value = _all_to_all_seq_to_head_batched(value, self._cp_world_size, self._cp_group)
        return query, key, value

    def _conv_qk_by_sequence(self, qk_states: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
        # Vectorized version of:
        # outputs = []
        # for start, end in zip(cu[:-1], cu[1:]):
        #     segment = F.pad(qk_states[:, :, start:end], (self.conv_kernel_size, 0))
        #     outputs.append(self.conv_qk_grouped(self.conv_qk_depthwise(segment)))
        # return torch.cat(outputs, dim=-1).transpose(1, 2)
        qk_states = qk_states.transpose(1, 2)

        if cu_seqlens is None or cu_seqlens.numel() <= 2:
            qk_states = F.pad(qk_states, (self.conv_kernel_size, 0))
            return self.conv_qk_grouped(self.conv_qk_depthwise(qk_states)).transpose(1, 2)

        B, C, S = qk_states.shape
        device = qk_states.device
        K = self.conv_kernel_size

        nseq = cu_seqlens.numel() - 1
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]

        seg_id = torch.repeat_interleave(
            torch.arange(nseq, device=device, dtype=cu_seqlens.dtype),
            lengths,
        )

        orig_idx = torch.arange(S, device=device, dtype=cu_seqlens.dtype)
        expanded_idx = orig_idx + K * (seg_id + 1)

        expanded_len = S + K * nseq
        expanded = qk_states.new_zeros(B, C, expanded_len)

        expanded.scatter_(
            dim=2,
            index=expanded_idx.to(torch.long)[None, None, :].expand(B, C, S),
            src=qk_states,
        )

        out = self.conv_qk_grouped(self.conv_qk_depthwise(expanded))

        gather_idx = (expanded_idx - K).to(torch.long)
        out = out.index_select(dim=2, index=gather_idx)

        return out.transpose(1, 2)

    def _delay_value_by_sequence(
        self,
        hidden_states: torch.Tensor,
        delayed_v_state: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        # Vectorized version of:
        # outputs = []
        # for start, end in zip(cu[:-1], cu[1:]):
        #     segment = delayed_v_state[:, start:end]
        #     outputs.append(torch.cat([recurrent_v_state, segment[:, :-1]], dim=1))
        # return torch.cat(outputs, dim=1)
        input_shape = hidden_states.shape[:-1]
        recurrent_v_state = self.v_proj_delayed(
            hidden_states.new_zeros(input_shape[0], 1, self.hidden_size)
        )

        if cu_seqlens is None or cu_seqlens.numel() <= 2:
            return torch.cat([recurrent_v_state, delayed_v_state[:, :-1]], dim=1)

        B, S, D = delayed_v_state.shape
        device = delayed_v_state.device

        idx = torch.arange(S, device=device)

        is_start = torch.zeros(S, device=device, dtype=torch.bool)
        is_start[cu_seqlens[:-1].to(torch.long)] = True

        prev_idx = (idx - 1).clamp_min(0)
        shifted = delayed_v_state.index_select(dim=1, index=prev_idx)

        return torch.where(
            is_start[None, :, None],
            recurrent_v_state.expand(B, S, D),
            shifted,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ):
        if self.cp_enabled:
            return self._forward_context_parallel(hidden_states, padding_mask)

        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states.dtype)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        projected_queries = self.q_proj(hidden_states)
        projected_keys = self.k_proj(hidden_states)
        qk_states = torch.cat([projected_queries, projected_keys], dim=-1)

        query_residual = projected_queries.view(*hidden_shape)
        key_residual = projected_keys.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        key_residual = _repeat_kv(key_residual, self.num_key_value_groups).transpose(1, 2)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(*input_shape, -1, self.num_key_value_groups, self.head_dim).mean(dim=-2)

        qk_states = self._conv_qk_by_sequence(qk_states, cu_seqlens)

        query_hidden_size = query_residual.shape[-2] * query_residual.shape[-1]
        query = qk_states[..., :query_hidden_size].view(*hidden_shape) + query_residual
        key = qk_states[..., query_hidden_size:].view(*hidden_shape) + key_residual

        value_current = self.v_proj_current(hidden_states)
        delayed_v_state = self.v_proj_delayed(hidden_states)
        value_delayed = self._delay_value_by_sequence(hidden_states, delayed_v_state, cu_seqlens)

        value = torch.cat([value_current, value_delayed], dim=-1).view(*hidden_shape)

        return query, key, value


class ZayaQKNorm(nn.Module):
    def __init__(self, config: ZayaConfig):
        super().__init__()
        scaling = config.head_dim**-0.5
        self.head_dim_scale = scaling**-1
        self.temp = nn.Parameter(torch.zeros(config.num_key_value_heads))
        self._cp_rank = 0
        self._cp_world_size = 1

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_eps = torch.finfo(query_states.dtype).eps
        query_states = query_states * (
            self.head_dim_scale / query_states.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps)
        )
        key_states = key_states * (self.head_dim_scale / key_states.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps))
        temp = self.temp
        if self._cp_world_size > 1:
            local_kv_heads = temp.shape[0] // self._cp_world_size
            start = self._cp_rank * local_kv_heads
            temp = temp[start : start + local_kv_heads]
        key_states = key_states * temp[None, None, :, None]
        return query_states, key_states


class ZayaRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: RotaryEmbeddingConfig, device=None):
        super().__init__(config, device=device)
        self.inv_freq = self.inv_freq.float()
        self.original_inv_freq = self.inv_freq.clone()

    def _apply(self, fn):
        super()._apply(fn)
        # always force RoPE in fp32
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device)
        self.inv_freq = inv_freq.float()
        self.original_inv_freq = self.inv_freq.clone()
        return self


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ZayaSPDAAttention(SDPAAttention):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        nn.Module.__init__(
            self
        )  # instead of initing with super().__init__() because we don't want to initialize the qkv projections
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.qkv_proj = ZayaCCAProjection(
            config=self.config,
            layer_idx=layer_idx,
        )
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.qk_norm = ZayaQKNorm(config)
        self._cp_group = None
        self._cp_rank = 0
        self._cp_world_size = 1

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        assert self.num_attention_heads % cp_world_size == 0
        assert self.num_key_value_heads % cp_world_size == 0
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        self.qkv_proj.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)
        self.qk_norm.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def _output_context_parallel(self, attn_output: torch.Tensor) -> torch.Tensor:
        attn_output = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1, self.head_dim)
        attn_output = _all_to_all_head_to_seq_batched(attn_output, self._cp_world_size, self._cp_group)
        return attn_output.flatten(-2)

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)
        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=causal_mask is None,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: dict[str, Any] | None = None,
        cca_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        mask_mapping = attention_mask or {}
        causal_mask = mask_mapping.get("causal")
        padding_mask = mask_mapping.get("padding")

        query_states, key_states, value_states = self.qkv_proj(hidden_states, padding_mask, cu_seqlens)

        query_states, key_states = self.qk_norm(query_states, key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.cp_enabled and cu_seqlens is None:
            # TODO: support packed sequences in Context Parallel
            causal_mask = None
        attn_output = self._attention_core(query_states, key_states, value_states, causal_mask)
        if self.cp_enabled:
            attn_output = self._output_context_parallel(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class ZayaFlashAttention(FlashAttention):
    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: ZayaConfig, layer_idx: int, flash_attn_version: int = 2):
        nn.Module.__init__(
            self
        )  # instead of initing with super().__init__() because we don't want to initialize the qkv projections
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.qkv_proj = ZayaCCAProjection(
            config=self.config,
            layer_idx=layer_idx,
        )
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.qk_norm = ZayaQKNorm(config)
        self._cp_group = None
        self._cp_rank = 0
        self._cp_world_size = 1

        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        assert self.num_attention_heads % cp_world_size == 0
        assert self.num_key_value_heads % cp_world_size == 0
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        self.qkv_proj.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)
        self.qk_norm.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def _output_context_parallel(self, attn_output: torch.Tensor) -> torch.Tensor:
        attn_output = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1, self.head_dim)
        attn_output = _all_to_all_head_to_seq_batched(attn_output, self._cp_world_size, self._cp_group)
        return attn_output.flatten(-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: dict[str, Any] | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        mask_mapping = attention_mask or {}
        padding_mask = mask_mapping.get("padding")

        query_states, key_states, value_states = self.qkv_proj(hidden_states, padding_mask, cu_seqlens)

        query_states, key_states = self.qk_norm(query_states, key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.cp_enabled:
            cu_seqlens = ULYSSES_PARAMS.get("cu_seqlens", cu_seqlens)
            max_seqlen = ULYSSES_PARAMS.get("max_seqlen", max_seqlen)
        attn_output = self._attention_core(query_states, key_states, value_states, cu_seqlens, max_seqlen)
        if self.cp_enabled:
            attn_output = self._output_context_parallel(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


def _get_zaya_attention(config: ZayaConfig, layer_idx: int) -> nn.Module:
    attn_impl = config._attn_implementation
    if attn_impl == "eager":
        attn_impl = "sdpa"
    match attn_impl:
        case "flash_attention_2":
            return ZayaFlashAttention(config, layer_idx, flash_attn_version=2)
        case "flash_attention_3":
            return ZayaFlashAttention(config, layer_idx, flash_attn_version=3)
        case "fa4":
            return ZayaFlashAttention(config, layer_idx, flash_attn_version=4)
        case "sdpa":
            return ZayaSPDAAttention(config, layer_idx)
        case _:
            raise ValueError(f"Zaya attention does not support '{config._attn_implementation}'.")


class ZayaDecoderLayer(nn.Module):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = _get_zaya_attention(config, layer_idx)
        self.mlp = ZayaMoE(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            router_hidden_size=config.router_hidden_size,
            norm_epsilon=config.norm_epsilon,
            use_grouped_mm=config.use_grouped_mm,
            use_eda=config.zaya_use_eda,
        )
        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.norm_epsilon))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.norm_epsilon))
        self.post_attention_residual_scale = ZayaResidualScaling(config)
        self.post_mlp_residual_scale = ZayaResidualScaling(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
        attention_mask: dict[str, Any] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        routed_experts: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        residual = self.post_attention_residual_scale(hidden_states, residual)
        hidden_states = self.post_attention_layernorm(residual.to(dtype=self.post_attention_layernorm.weight.dtype))

        hidden_states, prev_router_hidden_states = self.mlp(
            hidden_states,
            prev_router_hidden_states,
            routed_experts=routed_experts,
        )

        hidden_states = self.post_mlp_residual_scale(hidden_states, residual)

        return hidden_states, prev_router_hidden_states


class ZayaPreTrainedModel(PreTrainedModelPrimeRL):
    config_class = ZayaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ZayaDecoderATTLayer", "ZayaDecoderMLPLayer"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True


class ZayaModel(ZayaPreTrainedModel):
    def __init__(self, config: ZayaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ZayaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.input_hidden_states_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.input_hidden_states_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.norm_epsilon))
        rope_parameters = config.rope_parameters.get("hybrid", config.rope_parameters)
        rope_config = copy.copy(config)
        rope_config.rope_parameters = rope_parameters
        self.rotary_emb = ZayaRotaryEmbedding(
            RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings,
                rope_type=rope_parameters["rope_type"],
                model_config=rope_config,
            )
        )
        # no swa layers for 8B
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @property
    def cp_enabled(self) -> bool:
        return bool(self.layers) and getattr(self.layers[0].self_attn, "cp_enabled", False)

    @property
    def cp_group(self) -> dist.ProcessGroup | None:
        if not self.cp_enabled:
            return None
        return self.layers[0].self_attn._cp_group

    @property
    def cp_world_size(self) -> int:
        if not self.cp_enabled:
            return 1
        return self.layers[0].self_attn._cp_world_size

    def _prepare_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        batch_size, seq_length = inputs_embeds.shape[:2]
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full(
            (seq_length, seq_length),
            min_dtype,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)
        padding_mask = attention_mask[:, None, None, :].to(torch.bool)
        return causal_mask.masked_fill(~padding_mask, min_dtype)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is not None and not isinstance(attention_mask, dict) and attention_mask.ndim != 2:
            raise ValueError(
                "ZAYA CCA projection requires a 2D `attention_mask` to mask padding tokens before convolution."
            )

        flat_position_ids = position_ids.reshape(-1)
        is_packed = position_ids.shape[0] == 1 and (
            (flat_position_ids[1:] == 0).any() if flat_position_ids.numel() > 1 else False
        )
        use_flash = self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4")
        if use_flash or is_packed:
            cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            cu_seqlens = None
            max_seqlen = None

        if use_flash:
            causal_mask_mapping = dict.fromkeys(set(self.config.layer_types), None)
        elif isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        elif is_packed:
            seq_length = inputs_embeds.shape[1]
            min_dtype = torch.finfo(inputs_embeds.dtype).min
            causal_mask = torch.full(
                (seq_length, seq_length),
                min_dtype,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            segment_ids = (flat_position_ids == 0).cumsum(dim=0) - 1
            same_segment = segment_ids[:, None] == segment_ids[None, :]
            causal_mask = causal_mask.masked_fill(~same_segment, min_dtype)
            causal_mask = causal_mask[None, None, :, :]
            if attention_mask is not None:
                padding_mask_for_attn = attention_mask[:, None, None, -seq_length:].to(torch.bool)
                causal_mask = causal_mask.masked_fill(~padding_mask_for_attn, min_dtype)
            causal_mask_mapping = {"hybrid": causal_mask}
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "hybrid": create_causal_mask(**mask_kwargs),
            }

        padding_mask = None
        if attention_mask is not None and not isinstance(attention_mask, dict):
            padding_mask = attention_mask[:, -inputs_embeds.shape[1] :]
            if inputs_embeds.shape[1] == 1:
                padding_mask = None

        hidden_states = inputs_embeds
        rope_position_ids = position_ids
        if self.cp_enabled:
            rope_position_ids = gather_for_cp(position_ids.contiguous(), self.cp_group)
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        hidden_states = ((hidden_states + self.input_hidden_states_bias) * self.input_hidden_states_scale).to(torch.float32)
        prev_router_hidden_states = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            mask_mapping = {"causal": causal_mask_mapping[self.config.layer_types[layer_idx]], "padding": padding_mask}
            hidden_states, prev_router_hidden_states = decoder_layer(
                hidden_states,
                prev_router_hidden_states=prev_router_hidden_states,
                attention_mask=mask_mapping,
                position_embeddings=position_embeddings,
                routed_experts=routed_experts_layer,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.final_norm(hidden_states.to(dtype=self.final_norm.weight.dtype))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)


class ZayaForCausalLM(ZayaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ZayaConfig):
        super().__init__(config)
        self.model = ZayaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return _is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return _is_prime_state_dict(state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_prime_to_hf(state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_hf_to_prime(state_dict)

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_prime_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_prime(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_to_vllm(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_prime_to_vllm(state_dict)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        **kwargs,
    ) -> PrimeLmOutput:
        del cache_position, kwargs
        assert use_cache in (None, False), "use_cache is not supported for PrimeRL Zaya"
        assert past_key_values is None, "past_key_values is not supported for PrimeRL Zaya"
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if labels is not None:
            labels = labels[:, slice_indices]
        if type(self.lm_head) is nn.Linear:
            return PrimeLmOutput(logits=self.lm_head(hidden_states[:, slice_indices, :]))
        return self.lm_head(hidden_states[:, slice_indices, :], labels, temperature=temperature)

    def init_buffers_post_meta(self):
        for rotary_emb in (getattr(self.model, "rotary_emb", None), getattr(self.model, "swa_rotary_emb", None)):
            if rotary_emb is None:
                continue
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq = inv_freq
            rotary_emb.original_inv_freq = inv_freq

        for module in self.modules():
            if isinstance(module, ZayaMoE):
                module.tokens_per_expert = torch.zeros(
                    module.experts.num_experts,
                    dtype=torch.float32,
                    device=module.tokens_per_expert.device,
                )


__all__ = ["ZayaConfig", "ZayaForCausalLM", "ZayaModel", "ZayaPreTrainedModel"]