import functools
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from fla.modules.conv import causal_conv1d as fla_causal_conv1d
from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.attn import (
    flash_attn_3_varlen_func,
    flash_attn_4_varlen_func,
    flash_attn_varlen_func,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import FeedForward, MoE, MoEArgs
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.utils.cp import setup_cp_attention_params, shard_for_cp, shard_position_ids_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

from .configuration_qwen3_5_moe import Qwen3_5MoeConfig
from .converting_qwen3_5_moe import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)
from .mrope import build_qwen3_5_mrope_position_ids

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm variants
# ---------------------------------------------------------------------------


class Qwen3_5MoeRMSNorm(nn.Module):
    """RMSNorm with (1+weight) parameterization. Weight initialized to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# ---------------------------------------------------------------------------
# GatedDeltaNet linear attention
# ---------------------------------------------------------------------------


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention with Conv1d, beta/gamma gates, and chunk delta rule."""

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV convolution (depthwise)
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Time step projection
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Input projections
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def _build_cp_context(
        self,
        local_seq_len: int,
        device: torch.device,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_are_global: bool = False,
    ) -> "FLACPContext | None":
        """Build fla CP context from the local (sharded) sequence length."""
        cp_group = getattr(self, "cp_group", None)
        if cp_group is None:
            return None
        # Provenance flag rather than reading cu_seqlens back (per-layer CPU-GPU sync).
        if cu_seqlens is not None and cu_seqlens_are_global:
            global_cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        else:
            global_seq_len = local_seq_len * self.cp_world_size
            global_cu_seqlens = torch.tensor([0, global_seq_len], dtype=torch.int32, device=device)
        return build_cp_context(
            cu_seqlens=global_cu_seqlens,
            group=cp_group,
            conv1d_kernel_size=self.conv_kernel_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_are_global: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        cp_context = self._build_cp_context(seq_len, hidden_states.device, cu_seqlens, cu_seqlens_are_global)

        # Causal conv1d — must reset at sequence boundaries for packed batches,
        # otherwise the kernel-1 left pad leaks state across sequences.
        mixed_qkv, _ = fla_causal_conv1d(
            x=mixed_qkv.transpose(1, 2),
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens if cp_context is None else None,
            cp_context=cp_context,
        )
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if cp_context is not None:
            cu_seqlens = cp_context.cu_seqlens
            core_attn_out, _ = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
        else:
            core_attn_out, _ = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        # Gated RMSNorm
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Gated softmax attention (for full_attention layers)
# ---------------------------------------------------------------------------


@dataclass
class Qwen3_5MoeGatedAttentionConfig:
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    attention_bias: bool = False
    attention_dropout: float = 0.0


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Qwen3_5MoeGatedAttentionBase(nn.Module):
    """Base class for gated softmax attention (Q projects 2x: query + gate)."""

    def __init__(self, config: Qwen3_5MoeGatedAttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Q projects 2x: query + gate
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # QK normalization with (1+weight) parameterization
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def output_proj(
        self,
        attn_output: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = gate.shape[:-1]
        if attn_output.dim() == 4:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.contiguous().view(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen3_5MoeGatedFlashAttention(Qwen3_5MoeGatedAttentionBase):
    """Gated softmax attention using Flash Attention varlen functions."""

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: Qwen3_5MoeGatedAttentionConfig, flash_attn_version: int = 4):
        super().__init__(config)
        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def _compute_attention(self, q, k, v, cu_seqlens, max_seqlen):
        """Run the flash attention kernel. q/k/v are [total_tokens, heads, dim]."""
        kwargs: dict = {"causal": True}
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            kwargs["window_size"] = (sliding_window - 1, 0)
        if self._flash_attn_version == 4:
            # FA4's flash_attn_varlen_func has qv as the 4th positional arg,
            # so cu_seqlens must be passed as keyword args to avoid misalignment.
            kwargs["cu_seqlens_q"] = cu_seqlens
            kwargs["cu_seqlens_k"] = cu_seqlens
            out = self._flash_attn_call(q, k, v, **kwargs)
        else:
            out = self._flash_attn_call(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        return self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        return query_states, key_states, value_states, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        query_states, key_states, value_states, gate = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return self.output_proj(attn_output, gate), None


QWEN35MOE_ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(Qwen3_5MoeGatedFlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(Qwen3_5MoeGatedFlashAttention, flash_attn_version=3),
    "flash_attention_4": functools.partial(Qwen3_5MoeGatedFlashAttention, flash_attn_version=4),
}


def normalize_qwen3_5_attn_implementation(attn_impl: str) -> str:
    if attn_impl == "kernels-community/vllm-flash-attn3":
        return "flash_attention_3"
    return attn_impl


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


def _get_gated_attention(config: Qwen3_5MoeConfig) -> nn.Module:
    attn_config = Qwen3_5MoeGatedAttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
    )

    attn_impl = normalize_qwen3_5_attn_implementation(config._attn_implementation)
    config._attn_implementation = attn_impl

    if attn_impl not in QWEN35MOE_ATTN_IMPL2CLASS:
        supported = list(QWEN35MOE_ATTN_IMPL2CLASS.keys())
        raise ValueError(
            f"Qwen3.5-MoE attention does not support '{config._attn_implementation}'. "
            f"Supported implementations: {supported}."
        )

    return QWEN35MOE_ATTN_IMPL2CLASS[attn_impl](attn_config)


class Qwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        # Token mixer: either GatedDeltaNet or gated softmax attention
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = _get_gated_attention(config)

        # MoE: routed experts via shared MoE class (no shared experts in MoE itself)
        moe_args = MoEArgs(
            num_experts=config.num_experts,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=config.use_grouped_mm,
            load_balance_coeff=config.load_balance_coeff,
            fp8=getattr(config, "fp8", False),
        )
        self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)

        # Separate gated shared expert
        self.shared_expert = FeedForward(dim=config.hidden_size, hidden_dim=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        # Layer norms with (1+weight) parameterization
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
        routed_experts: Optional[torch.LongTensor] = None,
        cu_seqlens_are_global: bool = False,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Token mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cu_seqlens=cu_seqlens, cu_seqlens_are_global=cu_seqlens_are_global
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = residual + hidden_states

        # MLP: routed experts + gated shared expert
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Routed experts
        routed_output = self.mlp(hidden_states, routed_experts=routed_experts)

        # Gated shared expert
        bs, slen, dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, dim)
        shared_output = self.shared_expert(hidden_flat)
        shared_output = F.sigmoid(self.shared_expert_gate(hidden_flat)) * shared_output
        shared_output = shared_output.view(bs, slen, dim)

        hidden_states = residual + routed_output + shared_output
        return hidden_states


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------


class Qwen3_5MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5MoeConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            config.standardize_rope_params()
            rope_parameters = config.rope_parameters

        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        self.rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq.clone()

        self.mrope_section = rope_parameters.get("mrope_section")
        if self.mrope_section is None:
            self.mrope_section = self._scaled_default_mrope_section(self.inv_freq.numel())
        if not rope_parameters.get("mrope_interleaved", True):
            raise ValueError("Qwen3.5 MoE custom model expects interleaved MRoPE")
        if sum(self.mrope_section) != self.inv_freq.numel():
            raise ValueError(
                "Qwen3.5 mrope_section must sum to rotary_dim // 2: "
                f"mrope_section={self.mrope_section}, rotary_dim={self.inv_freq.numel() * 2}"
            )

    @staticmethod
    def _scaled_default_mrope_section(num_rotary_pairs: int) -> list[int]:
        default_section = [11, 11, 10]
        default_total = sum(default_section)
        scaled_section = [num_rotary_pairs * section // default_total for section in default_section]
        remainder = num_rotary_pairs - sum(scaled_section)
        remainder_order = sorted(
            range(len(default_section)),
            key=lambda idx: num_rotary_pairs * default_section[idx] % default_total,
            reverse=True,
        )
        for idx in remainder_order[:remainder]:
            scaled_section[idx] += 1
        return scaled_section

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3_5MoeConfig | None = None,
        device: Optional[torch.device] = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        rope_parameters = config.rope_parameters
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim != 3:
            raise ValueError(f"Qwen3.5 position_ids must be 2D or 3D, got shape={tuple(position_ids.shape)}")

        position_ids = position_ids.to(device=x.device)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self._apply_interleaved_mrope(freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


def _create_rotary_emb(config: Qwen3_5MoeConfig) -> Qwen3_5MoeRotaryEmbedding:
    return Qwen3_5MoeRotaryEmbedding(config)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class Qwen3_5MoePreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": Qwen3_5MoeDecoderLayer,
    }

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        attn_impl = normalize_qwen3_5_attn_implementation(attn_implementation or "flash_attention_3")
        if attn_impl not in QWEN35MOE_ATTN_IMPL2CLASS:
            supported = list(QWEN35MOE_ATTN_IMPL2CLASS.keys())
            raise ValueError(
                f"Qwen3.5-MoE attention does not support '{attn_implementation}'. Supported implementations: {supported}."
            )
        return attn_impl

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "mlp.experts.1.up_proj" in name
            or "mlp.experts.gate_up_proj" in name
            or "mlp.shared_expert.gate_proj" in name
            for name in state_dict.keys()
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in name for name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict


class Qwen3_5MoeModel(Qwen3_5MoePreTrainedModel):
    def __init__(self, config: Qwen3_5MoeConfig):
        config._attn_implementation = normalize_qwen3_5_attn_implementation(config._attn_implementation)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _create_rotary_emb(config)
        self.gradient_checkpointing = False

        self.post_init()

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size
        for layer in self.layers:
            if getattr(layer, "layer_type", None) == "linear_attention":
                layer.linear_attn.cp_group = cp_group
                layer.linear_attn.cp_rank = cp_rank
                layer.linear_attn.cp_world_size = cp_world_size

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_global: bool = False,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        flash_attn_enabled = self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4")
        if flash_attn_enabled:
            cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
                seq_lens.to(device=inputs_embeds.device),
                total_tokens=None if seq_lens_are_global else inputs_embeds.shape[1],
            )
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            seq_lens = seq_lens.to(device=inputs_embeds.device)
            if seq_lens.numel() > 1 and "full_attention" in self.config.layer_types:
                raise ValueError("Packed Qwen3.5 batches with full_attention layers require flash attention")
            cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
                seq_lens,
                total_tokens=None if seq_lens_are_global else inputs_embeds.shape[1],
            )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens_are_global = seq_lens_are_global

        for layer_idx, decoder_layer in enumerate(self.layers):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                routed_experts=routed_experts_layer,
                cu_seqlens_are_global=cu_seqlens_are_global,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# VLM composite model body
# ---------------------------------------------------------------------------


def _build_text_config(composite_config: PretrainedConfig) -> Qwen3_5MoeConfig:
    """Build custom PrimeRL text config from HF's composite VLM config."""
    text_dict = composite_config.text_config.to_dict()
    text_config = Qwen3_5MoeConfig(**text_dict)
    attn_impl = getattr(
        composite_config.text_config,
        "_attn_implementation",
        getattr(composite_config, "_attn_implementation", None),
    )
    if attn_impl is not None:
        text_config._attn_implementation = attn_impl
    return text_config


class Qwen3_5MoeVLMModel(nn.Module):
    """Composite VLM body: HF vision encoder + custom PrimeRL text model."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5MoeModel(_build_text_config(config))

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.language_model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    def _dummy_vision_inputs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Smallest valid vision input: a single merged token (grid [1, m, m])."""
        vcfg = self.config.vision_config
        m = vcfg.spatial_merge_size
        num_patches = m * m
        patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
        pixel_values = torch.zeros(num_patches, patch_dim, device=device, dtype=self.visual.dtype)
        grid_thw = torch.tensor([[1, m, m]], dtype=torch.long, device=device)
        return pixel_values, grid_thw

    def prepare_inputs_embeds_and_position_ids(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids are required when inputs_embeds are not provided")
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if image_grid_thw is not None and position_ids is not None and position_ids.ndim != 3:
            raise ValueError(
                f"Qwen3.5 multimodal forward requires 3D MRoPE position_ids; got shape={tuple(position_ids.shape)}"
            )

        # Always run the vision encoder for collective symmetry: under FSDP + EP
        # all ranks share one process group, so every rank must issue the vision
        # encoder's collectives in the same order each step. Text-only microbatches
        # keep the dummy embeds in the graph with zero contribution so trainable
        # vision modules also participate in backward collectives.
        has_images = pixel_values is not None
        vision_grid_thw = image_grid_thw
        if has_images:
            if input_ids is None:
                raise ValueError("input_ids are required when scattering Qwen3.5 image features")
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when pixel_values are provided")
            pixel_values = pixel_values.type(self.visual.dtype)
        else:
            pixel_values, vision_grid_thw = self._dummy_vision_inputs(inputs_embeds.device)

        vision_output = self.visual(pixel_values, grid_thw=vision_grid_thw, return_dict=True)
        image_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

        if has_images:
            image_mask = input_ids == self.config.image_token_id
            image_token_count = int(image_mask.sum().item())
            image_feature_count = int(image_embeds.shape[0])
            if image_token_count != image_feature_count:
                raise ValueError(
                    "Qwen VLM image token/feature mismatch before scatter: "
                    f"image_token_id={self.config.image_token_id}, "
                    f"image_tokens={image_token_count}, image_features={image_feature_count}, "
                    f"input_ids_shape={tuple(input_ids.shape)}, "
                    f"pixel_values_shape={tuple(pixel_values.shape)}, "
                    f"image_grid_thw_shape={tuple(image_grid_thw.shape) if image_grid_thw is not None else None}"
                )
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            inputs_embeds = inputs_embeds + image_embeds.sum() * 0.0

        if position_ids is None:
            if image_grid_thw is not None:
                if input_ids is None:
                    raise ValueError("input_ids are required to compute Qwen3.5 multimodal MRoPE positions")
                if mm_token_type_ids is None:
                    raise ValueError(
                        "Qwen3.5 multimodal forward requires mm_token_type_ids to compute MRoPE positions correctly"
                    )
                position_ids = build_qwen3_5_mrope_position_ids(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    seq_lens=seq_lens,
                )
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        return inputs_embeds, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_global: bool = False,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        inputs_embeds, position_ids = self.prepare_inputs_embeds_and_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            seq_lens=seq_lens,
        )

        cp_group = getattr(self.language_model, "_cp_group", None)
        if image_grid_thw is not None and cp_group is not None:
            cp_rank = self.language_model._cp_rank
            cp_world_size = self.language_model._cp_world_size
            setup_cp_attention_params(position_ids, cp_group=cp_group, cp_style="ulysses", seq_lens=seq_lens)
            inputs_embeds = shard_for_cp(inputs_embeds, cp_rank=cp_rank, cp_world_size=cp_world_size)
            position_ids = shard_position_ids_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
            if routed_experts is not None:
                routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_world_size)
            seq_lens_are_global = True

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_global=seq_lens_are_global,
        )


# ---------------------------------------------------------------------------
# Unified CausalLM / VLM class
# ---------------------------------------------------------------------------


def _has_vlm_keys(state_dict: dict[str, Tensor]) -> bool:
    return any(k.startswith("model.language_model.") for k in state_dict)


def _remap_lm_keys(state_dict: dict[str, Tensor], to_flat: bool = True) -> None:
    """Remap language model keys between VLM and flat format for weight conversion.

    to_flat=True:  model.language_model.* -> model.*
    to_flat=False: model.*               -> model.language_model.*

    Vision keys (model.visual.*) are never touched.
    """
    src = "model.language_model." if to_flat else "model."
    dst = "model." if to_flat else "model.language_model."
    for k in [k for k in list(state_dict.keys()) if k.startswith(src) and not k.startswith("model.visual.")]:
        state_dict[dst + k[len(src) :]] = state_dict.pop(k)


class Qwen3_5MoeForCausalLM(Qwen3_5MoePreTrainedModel, GenerationMixin):
    """Unified Qwen3.5 MoE model for both text-only and VLM configs.

    When config has a vision_config, creates a composite model with HF's frozen
    vision encoder + custom text model. Otherwise creates a text-only model.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _checkpoint_conversion_mapping = {}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._is_vlm = hasattr(config, "vision_config")

        if self._is_vlm:
            self.model = Qwen3_5MoeVLMModel(config)
            text_config = config.text_config
            self._tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
        else:
            self.model = Qwen3_5MoeModel(config)
            text_config = config

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        if self._is_vlm:
            return self.model.get_input_embeddings()
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        if self._is_vlm:
            self.model.set_input_embeddings(value)
        else:
            self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_context_parallel_attributes(self, cp_group, cp_rank: int, cp_world_size: int) -> None:
        self.model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)

    # ------------------------------------------------------------------
    # State dict detection & conversion (handles both text-only and VLM)
    # ------------------------------------------------------------------

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "mlp.experts.gate_up_proj" in name
            or "mlp.experts.1.up_proj" in name
            or "mlp.shared_expert.gate_proj" in name
            for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in name for name in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_tt_to_hf_moe(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_hf_to_tt_moe(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_tt_layer_to_hf(state_dict, layer_idx)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        vlm = _has_vlm_keys(state_dict)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=True)
        convert_hf_layer_to_tt(state_dict, layer_idx)
        if vlm:
            _remap_lm_keys(state_dict, to_flat=False)
        return state_dict

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Union[torch.Tensor, None] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_global: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None, "use_cache is not supported for custom qwen3_5_moe for now"
        assert past_key_values is None, "past_key_values is not supported for custom qwen3_5_moe for now"

        has_vlm_image_inputs = self._is_vlm and image_grid_thw is not None
        if position_ids is None and not has_vlm_image_inputs:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            elif input_ids is not None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        if self._is_vlm:
            outputs: MoeModelOutputWithPast = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
                seq_lens_are_global=seq_lens_are_global,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                routed_experts=routed_experts,
                seq_lens=seq_lens,
                seq_lens_are_global=seq_lens_are_global,
            )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Buffer init after meta-device loading
    # ------------------------------------------------------------------

    def init_buffers_post_meta(self):
        if self._is_vlm:
            lm_rope = self.model.language_model.rotary_emb
        else:
            lm_rope = self.model.rotary_emb

        if hasattr(lm_rope, "rope_init_fn"):
            inv_freq, lm_rope.attention_scaling = lm_rope.rope_init_fn(lm_rope.config, lm_rope.inv_freq.device)
            lm_rope.inv_freq.copy_(inv_freq)

        if self._is_vlm:
            vis_rope = self.model.visual.rotary_pos_emb
            if hasattr(vis_rope, "inv_freq"):
                dim = vis_rope.inv_freq.shape[0]
                inv_freq = 1.0 / (
                    10000.0
                    ** (torch.arange(0, dim * 2, 2, dtype=torch.float32, device=vis_rope.inv_freq.device) / (dim * 2))
                )
                vis_rope.inv_freq.copy_(inv_freq)


__all__ = [
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeModel",
    "Qwen3_5MoePreTrainedModel",
]
