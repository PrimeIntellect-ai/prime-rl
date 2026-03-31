import functools
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import FeedForward, MoE, MoEArgs
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig, apply_rotary_pos_emb

from .configuration_qwen3_5_moe import Qwen3_5MoeConfig
from .converting_qwen3_5_moe import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)

# Flash attention imports
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore

# Flash linear attention imports (for GatedDeltaNet fast path)
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None  # type: ignore

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.cp import FLACPContext, build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None  # type: ignore
    FusedRMSNormGated = None  # type: ignore
    FLACPContext = None  # type: ignore
    build_cp_context = None  # type: ignore

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


class Qwen3_5MoeRMSNormGated(nn.Module):
    """RMSNorm with SiLU gating for GatedDeltaNet output."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# GatedDeltaNet linear attention
# ---------------------------------------------------------------------------


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Pure-PyTorch fallback for chunk_gated_delta_rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


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

        # Gated RMSNorm on output (per-head)
        if FusedRMSNormGated is not None:
            self.norm = FusedRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        else:
            self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Input projections
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Select best available kernel
        self._causal_conv1d_fn = causal_conv1d_fn
        self._chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule

    def _build_cp_context(self, local_seq_len: int, device: torch.device) -> "FLACPContext | None":
        """Build fla CP context from the local (sharded) sequence length."""
        cp_group = getattr(self, "cp_group", None)
        if cp_group is None or build_cp_context is None:
            return None
        # Reconstruct global cu_seqlens: single contiguous sequence across all CP ranks
        global_seq_len = local_seq_len * self.cp_world_size
        global_cu_seqlens = torch.tensor([0, global_seq_len], dtype=torch.int32, device=device)
        return build_cp_context(
            cu_seqlens=global_cu_seqlens,
            group=cp_group,
            conv1d_kernel_size=self.conv_kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Causal conv1d
        if self._causal_conv1d_fn is not None:
            mixed_qkv = self._causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # Use fla's native CP when available, otherwise fall back to PyTorch kernel
        cp_context = self._build_cp_context(seq_len, hidden_states.device)
        if cp_context is not None:
            cu_seqlens = cp_context.cu_seqlens
            core_attn_out, _ = self._chunk_gated_delta_rule(
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
            core_attn_out, _ = self._chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
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


class Qwen3_5MoeGatedSDPAAttention(Qwen3_5MoeGatedAttentionBase):
    """Gated softmax attention using PyTorch's scaled_dot_product_attention."""

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

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states, key_states, value_states, gate

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)
        return F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=self.scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        query_states, key_states, value_states, gate = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(query_states, key_states, value_states)
        return self.output_proj(attn_output, gate), None


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
        args = [q, k, v, cu_seqlens, cu_seqlens]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen])
        kwargs: dict = {"causal": True}
        out = self._flash_attn_call(*args, **kwargs)
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
    "sdpa": Qwen3_5MoeGatedSDPAAttention,
    "flash_attention_2": functools.partial(Qwen3_5MoeGatedFlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(Qwen3_5MoeGatedFlashAttention, flash_attn_version=3),
    "fa4": functools.partial(Qwen3_5MoeGatedFlashAttention, flash_attn_version=4),
}


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

    attn_impl = config._attn_implementation
    if attn_impl == "eager":
        attn_impl = "sdpa"

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
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Token mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
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
# Model classes
# ---------------------------------------------------------------------------


def _create_rotary_emb(config: Qwen3_5MoeConfig) -> RotaryEmbedding:
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
    else:
        rope_type = "default"

    rotary_config = RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type=rope_type,
        model_config=config,
    )
    return RotaryEmbedding(rotary_config)


class Qwen3_5MoePreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Qwen3_5MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": Qwen3_5MoeDecoderLayer,
    }

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4"):
            flat_position_ids = position_ids.view(-1)
            seqlens = torch.cat(
                [
                    flat_position_ids[0:1],
                    flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
                    flat_position_ids[-1:] + 1,
                ]
            )
            max_seqlen = seqlens.max().item()
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            max_seqlen = None
            cu_seqlens = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                routed_experts=routed_experts_layer,
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

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
            image_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask = input_ids == self.config.image_token_id
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            routed_experts=routed_experts,
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None, "use_cache is not supported for custom qwen3_5_moe for now"
        assert past_key_values is None, "past_key_values is not supported for custom qwen3_5_moe for now"

        if position_ids is None:
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
                routed_experts=routed_experts,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                routed_experts=routed_experts,
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
