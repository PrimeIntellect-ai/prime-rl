import functools
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from prime_rl.trainer.models.fusions import fuse_qkv_projections

from .norms import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb

# flash-attention-2
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore

# flash-attention-3
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore


# Storage dtypes the trainer can replay for the inference KV cache, mapped to torch
# dtypes. ``fp8`` is vLLM's uncalibrated e4m3 cache; the straight-through cast below
# reproduces its unit-scale quantization error.
_KV_CACHE_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


# vLLM's vendored FA (3+), whose fp8 path the inference engines run. Imported eagerly
# at module load: a lazy import inside the forward would land after the trainer's
# process-group init, and importing vLLM mid-run tears down torch.distributed's
# default PG (vLLM's platform setup), breaking everything that follows.
try:
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_flash_attn_varlen_func
except ImportError:
    _vllm_flash_attn_varlen_func = None


class Fp8KVKernelAttention(torch.autograd.Function):
    """Attention through the engine's fp8 flash-attn kernel, with a bf16 straight-through backward.

    The forward quantizes Q/K/V to e4m3 (unit scale, matching vLLM's uncalibrated fp8
    cache and its per-tensor query quantization) and calls the same vendored FA3 entry
    point the engines call with descales of 1.0 — reproducing not just the quantized
    inputs but the kernel's own arithmetic (fp8 tensor-core matmuls, the in-kernel e4m3
    quantization of the attention probabilities). The fp8 kernel has no backward, so
    gradients re-run the standard bf16 kernel on the unquantized values: the forward
    matches the engine exactly (which is what the mismatch KL measures), while the
    backward treats the quantization as identity, like the value-level replay.
    """

    @staticmethod
    @torch._dynamo.disable(recursive=True)
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        fp8_dtype = torch.float8_e4m3fn
        q8, k8, v8 = (t.to(fp8_dtype).contiguous() for t in (q, k, v))
        num_seqs = cu_seqlens.numel() - 1
        num_kv_heads = k.shape[1]
        descale = torch.ones((num_seqs, num_kv_heads), device=q.device, dtype=torch.float32)
        softmax_scale = q.shape[-1] ** (-0.5)
        out, _, _, _ = torch.ops._vllm_fa3_C.fwd.default(
            q=q8,
            k=k8,
            v=v8,
            k_new=None,
            v_new=None,
            q_v=None,
            out=None,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_k_new=None,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            page_table=None,
            kv_batch_idx=None,
            leftpad_k=None,
            rotary_cos=None,
            rotary_sin=None,
            seqlens_rotary=None,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            softmax_scale=softmax_scale,
            is_causal=True,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            is_rotary_interleaved=True,
            scheduler_metadata=None,
            num_splits=0,
            pack_gqa=None,
            sm_margin=0,
            s_aux=None,
            cp_world_size=1,
            cp_rank=0,
            cp_tot_seqused_k=None,
        )
        ctx.save_for_backward(q, k, v)
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        return out

    @staticmethod
    @torch._dynamo.disable(recursive=True)
    def backward(ctx, dout: torch.Tensor):
        from prime_rl.trainer.models.layers.attn import FlashAttention

        q, k, v = ctx.saved_tensors
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = FlashAttention._funcs[3](
                q,
                k,
                v,
                ctx.cu_seqlens,
                ctx.cu_seqlens,
                ctx.max_seqlen,
                ctx.max_seqlen,
                causal=True,
            )
            out.backward(dout)
        return q.grad, k.grad, v.grad, None, None


def simulate_kv_cache_dtype(x: torch.Tensor, kv_cache_dtype: str | None) -> torch.Tensor:
    """Round-trip Q/K/V through the simulated KV-cache storage dtype.

    Quantized KV caches store K (post-RoPE) and V in 8 bits at unit scale, and
    vLLM additionally quantizes Q (per-tensor static scale) on the fp8 attention
    path; dequantizing all three back to the compute dtype reproduces the
    engine's quantization error in the trainer forward, keeping its logprobs
    aligned with the inference server's (the KV-cache analogue of router
    replay). The
    straight-through formulation keeps the backward exact: forward sees the
    quantized value, the gradient flows as if the cast were identity.
    """
    dtype = _KV_CACHE_DTYPE_MAP.get(kv_cache_dtype or "auto")
    if dtype is None or dtype == x.dtype:
        return x
    return x + (x.to(dtype).to(x.dtype) - x).detach()


@dataclass
class AttentionConfig:
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    is_causal: bool
    attention_bias: bool
    use_qk_norm: bool
    rms_norm_eps: float
    qk_norm_type: Literal["per_head", "per_layer"] = "per_head"
    output_bias: bool = False


# TODO: Does torch compile support config._attn_implementation forking?
# If so, we can combine FlashAttention variants into one class
# Otherwise, do ABC or something to make the signatures match


class FlashAttention(nn.Module):
    """Flash Attention"""

    supported_fusions = {"qkv": fuse_qkv_projections}

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__()
        self.head_dim = config.head_dim
        # Inference KV-cache storage dtype to replay in _compute_attention; set on
        # every FlashAttention module by setup_kv_cache_replay. None = no replay.
        self.kv_cache_dtype: str | None = None
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.qkv_sizes = (
            config.num_attention_heads * self.head_dim,
            config.num_key_value_heads * self.head_dim,
            config.num_key_value_heads * self.head_dim,
        )
        self.q_proj = nn.Linear(config.hidden_size, self.qkv_sizes[0], bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.qkv_sizes[1], bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.qkv_sizes[2], bias=config.attention_bias)
        self.register_module("qkv_proj", None)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.output_bias)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                self.q_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
                )
            else:
                self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
                self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query, key and value projections, from one packed GEMM when qkv is fused."""
        if self.qkv_proj is None:
            return self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        return self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, max_seqlen):
        """Run the flash attention kernel. q/k/v are [total_tokens, heads, dim]."""
        kv_cache_dtype = getattr(self, "kv_cache_dtype", None)
        if kv_cache_dtype == "fp8_kernel":
            # Kernel-level replay: run the engine's own fp8 flash-attn kernel on
            # unit-scale e4m3 Q/K/V, matching its forward numerics exactly.
            return _fp8kv_kernel_replay(q, k, v, cu_seqlens, max_seqlen)
        if kv_cache_dtype is not None:
            # Value-level replay: K is post-RoPE here, matching what vLLM quantizes
            # at cache-write time. Only K/V are replayed: although vLLM also quantizes
            # the query on the fp8 path, replaying Q adds uncorrelated bucket noise
            # against the engine's own Q values and measurably widens the mismatch.
            k = simulate_kv_cache_dtype(k, kv_cache_dtype)
            v = simulate_kv_cache_dtype(v, kv_cache_dtype)
        kwargs: dict = {"causal": True}
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            kwargs["window_size"] = (sliding_window - 1, 0)
        if self._flash_attn_version == 4:
            # FA4's flash_attn_varlen_func has qv as the 4th positional arg,
            # so cu_seqlens must be passed as keyword args to avoid misalignment.
            kwargs["cu_seqlens_q"] = cu_seqlens
            kwargs["cu_seqlens_k"] = cu_seqlens
            out, _ = self.func(q, k, v, **kwargs)
        else:
            out = self.func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.project_qkv(hidden_states)

        if self.use_qk_norm and self.qk_norm_type == "per_layer":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.use_qk_norm and self.qk_norm_type == "per_head":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary application instead of double transpose?
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        out = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)
        attn_output = out.contiguous().view(1, out.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


@torch._dynamo.disable
def _fp8kv_kernel_replay(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Run Fp8KVKernelAttention outside dynamo.

    The engine's fp8 flash-attn op declares mutating tensor args in its schema, which
    dynamo/AOT's functionalization cannot trace (it mangles the call into a pybind
    type error). Disabling this wrapper forces a clean graph break so the op always
    runs eagerly — inside compiled models too.
    """
    return Fp8KVKernelAttention.apply(q, k, v, cu_seqlens, max_seqlen)


def setup_kv_cache_replay(model: nn.Module, kv_cache_dtype: str | None) -> None:
    """Assign the simulated KV-cache storage dtype to every FlashAttention module.

    Mirrors setup_context_parallel's module-walk: modules hold the attribute with
    a None default and this walk sets it everywhere it applies. Models with
    custom attention paths (MLA, linear attention) keep the None default and
    simply skip the replay; their KV cache either does not exist (linear
    attention) or is quantized by a model-specific scheme this replay does not
    model.
    """
    if kv_cache_dtype is None or kv_cache_dtype == "auto":
        return
    for module in model.modules():
        if isinstance(module, FlashAttention):
            module.kv_cache_dtype = kv_cache_dtype


ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(FlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(FlashAttention, flash_attn_version=3),
    "flash_attention_4": functools.partial(FlashAttention, flash_attn_version=4),
}


def substitute_ring_attn(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
    attn_impl: str = "flash_attention_2",
) -> None:
    """Patch _compute_attention on FlashAttention variants to use ring attention."""
    from .ring_attn import ring_varlen_attention

    def _ring_compute_attention(self, q, k, v, cu_seqlens, max_seqlen):
        from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

        window_size = (-1, -1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            window_size = (sliding_window - 1, 0)

        out = ring_varlen_attention(
            q,
            k,
            v,
            cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
            cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
            max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
            max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
            local_k_slice=DATA_PARAMS["local_k_slice"],
            causal=True,
            window_size=window_size,
            group=process_group,
            heads_k_stride=heads_k_stride,
            attention_backend=attn_impl,
        )
        return out

    FlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    AfmoeFlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.gpt_oss.attention import substitute_gpt_oss_ring_attention

    substitute_gpt_oss_ring_attention(process_group, heads_k_stride)
