import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer
from prime_rl.trainer.models.layers.norms import LayerNorm, RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
from prime_rl.utils.cp import gather_for_cp

try:
    from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd as sparse_mla_bwd_tilelang
    from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface as sparse_mla_fwd_tilelang
except ImportError:
    sparse_mla_fwd_tilelang = None  # type: ignore
    sparse_mla_bwd_tilelang = None  # type: ignore

try:
    from prime_rl.trainer.models.kernels.sparse_mla_bucketed import sparse_mla_bwd_bucketed
    from prime_rl.trainer.models.kernels.sparse_mla_bucketed import sparse_mla_bwd_bucketed_sorted
    from prime_rl.trainer.models.kernels.sparse_mla_bucketed import sparse_mla_fwd_bucketed
except ImportError:
    sparse_mla_fwd_bucketed = None  # type: ignore
    sparse_mla_bwd_bucketed = None  # type: ignore
    sparse_mla_bwd_bucketed_sorted = None  # type: ignore

try:
    from prime_rl.trainer.models.kernels.sparse_mla_triton import sparse_mla_bwd as sparse_mla_bwd_triton
    from prime_rl.trainer.models.kernels.sparse_mla_triton import sparse_mla_fwd_interface as sparse_mla_fwd_triton
except ImportError:
    sparse_mla_fwd_triton = None  # type: ignore
    sparse_mla_bwd_triton = None  # type: ignore


def _select_sparse_mla_kernel():
    kernel = os.environ.get("PRIME_RL_SPARSE_MLA_KERNEL", "tilelang_bucketed_gemm_v1").lower()
    if kernel == "triton":
        if sparse_mla_fwd_triton is None or sparse_mla_bwd_triton is None:
            raise ImportError("PRIME_RL_SPARSE_MLA_KERNEL=triton but Triton sparse MLA kernels are unavailable")
        return sparse_mla_fwd_triton, sparse_mla_bwd_triton, "none"
    if kernel == "tilelang":
        if sparse_mla_fwd_tilelang is None or sparse_mla_bwd_tilelang is None:
            raise ImportError("PRIME_RL_SPARSE_MLA_KERNEL=tilelang but TileLang sparse MLA kernels are unavailable")
        return sparse_mla_fwd_tilelang, sparse_mla_bwd_tilelang, "none"
    if kernel == "tilelang_gemm_v1":
        if sparse_mla_fwd_tilelang is None or sparse_mla_bwd_tilelang is None:
            raise ImportError("PRIME_RL_SPARSE_MLA_KERNEL=tilelang_gemm_v1 but TileLang sparse MLA kernels are unavailable")

        def sparse_mla_bwd_tilelang_gemm_v1(q, kv, out, do, indices, lse, sm_scale=None):
            return sparse_mla_bwd_tilelang(
                q,
                kv,
                out,
                do,
                indices,
                lse,
                sm_scale=sm_scale,
                block_size=32,
                split_store=4,
                use_gemm_v1=True,
            )

        return sparse_mla_fwd_tilelang, sparse_mla_bwd_tilelang_gemm_v1, "none"
    if kernel == "tilelang_bucketed_gemm_v1":
        if sparse_mla_fwd_bucketed is None or sparse_mla_bwd_bucketed is None:
            raise ImportError(
                "PRIME_RL_SPARSE_MLA_KERNEL=tilelang_bucketed_gemm_v1 but bucketed TileLang sparse MLA kernels are unavailable"
            )
        return sparse_mla_fwd_bucketed, sparse_mla_bwd_bucketed, "none"
    if kernel == "tilelang_bucketed_bwd_sorted_gemm_v1":
        if sparse_mla_fwd_bucketed is None or sparse_mla_bwd_bucketed_sorted is None:
            raise ImportError(
                "PRIME_RL_SPARSE_MLA_KERNEL=tilelang_bucketed_bwd_sorted_gemm_v1 but bucketed TileLang sparse MLA kernels are unavailable"
            )
        return sparse_mla_fwd_bucketed, sparse_mla_bwd_bucketed_sorted, "none"
    if kernel == "tilelang_sorted_bs64":
        if sparse_mla_fwd_tilelang is None or sparse_mla_bwd_tilelang is None:
            raise ImportError(
                "PRIME_RL_SPARSE_MLA_KERNEL=tilelang_sorted_bs64 but TileLang sparse MLA kernels are unavailable"
            )

        def sparse_mla_bwd_tilelang_bs64(q, kv, out, do, indices, lse, sm_scale=None):
            return sparse_mla_bwd_tilelang(q, kv, out, do, indices, lse, sm_scale=sm_scale, block_size=64)

        return sparse_mla_fwd_tilelang, sparse_mla_bwd_tilelang_bs64, "sort"
    if kernel == "tilelang_sorted_gemm_v1":
        if sparse_mla_fwd_tilelang is None or sparse_mla_bwd_tilelang is None:
            raise ImportError(
                "PRIME_RL_SPARSE_MLA_KERNEL=tilelang_sorted_gemm_v1 but TileLang sparse MLA kernels are unavailable"
            )

        def sparse_mla_bwd_tilelang_gemm_v1(q, kv, out, do, indices, lse, sm_scale=None):
            return sparse_mla_bwd_tilelang(
                q,
                kv,
                out,
                do,
                indices,
                lse,
                sm_scale=sm_scale,
                block_size=32,
                split_store=4,
                use_gemm_v1=True,
            )

        return sparse_mla_fwd_tilelang, sparse_mla_bwd_tilelang_gemm_v1, "sort"
    if kernel == "tilelang_bwd_sorted_gemm_v1":
        if sparse_mla_fwd_tilelang is None or sparse_mla_bwd_tilelang is None:
            raise ImportError(
                "PRIME_RL_SPARSE_MLA_KERNEL=tilelang_bwd_sorted_gemm_v1 but TileLang sparse MLA kernels are unavailable"
            )

        def sparse_mla_bwd_tilelang_bwd_sorted(q, kv, out, do, indices, lse, sm_scale=None):
            return sparse_mla_bwd_tilelang(
                q,
                kv,
                out,
                do,
                indices,
                lse,
                sm_scale=sm_scale,
                block_size=32,
                split_store=4,
                use_gemm_v1=True,
            )

        return sparse_mla_fwd_tilelang, sparse_mla_bwd_tilelang_bwd_sorted, "bwd_sort"
    if kernel in {"tilelang_trim_gemm_v1", "tilelang_trim_bwd_sorted_gemm_v1"}:
        if sparse_mla_fwd_tilelang is None or sparse_mla_bwd_tilelang is None:
            raise ImportError(
                f"PRIME_RL_SPARSE_MLA_KERNEL={kernel} but TileLang sparse MLA kernels are unavailable"
            )

        def sparse_mla_bwd_tilelang_trim(q, kv, out, do, indices, lse, sm_scale=None):
            return sparse_mla_bwd_tilelang(
                q,
                kv,
                out,
                do,
                indices,
                lse,
                sm_scale=sm_scale,
                block_size=32,
                split_store=4,
                use_gemm_v1=True,
            )

        index_mode = "trim_bwd_sort" if kernel == "tilelang_trim_bwd_sorted_gemm_v1" else "trim"
        return sparse_mla_fwd_tilelang, sparse_mla_bwd_tilelang_trim, index_mode
    raise ValueError(
        f"Unsupported PRIME_RL_SPARSE_MLA_KERNEL={kernel!r}; "
        "expected 'tilelang', 'tilelang_gemm_v1', 'tilelang_bucketed_gemm_v1', "
        "'tilelang_bucketed_bwd_sorted_gemm_v1', "
        "'tilelang_sorted_bs64', 'tilelang_sorted_gemm_v1', "
        "'tilelang_bwd_sorted_gemm_v1', 'tilelang_trim_gemm_v1', "
        "'tilelang_trim_bwd_sorted_gemm_v1', or 'triton'"
    )


def _trim_trailing_invalid_blocks(indices: torch.Tensor, kv_seq_len: int, block_size: int = 64) -> torch.Tensor:
    max_valid_index = kv_seq_len - 2
    max_valid = int((indices <= max_valid_index).sum(dim=-1).max().item())
    effective_topk = max(block_size, min(indices.shape[-1], ((max_valid + block_size - 1) // block_size) * block_size))
    if effective_topk == indices.shape[-1]:
        return indices
    return indices[..., :effective_topk].contiguous()


@dataclass(frozen=True)
class SparseMlaAttentionArgs:
    hidden_size: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    attention_bias: bool
    rms_norm_eps: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int


class _SparseMLA(torch.autograd.Function):
    """Autograd wrapper for sparse MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        sparse_mla_fwd_interface, sparse_mla_bwd, index_mode = _select_sparse_mla_kernel()
        if index_mode == "sort":
            indices = torch.sort(indices, dim=-1).values.contiguous()
            bwd_indices = indices
        elif index_mode in {"trim", "trim_bwd_sort"}:
            indices = _trim_trailing_invalid_blocks(indices, kv.shape[1], block_size=64)
            bwd_indices = indices
            if index_mode == "trim_bwd_sort":
                bwd_indices = torch.sort(indices, dim=-1).values.contiguous()
        elif index_mode == "bwd_sort":
            bwd_indices = torch.sort(indices, dim=-1).values.contiguous()
        else:
            bwd_indices = indices
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, out, bwd_indices, lse)
        ctx.sm_scale = sm_scale
        ctx.sparse_mla_bwd = sparse_mla_bwd
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, out, indices, lse = ctx.saved_tensors
        sparse_mla_bwd = ctx.sparse_mla_bwd
        dq, dkv = sparse_mla_bwd(q, kv, out, do.contiguous(), indices, lse, sm_scale=ctx.sm_scale)
        return dq, dkv, None, None


def apply_rope_interleave_single(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = t.shape
    t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return (t * cos) + (rotate_half(t) * sin)


class Indexer(nn.Module):
    def __init__(self, args: SparseMlaAttentionArgs):
        super().__init__()
        self.n_head = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=args.attention_bias)
        self.k_norm = LayerNorm(dim=self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_head, bias=False)
        self.weight_scale = (self.head_dim**-0.5) * (self.n_head**-0.5)

    @torch.no_grad()
    def compute_sparse_indices(
        self,
        hidden_states_local: torch.Tensor,
        q_latent_local: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        index_topk: int,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
        cp_group: dist.ProcessGroup | None,
        cp_world_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        assert index_topk % 64 == 0, f"index_topk must be divisible by 64 (block_I), got {index_topk}"

        s_local = hidden_states_local.shape[1]

        q_idx = self.wq_b(q_latent_local[0]).view(s_local, self.n_head, self.head_dim)
        k_idx_local = self.k_norm(self.wk(hidden_states_local[0]))
        w = self.weights_proj(hidden_states_local[0])

        if cp_world_size > 1:
            k_idx = gather_for_cp(k_idx_local.unsqueeze(0), cp_group).squeeze(0)
        else:
            k_idx = k_idx_local

        cos_full, sin_full = position_embeddings_full
        if cp_world_size > 1:
            cos_local = cos_full[:, cp_rank * s_local : (cp_rank + 1) * s_local, :]
            sin_local = sin_full[:, cp_rank * s_local : (cp_rank + 1) * s_local, :]
        else:
            cos_local, sin_local = cos_full, sin_full

        q_pe = q_idx[..., : self.rope_dim]
        q_nope = q_idx[..., self.rope_dim :]
        k_pe = k_idx[..., : self.rope_dim]
        k_nope = k_idx[..., self.rope_dim :]

        q_pe = q_pe.unsqueeze(0).transpose(1, 2)  # [1, H, S_local, rope_dim]
        q_pe = apply_rope_interleave_single(q_pe, cos_local, sin_local)
        q_pe = q_pe.transpose(1, 2).squeeze(0)

        k_pe = k_pe.unsqueeze(0).unsqueeze(1)  # [1, 1, S_full, rope_dim]
        k_pe = apply_rope_interleave_single(k_pe, cos_full, sin_full)
        k_pe = k_pe.squeeze(1).squeeze(0)

        q_idx = torch.cat([q_pe, q_nope], dim=-1)
        k_idx = torch.cat([k_pe, k_nope], dim=-1)

        indices = fp8_indexer(q_idx, k_idx, w, ks, ke, index_topk, self.weight_scale)
        # indices shape: [S_local, topk] in K's coordinate space (sentinel = s_full)
        # KV passed to sparse MLA has length s_full + 1 (sentinel zeros at index s_full).
        return indices.view(1, s_local, 1, index_topk)


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, args: SparseMlaAttentionArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim

        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=args.attention_bias)
        self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=args.q_lora_rank, eps=args.rms_norm_eps))
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=self.kv_lora_rank, eps=args.rms_norm_eps))
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, args.hidden_size, bias=args.attention_bias)
        self.indexer = Indexer(args)
        self.scaling = self.qk_head_dim ** (-0.5)

        self._cp_group: dist.ProcessGroup | None = None
        self._cp_rank: int = 0
        self._cp_world_size: int = 1

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def attn_projections(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return q_latent, self.kv_a_layernorm(k_compressed), k_rope

    def mla_up_proj(
        self,
        q_latent_local: torch.Tensor,
        k_compressed_normed_full: torch.Tensor,
        k_rope_full: torch.Tensor,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, s_local, _ = q_latent_local.shape
        s_full = k_compressed_normed_full.shape[1]

        q_full = self.q_b_proj(q_latent_local).view(batch_size, s_local, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        cos_full, sin_full = position_embeddings_full
        if self.cp_enabled:
            cos_local = cos_full[:, self._cp_rank * s_local : (self._cp_rank + 1) * s_local, :]
            sin_local = sin_full[:, self._cp_rank * s_local : (self._cp_rank + 1) * s_local, :]
        else:
            cos_local, sin_local = cos_full, sin_full

        q_rope_r = q_rope.transpose(1, 2)  # [B, H, S_local, rope_dim]
        q_rope_r = apply_rope_interleave_single(q_rope_r, cos_local, sin_local)
        q_rope = q_rope_r.transpose(1, 2)

        k_rope_r = k_rope_full.unsqueeze(1)  # [B, 1, S_full, rope_dim]
        k_rope_r = apply_rope_interleave_single(k_rope_r, cos_full, sin_full)
        k_rope_full = k_rope_r.squeeze(1)

        kv_b_w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = kv_b_w[:, : self.qk_nope_head_dim, :]
        w_v = kv_b_w[:, self.qk_nope_head_dim :, :]
        q_absorbed = torch.einsum("bshd,hdk->bshk", q_nope, w_k_nope)

        sparse_q = torch.cat([q_absorbed, q_rope], dim=-1)
        sparse_kv = torch.cat([k_compressed_normed_full, k_rope_full], dim=-1).unsqueeze(2)

        sentinel = torch.zeros(batch_size, 1, 1, sparse_kv.shape[-1], dtype=sparse_kv.dtype, device=sparse_kv.device)
        sparse_kv = torch.cat([sparse_kv, sentinel], dim=1)
        assert sparse_kv.shape[1] == s_full + 1
        return sparse_q, sparse_kv, w_v

    def _mla_unabsorb(self, out: torch.Tensor, w_v: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bshk,hdk->bshd", out, w_v)

    def output_proj(self, attn_output: torch.Tensor, w_v: torch.Tensor) -> torch.Tensor:
        attn_output = self._mla_unabsorb(attn_output, w_v)
        batch_size, total_tokens = attn_output.shape[:2]
        attn_output = attn_output.reshape(batch_size, total_tokens, -1)
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None = None,
        ke: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_latent, k_compressed_normed, k_rope = self.attn_projections(hidden_states)

        if self.cp_enabled:
            k_compressed_normed = gather_for_cp(k_compressed_normed, self._cp_group)
            k_rope = gather_for_cp(k_rope, self._cp_group)

        indices = self.indexer.compute_sparse_indices(
            hidden_states_local=hidden_states,
            q_latent_local=q_latent,
            ks=ks,
            ke=ke,
            index_topk=self.args.index_topk,
            position_embeddings_full=position_embeddings,
            cp_group=self._cp_group,
            cp_world_size=self._cp_world_size,
            cp_rank=self._cp_rank,
        )

        sparse_q, sparse_kv, w_v = self.mla_up_proj(
            q_latent_local=q_latent,
            k_compressed_normed_full=k_compressed_normed,
            k_rope_full=k_rope,
            position_embeddings_full=position_embeddings,
        )

        out = _SparseMLA.apply(sparse_q, sparse_kv, indices, self.scaling)
        return self.output_proj(out, w_v), None
