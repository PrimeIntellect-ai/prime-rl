from dataclasses import dataclass

import torch
from torch import nn

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer
from prime_rl.trainer.models.layers.norms import LayerNorm, RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb_interleave

try:
    from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
    from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface
except ImportError:
    sparse_mla_fwd_interface = None  # type: ignore
    sparse_mla_bwd = None  # type: ignore


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
    """Autograd wrapper for tilelang sparse MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, out, indices, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, out, indices, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(q, kv, out, do.contiguous(), indices, lse, sm_scale=ctx.sm_scale)
        return dq, dkv, None, None


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
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        index_topk: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[1]
        assert index_topk % 64 == 0, f"index_topk must be divisible by 64 (block_I), got {index_topk}"

        q_idx = self.wq_b(q_latent[0]).view(total_tokens, self.n_head, self.head_dim)
        k_idx = self.k_norm(self.wk(hidden_states[0]))
        w = self.weights_proj(hidden_states[0])

        q_pe = q_idx[..., : self.rope_dim]
        q_nope = q_idx[..., self.rope_dim :]
        k_pe = k_idx[..., : self.rope_dim]
        k_nope = k_idx[..., self.rope_dim :]

        cos, sin = position_embeddings
        q_pe = q_pe.unsqueeze(0).transpose(1, 2)
        k_pe = k_pe.unsqueeze(0).unsqueeze(1)
        q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin)
        q_pe = q_pe.transpose(1, 2).squeeze(0)
        k_pe = k_pe.squeeze(1).squeeze(0)

        q_idx = torch.cat([q_pe, q_nope], dim=-1)
        k_idx = torch.cat([k_pe, k_nope], dim=-1)

        indices = fp8_indexer(q_idx, k_idx, w, ks, ke, index_topk, self.weight_scale)
        return indices.view(1, total_tokens, 1, index_topk)


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

    def attn_projections(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return q_latent, self.kv_a_layernorm(k_compressed), k_rope

    def mla_up_proj(
        self,
        q_latent: torch.Tensor,
        k_compressed_normed: torch.Tensor,
        k_rope: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, total_tokens, _ = q_latent.shape
        q_full = self.q_b_proj(q_latent).view(batch_size, total_tokens, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_rope_r = q_rope.transpose(1, 2)
        k_rope_r = k_rope.unsqueeze(1)
        cos, sin = position_embeddings
        q_rope_r, k_rope_r = apply_rotary_pos_emb_interleave(q_rope_r, k_rope_r, cos, sin)
        q_rope = q_rope_r.transpose(1, 2)
        k_rope = k_rope_r.squeeze(1)

        kv_b_w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = kv_b_w[:, : self.qk_nope_head_dim, :]
        w_v = kv_b_w[:, self.qk_nope_head_dim :, :]
        q_absorbed = torch.einsum("bshd,hdk->bshk", q_nope, w_k_nope)

        sparse_q = torch.cat([q_absorbed, q_rope], dim=-1)
        sparse_kv = torch.cat([k_compressed_normed, k_rope], dim=-1).unsqueeze(2)

        sentinel = torch.zeros(batch_size, 1, 1, sparse_kv.shape[-1], dtype=sparse_kv.dtype, device=sparse_kv.device)
        sparse_kv = torch.cat([sparse_kv, sentinel], dim=1)
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

        indices = self.indexer.compute_sparse_indices(
            hidden_states, q_latent, ks, ke, self.args.index_topk, position_embeddings
        )

        sparse_q, sparse_kv, w_v = self.mla_up_proj(
            q_latent,
            k_compressed_normed,
            k_rope,
            position_embeddings=position_embeddings,
        )

        out = _SparseMLA.apply(sparse_q, sparse_kv, indices, self.scaling)
        return self.output_proj(out, w_v), None
