"""Shared DeepSeek Sparse Attention (DSA) building blocks.

Architecture-agnostic pieces of DSA: the lightning `Indexer`, the sparse MLA
autograd wrapper, and the combined `SparseMlaAttention` module (MLA + indexer +
top-k sparse attention, with a dense-attention fallback used to convert an
existing dense-MLA checkpoint to DSA via continued pretraining). Any MLA-based
family (GLM's ``glm_moe_dsa``, DeepSeek-V3, Kimi K2, ...) can build on this
directly; only the surrounding decoder-layer/config plumbing is per-family.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer
from prime_rl.trainer.models.layers.norms import LayerNorm, RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
from prime_rl.utils.cp import gather_for_cp
from prime_rl.utils.vlm import get_language_model

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
    use_index_cache: bool = False
    skip_topk: bool = False
    use_sparse_attn: bool = True
    """Whether to attend only over the indexer's top-k selection (DSA) or run ordinary
    dense causal attention over the full sequence. Dense mode is used only to convert
    an existing dense-MLA checkpoint to DSA: the model runs (and is trained) exactly as
    it was before conversion, while the indexer is separately trained (via `train_indexer`)
    to predict which keys the real dense attention would attend to."""
    train_indexer: bool = False
    """Compute and stash a differentiable indexer-vs-attention KL term this forward pass
    (see `compute_indexer_kl_loss`). Off by default: zero overhead for ordinary training
    and inference."""


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


def apply_rope_interleave_single(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = t.shape
    t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return (t * cos) + (rotate_half(t) * sin)


def _apply_rope_qk(
    q_rope: torch.Tensor,
    k_rope_full: torch.Tensor,
    cos_q: torch.Tensor,
    sin_q: torch.Tensor,
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for the MLA query (locally-sharded under CP) and key (always full-sequence,
    shared across heads) halves. Shared by `mla_up_proj` (CP-aware: `cos_q`/`sin_q` may be
    a CP-local slice of `cos_full`/`sin_full`) and `_dense_attention` (no CP support, so
    callers pass `cos_full`/`sin_full` for both)."""
    q_rope_r = apply_rope_interleave_single(q_rope.transpose(1, 2), cos_q, sin_q)  # [B, H, S_local, rope_dim]
    q_rope = q_rope_r.transpose(1, 2)

    k_rope_r = apply_rope_interleave_single(k_rope_full.unsqueeze(1), cos_full, sin_full)  # [B, 1, S_full, rope_dim]
    k_rope_full = k_rope_r.squeeze(1)
    return q_rope, k_rope_full


def _valid_key_mask(n_keys: int, ks: torch.Tensor, ke: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Causal + document-boundary mask from packed-sequence bounds: key `s` is valid for
    query `t` iff `ks[t] <= s < ke[t]`."""
    key_idx = torch.arange(n_keys, device=device)
    return (key_idx.unsqueeze(0) >= ks.unsqueeze(1)) & (key_idx.unsqueeze(0) < ke.unsqueeze(1))


def _heads_to_target_distribution(probs: torch.Tensor) -> torch.Tensor:
    """Indexer-KL reference target: per-head attention probabilities (head dim 1) summed
    across heads and L1-renormalized, detached (the paper trains the indexer against the
    real attention distribution without letting that supervision backprop into the base
    model's own attention weights)."""
    target = probs.sum(dim=1)
    return (target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)).detach()


def _masked_kl_div(log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || exp(log_probs)), summed over keys and averaged over queries.

    Both `target` and `log_probs` are exactly 0 / -inf outside the causal/varlen window.
    Naively computing `target * (target.log() - log_probs)` gives `0 * (-inf - -inf) = nan`
    there despite the true limit being 0 — mask before multiplying rather than relying on
    `F.kl_div`'s own zero handling. `target` is already exactly 0 wherever masked, so the
    product needs no further masking once its factors are safe.
    """
    valid = target > 0
    safe_log_target = torch.where(valid, target.clamp_min(1e-12).log(), torch.zeros_like(target))
    safe_log_probs = torch.where(valid, log_probs, torch.zeros_like(log_probs))
    kl_per_key = target * (safe_log_target - safe_log_probs)
    return kl_per_key.sum(dim=-1).mean()


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

    def _init_indexer_parameters(self, generator: torch.Generator | None = None, std: float = 0.02) -> None:
        """Re-initialize a freshly-bootstrapped indexer's weights.

        A dense predecessor checkpoint has no `indexer.*` keys at all — mirrors LoRA's
        post-load `_init_lora_parameters` (`trainer/lora.py`), called the same way for the
        same reason: meta-device materialization leaves params `dcp_load` never wrote as
        uninitialized garbage, not a sane random init, so skipping this after
        `strip_indexer_from_state_dict` would silently train from noise.
        """
        for linear in (self.wq_b, self.wk, self.weights_proj):
            nn.init.normal_(linear.weight, mean=0.0, std=std, generator=generator)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.ones_(self.k_norm.weight)
        nn.init.zeros_(self.k_norm.bias)

    def _project_and_rope(
        self,
        hidden_states_local: torch.Tensor,
        q_latent_local: torch.Tensor,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
        cp_group: dist.ProcessGroup | None,
        cp_world_size: int,
        cp_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared query/key projection + RoPE prep for `compute_sparse_indices` and `score`.

        Returns `(q_idx, k_idx, w)`: post-RoPE query `[s_local, n_head, head_dim]`, post-RoPE
        key `[S_full, head_dim]` (shared across heads), and per-head weights `[s_local, n_head]`.
        """
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
        return q_idx, k_idx, w

    def _logits_from_projections(
        self, q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable indexer logits `I[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s])` from
        already-projected q/k/w — same formula `fp8_indexer` selects top-k with (its own
        `weight_scale` factor is a no-op there too: folded fp8 quantization scales cancel it
        out, see `fp8_indexer.fp8_indexer`'s comment), computed here in full precision."""
        raw = F.relu(torch.einsum("thd,sd->ths", q_idx, k_idx))  # [s_local, H, S_full]
        logits = torch.einsum("ths,th->ts", raw, w)  # [s_local, S_full]
        valid = _valid_key_mask(k_idx.shape[0], ks, ke, logits.device)
        return logits.masked_fill(~valid, float("-inf"))

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
        q_idx, k_idx, w = self._project_and_rope(
            hidden_states_local, q_latent_local, position_embeddings_full, cp_group, cp_world_size, cp_rank
        )
        indices = fp8_indexer(q_idx, k_idx, w, ks, ke, index_topk, self.weight_scale)
        # indices shape: [S_local, topk] in K's coordinate space (sentinel = s_full)
        # KV passed to sparse MLA has length s_full + 1 (sentinel zeros at index s_full).
        return indices.view(1, s_local, 1, index_topk)

    def compute_indices_and_logits(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`compute_sparse_indices` + `score` combined, projecting q/k/w once instead of
        twice. Used when both `use_sparse_attn` and `train_indexer` are on (stage B:
        sparse-adaptation), the only case where both the real top-k selection and the
        differentiable KL logits are needed from the same forward pass."""
        assert index_topk % 64 == 0, f"index_topk must be divisible by 64 (block_I), got {index_topk}"

        s_local = hidden_states_local.shape[1]
        q_idx, k_idx, w = self._project_and_rope(
            hidden_states_local, q_latent_local, position_embeddings_full, cp_group, cp_world_size, cp_rank
        )
        with torch.no_grad():
            indices = fp8_indexer(q_idx.detach(), k_idx.detach(), w.detach(), ks, ke, index_topk, self.weight_scale)
        logits = self._logits_from_projections(q_idx, k_idx, w, ks, ke)
        return indices.view(1, s_local, 1, index_topk), logits

    def score(
        self,
        hidden_states_local: torch.Tensor,
        q_latent_local: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
        cp_group: dist.ProcessGroup | None,
        cp_world_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        """Differentiable indexer logits, kept in the autograd graph so a KL loss against a
        real attention distribution can train the indexer (see `_logits_from_projections`).
        Used standalone in dense mode (stage A: indexer warm-up); `compute_indices_and_logits`
        covers the sparse-mode case where the real top-k selection is also needed."""
        q_idx, k_idx, w = self._project_and_rope(
            hidden_states_local, q_latent_local, position_embeddings_full, cp_group, cp_world_size, cp_rank
        )
        return self._logits_from_projections(q_idx, k_idx, w, ks, ke)


class SparseMlaAttention(nn.Module):
    """DeepSeek/GLM-style MLA attention with a DSA lightning indexer.

    `use_sparse_attn` (per `SparseMlaAttentionArgs`) switches between the DSA top-k sparse
    path and an ordinary dense-causal fallback over the same MLA projections — the fallback
    is what lets a dense-MLA checkpoint train under this exact module during DSA conversion.
    """

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
        # IndexShare (GLM-5.2): layers that reuse cached indices carry no indexer weights.
        self.indexer = Indexer(args) if not args.skip_topk else None
        self.use_index_cache = args.use_index_cache
        self.skip_topk = args.skip_topk
        self.use_sparse_attn = args.use_sparse_attn
        self.train_indexer = args.train_indexer
        self.scaling = self.qk_head_dim ** (-0.5)

        # Stashed here (not returned from forward()) during a train_indexer pass, mirroring
        # how MoE load-balance stats are stashed on the router module. Read back by
        # `compute_indexer_kl_loss`, which walks `language_model.layers`.
        self._indexer_kl_loss: torch.Tensor | None = None

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

    def _kv_b_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Split `kv_b_proj`'s weight into its nope-K and V per-head slices."""
        kv_b_w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = kv_b_w[:, : self.qk_nope_head_dim, :]
        w_v = kv_b_w[:, self.qk_nope_head_dim :, :]
        return w_k_nope, w_v

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

        q_rope, k_rope_full = _apply_rope_qk(q_rope, k_rope_full, cos_local, sin_local, cos_full, sin_full)

        w_k_nope, w_v = self._kv_b_weights()
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

    def _dense_attention(
        self,
        q_latent_local: torch.Tensor,
        k_compressed_normed_full: torch.Tensor,
        k_rope_full: torch.Tensor,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor,
        ke: torch.Tensor,
        need_target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Ordinary (non-absorbed, non-sparse) causal MLA attention.

        Decompresses `kv_b_proj` into real per-head nope-K/V (rather than folding it into
        Q for the latent-space "weight absorption" trick the sparse path uses) and runs a
        manual masked softmax so the per-token, per-head attention distribution is available
        for the indexer's KL-loss target (only when `need_target`). Used only for DSA
        conversion's dense-mode stage (short, cheap by design — see docs), so the O(S^2)
        score matrix is intentional, not an oversight. CP is not supported in dense mode
        (raising here is a last resort — `setup_sparse_mla_cp` refuses this combination at
        setup time instead, before training starts).
        """
        if self.cp_enabled:
            raise NotImplementedError("Dense-mode DSA-conversion attention does not support context parallelism yet.")

        batch_size, s_local, _ = q_latent_local.shape
        assert batch_size == 1, "SparseMlaAttention assumes a single packed sequence (batch_size == 1)."
        s_full = k_compressed_normed_full.shape[1]
        assert s_local == s_full, "Dense mode requires the full (unsharded) sequence."

        q_full = self.q_b_proj(q_latent_local).view(batch_size, s_local, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        cos_full, sin_full = position_embeddings_full
        q_rope, k_rope_full = _apply_rope_qk(q_rope, k_rope_full, cos_full, sin_full, cos_full, sin_full)

        w_k_nope, w_v = self._kv_b_weights()
        k_nope = torch.einsum("bsk,hdk->bshd", k_compressed_normed_full, w_k_nope)
        v = torch.einsum("bsk,hdk->bshd", k_compressed_normed_full, w_v)

        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, S, H, D]
        k_rope_expanded = k_rope_full.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k = torch.cat([k_nope, k_rope_expanded], dim=-1)  # [B, S, H, D]

        valid = _valid_key_mask(s_full, ks, ke, q.device)

        # [B, H, S, S]: manual (not fused) so per-head softmax weights are available below.
        scores = torch.einsum("bshd,bthd->bhst", q, k).float() * self.scaling
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_probs = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum("bhst,bthd->bshd", attn_probs.to(v.dtype), v)

        target = _heads_to_target_distribution(attn_probs) if need_target else None

        attn_output = attn_output.reshape(batch_size, s_local, -1)
        return self.o_proj(attn_output), target

    def _sparse_indexer_kl_target(
        self, sparse_q: torch.Tensor, sparse_kv: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Reference distribution for Stage B (sparse-adaptation): softmax restricted to the
        actually-selected top-k keys, computed in the same absorbed latent space + scale the
        real sparse kernel uses. Cheap: only `index_topk` keys per query, independent of
        context length."""
        # sparse_q: [1, S, H, D], sparse_kv: [1, S_full+1, 1, D], indices: [1, S, 1, topk]
        idx = indices.squeeze(0).squeeze(1)  # [S, topk]
        sentinel_id = sparse_kv.shape[1] - 1
        gathered = sparse_kv.squeeze(0).squeeze(1)[idx.clamp_max(sentinel_id)]  # [S, topk, D]
        scores = torch.einsum("shd,skd->shk", sparse_q.squeeze(0), gathered).float() * self.scaling
        scores = scores.masked_fill((idx == sentinel_id).unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)  # [S, H, topk]
        return _heads_to_target_distribution(probs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None = None,
        ke: torch.Tensor | None = None,
        cached_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._indexer_kl_loss = None
        compute_kl = self.train_indexer and not self.skip_topk

        q_latent, k_compressed_normed, k_rope = self.attn_projections(hidden_states)

        if self.cp_enabled:
            k_compressed_normed = gather_for_cp(k_compressed_normed, self._cp_group)
            k_rope = gather_for_cp(k_rope, self._cp_group)

        logits = None
        indices = cached_indices
        if not self.skip_topk and self.use_sparse_attn:
            if compute_kl:
                indices, logits = self.indexer.compute_indices_and_logits(
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
            else:
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

        if self.use_sparse_attn:
            sparse_q, sparse_kv, w_v = self.mla_up_proj(
                q_latent_local=q_latent,
                k_compressed_normed_full=k_compressed_normed,
                k_rope_full=k_rope,
                position_embeddings_full=position_embeddings,
            )
            out = _SparseMLA.apply(sparse_q, sparse_kv, indices, self.scaling)
            attn_output = self.output_proj(out, w_v)
            target = self._sparse_indexer_kl_target(sparse_q, sparse_kv, indices) if compute_kl else None
        else:
            attn_output, target = self._dense_attention(
                q_latent, k_compressed_normed, k_rope, position_embeddings, ks, ke, need_target=compute_kl
            )
            if compute_kl:
                logits = self.indexer.score(
                    hidden_states_local=hidden_states,
                    q_latent_local=q_latent,
                    ks=ks,
                    ke=ke,
                    position_embeddings_full=position_embeddings,
                    cp_group=self._cp_group,
                    cp_world_size=self._cp_world_size,
                    cp_rank=self._cp_rank,
                )

        if compute_kl and target is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            self._indexer_kl_loss = _masked_kl_div(log_probs, target)

        cached_indices = indices if self.use_index_cache else None
        return attn_output, cached_indices


def compute_indexer_kl_loss(model: nn.Module, coeff: float) -> torch.Tensor | None:
    """Sum the per-layer indexer-vs-attention KL terms stashed by `SparseMlaAttention.forward`
    (only populated when `train_indexer=True`), mirroring how MoE load-balance stats are
    stashed on each router and later walked/summed. Returns `None` if no layer produced a term."""
    language_model = get_language_model(model)
    terms = []
    for layer in language_model.layers:
        self_attn = getattr(layer, "self_attn", None)
        kl = getattr(self_attn, "_indexer_kl_loss", None)
        if kl is not None:
            terms.append(kl)
    if not terms:
        return None
    return coeff * torch.stack(terms).sum()


def strip_indexer_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip `indexer.*` keys from a state dict, mirroring `strip_lora_from_state_dict`
    (`trainer/lora.py`). Bootstrapping a dense checkpoint's weights into a DSA-capable model
    class means the indexer's keys are absent from the source entirely (a dense predecessor
    never had one) — without stripping them first, `load_dcp_from_hf`'s strict DCP load
    raises `Missing key in checkpoint state_dict`. Pair with `Indexer._init_indexer_parameters`
    (called the same way `_init_lora_parameters` is, right after the load succeeds) to give
    the stripped params a real init instead of leaving them as meta-materialized garbage."""
    return {key: value for key, value in state_dict.items() if "indexer." not in key}


__all__ = [
    "SparseMlaAttentionArgs",
    "Indexer",
    "SparseMlaAttention",
    "compute_indexer_kl_loss",
    "strip_indexer_from_state_dict",
]
