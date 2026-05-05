"""Ulysses-style context parallelism via all-to-all on the head/seq dimensions.

Ulysses is the simpler alternative to ring attention for context parallelism:

    sequence-sharded Q/K/V                 head-sharded Q/K/V
    [S/cp, H,    D]   ── all-to-all ──▶   [S, H/cp, D]
                         (heads ↔ seq)
                              │
                              ▼
                       run vanilla local attention
                       on the *full* sequence with
                       *fewer* heads (any kernel works)
                              │
                              ▼
    [S/cp, H,    D]  ◀── all-to-all ──   [S, H/cp, D]
                         (seq ↔ heads)

Key benefit: the attention kernel itself does not need to be CP-aware. The
all-to-all is purely on Q/K/V tensors, so this works out of the box with
softmax flash-attn, linear attention, mamba, etc., without rewriting kernels.

Constraints:
- cp_size must divide both num_attention_heads and num_key_value_heads.
- Sequence length must be divisible by cp_size.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

# Populated by `update_ulysses_params` before each forward pass. Mirrors
# ring_flash_attn's DATA_PARAMS pattern so the patched attention path can
# reach the *full* (un-sharded) cu_seqlens / max_seqlen at call time.
ULYSSES_PARAMS: dict = {}


def update_ulysses_params(cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
    ULYSSES_PARAMS["cu_seqlens"] = cu_seqlens
    ULYSSES_PARAMS["max_seqlen"] = int(max_seqlen)


def _all_to_all_seq_to_head(
    t: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Redistribute [S_local, H, D] -> [S_global, H_local, D].

    Splits the head dim into cp_size groups and exchanges them so each rank
    ends up with the full sequence but only H/cp_size heads. Differentiable.
    """
    s_local, h, d = t.shape
    assert h % cp_size == 0, f"num_heads ({h}) must be divisible by cp_size ({cp_size})"
    h_local = h // cp_size

    # [S_local, cp_size, H_local, D] -> [cp_size, S_local, H_local, D]
    t = t.reshape(s_local, cp_size, h_local, d).transpose(0, 1).contiguous()
    output = torch.empty_like(t)
    out = dist_nn.all_to_all_single(output, t, group=cp_group)
    # out[i] is the chunk that source-rank i had at position my_rank, i.e.
    # source-rank i's local sequence shard for *my* head slice.
    return out.reshape(cp_size * s_local, h_local, d)


def _all_to_all_head_to_seq(
    t: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Inverse of `_all_to_all_seq_to_head`: [S_global, H_local, D] -> [S_local, H, D]."""
    s_global, h_local, d = t.shape
    assert s_global % cp_size == 0
    s_local = s_global // cp_size
    h = h_local * cp_size

    t = t.reshape(cp_size, s_local, h_local, d).contiguous()
    output = torch.empty_like(t)
    out = dist_nn.all_to_all_single(output, t, group=cp_group)
    # out[s']: original chunk for sequence-rank s' (which now becomes head-rank s').
    return out.transpose(0, 1).contiguous().reshape(s_local, h, d)


def ulysses_flash_attn_varlen_func(
    flash_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    flash_attn_version: int = 2,
    window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    deterministic: bool | None = None,
) -> torch.Tensor:
    """Run varlen flash attention under Ulysses CP.

    `cu_seqlens_*` and `max_seqlen_*` describe the *full* (un-sharded) sequence,
    because after the seq->head all-to-all each rank holds the full sequence.
    """
    q = _all_to_all_seq_to_head(q, cp_size, cp_group)
    k = _all_to_all_seq_to_head(k, cp_size, cp_group)
    v = _all_to_all_seq_to_head(v, cp_size, cp_group)

    kwargs: dict = {"causal": causal}
    if window_size != (-1, -1):
        kwargs["window_size"] = window_size
    if softmax_scale is not None:
        kwargs["softmax_scale"] = softmax_scale
    if dropout_p:
        kwargs["dropout_p"] = dropout_p
    if deterministic is not None:
        kwargs["deterministic"] = deterministic

    if flash_attn_version == 4:
        # FA4 takes cu_seqlens as keyword args (qv positional collides otherwise).
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k
        out = flash_fn(q, k, v, **kwargs)
    else:
        out = flash_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)
    if isinstance(out, tuple):
        out = out[0]

    return _all_to_all_head_to_seq(out, cp_size, cp_group)


def substitute_ulysses_attn(
    process_group: dist.ProcessGroup,
    attn_impl: str = "flash_attention_2",
) -> None:
    """Patch FlashAttention._compute_attention to do Ulysses all-to-all + local FA.

    Mirrors `substitute_ring_attn` so swapping CP styles is a one-line change
    in the trainer.
    """
    cp_size = dist.get_world_size(group=process_group)

    # Resolve flash kernel + version from attn_impl.
    if attn_impl == "fa4":
        from flash_attn.cute import flash_attn_varlen_func as flash_fn

        flash_attn_version = 4
        flash_fn = torch._dynamo.disable(flash_fn)
    elif attn_impl == "flash_attention_3":
        from flash_attn_interface import flash_attn_varlen_func as flash_fn

        flash_attn_version = 3
    else:
        from flash_attn import flash_attn_varlen_func as flash_fn

        flash_attn_version = 2

    def _ulysses_compute_attention(self, q, k, v, cu_seqlens, max_seqlen):
        # cu_seqlens / max_seqlen passed in are for the *local* sharded sequence;
        # ulysses needs the *full* ones (each rank holds the full seq after a2a).
        cu_seqlens_full = ULYSSES_PARAMS["cu_seqlens"]
        max_seqlen_full = ULYSSES_PARAMS["max_seqlen"]

        window_size = (-1, -1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            window_size = (sliding_window - 1, 0)

        return ulysses_flash_attn_varlen_func(
            flash_fn,
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_full,
            cu_seqlens_k=cu_seqlens_full,
            max_seqlen_q=max_seqlen_full,
            max_seqlen_k=max_seqlen_full,
            causal=True,
            cp_group=process_group,
            cp_size=cp_size,
            flash_attn_version=flash_attn_version,
            window_size=window_size,
        )

    from prime_rl.trainer.models.layers.attn import FlashAttention

    FlashAttention._compute_attention = _ulysses_compute_attention

    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    AfmoeFlashAttention._compute_attention = _ulysses_compute_attention

    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedFlashAttention

    Qwen3_5MoeGatedFlashAttention._compute_attention = _ulysses_compute_attention


def substitute_hf_ulysses_attn(process_group: dist.ProcessGroup) -> None:
    """Patch HF's `_flash_attention_forward` to use Ulysses all-to-all + local FA2.

    Used for HF (non-custom) model paths, e.g. qwen2/qwen3 dense models. Mirrors
    ring_flash_attn's `substitute_hf_flash_attn` but with all-to-all instead of
    ring all-gather.
    """
    import transformers
    import transformers.modeling_flash_attention_utils
    from flash_attn import flash_attn_varlen_func

    cp_size = dist.get_world_size(group=process_group)

    def _ulysses_flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids=None,
        softmax_scale=None,
        sliding_window=None,
        use_top_left_mask: bool = False,
        softcap=None,
        deterministic=None,
        **kwargs,
    ):
        assert is_causal, "ulysses CP only supports causal attention"
        assert softcap is None, "ulysses CP path does not support softcap"
        assert query_states.size(0) == 1, "varlen data should be processed with batch=1"

        cu_seqlens = ULYSSES_PARAMS["cu_seqlens"]
        max_seqlen = ULYSSES_PARAMS["max_seqlen"]

        window_size = (-1, -1)
        if sliding_window is not None and key_states.shape[1] > sliding_window:
            window_size = (sliding_window, sliding_window)

        out = ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=process_group,
            cp_size=cp_size,
            flash_attn_version=2,
            window_size=window_size,
            softmax_scale=softmax_scale,
            dropout_p=dropout,
            deterministic=deterministic,
        )
        return out.unsqueeze(0)

    transformers.modeling_flash_attention_utils._flash_attention_forward = _ulysses_flash_attention_forward

    # Newer transformers route attention through ALL_ATTENTION_FUNCTIONS["flash_attention_2"].
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        ALL_ATTENTION_FUNCTIONS = None

    if ALL_ATTENTION_FUNCTIONS is not None:
        def _ulysses_flash_attention_forward_v2(
            module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask=None,
            dropout: float = 0.0,
            scaling=None,
            sliding_window=None,
            softcap=None,
            **kw,
        ):
            # Match HF v2 entrypoint: query/key/value arrive as [B, H, S, D].
            seq_len = query.shape[2]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            kw.pop("is_causal", None)
            attn_out = _ulysses_flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query_length=seq_len,
                is_causal=module.is_causal,
                dropout=dropout,
                softmax_scale=scaling,
                sliding_window=sliding_window,
                softcap=softcap,
            )
            return attn_out, None

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _ulysses_flash_attention_forward_v2
