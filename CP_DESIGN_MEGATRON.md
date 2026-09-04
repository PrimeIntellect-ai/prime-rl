# Context

Megatron-LM's `dev` branch already has working CP support for DeepSeek-V4-Flash-style attention, unlike prime-rl, which currently rejects CP for DS V4 entirely (see `CP_DESIGN.md`). Plain MLA gets CP for free via standard ring attention: `cp_partition_mode="zigzag"` routes through TransformerEngine's P2P ring attention (`cp_comm_type` defaults to `"p2p"`, `megatron/core/extensions/transformer_engine.py:1746,1825`), with a native all-gather-based fallback (`AllGatherComm`, `AttentionFuncionWithContextParallel`) in `megatron/core/transformer/dot_product_attention_context_parallel.py:108-133,150-154`, invoked from the non-TE `DotProductAttention.forward` (`megatron/core/transformer/dot_product_attention.py:183-194`). Plain DSA (DeepSeek-v3.2-style sparse attention, no sliding window, no HCA) gets CP even more simply, via an all-gather of K/V across the CP group plus an "undo zigzag reordering" step (`megatron/core/transformer/experimental_attention_variant/dsa.py:2209-2267`, comment at line 2215): there's no cross-rank pooling to worry about, so a global K/V materialization per rank is sufficient.

The actual DSv4-Hybrid/CSA path, Megatron's analog of what prime-rl is designing for, needs bespoke CP support because compression pooling only makes sense over sequentially contiguous token blocks, which conflicts with zigzag's interleaved causal load-balancing. It's implemented under `megatron/core/transformer/experimental_attention_variant/` (`deepseek_v4_hybrid_attention.py`, `csa.py`, `csa_utils/cp_utils.py`, `dsa_layout.py`), merged via PR #5087 (`bfa33263c`, "[dev] [DeepSeek-v4] Context Parallel support"). As with prime-rl's `PLAN.md`, there's a known list of open items: causal load-imbalance under contiguous partitioning is explicitly measured (peak activation memory only drops to 65% of the CP=1 baseline at CP=2, versus an ideal 50%, and to 35% at CP=4, versus an ideal 25%; `_DSV4_CP_MEMORY_RATIO_LIMITS = {2: 0.65, 4: 0.35}`, `tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_hybrid_attention_cp.py:52`), and an unmerged, stale PR (#6058, `4ddc698f2`, adds `cp_balanced_indexer.py`) attempts a fix but hasn't landed on `dev`. There is no design doc in the Megatron repo itself; every claim below comes from reading the code, tests, and PR/commit history directly.

# Components

## 1. Context-parallel data chunking

**Problem.** Under `cp_partition_mode="contiguous"`, a rank's shard `[global_start, global_start + l_local)` is disjoint, not overlapping: it doesn't already hold the raw hidden states a sliding-window layer needs from before `global_start`, nor does its local compressor output cover the compressed entries other ranks' queries need to select over. Both gaps require runtime communication, unlike prime-rl's pre-existing local overlap.

**Key fact**: prime-rl gives every CP rank a redundant, overlapping slice of an already-fully-resident micro-batch (`CP_DESIGN.md` component 1), so its halo needs no runtime exchange. Megatron shards disjointly instead: `cp_partition_mode="contiguous"` is the only mode compatible with `experimental_attention_variant="dsv4_hybrid"` (`megatron/core/transformer/transformer_config.py:1719-1724` rejects `zigzag` for `dsv4_hybrid`; `transformer_config.py:1703-1710` requires `contiguous` to pair with `dsv4_hybrid`, `gdn`, or `kda`). Contiguous blocks are what let compression pooling operate over sequentially contiguous token ranges at all; the cost is that boundary and global-compressed context must be fetched at runtime instead of being free.

**Design**:
- Left-boundary-only P2P exchange (`_LeftBoundaryExchange`, `megatron/core/transformer/experimental_attention_variant/csa_utils/cp_utils.py:124-188`): rank `r` sends its own last `d_window = max(csa_window_size, d_comp)` hidden-state rows to rank `r+1` and receives the analogous slice from rank `r-1` (`exchange_cp_boundary_hidden`, `cp_utils.py:191-202`). There's no right-side exchange: a compressed group is owned by whichever rank holds the group's *last* token (`prepare_cp_compressor_input`, `cp_utils.py:245-257`), so a group straddling a seam is reassigned entirely to the later rank rather than split, sidestepping the right-side halo prime-rl's `owned_by` needs.
- Global compressed-KV and indexer-K all-gather (`prepare_cp_compressor_input`, `compute_cp_indexer_topk`, `cp_utils.py:210-274,304-400`): each rank packs its compressor output into a fixed-capacity, alignment-padded buffer (`c_cap`) before an all-gather across the CP group, so top-k sees the whole sequence's compressed entries, not just a local shard. `seq_to_rank_row` remaps between the logical (sequence-major) row order and the physical (rank-major, fixed-capacity) storage the all-gather produces.
- This all-gather step has no analog in prime-rl's current halo-based design, which keeps CSA/indexer work local to each rank's extended, overlapping slice, at the cost of redundant compute instead of communication.

**`CompressedSparseAttention._forward_thd_cp`** (`megatron/core/transformer/experimental_attention_variant/csa.py:2537-2896`) is the THD CP-specific forward path that stitches the boundary-exchanged, local raw, and all-gathered compressed KV together before calling the same sparse-attention kernel the non-CP path uses.

```python
def _forward_thd_cp(self, query, key, x, qr, boundary_hidden, boundary_kv, packed_seq_params):
    cp_group = self.pg_collection.cp
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    l_local = query.shape[0]
    global_start = cp_rank * l_local                    # contiguous shard offset; no zigzag remap needed
    kv_local = key.squeeze(-2).squeeze(1)                # this rank's raw K/V; identical to the non-CP path
    d_window = boundary_hidden.shape[0]                  # boundary_hidden/boundary_kv already P2P-exchanged (CP-only, step 1)

    ratio = self.compress_ratio
    if self.compressor is not None and ratio > 1:
        hidden_compact, compressed_group_ids, seq_to_rank_row = cp_utils.prepare_cp_compressor_input(
            x, boundary_hidden, cu_seqlens, cu_seqlens_compressed, global_start, cp_size, ratio,
        )                                                 # fixed-capacity local compressor input, boundary-aware (CP-only)
        compressed_kv_local, _ = self.compressor._forward_thd(hidden_compact, cu_seqlens, ...)  # same compressor as non-CP
        compressed_kv_rank_major = gather_from_sequence_parallel_region(compressed_kv_local, group=cp_group)  # CP-only (step 2)

        if self.indexer is not None:
            k_indexer_local, _ = indexer.compressor._forward_thd(hidden_compact, ...)
            k_indexer_rank_major = gather_from_sequence_parallel_region(k_indexer_local, group=cp_group)  # CP-only
            k_indexer_seq_major = k_indexer_rank_major[seq_to_rank_row.clamp_min(0)]  # undo fixed-capacity packing (CP-only)
            compressed_topk, indexer_layout = cp_utils.compute_cp_indexer_topk(
                q_indexer_cp, weights_indexer_cp, k_indexer_seq_major, ...
            )                                             # top-k now runs over the *global* compressed sequence

    kv_full_thd = torch.cat((boundary_kv, kv_local, compressed_kv_rank_major), dim=0)  # stitch: exchanged + local + gathered (step 3)
    topk_idxs, topk_length, _ = csa_cp_layout_kernels.build_attention_indices(
        cu_seqlens, global_start, l_local, d_window, self.window_size, ratio, ...,
        compressed_topk, seq_to_rank_row=seq_to_rank_row,
    )                                                     # remaps logical topk ids into kv_full_thd's physical rows (CP-only)
    return csa_sparse_attn(query, kv_full_thd, self.attn_sink.float(), topk_idxs, self.softmax_scale, ...)  # step 4, shared kernel
```

**Open follow-ups**: contiguous partitioning reintroduces exactly the causal load imbalance zigzag was designed to avoid; peak memory only drops to 65% of the CP=1 baseline at CP=2 (ideal: 50%) and 35% at CP=4 (ideal: 25%) (`_DSV4_CP_MEMORY_RATIO_LIMITS`, `test_dsv4_hybrid_attention_cp.py:52`), and the unmerged PR #6058 (`4ddc698f2`, adds `cp_balanced_indexer.py`, a per-sequence zigzag-balanced indexer) attempts a fix but hasn't landed. Separately, the fixed-capacity `c_cap` buffers exist for CUDA-graph safety, but replay must still tolerate padding shifting between capture and replay because the CP path rebuilds compressed-row metadata from device `cu_seqlens_padded` rather than capture-time host sizes (`test_dsv4_hybrid_attention_cp.py:70-74`); whether that fixed-capacity approach is even necessary for prime-rl, which hasn't committed to CUDA graphs for this path, is untouched here.
