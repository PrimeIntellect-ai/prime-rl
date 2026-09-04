# Context

Megatron-LM's `dev` branch already has CP support for DeepSeek-V4-Flash-style attention. Plain MLA gets CP for free via standard zigzag ring attention (P2P, TransformerEngine). Plain DSA (DeepSeek-v3.2-style: sparse, no sliding window, no HCA) gets CP even more simply — an all-gather of K/V across the CP group, no pooling involved, so a global materialization per rank is sufficient (the same shape of solution as GLM's DSA in `CP_DESIGN.md`).

The DSv4-Hybrid/CSA path — Megatron's analog of what `CP_DESIGN.md` is designing — needs bespoke support because compression pooling only makes sense over sequentially contiguous token blocks, which conflicts with zigzag's interleaved causal load-balancing. So this path forces `cp_partition_mode="contiguous"` instead of zigzag, trading causal load-balance for pooling correctness. There's no design doc in the Megatron repo; this summarizes their code, tests, and PR history directly (merged PR #5087, `bfa33263c`).

# Components

## 1. Context-parallel data chunking (DSv4-Hybrid/CSA path)

**Problem**: under contiguous (disjoint) sharding, a rank's shard doesn't hold the raw hidden states a sliding-window layer needs from just before its start, nor does its local compressor output cover the compressed entries other ranks' queries need to select over. Unlike prime-rl's current all-gather-only design, Megatron ships real data across ranks at runtime rather than working from an already-resident full sequence.

**Design**:
- **Left-boundary P2P exchange**: rank `r` sends its own last `d_window = max(csa_window_size, d_comp)` hidden-state rows to rank `r+1`, and receives the analogous slice from rank `r-1`. One direction only — there's no right-side exchange, because a compressed group that straddles a seam is owned entirely by whichever rank holds the group's *last* token, not split between ranks. That's a different resolution than the "first token owns it" convention `CP_DESIGN.md`'s (now-superseded) halo design used, and it's what lets the exchange stay one-directional.
- **Global compressed-KV and indexer-K all-gather**: each rank packs its compressor output into a fixed-capacity, padding-aligned buffer, then all-gathers across the CP group, so top-k sees the whole sequence's compressed entries. A remap step translates between logical (sequence-order) and physical (rank-major, fixed-capacity) row order.
- The forward path stitches boundary-exchanged KV, local raw KV, and all-gathered compressed KV into one buffer, remaps top-k indices into that buffer's physical rows, then calls the *same* sparse-attention kernel the non-CP path uses — same "pare down to what the shared kernel needs" shape as `CP_DESIGN.md`'s components.

**Pseudocode**, schematic:

```python
kv_local = key_local                                    # this rank's raw K/V, unchanged from non-CP
boundary_kv = p2p_exchange_left(kv_local, d_window)      # one rank's tail -> next rank's boundary input

if compress_ratio > 1:
    compressed_kv_local = compressor(pack_boundary_aware(hidden_states_local, boundary_hidden))
    compressed_kv = gather_for_cp(compressed_kv_local, cp_group)         # fixed-capacity, then remapped to sequence order

    k_indexer_local = indexer_compressor(pack_boundary_aware(hidden_states_local, boundary_hidden))
    k_indexer = gather_for_cp(k_indexer_local, cp_group)
    top_k_indices = indexer_topk(q_indexer_local, k_indexer)             # global top-k, remapped to stitched-buffer rows

kv_full = concat(boundary_kv, kv_local, compressed_kv)   # stitched buffer
out_local = sparse_attn(q_local, kv_full, top_k_indices)  # same kernel as non-CP
```

**Open follow-ups**: contiguous partitioning reintroduces the causal load imbalance zigzag exists to avoid — measured peak memory only drops to 65% of the CP=1 baseline at CP=2 (ideal 50%) and 35% at CP=4 (ideal 25%). An unmerged PR (#6058, adds a zigzag-balanced indexer) attempts a fix but hasn't landed on `dev`. Separately, the fixed-capacity buffers exist for CUDA-graph safety, which prime-rl hasn't committed to for this path — whether that fixed-capacity approach is even necessary here is untouched by this summary.
