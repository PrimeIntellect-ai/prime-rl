# Context

DeepSeek V4 currently hard-rejects context parallelism (CP): `cp_support()` (`modeling_deepseek_v4.py:117-123`) returns no styles, because its sliding-window mask and compressed-attention layouts (`PackedContext`/`CompressionLayout`, `attention.py`) are built from post-shard (local) document boundaries. This document sketches the architecture for lifting that, one component per section. See `PLAN.md` for still-open decision points not covered here.

Precedent: GLM's DSA (`glm_moe_dsa`) already supports CP — shard Q, all-gather K, run top-k/attention locally, with boundaries reconstructed via an all-gather on `position_ids`. DS V4 adds bounded sliding-window layers and token compression (CSA/HCA), which pool raw tokens into entries before GLM's kind of retrieval even applies.

# Components

## 1. Context-parallel attention: all-gather strategy

**Decision**: adopt the naive all-gather-only strategy for v1, mirroring GLM's `glm_moe_dsa` CP implementation for consistency with the rest of the repo. A cheaper halo-based alternative was designed and costed but is deferred (see below) to avoid premature optimization.

**Design**: Q shards disjointly via the existing `shard_for_cp` — no halo, no overlapping slice, no ownership/padding scheme. Per layer, branching only on layer type:
- **SWA-only layers**: all-gather `kv` (`self.kv_proj(hidden_states)`, `attention.py:831`, 512-wide/token) across the CP group. Sufficient on its own — SWA's attention never needs anything wider than the shared K=V vector it already computes per token.
- **SWA+{CSA,HCA} layers**: all-gather `hidden_states` (4096-wide/token) instead. `kv` for that layer's local-window component becomes a free local projection of the now-fully-gathered `hidden_states`; the compressor (`compress()`, `attention.py:608-617`) runs completely unmodified against it, since every rank now has the whole sequence's raw input to pool from.
- `PackedContext.build` needs **no CP-awareness at all** under this scheme — call it with the full, pre-shard `seq_lens` (already passed today via the currently-rejected `seq_lens_are_pre_shard`) exactly as the non-CP path does, since `kv`/`hidden_states` are now globally resident too. Only Q stays sharded; slice the attention output down to the local true range at the very end.
- The all-gather uses the existing `gather_for_cp` helper (`src/prime_rl/utils/cp.py:113-114`, a differentiable `all_gather` along `dim=1`) — the same primitive GLM's CP path already uses.

**Cost** (per layer, 128k tokens, bf16, vs. GLM's own gather of 176 MiB): SWA-only layers are actually cheaper (128 MiB); SWA+CSA/HCA layers — the majority of the model's layers — are ~5.8x more expensive (1024 MiB). Accepted for v1 to reuse existing non-CP code paths unmodified.

**Deferred optimization**: halo-extended local slicing + local compression + gather of only the post-compression entries, costed at ~40 MiB/layer for CSA. Revisit if CP communication becomes a measured bottleneck.

## 2. Sliding-window-only layers

**Design**:
- Q stays local throughout — never gathered.
- `kv = kv_proj(hidden_states)`, computed locally then all-gathered (differentiable) to the full sequence.
- `PackedContext`/document-boundary metadata is already global, model-level, unchanged from non-CP.
- Once `kv` and the metadata are fully realized globally, pare down to the local Q range and hand off to whatever windowed-attention kernel is in use — no mask materialization assumed here.

**Pseudocode**, schematic:

```python
kv = gather_for_cp(kv_proj(hidden_states_local), cp_group)   # fully realized, global

out_local = windowed_attn(q_local, kv, packed)   # kernel call; slices to local Q range internally
```

**Open follow-ups**: nothing CP-specific beyond the gather itself — whatever windowed-attention kernel replaces the dense-mask path just needs to accept a local Q range against a global K, which is the same shape of problem any non-CP windowed kernel already solves.

## 3. CSA layers

**Design**:
- `hidden_states` gets all-gathered (per Component 1's SWA+CSA/HCA branch) — this alone is sufficient for the compressor and indexer to run completely unchanged, since they now see the whole sequence exactly as the non-CP path does.
- Everything Q-side stays local: `q_residual` (the indexer's own query input) and the local-window `kv` are both local-range only. Everything K/entries-side is global: `compressed_kv` and the indexer's top-k both naturally come out as "local queries against global entries," the same shard-Q/gather-K/local-retrieval shape as GLM and Component 2.
- No new communication beyond the Component 1 gather — CSA doesn't need its own separate gather step.

**Pseudocode**, schematic:

```python
hidden_states = gather_for_cp(hidden_states_local, cp_group)   # fully realized, global

compressed_kv, top_k_indices = compressor(hidden_states, q_residual_local, packed)
# entries computed over the global hidden_states; indexer scores local queries against them,
# so top_k_indices comes out already local-Q-sized

out_local = sparse_attn(q_local, kv_local_window, compressed_kv, top_k_indices)   # kernel call
```

**Open follow-ups**: whether the indexer's own per-query weighting (today's `weights_proj(hidden_states)`) should read from the global or a locally-sliced `hidden_states` — logically it's Q-side data, so it should use the local slice, but that's a slicing detail for whoever implements this, not an architecture question.

## 4. HCA layers

**Design**:
- Same `hidden_states` gather as CSA (Component 1's SWA+CSA/HCA branch) — no separate gather step needed.
- No indexer, no top-k: every local query attends over *all* global compressed entries plus its local window, restricted only by a causal threshold (an entry is readable once its source tokens are entirely at or before the query) — same "local queries against global entries" shape as CSA, just without a selection step.
- Q-side stays local (the query itself and its causal-threshold position); K/entries-side is global (`compressed_kv` over the whole sequence).

**Pseudocode**, schematic:

```python
hidden_states = gather_for_cp(hidden_states_local, cp_group)   # fully realized, global

compressed_kv, causal_info = compressor(hidden_states, packed)   # entries + per-query readability,
                                                                   # global-entries / local-query shaped

out_local = dense_attn(q_local, kv_local_window, compressed_kv, causal_info)   # kernel call, no selection step
```

**Open follow-ups**: none beyond Component 1's gather — HCA is strictly simpler than CSA, since there's no index-remapping/sentinel convention to design at all here.
