# DSv4 sparse attention: recompiles from a dynamic HCA gather width

All numbers are from one H200 node (SLURM job 1341), V4 Flash attention shapes (64 heads,
`head_dim` 512, one KV head), bfloat16, tilelang 0.1.12, base commit `ae37b35a3`. Times are
absolute; lower is better everywhere. "Tokens" is tokens per GPU (one packed row, no CP).

## Phase 0: the stall

### Cost of one new gather width (`stall.py`, 4096 tokens)

Three cache states for the first call at a width never seen before.

| State                                  | First fwd (s) | First bwd (s) | Steady fwd+bwd (ms) |
|----------------------------------------|---------------|---------------|---------------------|
| Cold: empty `TILELANG_CACHE_DIR`       | 6.3           | 5.1           | 5-11                |
| Warm disk cache, fresh process         | 0.06          | 0.04          | 5-11                |
| Same process, second call              | 0.002-0.003   | 0.003-0.009   | 5-11                |

The cold cost is flat in the width (192 through 1152 all within 0.1 s of each other) and is a
compile, not a disk load: a warm disk cache brings it down to about 0.1 s per width (tracing plus
loading the cached binary), and the in-process memo (`JITImpl._kernel_cache`) makes a repeat free.
Neither matches the Slack "~2 s" exactly: a new width costs about 11 s on a node whose
`~/.tilelang/cache` is cold and about 0.1 s on one that already holds it.

### Recurrence over a stream of packings (`bench.py`, Phase 1 baseline below)

Each run feeds 40 log-normal packings (median document 4096 tokens, sigma 1.25, one layer,
fwd+bwd) through a fresh cache. A compile is any call into `JITImpl.compile`; each new width costs
two (fwd and bwd), and the first step also pays the width-independent `preprocess`/`postprocess`.

| Layer   | Tokens  | Widths seen | Compiles | Compile stall total (s) | Last step that compiled |
|---------|---------|-------------|----------|-------------------------|-------------------------|
| HCA     | 65536   | 6           | 14       | 70.7                    | 12                      |
| HCA     | 131072  | 10          | 22       | 112.7                   | 23                      |
| HCA     | 262144  | 16          | 34       | 179.7                   | 35                      |
| CSA     | 65536   | 1           | 4        | 15.2                    | 0                       |
| CSA     | 131072  | 1           | 4        | 15.4                    | 0                       |
| Sliding | 131072  | 1           | 4        | 15.2                    | 0                       |

Only HCA varies the width, as expected. The stall does stop once every width has been seen, but at
256k tokens 40 steps saw only 16 of the 33 reachable widths and it was still compiling at step 35:
a long tail of first-seen widths trickles in over a run. Six concurrent compiles on one node made
each one about 5.5 s rather than the 11 s measured alone above.

### "Dynamic `topk` is too slow" (`dyn_topk.py`, 32768 tokens)

`dynamic`: `topk` is a `T.dynamic` shape (`dyn_fwd.py`, `dyn_bwd.py`). `tiled`: `Indices` is
passed as a free view `(B, S, G, n_tiles, tile)` and only `n_tiles` is dynamic (`tiled_fwd.py`,
`tiled_bwd.py`). Indices are HCA-like: a per-query valid prefix of uniform length in `[0, width]`.

| Width | Static fwd (ms) | Dynamic fwd (ms) | Tiled fwd (ms) |
|-------|-----------------|------------------|----------------|
| 128   | 4.78            | 4.90             | 4.83           |
| 192   | 5.62            | 5.65             | 5.54           |
| 256   | 6.41            | 6.41             | 6.25           |
| 384   | 7.91            | 7.92             | 7.69           |
| 640   | 10.99           | 10.94            | 10.56          |
| 1152  | 17.05           | 16.93            | 16.17          |

| Width | Static bwd (ms) | Dynamic bwd (ms) | Tiled bwd (ms) |
|-------|-----------------|------------------|----------------|
| 128   | 8.51            | 10.54            | 8.49           |
| 192   | 11.50           | 14.68            | 11.55          |
| 256   | 14.59           | 18.62            | 14.59          |
| 384   | 20.70           | 26.69            | 20.71          |
| 640   | 32.86           | 42.55            | 32.88          |
| 1152  | 57.31           | 73.89            | 56.85          |

Both variants match the static kernel bit for bit on `out`, `lse` and `dq`; `dkv` differs at the
level of float32 atomic-add ordering (at most 0.06), since `dKV` is scattered with atomics whose
order is not fixed.

The claim holds for the backward only: a dynamic `topk` costs nothing in the forward and 24-29% in
the backward. The generated CUDA shows why. With `topk` symbolic, tilelang cannot prove
`i_i * block + bi_i < topk`, so it wraps every `Indices` read (mask, gather, and the index feeding
each `dKV` atomic) in that bounds check and loses the vectorized mask load. The `tiled` layout
makes the in-tile offset static and the tile index bounded by the loop, so the checks vanish and
it runs at static speed (the forward is up to 5% faster at wide widths). One compile covers every
width.

## Phase 1: baseline (`bench.py`, same packings as above)

Kernel time is the sparse attention op alone, re-run on the inputs the layer produced, averaged
over all 40 steps. Layer time is the whole attention layer's fwd+bwd wall time, median over
compile-free steps in the last three quarters of the run.

| Layer   | Tokens  | Kernel fwd (ms) | Kernel bwd (ms) | Layer fwd+bwd (ms) | Peak memory (GiB) |
|---------|---------|-----------------|-----------------|--------------------|-------------------|
| HCA     | 65536   | 15.1            | 44.2            | 192                | 26.8              |
| HCA     | 131072  | 37.2            | 116.2           | 400                | 53.4              |
| HCA     | 262144  | 88.8            | 284.8           | 894                | 107.2             |
| CSA     | 65536   | 21.9            | 75.8            | 281                | 27.1              |
| CSA     | 131072  | 42.8            | 151.5           | 596                | 53.8              |
| Sliding | 131072  | 18.3            | 49.4            | 334                | 52.4              |

## Phase 2

Each candidate is re-run through `sweep.sh` on the same 40 packings per config (same seed) with a
fresh tilelang cache. Kernel times are means over all 40 steps, so they compare like for like
across candidates. The layer wall-time median is not directly comparable to the baseline's, since
the baseline's median excludes its compiling steps, which are a different subset of packings.

### Fix 4: tiled `Indices`, dynamic tile count (label `tiled`)

The `tiled_{fwd,bwd}.py` layout moved into `dsv4_sparse_attn_{fwd,bwd}.py`: the op passes
`Indices` as a view `(B, S, G, K / tile, tile)` and `topk` is no longer a compile key.

| Layer   | Tokens  | Compiles (base -> fix) | Compile stall s (base -> fix) | Last compile step (base -> fix) |
|---------|---------|------------------------|-------------------------------|---------------------------------|
| HCA     | 65536   | 14 -> 4                | 70.7 -> 16.1                  | 12 -> 0                         |
| HCA     | 131072  | 22 -> 4                | 112.7 -> 15.3                 | 23 -> 0                         |
| HCA     | 262144  | 34 -> 4                | 179.7 -> 15.2                 | 35 -> 0                         |
| CSA     | 131072  | 4 -> 4                 | 15.4 -> 15.5                  | 0 -> 0                          |

| Layer   | Tokens  | Kernel fwd ms (base -> fix) | Kernel bwd ms (base -> fix) | Peak GiB (base -> fix) |
|---------|---------|-----------------------------|-----------------------------|------------------------|
| HCA     | 65536   | 15.1 -> 14.8                | 44.2 -> 44.3                | 26.8 -> 26.8           |
| HCA     | 131072  | 37.2 -> 35.8                | 116.2 -> 115.9              | 53.4 -> 53.4           |
| HCA     | 262144  | 88.8 -> 85.0                | 284.8 -> 284.4              | 107.2 -> 107.2         |
| CSA     | 65536   | 21.9 -> 21.0                | 75.8 -> 75.8                | 27.1 -> 27.1           |
| CSA     | 131072  | 42.8 -> 41.1                | 151.5 -> 151.8              | 53.8 -> 53.8           |
| Sliding | 131072  | 18.3 -> 18.6                | 49.4 -> 49.4                | 52.4 -> 52.4           |

Every layer now compiles its four kernels (fwd, bwd, `preprocess`, `postprocess`) once, on the
first step, whatever the packing. Kernel time is unchanged in the backward and 2-4% lower in the
forward. The 92 kernel and model tests pass unchanged.
