# DS V4 kernel work, phase 0: measure before optimizing

Measurements, analysis, and a ranked target list for the DeepSeek V4 kernel effort. **No model
code was changed.** Everything here was produced on this branch from the real
`DeepSeek-V4-Flash-0731` config, batch 1, bf16, one NVIDIA H200 (sm90, 143771 MiB) per measured
point, no parallelism.

| file | what it holds |
|---|---|
| `findings-vs-plan.md` | where the handoff plan disagreed with the repository, and what changed |
| `measured.md` | the sweep: peak memory, OOM ceilings, time split, top allocations |
| `memory-model.md` | closed-form bytes per tensor, fitted and validated to 1.1% |
| `static-analysis.md` | by-eye bottlenecks from four reviews, annotated with what measurement said |
| `megatron-survey.md` | what Megatron-LM `dev` already has for DS V4, and how portable it is |
| `before-after.md` | phase 1: what the fused CSA attention kernel measured, against this baseline |
| `bench/` | the throwaway harness, the sweep driver, and the renderer |

## Three structural findings

**1. One tensor is the whole problem.** Define `S(t) = 2 * h * t * (t + e + 1)`, the size of one
`(1, h, t, t+e+1)` bf16 attention score tensor. A least-squares fit over all 14 surviving
attention points, across all three layer types, gives

```
fwd_peak = 3.005 * S(t) + 405 KB/token        worst error 1.1%
retained = 0.996 * S(t) + 364 KB/token        worst error 1.9%
bwd_peak = 3.979 * S(t) + 329 KB/token        worst error 3.5%
```

Three simultaneous copies at the forward peak, one retained, four at the backward peak. A single
CSA decoder layer at `t = 8192` has a 34.50 GB forward transient, of which **33.35 GB is the
attention block** and roughly 1 GB is everything else including the entire 256-expert MoE. That
ratio is the answer to "how much of the layer can this work reach": essentially all of it.

**2. No stock flash kernel accepts `head_dim = 512` on H200.** FA4 is installed (flash-attn
2.8.3). `_validate_head_dims` in `flash_attn/cute/interface.py:112` gates the
`is_deepseek_mla_absorbed_shape` path (`head_dim_v == 512`) to `compute_capability in [10, 11]`;
the `compute_capability == 9` branch asserts `8 <= head_dim <= 256`. H200 is sm90. So the local
sliding-window half needs either a banded custom kernel or the absorbed-MLA reformulation, not a
drop-in `flash_attn_varlen_func`.

**3. The window part and the top-k entry part share one softmax**, including a per-head learnable
sink logit (`attention.py:158-162`). A sparse-MLA kernel covers only the entry half. The natural
decomposition is to run the two halves separately, each returning its LSE, and combine, folding
the sink into the combination. This is the design decision the next phase turns on.

## A correctness finding that is not about performance

**The Lightning Indexer receives no gradient.** `DeepseekV4Indexer.forward` ends at
`scores.topk(...).indices`, an int64 tensor, and those indices feed a `scatter_` of constant
`0.0` / `-inf` values into `block_bias`. Neither step is differentiable, so the entire indexer
subgraph is unreachable from the loss.

Verified directly rather than inferred: after a real CSA-layer forward and backward at
`t = 2048`, all 6 indexer parameters have `p.grad is None`, while all 4 CSA-compressor parameters
and all 8 attention parameters have nonzero gradients.

```
indexer                         :   6 params,   0 have .grad,   0 nonzero
csa compressor (non-indexer)    :   4 params,   4 have .grad,   4 nonzero
attention proper                :   8 params,   8 have .grad,   8 nonzero
```

Megatron trains its equivalent with an auxiliary KL loss between the indexer's score distribution
and the attention distribution (`dsa_indexer_loss_coeff: 1e-2` in its V4 recipe). prime-rl has no
such loss. For RL from a pretrained checkpoint a frozen indexer may be the right choice, but it
should be a deliberate one rather than an accident of the top-k being non-differentiable. This is
out of scope for a measurement phase; it is flagged for the author.

The silver lining is that this is *why* the indexer's 12.46 GB forward transient at `t = 8192`
never appears in the peak: with no graph holding them, its three fp32 score copies are freed
before the attention core allocates. The indexer peak is a `max` with the attention peak, not a
sum.

## OOM ceilings, measured

Single module, single H200, no parallelism, no activation checkpointing.

| module | largest `t` that fits | first `t` that OOMs |
|---|---:|---:|
| `attn-csa` | 12288 | 16384 |
| `attn-hca` | 16384 | 24576 |
| `attn-sliding` | 16384 | 24576 |
| `indexer-scorer` | 16384 | 24576 |
| `indexer` | 24576 | 32768 |
| `compressor-csa` | 24576 | 32768 |
| `compressor-hca`, `hyperconnection`, `rmsnorm`, `rotary`, `packed-context` | 32768 | none in sweep |

**The production aspiration of a 64k packed row is 13x to 16x beyond a single H200**, and
activation checkpointing does not help: it removes the `0.996 * S` retained term, not the
`3.005 * S` transient peak inside one attention call. Measured at `t = 8192` on a full decoder
layer, `ac="full"` cuts retained memory from 18.23 GB to 0.33 GB and leaves the backward peak
unchanged at 56.03 GB. Reaching 64k requires removing the quadratic term outright, or context
parallelism, which prime-rl correctly rejects for DS V4 today
(`modeling_deepseek_v4.py:241-245`). Megatron reaches 64k only with CP16, and its own arithmetic
makes CP a prerequisite rather than an optimization (see `megatron-survey.md`).

## Ranked target list

Bytes are per layer at `b = 1`, bf16. "at 64k" is projected from the validated model, not
measured. Times are `do_bench` medians at `t = 8192`.

| # | target | bytes saved at `t=8192` | projected at `t=65536` | time saved | reuse or write | depends on |
|---|---|---|---|---|---|---|
| 1 | **Banded + gathered flash attention core** replacing `eager_attention_with_sinks`. **Done for CSA**, see `before-after.md`; sliding untouched | measured 18.7 GB fwd and 36.5 GB bwd per CSA layer; 24.0 GB (sliding) projected | ceiling moved 12288 to 24576 per CSA layer, and the forward is now indexer-bound, not attention-bound; 1539 GB (sliding) projected | measured 229 of 290 ms fwd+bwd per CSA layer, 6.0x at t=12288 | **written**: no stock kernel takes `head_dim 512` on sm90, so this is a TileLang kernel | needed #2 first (index representation) |
| 2 | **Sparse index representation**: keep `top_k_indices`, drop the dense `block_bias` and the mask `cat`. **Done for CSA**, HCA still renders a dense bias | 0.19 GB per CSA layer | 12 GB per CSA layer | small on its own, and it is what made #1 possible | **written**: the index contract is `SparseAttnInputs` in `deepseek_v4/attention.py` | none; enabling change for #1 |
| 3 | **Drop the dense sliding mask** for `(cu_seqlens, sliding_window)` | 0.8 GB transient, 0.12 GB resident (once per forward, not per layer) | 52 GB transient, 8.0 GB resident | 1.6 ms per forward | **write**, trivial once #1 consumes bounds | #1 |
| 4 | **mHC fused norm + projection** replacing the fp32 flatten at `hyperconnections.py:51` | 1.68 GB per instance, x86 instances | 13.3 GB per instance | ~0.7 of 8.1 ms per instance | **partial reuse**: `quack.rmsnorm(x, None, ...)` is a one-line drop-in for the norm; Megatron's `fused_proj_rms_compute_h` is cuTile-only and unavailable on H200 | none |
| 5 | **Fused Sinkhorn** replacing the 39-step loop | 0.03 GB saved state per instance | 0.20 GB per instance | launch-bound: 10,234 launches to 86 per forward | **reuse**: Megatron `fused_sinkhorn`, Triton, sm90-clean, semantics verified byte-identical | none |
| 6 | **Indexer einsum**: fold `softmax_scale` into `weights`, replace the product-then-sum | 8.0 GB transient per CSA layer | 512 GB transient per CSA layer | part of 17.9 ms | **write**, two lines for two of three copies | none |
| 7 | **In-place partial RoPE** replacing the 512-channel `cat` for a 64-channel rotation | ~2.8 GB churn per layer | ~22 GB per layer | part of the per-layer elementwise chain | **adapt**: Megatron's `fused_mla_rope_inplace` refuses adjacent-pair interleaving, which is what prime-rl uses | none |
| 8 | **Fused `h_aggregate` / `h_post_bda`** for the mHC stream mixing | 1.1 GB per layer | 8.8 GB per layer | ~15 of 42 ms per forward | **reuse**: Megatron, Triton, sm90-clean | none |

Rows 1, 2, 3 and 6 are quadratic in `t` and so grow by 64x from `t=8192` to `t=65536`; rows 4, 5,
7 and 8 are linear and grow by 8x.

Items 1 through 3 are one project and are the only ones that move the OOM ceiling. Items 4, 5 and
8 are independent, cheap, and reusable from Megatron today. Item 6 is two lines for half its
benefit.

## Reproducing

```bash
uv sync --all-extras                          # torch lives in the `gpu` extra
./notes/ds-v4-kernels/bench/sweep.sh          # ~40 min across 8 GPUs, idempotent
uv run notes/ds-v4-kernels/bench/render.py    # regenerate the tables in measured.md
```

Raw per-point JSON lands in `bench/raw/`; chrome traces and logs go to `outputs/ds-v4-kernels/`,
which is gitignored.

## Verification performed

1. `torch.cuda.device_count()` reports 8, `get_device_name(0)` is `NVIDIA H200`, capability
   `(9, 0)`. **Pass.**
2. Harness smoke test on the toy config: builds, forward and backward run, peaks are plausible for
   a 62k-parameter module. **Pass.**
3. Parameter-count guard: the harness asserts `head_dim == 512`, `num_attention_heads == 64`,
   `sliding_window == 128`, the 2 / 21 / 20 layer mix, and `> 1e8` parameters for any attention
   module. Measured 107.0M (sliding), 111.2M (HCA), 126.1M (CSA). **Pass.**
4. Analytic model predicts measured peak: **1.1% worst error** across all 14 attention points,
   against the 20% the plan asked for. **Pass.**
5. Timing stability: **initially failed** at 19.9% spread. `do_bench`'s `warmup` and `rep` are
   milliseconds of total work, not iteration counts, so the default `rep=100` bought one or two
   reps at these shapes and collapsed `p20`/`p50`/`p80` onto each other. With
   `--warmup 300 --rep 3000` the median spread over three repeated invocations is **0.13%**. The
   whole timing sweep was re-run at the corrected budget and `sweep.sh` now sets it. **Pass after
   correction.**
6. Cross-GPU reproducibility: `attn-sliding` at `t = 8192` on GPU 1 and GPU 2 gives bit-identical
   peaks (`fwd 29508342272`, `bwd 37272000000`). **Pass.**
7. Profiler attribution corroborates the dispatch-mode allocation log: both name the same five
   10.00 GB `(1, 64, 8192, 10241)` tensors for `attn-csa`. **Pass.**

One caveat on the allocation log: it keys on storage `data_ptr` and skips pointers already seen,
so an op landing on a recycled block is not recorded twice. That is why `aten.add` (the mask add)
and `aten._softmax` do not appear in the top-allocation table even though the fitted coefficient
of 3.005 says they exist. The log undercounts; it does not overcount.
