# DS V4 kernel work: attention on the fused kernel, measured

What the fused sparse-attention kernel did to the attention block, measured the same way phase 0
measured the baseline. Phase 1 converted CSA and is below; phase 2 converted the sliding and HCA
layers and is at the end. Conditions for both: one `DeepseekV4Attention` of the real
`DeepSeek-V4-Flash-0731` config, at the layer index that config gives its type, batch 1, bf16, one
H200 per point, no parallelism, no activation checkpointing. Memory columns are net of the
module's own parameter and buffer baseline. Timing is `do_bench` with
`--warmup 300 --rep 3000`, whose p20/p80 spread is under 0.7% on every point below except `gather`
at 4096, where it is 1.3%.

Three implementations, selected here by `sweep.sh`'s `IMPLS` and the harness's `--attn-impl`:

- `eager`: the dense additive mask, quadratic in sequence length. The phase-0 baseline, and what
  `auto` falls back to when the kernel cannot run.
- `gather`: a PyTorch reference over the explicit index tensor, computing in the dtype it is
  handed. An oracle for the kernel when fed float32, and an explicit choice for anyone who wants
  the sparse semantics without the kernel; `auto` never selects it.
- `kernel`: the fused TileLang kernel, which `auto` selects whenever it can run.

## Peak memory

Split forward from backward rather than putting ten columns on one line. `vs eager` is
`eager / impl`, so above 1 is an improvement and below 1 is a regression; it is undefined once
eager OOMs, which is itself the result.

Forward peak:

| t | eager | gather | vs eager | kernel | vs eager |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2.70 GB | 2.56 GB | 1.05x | **1.41 GB** | **1.91x** |
| 4096 | 9.14 GB | 5.08 GB | 1.80x | **4.31 GB** | **2.12x** |
| 8192 | 33.35 GB | 14.61 GB | 2.28x | **14.61 GB** | **2.28x** |
| 12288 | 72.65 GB | 30.95 GB | 2.35x | **30.95 GB** | **2.35x** |
| 16384 | OOM | 53.32 GB | fits | **53.32 GB** | **fits** |
| 24576 | OOM | 116.15 GB | fits | **116.15 GB** | **fits** |

Backward peak:

| t | eager | gather | vs eager | kernel | vs eager |
|---:|---:|---:|---:|---:|---:|
| 2048 | 3.24 GB | 4.77 GB | 0.68x | **1.62 GB** | **2.00x** |
| 4096 | 11.29 GB | 9.36 GB | 1.21x | **3.04 GB** | **3.71x** |
| 8192 | 42.39 GB | 18.52 GB | 2.29x | **5.86 GB** | **7.23x** |
| 12288 | 93.49 GB | 27.69 GB | 3.38x | **8.69 GB** | **10.76x** |
| 16384 | OOM | 36.86 GB | fits | **11.52 GB** | **fits** |
| 24576 | OOM | 55.20 GB | fits | **17.18 GB** | **fits** |

The kernel's advantage widens with `t` on both, and much faster on the backward, because eager
carries four simultaneous score tensors there against three in the forward. `gather` beats eager
on the forward at every length and on the backward from 4096 up; the one point where it is worse
is its backward peak at 2048, by 1.5x, and the section below says why.

| module | attn | largest `t` that fits | first `t` that OOMs |
|---|---|---:|---:|
| `attn-csa` | eager | 12288 | 16384 |
| `attn-csa` | gather | 24576 | 32768 |
| `attn-csa` | kernel | 24576 | none in sweep |

## Time

| t | eager fwd | kernel fwd | eager fwd+bwd | kernel fwd+bwd | speedup |
|---:|---:|---:|---:|---:|---:|
| 2048 | 13.48 ms | 5.47 ms | 26.14 ms | 16.14 ms | 1.6x |
| 4096 | 45.24 ms | 12.11 ms | 84.90 ms | 29.19 ms | 2.9x |
| 8192 | 149.93 ms | 31.30 ms | 289.77 ms | 61.05 ms | 4.7x |
| 12288 | 326.93 ms | 58.08 ms | 622.60 ms | 103.79 ms | 6.0x |
| 16384 | OOM | 93.21 ms | OOM | 152.14 ms | - |
| 24576 | OOM | 189.89 ms | OOM | 279.51 ms | - |

The speedup grows with `t` because eager is quadratic and the kernel is linear: a CSA query reads
`sliding_window + index_topk + 1 = 641` keys regardless of how long the row gets.

## The backward is where the win is, and the forward is no longer attention-bound

The backward peak fell by 10.8x at `t = 12288`, from 93.49 GB to 8.69 GB, and grows linearly:
17.18 GB at 24576, where eager would have needed roughly 370 GB by the phase-0 model. Retained
memory after the forward tells the same story, 26.76 GB against 5.03 GB at 12288.

The forward peak is a different matter. It fell only 2.3x at 12288, and **`gather` and `kernel`
report exactly the same forward peak at every length from 8192 up**: 14.61, 30.95, 53.32 and
116.15 GB. Two implementations with completely different attention memory do not agree to the
hundredth of a gigabyte four times over by chance. They agree because neither is what peaks: the
Lightning Indexer is. Phase 0 measured the indexer alone at 12.46, 27.74, 49.04 and 109.75 GB at
those four lengths (`measured.md`), just under each of the figures above.

So the shape of the problem has changed. Attention was the ceiling; now the indexer is. Its fp32
scorer materializes `scores[b,s,h,e]` over `h = 64` heads and `e = t/4` entries, which is 17.2 GB
per copy at `t = 16384` and quadratic in `t`, exactly the term the attention kernel just removed
from the other half of the layer. `24576` fits only because the GPU has 143 GB.

That made `DeepseekV4IndexerScorer` the next lever rather than anything in attention, and it was
cheaper to fix than attention was. See the follow-on section below.

## Follow-on: the indexer scorer, measured

`DeepseekV4Indexer.forward` now runs under `torch.no_grad()`, which is free but licenses the two
changes that are not: the scorer mutates its `(batch, seq, heads, entries)` intermediate in place
instead of copying it, and the two constant scales ride along on the per-head weights instead of
costing a separate pass. Same harness, same conditions as above.

| t | fwd peak before | after | vs before | fwd before | after | vs before |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.85 GB | **0.34 GB** | **2.47x** | 1.79 ms | 1.66 ms | 1.08x |
| 4096 | 3.21 GB | **1.19 GB** | **2.70x** | 5.24 ms | 4.73 ms | 1.11x |
| 8192 | 12.46 GB | **4.38 GB** | **2.85x** | 17.89 ms | 16.03 ms | 1.12x |
| 12288 | 27.74 GB | **9.56 GB** | **2.90x** | 38.36 ms | 33.76 ms | 1.14x |
| 16384 | 49.04 GB | **16.75 GB** | **2.93x** | 69.86 ms | 58.95 ms | 1.19x |
| 24576 | 109.75 GB | **37.13 GB** | **2.96x** | 150.83 ms | 132.68 ms | 1.14x |
| 32768 | OOM | **65.51 GB** | **fits** | OOM | 226.37 ms | **runs** |

Three copies of the intermediate become one, so the ratio approaches 3x as that term comes to
dominate, and the indexer clears 32768 for the first time. The 8.08 GB saved at 8192 is what item 6
of the ranked target list predicted analytically (8.0 GB).

Selection is unchanged. The in-place rewrite is bitwise identical to the previous scorer. Folding
the scales perturbs scores by at most 7.2e-7, which left the selected entry sets identical at 2048
and 8192 and on two packed multi-document rows, and reordered 102 of 4.19M slots within their own
top-k, every one of them a near-tie with a score gap at or below 2.4e-7. Nothing downstream depends
on that order: the indices become gather slots and attention softmaxes over all of them.

## `gather` is a memory win and a large time loss, and the time loss is not the dtype

`gather` now reaches 24576, the same length the kernel does, and outlives `eager` by two steps.
Its cost is linear where eager's is quadratic: eager's `(1, 64, t, t + t/4)` bf16 logits alone are
42.9 GB at 16384, before the sink concatenation, the row-max subtraction and the softmax each copy
it, while `gather`'s `(batch, seq_len, n_slots, head_dim)` bf16 gather is 0.66 MB per token
regardless of `t`. It also stops being what peaks: from 8192 up its forward peak is the Lightning
Indexer's, identical to the kernel's, and its OOM at 32768 is the indexer's too, the allocation it
dies on being exactly the 64 GiB fp32 `scores[b,s,h,e]`.

The time did not follow. `gather` is **21x slower than eager at 2048** and 1.9x at 12288, against
21x and 2.1x for the float32 version this replaced: a 2% to 8% improvement, which is to say none.
Computing in float32 was never what cost the time. The forward alone was already faster than
eager's (10.94 ms against 13.48 ms at 2048, and 3.6x faster at 12288) and got faster still; the
whole loss is the backward, 536.61 ms of the 547.55 ms at 2048, or 42x eager's 12.66 ms.

That backward is dominated by the scatter-add that differentiates the gather, and specifically by
the duplicate indices in it. Every slot a query does not use holds the sentinel, so one KV
position absorbs a large fraction of all `seq_len * n_slots` gradient contributions, and
`indexing_backward_kernel` sorts and serially reduces the runs it finds. Benchmarked at the CSA
layer's own shapes, sweeping only the sentinel fraction, this function's forward+backward goes
16.9 ms at 0% sentinel, 336.9 ms at 25%, 657.4 ms at 50% and 1174.2 ms at 90%, while its forward
stays at 6.3 ms throughout. In float32 the same sweep reads 27.5, 249.8, 468.3 and 821.1 ms:
**bfloat16 is the faster forward and the slower backward**, because the serial reduction runs on
narrower values without running fewer of them, and the two effects roughly cancel at the layer
level.

So the honest summary is that this change bought memory and a step of context, not speed. That is
acceptable for what `gather` is: an oracle the kernel is tested against, and an explicitly
selectable path for anyone who wants the sparse memory profile without the kernel, mainly long
context on a machine that cannot build tilelang. `auto` never picks it, so a 21x gap at short
context is the price of asking for it rather than something a default can inflict. Removing the gap would mean replacing
the advanced-indexing backward with something that does not scatter into a single hot index, which
is the kernel, which already exists.

## Phase 2: sliding and HCA on the same kernel

The other 22 layers of 43 now gather too, so no attention layer of any type builds a dense mask.
Neither conversion needed a new kernel. A sliding layer passes no entries at all, so its buffer is
the token stream plus the sentinel; HCA's readable entries are the contiguous run its document has
completed, which the compressor computes arithmetically instead of rendering as a bias. Same
harness and conditions, `layer_idx = 0` for sliding and `3` for HCA, the real config's own choice.

Forward peak:

| t | sliding eager | kernel | vs eager | HCA eager | kernel | vs eager |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2.30 GB | **0.99 GB** | **2.33x** | 2.33 GB | **1.00 GB** | **2.34x** |
| 4096 | 7.56 GB | **1.94 GB** | **3.90x** | 7.66 GB | **1.96 GB** | **3.91x** |
| 8192 | 27.09 GB | **3.85 GB** | **7.04x** | 27.45 GB | **3.89 GB** | **7.06x** |
| 12288 | 58.62 GB | **5.76 GB** | **10.18x** | 59.39 GB | **5.82 GB** | **10.21x** |
| 16384 | 102.15 GB | **7.67 GB** | **13.32x** | 103.49 GB | **7.74 GB** | **13.36x** |
| 24576 | OOM | **11.49 GB** | **fits** | OOM | **11.60 GB** | **fits** |
| 32768 | OOM | **15.31 GB** | **fits** | OOM | **15.46 GB** | **fits** |

Backward peak:

| t | sliding eager | kernel | vs eager | HCA eager | kernel | vs eager |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 2.72 GB | **1.61 GB** | **1.69x** | 2.74 GB | **1.61 GB** | **1.70x** |
| 4096 | 9.25 GB | **3.02 GB** | **3.06x** | 9.33 GB | **3.03 GB** | **3.08x** |
| 8192 | 34.32 GB | **5.85 GB** | **5.87x** | 34.60 GB | **5.86 GB** | **5.91x** |
| 12288 | 75.39 GB | **8.68 GB** | **8.69x** | 76.00 GB | **8.68 GB** | **8.75x** |
| 16384 | 132.45 GB | **11.50 GB** | **11.51x** | 133.51 GB | **11.51 GB** | **11.60x** |
| 24576 | OOM | **17.16 GB** | **fits** | OOM | **17.17 GB** | **fits** |
| 32768 | OOM | **22.82 GB** | **fits** | OOM | **22.82 GB** | **fits** |

Forward plus backward time, `do_bench` p50:

| t | sliding eager | kernel | speedup | HCA eager | kernel | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 20.7 ms | 8.1 ms | 2.6x | 22.1 ms | 9.5 ms | 2.3x |
| 4096 | 65.9 ms | 15.3 ms | 4.3x | 69.6 ms | 17.5 ms | 4.0x |
| 8192 | 228.5 ms | 29.7 ms | 7.7x | 236.4 ms | 32.9 ms | 7.2x |
| 12288 | 475.3 ms | 44.0 ms | 10.8x | 478.1 ms | 48.9 ms | 9.8x |
| 16384 | 811.9 ms | 58.5 ms | 13.9x | 833.1 ms | 64.4 ms | 12.9x |
| 24576 | OOM | 87.6 ms | - | OOM | 96.3 ms | - |
| 32768 | OOM | 118.2 ms | - | OOM | 128.8 ms | - |

| module | attn | largest `t` that fits | first `t` that OOMs |
|---|---|---:|---:|
| `attn-sliding` | eager | 16384 | 24576 |
| `attn-sliding` | kernel | 32768 | none in sweep |
| `attn-hca` | eager | 16384 | 24576 |
| `attn-hca` | kernel | 32768 | none in sweep |

## Both are linear, and they cost almost the same

Every kernel column above doubles when `t` doubles, to within a percent: sliding runs 3.85, 7.67
and 15.31 GB at 8192, 16384 and 32768. The quadratic term is gone rather than reduced, so the
ratio against eager keeps growing instead of settling, 2.3x at 2048 and 13.4x at 16384.

Sliding and HCA land within 1% of each other at every point, which is the arithmetic working out
rather than a coincidence. Peak is set by the slot count, and the harness packs 8192-token
documents, so HCA affords `8192 / 128 = 64` picks against sliding's none: 192 slots to 128, both
far below CSA's 640. That is also why CSA still peaks higher, and why HCA's slot count is 192 at
every length in the table. It follows the longest document, not the packed row.

## The remaining ceiling is the indexer, and it is not attention's to move

Attention at 8192 now costs 3.85 GB forward on a sliding layer. Measured on this branch at the same
lengths, `indexer` costs 0.85, 3.21, 12.46, 27.74, 49.04 and 109.75 GB, and OOMs at 32768 where
attention runs at 15.31 GB. So the attention block is no longer what binds a long row anywhere in
the sweep; the Lightning Indexer's fp32 `scores[b,s,h,e]` is, and it is quadratic in `t` for the
same reason attention used to be.

Those indexer figures are the phase-0 baseline, not the ones in the follow-on section above. The
scorer rewrite that section reports is not on this branch; it lives on `feat/ds-v4-kernels-indexer`
and has not reached `feat/ds-v4-kernels`. Read the two sets of numbers accordingly.

CSA re-measures identically to phase 1 at every length, 1.41, 4.31, 14.61, 30.95, 53.32 and
116.15 GB forward, so deriving the slot count from the index tensors rather than the config cost it
nothing.

## What this does not do

No attention layer of any type builds a dense mask now, and `PackedContext` carries nothing wider
than `O(S)` per query. What that does not buy is a 64k model: these are single modules on one GPU,
the indexer still holds a quadratic term, and the 43-layer model's retained activations were never
in this measurement at all. Context parallelism remains blocked, for a reason the dense mask was
only ever the descriptor of: the sliding window is built from post-shard document boundaries, which
global ones cannot address.

## Reproducing

```bash
uv sync --all-extras
IMPLS="eager kernel" MODULES="attn-csa attn-hca attn-sliding" ./notes/ds-v4-kernels/bench/sweep.sh
uv run notes/ds-v4-kernels/bench/render.py
```

The whole sweep is about 20 minutes. `sweep.sh` honours `CUDA_VISIBLE_DEVICES`; leave it unset to
use every device, and never co-locate two points on one, since each reads a global peak. Raw
per-point JSON lands in `bench/raw/`, which is gitignored; the committed artifact is this file.

## Verification performed

1. The `eager` rows reproduce phase 0. Re-measured independently at 2048, 8192 and 12288 with the
   new `--attn-impl` axis: agrees to 0.08% on memory and 0.5% on time. The axis plumbing does not
   perturb the eager path. **Pass.**
2. The `kernel` rows really ran the kernel. Each layer resolves `dsv4_attn` at construction, so a
   plumbing mistake would profile eager under a kernel label. Verified by counting calls into all
   three module-level entry points and by checking the CUDA trace: exactly one implementation
   fires per label, and the tilelang `main_kernel` appears only under `kernel`. **Pass.**
3. Timing stability: p20/p80 spread under 0.7% on every point at `--warmup 300 --rep 3000`, except
   `gather` at 4096, where it is 1.3%. `gather` at 24576 reports a zero spread because one
   iteration there costs most of the 3000 ms budget, so its `p20 = p50 = p80` is a single rep and
   not a converged quantile. **Pass, with that one point weaker than the rest.**
4. The indexer attribution above is corroborated by an independent phase-0 measurement of the
   indexer in isolation, not inferred from the attention rows. **Pass.**
5. The phase-2 baseline reproduces phase 0 before any code changed: 27.09 GB forward at 8192 for
   sliding against `measured.md`'s 27.09, and 27.45 against 27.44 for HCA. The second differs in
   the last digit shown, which is allocator noise at 0.04%. **Pass.**
6. CSA is unchanged by phase 2. Re-measured on the finished tree, its kernel rows reproduce phase
   1 exactly at all six lengths, so deriving the slot count from the index tensors rather than
   from the config neither cost nor saved anything. **Pass.**
7. All three layer types reach the keys the dense rules admit, as integer set equality per query
   with no tolerance, on five document layouts including rows whose leading document is too short
   to compress. The dense side is built from the document lengths, never from `window_indices` or
   from the index tensor, so the comparison cannot be satisfied by agreeing with itself. **Pass.**
8. The kernel rows really ran the kernel, now checked at the dispatcher: a `TorchDispatchMode`
   counts `prime_rl::dsv4_sparse_attn` and the parity tests assert one call on the kernel module
   and none on the eager ones, so a silent fallback fails rather than passing. **Pass.**
9. HCA's slot count follows the longest document, not the packed row. It is 192 at every length
   from 2048 to 32768 with the harness's 8192-token documents, and takes five distinct values
   across documents from 64 to 65536 tokens, so the minimum-width rule costs few recompiles.
   **Pass.**
