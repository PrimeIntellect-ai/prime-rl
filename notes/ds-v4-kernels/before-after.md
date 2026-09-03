# DS V4 kernel work, phase 1: CSA attention, measured

What the fused sparse-attention kernel did to a CSA layer, measured the same way phase 0 measured
the baseline: one `DeepseekV4Attention` at `layer_idx = 2` of the real `DeepSeek-V4-Flash-0731`
config, batch 1, bf16, one H200 per point, no parallelism, no activation checkpointing. Memory
columns are net of the module's own parameter and buffer baseline. Timing is `do_bench` with
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
from the other half of the layer. `24576` fits only because an H200 has 143 GB.

That makes `DeepseekV4IndexerScorer` the next lever rather than anything in attention. It is
cheaper to fix than attention was: the scores feed only a top-k, nothing retains them (the indexer
receives no gradient, see `README.md`), and item 6 of the ranked target list already notes that
folding `softmax_scale` into the head weights removes two of its three copies for two lines of
change.

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

## What this does not do

This phase converts the 21 CSA layers of 43. The 20 HCA layers and 2 sliding layers still build
dense score tensors, and `PackedContext.attention_mask`, the dense `(1, 1, t, t)` sliding mask,
still exists because they consume it. The model-level ceiling therefore does not move to 64k. What
moved is the per-layer ceiling for half the layers, from 12288 to 24576, plus a proven kernel and
index contract for HCA to reuse.

## Reproducing

```bash
uv sync --all-extras
IMPLS="eager gather kernel" MODULES="attn-csa" ./notes/ds-v4-kernels/bench/sweep.sh
uv run notes/ds-v4-kernels/bench/render.py
```

The whole sweep is about 20 minutes. Raw per-point JSON lands in `bench/raw/`, which is gitignored;
the committed artifact is this file.

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
