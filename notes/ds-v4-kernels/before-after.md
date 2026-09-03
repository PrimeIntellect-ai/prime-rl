# DS V4 kernel work, phase 1: CSA attention, measured

What the fused sparse-attention kernel did to a CSA layer, measured the same way phase 0 measured
the baseline: one `DeepseekV4Attention` at `layer_idx = 2` of the real `DeepSeek-V4-Flash-0731`
config, batch 1, bf16, one H200 per point, no parallelism, no activation checkpointing. Memory
columns are net of the module's own parameter and buffer baseline. Timing is `do_bench` with
`--warmup 300 --rep 3000`, whose p20/p80 spread is under 0.4% on every point below.

Three implementations, selected by `PRIME_RL_DSV4_ATTN`:

- `eager`: the dense additive mask, quadratic in sequence length. The phase-0 baseline.
- `gather`: a float32 PyTorch reference over the explicit index tensor. An oracle for the kernel,
  and a fallback for installs without tilelang.
- `kernel`: the fused TileLang kernel.

## Peak memory

Split forward from backward rather than putting ten columns on one line. `vs eager` is
`eager / impl`, so above 1 is an improvement and below 1 is a regression; it is undefined once
eager OOMs, which is itself the result.

Forward peak:

| t | eager | gather | vs eager | kernel | vs eager |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2.70 GB | 4.78 GB | 0.56x | **1.41 GB** | **1.91x** |
| 4096 | 9.14 GB | 9.52 GB | 0.96x | **4.31 GB** | **2.12x** |
| 8192 | 33.35 GB | 19.01 GB | 1.75x | **14.61 GB** | **2.28x** |
| 12288 | 72.65 GB | 30.95 GB | 2.35x | **30.95 GB** | **2.35x** |
| 16384 | OOM | 53.32 GB | fits | **53.32 GB** | **fits** |
| 24576 | OOM | OOM | - | **116.15 GB** | **fits** |

Backward peak:

| t | eager | gather | vs eager | kernel | vs eager |
|---:|---:|---:|---:|---:|---:|
| 2048 | 3.24 GB | 8.93 GB | 0.36x | **1.62 GB** | **2.00x** |
| 4096 | 11.29 GB | 17.67 GB | 0.64x | **3.04 GB** | **3.71x** |
| 8192 | 42.39 GB | 35.15 GB | 1.21x | **5.86 GB** | **7.23x** |
| 12288 | 93.49 GB | 52.63 GB | 1.78x | **8.69 GB** | **10.76x** |
| 16384 | OOM | 70.11 GB | fits | **11.52 GB** | **fits** |
| 24576 | OOM | OOM | - | **17.18 GB** | **fits** |

The kernel's advantage widens with `t` on both, and much faster on the backward, because eager
carries four simultaneous score tensors there against three in the forward. `gather` is worse than
eager at short context and only overtakes it at 8192, for the reason in the section below.

| module | attn | largest `t` that fits | first `t` that OOMs |
|---|---|---:|---:|
| `attn-csa` | eager | 12288 | 16384 |
| `attn-csa` | gather | 16384 | 24576 |
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
report exactly the same forward peak at 12288 and 16384**, 30.95 GB and 53.32 GB. Two
implementations with completely different attention memory cannot agree by chance. They agree
because neither is what peaks: the Lightning Indexer is. Phase 0 measured the indexer alone at
27.74 GB, 49.04 GB and 109.75 GB at those three lengths (`measured.md`), just under each of the
figures above.

So the shape of the problem has changed. Attention was the ceiling; now the indexer is. Its fp32
scorer materializes `scores[b,s,h,e]` over `h = 64` heads and `e = t/4` entries, which is 17.2 GB
per copy at `t = 16384` and quadratic in `t`, exactly the term the attention kernel just removed
from the other half of the layer. `24576` fits only because an H200 has 143 GB.

That makes `DeepseekV4IndexerScorer` the next lever rather than anything in attention. It is
cheaper to fix than attention was: the scores feed only a top-k, nothing retains them (the indexer
receives no gradient, see `README.md`), and item 6 of the ranked target list already notes that
folding `softmax_scale` into the head weights removes two of its three copies for two lines of
change.

## `gather` is a memory win and a large time loss

`gather` outlives `eager` by one step, fitting at 16384 where eager does not. That is not
what a float32 path costs 1.3 MB per token would suggest, and the reason is that its cost is
linear while eager's is quadratic: eager's `(1, 64, t, t + t/4)` bf16 logits alone are 42.9 GB at
16384, before the sink concatenation, the row-max subtraction and the softmax each copy it.

It pays for that in time. `gather` is 21x slower than eager at 2048 and 2.3x at 12288, and its
backward dominates completely, 543 ms of 559 ms at 2048. A float32 `einsum` pair holding a
`(batch, seq_len, n_slots, head_dim)` saved tensor is simply an expensive way to compute
attention. That is fine for what it is: an oracle the kernel is tested against, and a fallback for
installs without tilelang, where correctness matters and long context is not on offer anyway.

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
2. The `kernel` rows really ran the kernel. `attn_impl` is snapshotted per layer at construction,
   so a plumbing mistake would profile eager under a kernel label. Verified by counting calls into
   all three module-level entry points and by checking the CUDA trace: exactly one implementation
   fires per label, and the tilelang `main_kernel` appears only under `kernel`. **Pass.**
3. Timing stability: p20/p80 spread under 0.4% on every point, at `--warmup 300 --rep 3000`.
   **Pass.**
4. The indexer attribution above is corroborated by an independent phase-0 measurement of the
   indexer in isolation, not inferred from the attention rows. **Pass.**
