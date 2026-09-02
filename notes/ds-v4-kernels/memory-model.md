# Analytic memory model, and what it says about 64k

Closed-form activation bytes per tensor as a function of packed length `t`, fitted and validated
against every surviving point in `measured.md`. This is what lets the report say something
defensible about 64k without running it, since 64k does not fit.

All figures are `b = 1`, bf16 activations, and the real V4-Flash config: `h = 64` attention heads,
`d = 512` head dim, `H = 4096` hidden, `hc = 4` residual streams, `w = 128` sliding window,
`index_topk = 512`, `index_n_heads = 64`.

## Compressed entry count

```
e = t / 4       CSA layers      (compress_rate 4)
e = t / 128     HCA layers      (compress_rate 128)
e = 0           sliding layers  (no compressor)
```

`n_series = 2` on the CSA compressor does **not** double this. It widens each entry's pooling
window from `compress_rate` to `2 * compress_rate` slots (`_overlap_with_previous_window`,
`attention.py:388-406`); the entry count is `L_doc // compress_rate` either way
(`CompressionLayout.build`, `attention.py:207`).

## Per-tensor closed forms

The unit that governs everything is one attention score tensor:

```
S(t) = 2 * h * t * (t + e + 1)          bytes, bf16     <- the (1, h, t, t+e+1) logits
```

| tensor | shape | dtype | bytes | scaling |
|---|---|---|---|---|
| attention logits | `(1, h, t, t+e+1)` | bf16 | `S(t) = 128 t (t+e+1)` | quadratic |
| sliding mask | `(1, 1, t, t)` | bf16 | `2 t^2` | quadratic, lives all 43 layers |
| sliding-mask build transients | `(t,t)` int64 + 5x `(t,t)` bool | mixed | `13 t^2` | quadratic, transient |
| CSA `block_bias` | `(1, 1, t, e+1)` | bf16 | `2 t (t/4)` = `0.5 t^2` | quadratic |
| concatenated mask | `(1, 1, t, t+e)` | bf16 | `2.5 t^2` (CSA) | quadratic, per layer |
| indexer scores | `(1, t, 64, e)` | **fp32** | `256 t e` = `64 t^2` (CSA) | quadratic, **x3 copies** |
| `q` | `(1, h, t, d)` | bf16 | `65536 t` | linear |
| `attn_out` | `(1, t, h, d)` | bf16 | `65536 t` | linear |
| `kv` | `(1, 1, t, d)` | bf16 | `1024 t` | linear |
| mHC fp32 flatten | `(1, t, hc*H)` | **fp32** | `65536 t` | linear, per hyper-connection (86) |
| residual streams | `(1, t, hc, H)` | bf16 | `32768 t` | linear, across all 43 layers |
| Sinkhorn trajectory | 39x `(1, t, hc, hc)` | fp32 | `~3248 t` | linear, per hyper-connection |

## The fitted model

Least squares over all 14 surviving attention points (3 layer types x 4-5 lengths), regressing
peak-over-baseline on `[S(t), t]`:

```
fwd_peak          = 3.005 * S(t) + 405 KB/token       worst error 1.1%
retained_after_fwd= 0.996 * S(t) + 364 KB/token       worst error 1.9%
bwd_peak          = 3.979 * S(t) + 329 KB/token       worst error 3.5%
```

Per-point residuals are in the table below; every one is under 1.2%. That is far inside the 20%
the plan asked for, and it holds across sliding, CSA and HCA with a single pair of coefficients,
which is the real evidence that `S(t)` is the right unit.

| module | t | S GB | predicted GB | measured GB | error |
|---|---:|---:|---:|---:|---:|
| `attn-sliding` | 2048 | 0.50 | 2.30 | 2.30 | -0.0% |
| `attn-sliding` | 4096 | 2.00 | 7.60 | 7.56 | +0.5% |
| `attn-sliding` | 8192 | 8.00 | 27.21 | 27.09 | +0.4% |
| `attn-sliding` | 12288 | 18.00 | 58.85 | 58.62 | +0.4% |
| `attn-sliding` | 16384 | 32.00 | 102.51 | 102.15 | +0.4% |
| `attn-hca` | 8192 | 8.06 | 27.40 | 27.44 | -0.1% |
| `attn-hca` | 16384 | 32.25 | 103.26 | 103.47 | -0.2% |
| `attn-csa` | 2048 | 0.63 | 2.67 | 2.70 | -1.1% |
| `attn-csa` | 8192 | 10.00 | 33.22 | 33.35 | -0.4% |
| `attn-csa` | 12288 | 22.50 | 72.37 | 72.65 | -0.4% |

The coefficients read directly as structure. **Three** simultaneous full-size score buffers at the
forward peak, **one** retained for backward (the softmax output, which is what
`_softmax_backward_data` needs), and **four** at the backward peak (the retained one, the
recomputed logits, the incoming gradient, and the softmax-backward output). The attribution log
confirms the identity of each: `aten.bmm`, `aten.mul`, `aten.sub`, `aten.cat` in forward and
`aten._softmax_backward_data` in backward, all at exactly `(1, 64, 8192, 10241)` for CSA.

## The indexer peak is not additive, and the reason matters

The obvious model, "CSA peak = attention peak + indexer peak", overestimates by 28%. The
measurement says otherwise:

| t | indexer alone | `3.005 * S` | `attn-csa` measured | sum would be |
|---:|---:|---:|---:|---:|
| 4096 | 3.21 GB | 8.00 GB | 9.14 GB | 11.22 GB |
| 8192 | 12.46 GB | 32.00 GB | 33.35 GB | 44.46 GB |
| 12288 | 27.74 GB | 72.00 GB | 72.65 GB | 99.74 GB |

The indexer's three fp32 `(1, t, 64, e)` tensors, 12 GB at `t = 8192`, are **freed before the
attention core allocates**. They are freed because nothing holds their autograd graph: the
indexer's only output is `topk(...).indices`, an int64 tensor with no `grad_fn`, so the entire
subgraph becomes unreachable the moment `DeepseekV4Indexer.forward` returns. Measured directly:
`indexer` has a 12.46 GB forward peak at `t = 8192` and retains 0.03 GB.

This has a consequence well beyond memory accounting, recorded in `README.md`: the Lightning
Indexer receives **no gradient at all**.

So the CSA peak is a `max`, not a sum:

```
peak_CSA(t) = max( 3.005 * S(t),  3 * 256 * t * e )  +  linear
            = 3.005 * S(t) + linear        for all t, since 3.005 * S > 192 t^2 > 3 * 64 t^2
```

## Projection to 64k

Using `fwd_peak = 3.005 * S(t) + 405 KB/token`, and noting an H200 has 143771 MiB = 140.4 GiB
total, of which roughly 120 GiB is available for activations after weights and gradients:

| t | layer | `S(t)` | predicted fwd peak | fits in 120 GiB? |
|---:|---|---:|---:|---|
| 16384 | sliding | 32.0 GiB | 102.5 GiB | yes (measured 102.2) |
| 16384 | CSA | 40.0 GiB | 126.5 GiB | no (measured: OOM) |
| 32768 | sliding | 128.0 GiB | 397.3 GiB | no |
| 65536 | sliding | 512.0 GiB | 1563 GiB | no, by 13x |
| 65536 | CSA | 640.0 GiB | 1949 GiB | no, by 16x |

**64k is 13-16x beyond a single H200, and no amount of activation checkpointing changes that**,
because the three-copy peak is a transient *inside* one attention call. Checkpointing removes the
`0.996 * S` retained term, not the `3.005 * S` peak. The decoder-layer measurement confirms this
directly: at `t = 8192`, `ac="full"` cuts retained memory from 18.23 GB to 0.33 GB and leaves the
backward peak at 56.03 GB, identical to `ac="none"`.

## What a sparse kernel changes

The whole quadratic term exists to score key columns that are then masked out. The real key set
per query is `w + index_topk + 1 = 128 + 512 + 1 = 641` for CSA, and `w + (entries readable) + 1`
for HCA. Replacing the dense score tensor with a banded-plus-gathered flash kernel that never
materializes logits gives

```
S_sparse(t) = 0            (streamed in registers, never written to HBM)
peak(t)     = linear terms only, ~405 KB/token
```

At `t = 65536` that is `405 KB * 65536 = 25.3 GiB` of linear terms, which fits. The linear terms
themselves are then worth attacking (the rotary `torch.cat` and the mHC fp32 flatten dominate
them), but they are not what blocks 64k. The quadratic term is, and it is entirely removable:
`641 / 81921 = 0.8%` of the CSA score columns at `t = 65536` are ever unmasked.

## Non-attention modules, measured and linear

Each of these is linear in `t`, and each is reported per instance. Multiply by the instance count
in a 43-layer model to get the whole-model figure.

| module | instances | fwd peak at `t=32768` | bytes/token | note |
|---|---:|---:|---:|---|
| `packed-context` | 1 | 14.00 GB | quadratic, `13 t^2` | the dense mask build; transient |
| `hyperconnection` | 86 | 6.64 GB | 217 KB | fp32 flatten dominates |
| `compressor-hca` | 20 | 0.19 GB | 6.2 KB | genuinely cheap |
| `rmsnorm` | 87 | 0.25 GB | 8.2 KB | quack-fused, cheap |
| `rotary` | 1 per rope type | 0.02 GB | 0.6 KB | negligible |

`packed-context` is the exception that is not linear: `build_sliding_window_mask` materializes an
int64 `(t,t)` distance matrix plus five `(t,t)` bool tensors, so it peaks at roughly `13 t^2` and
measures 14.00 GB at `t = 32768` against a predicted 13.6 GB. It retains only the `2 t^2` bf16
mask, 2.01 GB, but that one lives for the entire 43-layer forward.
