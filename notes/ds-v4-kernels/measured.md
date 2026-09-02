# Measured: DS V4 module sweep on one H200

Every number here comes from `bench/profile_ds_v4.py` on a single NVIDIA H200 (sm90, 143771 MiB),
batch 1, bf16, the real `DeepSeek-V4-Flash-0731` config. Regenerate with `bench/sweep.sh` and
re-render this file with `uv run notes/ds-v4-kernels/bench/render.py`.

## How to read it

Three separate processes measure each point, because each instrument perturbs the others:

- **memory**: one forward, then one forward+backward, no profiler, no memory recording. Peaks are
  a high-water mark, so a single iteration is the right number of iterations.
- **timing**: `triton.testing.do_bench` with `warmup=300, rep=3000` (milliseconds of work, not
  iteration counts), quantiles `p20/p50/p80`, `grad_to_none` set to the module's parameters.
  Forward is timed under `no_grad`; `bwd ms` is the difference between the two calls.
- **attribution**: a `TorchDispatchMode` allocation log plus a `torch.profiler` chrome trace.
  Its timings are discarded.

All memory columns are **net of `baseline_bytes`**, the module's own parameters and inputs, so
they measure activation cost alone. `retained after fwd` is what actually crosses into backward;
that is the only column activation checkpointing removes.

## Document layout

Documents are 8192 tokens where `t` permits and a single document below that, so `t = 2048` and
`t = 4096` are one short document, `t = 8192` is one full document, and `t = 24576` is three.
`PackedContext` behaviour depends on this: the sliding-window mask and every compressed-entry
layout are clipped at document boundaries.

## Layer indices

Taken from the real checkpoint's `compress_ratios`, not from `DeepseekV4Config()` defaults:
sliding = 0, CSA = 2, HCA = 3 (see `findings-vs-plan.md` section 3). The `decoder-*` rows use the
first layer of each type past the three hash-routed bootstrap layers, so CSA = 4 and HCA = 3.


## OOM ceiling per module

Peak allocations are reported net of the module's own parameters and inputs, which is what `baseline_bytes` records.

| module | largest t that fits | first t that OOMs |
|---|---:|---:|
| `attn-csa` | 12288 | 16384 |
| `attn-hca` | 16384 | 24576 |
| `attn-sliding` | 16384 | 24576 |
| `indexer` | 24576 | 32768 |
| `indexer-scorer` | 16384 | 24576 |
| `compressor-csa` | 24576 | 32768 |
| `compressor-hca` | 32768 | none in sweep |
| `hyperconnection` | 32768 | none in sweep |
| `rmsnorm` | 32768 | none in sweep |
| `rotary` | 32768 | none in sweep |
| `packed-context` | 32768 | none in sweep |
| `decoder-csa` | 8192 | none in sweep |
| `decoder-hca` | 8192 | none in sweep |

### `attn-csa`

126.1M parameters. layer_idx=2

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 2.70 | 1.36 | 3.24 | 13.48 | 26.14 | 12.66 | 26.13/26.16 |
| 4096 | ok | 9.14 | 3.94 | 11.29 | 45.24 | 84.90 | 39.65 | 84.82/84.93 |
| 8192 | ok | 33.35 | 12.85 | 42.39 | 149.93 | 289.77 | 139.84 | 288.87/289.90 |
| 12288 | ok | 72.65 | 26.76 | 93.49 | 326.93 | 622.60 | 295.68 | 621.81/623.64 |
| 16384 | oom | - | - | - | - | - | - | - |

### `attn-hca`

111.2M parameters. layer_idx=3

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 2.32 | 1.23 | 2.74 | 10.25 | 22.03 | 11.78 | 22.02/22.04 |
| 4096 | ok | 7.66 | 3.44 | 9.33 | 33.27 | 69.61 | 36.34 | 69.59/69.63 |
| 8192 | ok | 27.44 | 10.87 | 34.61 | 114.97 | 235.92 | 120.95 | 235.47/236.96 |
| 12288 | ok | 59.38 | 22.34 | 76.00 | 227.29 | 477.85 | 250.55 | 477.53/478.45 |
| 16384 | ok | 103.47 | 37.84 | 133.51 | - | - | - | - |
| 24576 | oom | - | - | - | - | - | - | - |

### `attn-sliding`

107.0M parameters. layer_idx=0

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 2.30 | 1.22 | 2.72 | 9.83 | 20.63 | 10.80 | 20.62/20.64 |
| 4096 | ok | 7.56 | 3.41 | 9.25 | 32.51 | 65.82 | 33.31 | 65.80/65.85 |
| 8192 | ok | 27.09 | 10.78 | 34.32 | 110.27 | 228.17 | 117.89 | 227.60/228.63 |
| 12288 | ok | 58.62 | 22.15 | 75.39 | 228.54 | 474.51 | 245.97 | 473.77/475.03 |
| 16384 | ok | 102.15 | 37.53 | 132.45 | 391.40 | 810.96 | 419.56 | 810.69/812.21 |
| 24576 | oom | - | - | - | - | - | - | - |

### `indexer`

10.7M parameters. forward only: output is int64 topk indices

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.85 | 0.01 | - | 1.79 | - | - | - |
| 4096 | ok | 3.21 | 0.02 | - | 5.24 | - | - | - |
| 8192 | ok | 12.46 | 0.03 | - | 17.89 | - | - | - |
| 12288 | ok | 27.74 | 0.05 | - | 38.36 | - | - | - |
| 16384 | ok | 49.04 | 0.06 | - | 69.86 | - | - | - |
| 24576 | ok | 109.75 | 0.09 | - | 150.83 | - | - | - |
| 32768 | oom | - | - | - | - | - | - | - |

### `indexer-scorer`

0.3M parameters. fp32 matmul + relu + weighted sum

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.82 | 0.57 | 1.07 | 1.09 | 3.09 | 2.00 | 3.08/3.09 |
| 4096 | ok | 3.14 | 2.14 | 4.16 | 4.04 | 11.42 | 7.37 | 11.41/11.43 |
| 8192 | ok | 12.32 | 8.32 | 16.38 | 15.70 | 44.29 | 28.59 | 44.25/44.33 |
| 12288 | ok | 27.52 | 18.52 | 36.66 | 34.72 | 100.27 | 65.55 | 100.26/100.34 |
| 16384 | ok | 48.76 | 32.76 | 65.01 | 61.95 | 172.18 | 110.22 | 172.13/172.20 |
| 24576 | oom | - | - | - | - | - | - | - |

### `compressor-csa`

19.1M parameters. layer_idx=2

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.87 | 0.02 | 0.09 | 2.25 | 4.73 | 2.48 | 4.31/5.82 |
| 4096 | ok | 3.25 | 0.03 | 0.14 | 5.61 | 6.30 | 0.70 | 6.30/6.31 |
| 8192 | ok | 12.53 | 0.07 | 0.24 | 18.54 | 19.81 | 1.27 | 19.80/19.82 |
| 12288 | ok | 27.84 | 0.10 | 0.33 | 39.29 | 40.81 | 1.52 | 40.79/40.82 |
| 16384 | ok | 49.18 | 0.13 | 0.43 | 68.25 | 70.20 | 1.95 | 70.19/70.21 |
| 24576 | ok | 109.95 | 0.20 | 0.62 | 152.60 | 169.45 | 16.85 | 156.68/175.97 |
| 32768 | oom | - | - | - | - | - | - | - |

### `compressor-hca`

4.3M parameters. layer_idx=3

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.01 | 0.01 | 0.09 | 0.72 | 2.33 | 1.61 | 2.29/3.14 |
| 4096 | ok | 0.02 | 0.02 | 0.13 | 0.97 | 3.40 | 2.43 | 3.37/3.44 |
| 8192 | ok | 0.05 | 0.03 | 0.23 | 0.72 | 2.10 | 1.38 | 2.08/2.12 |
| 12288 | ok | 0.07 | 0.05 | 0.32 | 0.73 | 3.10 | 2.38 | 2.75/3.37 |
| 16384 | ok | 0.09 | 0.06 | 0.41 | 0.72 | 3.39 | 2.67 | 3.36/3.42 |
| 24576 | ok | 0.14 | 0.09 | 0.60 | 0.71 | 2.12 | 1.40 | 2.11/2.42 |
| 32768 | ok | 0.19 | 0.13 | 0.79 | 0.89 | 2.67 | 1.78 | 2.66/2.67 |

### `hyperconnection`

0.4M parameters. backward through `collapsed`

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.45 | 0.30 | 0.89 | 0.98 | 2.48 | 1.50 | 2.48/3.31 |
| 4096 | ok | 0.86 | 0.56 | 1.72 | 1.48 | 4.37 | 2.89 | 4.36/4.37 |
| 8192 | ok | 1.68 | 1.10 | 3.38 | 2.55 | 8.13 | 5.58 | 8.13/8.14 |
| 12288 | ok | 2.51 | 1.63 | 5.03 | 3.80 | 12.08 | 8.28 | 12.06/12.09 |
| 16384 | ok | 3.34 | 2.16 | 6.69 | 4.76 | 15.74 | 10.98 | 15.73/15.74 |
| 24576 | ok | 4.99 | 3.22 | 10.00 | 6.94 | 23.29 | 16.35 | 23.28/23.30 |
| 32768 | ok | 6.64 | 4.29 | 13.31 | 9.01 | 30.73 | 21.72 | 30.72/30.74 |

### `rmsnorm`

0.0M parameters. 

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.02 | 0.02 | 0.17 | 0.02 | 0.91 | 0.89 | 0.56/0.96 |
| 4096 | ok | 0.03 | 0.03 | 0.34 | 0.02 | 0.92 | 0.90 | 0.66/1.31 |
| 8192 | ok | 0.06 | 0.06 | 0.69 | 0.04 | 0.82 | 0.78 | 0.73/0.92 |
| 12288 | ok | 0.09 | 0.09 | 1.03 | 0.06 | 1.06 | 1.01 | 1.06/1.39 |
| 16384 | ok | 0.13 | 0.13 | 1.38 | 0.07 | 1.39 | 1.32 | 1.39/1.39 |
| 24576 | ok | 0.19 | 0.19 | 2.06 | 0.10 | 2.05 | 1.95 | 2.05/2.05 |
| 32768 | ok | 0.25 | 0.25 | 2.75 | 0.13 | 2.70 | 2.57 | 2.70/2.70 |

### `rotary`

0.0M parameters. buffers only, no parameters

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.00 | 0.00 | - | 0.11 | - | - | - |
| 4096 | ok | 0.00 | 0.00 | - | 0.12 | - | - | - |
| 8192 | ok | 0.00 | 0.00 | - | 0.12 | - | - | - |
| 12288 | ok | 0.01 | 0.00 | - | 0.12 | - | - | - |
| 16384 | ok | 0.01 | 0.00 | - | 0.06 | - | - | - |
| 24576 | ok | 0.01 | 0.00 | - | 0.12 | - | - | - |
| 32768 | ok | 0.02 | 0.00 | - | 0.10 | - | - | - |

### `packed-context`

0.0M parameters. once per model forward

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2048 | ok | 0.05 | 0.01 | - | 1.20 | - | - | - |
| 4096 | ok | 0.22 | 0.03 | - | 1.60 | - | - | - |
| 8192 | ok | 0.88 | 0.13 | - | 1.62 | - | - | - |
| 12288 | ok | 1.97 | 0.28 | - | 2.88 | - | - | - |
| 16384 | ok | 3.50 | 0.50 | - | 4.25 | - | - | - |
| 24576 | ok | 7.88 | 1.13 | - | 8.56 | - | - | - |
| 32768 | ok | 14.00 | 2.01 | - | 14.25 | - | - | - |

### `decoder-csa (ac=full)`

6595.6M parameters. layer_idx=4, ac=full

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096 | ok | 8.43 | 0.18 | 24.14 | - | - | - | - |
| 8192 | ok | 31.92 | 0.33 | 56.03 | - | - | - | - |

### `decoder-csa`

6595.6M parameters. layer_idx=4, ac=none

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096 | ok | 9.72 | 6.67 | 24.13 | - | - | - | - |
| 8192 | ok | 34.50 | 18.23 | 56.03 | - | - | - | - |

### `decoder-hca (ac=full)`

6580.7M parameters. layer_idx=3, ac=full

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096 | ok | 6.94 | 0.16 | 22.18 | - | - | - | - |
| 8192 | ok | 26.00 | 0.28 | 48.25 | - | - | - | - |

### `decoder-hca`

6580.7M parameters. layer_idx=3, ac=none

| t | status | fwd peak GB | retained after fwd GB | bwd peak GB | fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096 | ok | 8.23 | 6.16 | 22.18 | - | - | - | - |
| 8192 | ok | 28.59 | 16.26 | 48.24 | - | - | - | - |

## Top allocations across the sweep

From the `TorchDispatchMode` allocation log, which keys on storage `data_ptr` so a view of a storage already counted is not counted twice.

| GB | module | t | phase | op | shape | dtype |
|---:|---|---:|---|---|---|---|
| 10.00 | `attn-csa` | 8192 | forward | `aten.cat.default` | (1, 64, 8192, 10241) | bfloat16 |
| 10.00 | `attn-csa` | 8192 | forward | `aten.sub.Tensor` | (1, 64, 8192, 10241) | bfloat16 |
| 10.00 | `attn-csa` | 8192 | backward | `aten._softmax_backward_data.default` | (1, 64, 8192, 10241) | bfloat16 |
| 10.00 | `attn-csa` | 8192 | forward | `aten.bmm.default` | (64, 8192, 10240) | bfloat16 |
| 10.00 | `attn-csa` | 8192 | forward | `aten.mul.Tensor` | (1, 64, 8192, 10240) | bfloat16 |
| 8.06 | `attn-hca` | 8192 | forward | `aten.cat.default` | (1, 64, 8192, 8257) | bfloat16 |
| 8.06 | `attn-hca` | 8192 | forward | `aten.sub.Tensor` | (1, 64, 8192, 8257) | bfloat16 |
| 8.06 | `attn-hca` | 8192 | backward | `aten._softmax_backward_data.default` | (1, 64, 8192, 8257) | bfloat16 |
| 8.06 | `attn-hca` | 8192 | forward | `aten.bmm.default` | (64, 8192, 8256) | bfloat16 |
| 8.06 | `attn-hca` | 8192 | forward | `aten.mul.Tensor` | (1, 64, 8192, 8256) | bfloat16 |
| 8.00 | `attn-sliding` | 8192 | forward | `aten.cat.default` | (1, 64, 8192, 8193) | bfloat16 |
| 8.00 | `attn-sliding` | 8192 | forward | `aten.sub.Tensor` | (1, 64, 8192, 8193) | bfloat16 |
| 8.00 | `attn-sliding` | 8192 | backward | `aten._softmax_backward_data.default` | (1, 64, 8192, 8193) | bfloat16 |
| 8.00 | `attn-sliding` | 8192 | forward | `aten.bmm.default` | (64, 8192, 8192) | bfloat16 |
| 8.00 | `attn-sliding` | 8192 | forward | `aten.mul.Tensor` | (1, 64, 8192, 8192) | bfloat16 |
| 8.00 | `indexer-scorer` | 8192 | backward | `aten.bmm.default` | (8192, 128, 2048) | float32 |
| 4.00 | `attn-csa` | 8192 | forward | `aten.bmm.default` | (8192, 64, 2048) | float32 |
| 4.00 | `attn-csa` | 8192 | forward | `aten.relu.default` | (1, 8192, 64, 2048) | float32 |
| 4.00 | `attn-csa` | 8192 | forward | `aten.mul.Tensor` | (1, 8192, 64, 2048) | float32 |
| 4.00 | `indexer-scorer` | 8192 | forward | `aten.bmm.default` | (8192, 64, 2048) | float32 |
