# Qwen3-235B-A22B: SFT max trainer perf

A throughput reference for the SFT trainer on a large MoE: `Qwen/Qwen3-235B-A22B-Thinking-2507` on 4 nodes x 8 B300 at 16k context. It applies the [Max-Throughput Recipe](../../../docs/scaling.md#max-throughput-recipe): everything on GPU, no offloading, `fullgraph` compile. It runs 20 steps on balanced routing and writes no checkpoint. Use it to check a cluster or a trainer change against a known number, around 46% MFU.

| Config | What it runs | Topology |
|---|---|---|
| [`sft-max-perf.toml`](sft-max-perf.toml) | 20 SFT steps, seq 16384, EP 8, full AC, fp32 AdamW, balanced routing | 4 train nodes |

## Requirements

- A Slurm cluster with 4 nodes x 8 B300 and a shared filesystem.
- `HF_TOKEN` exported in the launch shell. The first run downloads the checkpoint and builds the HF -> prime-rl conversion cache, which `dist_timeout_seconds = 14400` covers.

## Launch

```bash
uv run sft @ examples/advanced/qwen3-235b-a22b/sft-max-perf.toml
```

Read `time/step`, `perf/mfu`, `perf/throughput_per_gpu`, and `perf/peak_memory` from `outputs/<run>/monitors/file/metrics.jsonl`, skipping the first few warmup steps.

Variants:

- Real routing: remove `[model.debug]`. Expect it to be noticeably slower, since routing imbalance dominates at this size.
- Other clusters: set `[slurm] partition` and point `HF_HOME` at a shared cache via `[env_vars]`.
