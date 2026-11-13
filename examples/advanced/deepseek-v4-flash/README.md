# DeepSeek-V4-Flash

Agentic RL on [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (284B MoE, 13B active) at 131k context. It trains on two tasksets at equal ratio, `terminal-lego` (13,825 terminal tasks) and `swelego` (4,323 SWE tasks), with a `bash` harness in sandboxes and SWE-Bench Verified evals. A linear length penalty scales with each group's pass rate, so groups where every rollout solves the task still train: shorter solutions get the higher advantage. The model is small for its class and has a very small KV-cache footprint, so this recipe aims for high concurrency: at least 1024 rollouts in flight.

The recipe is a hardware-agnostic base plus one overlay per GPU type. The base holds the model, task, algorithm, and inference layout. The overlay holds the topology, the precision, and the kernels:

| Config | What it sets |
|---|---|
| [`rl.toml`](rl.toml) | Base: `terminal-lego` + `swelego` training with a linear length penalty, `swebench-verified` evals, Muon, batch 512 / group 16, DEP8 inference (one replica per node: attention data parallel, experts sharded 8 ways). Does not parse standalone. |
| [`h200.toml`](h200.toml) | 8×H200 nodes: 4 trainer + 2 inference nodes, blockwise FP8 trainer and online FP8 vLLM, DeepEP low-latency all-to-all, 1024+ rollouts in flight. |

```bash
uv run rl @ examples/advanced/deepseek-v4-flash/rl.toml @ examples/advanced/deepseek-v4-flash/h200.toml
```

## Tasks

The base model solves most of both tasksets, so the length penalty does much of the work. Baseline with the bash harness, 8 random tasks × 8 rollouts each, temperature 1.0, on the H200 inference setup below:

| Taskset | Solve rate | Tasks solved by all / some / none | Turns p50 / p90 | Episode tokens p50 / p90 / max |
|---|---|---|---|---|
| `terminal-lego` | 0.95 | 5 / 3 / 0 | 6 / 10 | 3k / 6k / 20k |
| `swelego` | 0.83 | 5 / 3 / 0 | 33 / 47 | 23k / 39k / 47k |

No rollout came close to the 131k context limit.

## H200 setup

### Trainer: 4 nodes

Four nodes is the minimum. The optimizer step holds the FP32 master weights, the FP32 gradients, and the Muon momentum on the GPU at the same time, about 12 bytes per parameter (≈107 GB per GPU on 32 GPUs). Three nodes would need ≈142 GB per GPU. Optimizer offload (the default) keeps the momentum in CPU RAM between steps. Full offload is not used: it supports AdamW and SignSGD only.

| Context parallelism (4 nodes, 131k sequences) | Step time per 1M tokens | Throughput | Peak memory |
|---|---|---|---|
| `cp = 8` | 22.8 s | 46.8k tok/s | 115.6 GiB |
| `cp = 4` (used) | 19.9 s | 55.3k tok/s | 115.4 GiB |

Measured with fake data and balanced routing, FP8, full activation checkpointing and activation offloading (both default). The peak comes from the optimizer step, so a lower CP degree does not raise it.

### Precision: FP8 on both sides

The checkpoint's dense weights sit on the FP8 grid (DeepSeek ships them as FP8), and the bf16 re-export keeps those values. If inference re-quantizes the weights and the trainer does not, small updates round away on the inference side and the two policies drift apart. The H200 overlay therefore quantizes both sides the same way:

- Trainer: dense linears in DeepGEMM blockwise FP8 (`[trainer.model.quantization] type = "fp8"`) and routed experts in DeepGEMM FP8 grouped GEMMs (`moe.compute.type = "deepgemm_fp8"`). The FP8 trainer has the same peak memory as bf16, and its steps take about 5% longer (22.8 s vs 21.8 s at `cp = 8`).
- vLLM: online per-block FP8 (`quantization = "fp8_per_block"`) with FP32 block scales (`VLLM_USE_DEEP_GEMM_E8M0 = "0"`; vLLM defaults to power-of-two scales, which the trainer only uses on Blackwell).
- The attention compressor and the indexer stay bf16 on both sides.

FP8 also halves the inference weights (41 GiB vs 79 GiB per GPU), which leaves more room for KV cache.

### Inference: one DEP8 replica per node

Each replica runs eight data-parallel attention ranks and shards the 256 experts eight ways. All-to-all choice on one 8×H200 node, FP8, `gpu_memory_utilization = 0.85`:

| `all2all_backend` | KV cache per GPU | Startup | 1024 requests, 4k in / 1k out | 2048 requests, 16k in / 512 out |
|---|---|---|---|---|
| `allgather_reducescatter` (vLLM default) | 1.3M tokens | 19 min | 248 s | 582 s |
| `deepep_low_latency` (used) | 12.9M tokens | 9 min | 225 s | 638 s |
| `deepep_high_throughput` | fails at startup (DeepEP CPU recv timeout) | | | |
| `flashinfer_nvlink_one_sided` | not supported on Hopper | | | |

With DeepEP low-latency, one node holds about 100M KV tokens, which is roughly 780 full 131k-token contexts. KV cache does not limit concurrency at 1024 rollouts in flight. Prefill-heavy load runs about 10% slower than with the default backend, but multi-turn rollouts gain more from the larger prefix cache. For comparison, a bf16 replica (no quantization, default backend) holds 0.56M KV tokens per GPU and takes 339 s and 782 s on the two loads.

### Scaling

The trainer stays at 4 nodes. To generate rollouts faster, add inference replicas:

```bash
uv run rl @ examples/advanced/deepseek-v4-flash/rl.toml @ examples/advanced/deepseek-v4-flash/h200.toml \
  --deployment.num-infer-replicas 4
```

Compare the trainer and orchestrator step times: if the trainer waits for batches, add replicas.

## Requirements

- A Slurm cluster with 8-GPU nodes and a shared filesystem, mounted at `/shared` in this guide. The H200 overlay needs 6 nodes.
- About 600 GB of disk for the bf16 checkpoint on the shared filesystem, and about 2.3 TB per trainer checkpoint (FP32 master weights plus Muon momentum).
- **Sandboxes.** Rollout and eval agents run in [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview). Install the `prime` CLI and log in:

```bash
uv tool install prime
prime login   # or: prime config set-api-key <your-key>
```

  To use your own infrastructure, set `env.agent.runtime` on each source to a runtime your environments support (e.g. a local Docker backend).

- `WANDB_API_KEY`, exported in the launch shell (the launcher passes its environment to every component).

## Setup

Clone prime-rl onto the shared filesystem and install everything, including the environments and DeepEP (part of the default extras):

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras --all-packages
```

In `rl.toml`, set the two `# FILL IN` values: `output_dir` and `[slurm] partition`.

## Monitor

```bash
uv run dashboard /shared/outputs   # serves http://localhost:7788
```

Watch `mismatch_kl` on the trainer: with matched FP8 it should stay at bf16-run levels. Watch the in-flight count and the trainer vs orchestrator step times on the orchestrator.

## Checkpoint and resume

`rl.toml` checkpoints every 50 steps and keeps the latest. To resume, re-run the same command with `--resume` and a stable `--run.name`. See [Training](../../../docs/training.md) for resume and export.
