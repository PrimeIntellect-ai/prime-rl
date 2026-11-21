# Advanced model configs

RL configs per model at 256k context, layered as in [`deepseek-v4-flash/`](deepseek-v4-flash/):

1. `<model>/rl.toml`: the model definition: the model, its optimizer, and its inference engine
   settings that do not depend on hardware (context length, parsers, prefix caching, router replay).
2. `<model>/gpu/h200.toml`: the compute overlay: trainer nodes and parallelism, fit to one
   262144-token sample per micro batch, and one inference node per trainer node. The trainer keeps
   the defaults: optimizer offload, full activation checkpointing and activation offloading.
3. `data/`: the data overlay: `scaleswe.toml` (ScaleSWE with the bash harness, bash + edit tools,
   eval on SWE-bench Verified), `scaleswe-short.toml` (the same, 20-turn cap), `math.toml`
   (INTELLECT-3 math, eval on AIME 2025).

```bash
uv run rl @ examples/advanced/<model>/rl.toml @ examples/advanced/<model>/gpu/h200.toml @ examples/advanced/data/scaleswe.toml \
    --slurm.partition <partition> --output-dir <shared-dir>
```

For a trainer-only run, replace the data overlay with `fake.toml` (see [Fit a trainer](#fit-a-trainer)).

Trainer fits: trainer only, fake 256k samples, random weights, balanced MoE routing, 8x H200 nodes.
Every fit uses Muon, ep 8 (MoE models), cp 8, and the default optimizer offload, full activation
checkpointing and activation offloading.

| Model | Trainer nodes | Throughput | Throughput per GPU | Peak memory | Notes |
|---|---|---|---|---|---|
| [Muse Glimmer 30B](muse-glimmer-30b/rl.toml) | 1 | 12.5k tok/s | 1,569 tok/s | 52.2 GiB (reserved) | lots of headroom |
| [Qwen3.8 27B](qwen3.8-27b/rl.toml) | 1 | 8.7k tok/s | 1,084 tok/s | 73.2 GiB (reserved) | |
| [Nemotron 3 Super](nemotron-3-super/rl.toml) | 2 | 29.5k tok/s | 1,842 tok/s | 106.3 GiB (reserved) | |
| [Qwen3.8 Flash Next](qwen3.8-flash-next/rl.toml) | 3 | 39.1k tok/s | 1,629 tok/s | 109.3 GiB allocated | 2 nodes ran out of memory |
| [GLM-5.3 Flash](glm-5.3-flash/rl.toml) | 6 | 61.0k tok/s | 1,270 tok/s | 94.6 GiB allocated | 4 and 5 nodes ran out of memory |

"Reserved" includes PyTorch's allocator cache, so real use is lower; "allocated" is the peak of
live tensors (`perf/peak_allocated_memory`). Both include memory outside the allocator (NCCL, CUDA
context). The inference configs are not fitted yet, and no recipe has run on real weights and data yet.

## Fit a trainer

`fake.toml` runs the trainer alone on fake 256k samples with random weights and balanced routing:

```bash
uv run rl @ examples/advanced/<model>/rl.toml @ examples/advanced/<model>/gpu/h200.toml @ examples/advanced/fake.toml \
    --slurm.partition <partition> --output-dir <shared-dir>
```

MoE models add `--trainer.model.debug.force-balanced-routing`. Read tok/s and `perf/peak_allocated_memory` of steps 2+ in the trainer's `metrics.jsonl`; step 1
includes `torch.compile`. `perf/peak_memory` counts the allocator's cache and reads near-full whenever
memory is tight. Keep about 10% of GPU memory free: real routing is uneven and peaks higher than
balanced routing.
