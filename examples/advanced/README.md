# Advanced model configs

RL configs per model at 256k context, layered as in [`deepseek-v4-flash/`](deepseek-v4-flash/):

1. `<model>/rl.toml`: the model definition: the model, its optimizer, and its inference engine
   settings that do not depend on hardware (context length, parsers, prefix caching, router replay).
2. `<model>/gpu/h200_<N>nodes.toml`: a compute overlay for N trainer nodes (a model can have several; `b300_<N>nodes.toml` for B300): trainer nodes and parallelism, fit to one
   262144-token sample per micro batch, and one inference node per trainer node. The trainer keeps
   the defaults: optimizer offload, full activation checkpointing and activation offloading.
3. `data/`: the data overlay: `scaleswe.toml` (ScaleSWE with the bash harness, bash + edit tools,
   eval on SWE-bench Verified), `scaleswe-short.toml` (the same, 20-turn cap), `math.toml`
   (INTELLECT-3 math, eval on AIME 2025).

```bash
uv run rl @ examples/advanced/<model>/rl.toml @ examples/advanced/<model>/gpu/h200_<N>nodes.toml @ examples/advanced/data/scaleswe.toml \
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
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/rl.toml) | 1 | | | | not tested in this layout |
| [DeepSeek-V4-Flash](deepseek-v4-flash/rl.toml) | 4 | | | | not tested in this layout; FP8 trainer and inference, also on B300 (MXFP8) |
| [Nemotron 3.5 Super](nemotron-3.5-super/rl.toml) | 4 | | | | not tested in this layout |
| [Inkling 2 Small](inkling2/rl.toml) | 6 | | | | not tested in this layout; 16k context, as the trainer has no context parallelism for Inkling 2 |

"Reserved" includes PyTorch's allocator cache, so real use is lower; "allocated" is the peak of
live tensors (`perf/peak_allocated_memory`). Both include memory outside the allocator (NCCL, CUDA
context). The inference configs are not fitted yet, and no recipe has run on real weights and data yet.
The models not tested in this layout take their settings from the configs named in their overlay
headers.

## ScaleSWE RL

Real weights, `data/scaleswe.toml` (batch 128, group 8), steady steps after warm-up:

| Model | Trainer + inference | Step time | Trainer waiting | Decode per inference node | Peak allocated | Mismatch KL |
|---|---|---|---|---|---|---|
| Muse Glimmer 30B | 1 (cp 4) + 1 (TP2 × DP4) | 12-14 min | ~65% | 13.5k tok/s | 48 GiB | 0.0003 |
| Muse Glimmer 30B | 1 (cp 4) + 2 (TP2 × DP4) | ~7 min (3-16) | 0-60% | 14k tok/s | 60 GiB | 0.0003 |
| GLM-5.3 Flash | 6 (cp 8) + 6 (DEP8), 600 episodes in flight | 12.5-17 min | ~80% | 1.2-1.5k tok/s | 91 GiB | 0.006 |
| Qwen3.8 Flash Next | 3 (cp 8) + 3 (DEP8) | 1.5 h (step 1 only) | 93% | 2-3.5k tok/s | 102 GiB | 0.0036 |
| Nemotron 3 Super | 2 (cp 8, no AC offload) + 4 (DEP8) | 5-6.5 min | ~75% | 3.5k tok/s | 105 GiB | 0.0005 |
| Qwen3.8 27B | 1 (cp 8, no AC offload) + 1 (TP2 × DP4) | 1-2.5 h | 85-90% | 3k tok/s | 52 GiB | 0.0001 |

The hybrid linear-attention and Mamba models decode several times slower per inference node than
Muse, so their trainers wait for rollouts most of each step; they need more inference nodes per
trainer node. GLM-5.3 Flash needs a cap on episodes in flight (600 here): at 1,536 its small KV cache
(0.9M tokens per DP rank) filled, preempted ~17k requests and doubled the step time.

## Inference layout

Muse Glimmer 30B on math (real weights, ~170 prompt and ~14k output tokens per sample), one 8x H200
inference node, decode throughput in the first minutes of decode, no preemptions:

| Layout | Requests in flight | Decode per node | Decode per GPU | Peak KV cache use |
|---|---|---|---|---|
| TP8 | ~270 (adaptive) | 12.6k tok/s | 1,580 tok/s | 56% |
| TP8 | 512 | 16.0k tok/s | 2,000 tok/s | 49% |
| TP2 × DP4 | 2,048 | 44.7k tok/s | 5,590 tok/s | 49% |

Decode reads every sequence's KV cache each step, and Muse has 2 KV heads: TP above 2 makes several
GPUs read the same heads, and TP8 adds two 8-way all-reduces per layer. More, smaller engines with
many requests in flight decode much faster per GPU.

## Fit a trainer

`fake.toml` runs the trainer alone on fake 256k samples with random weights and balanced routing:

```bash
uv run rl @ examples/advanced/<model>/rl.toml @ examples/advanced/<model>/gpu/h200_<N>nodes.toml @ examples/advanced/fake.toml \
    --slurm.partition <partition> --output-dir <shared-dir>
```

MoE models add `--trainer.model.debug.force-balanced-routing`. Multi-node overlays add
`--trainer.data.fake.batch-size <N>` (one micro batch per data-parallel rank); each overlay's header
has the full command. Read tok/s and `perf/peak_allocated_memory` of steps 2+ in the trainer's `metrics.jsonl`; step 1
includes `torch.compile`. `perf/peak_memory` counts the allocator's cache and reads near-full whenever
memory is tight. Keep about 10% of GPU memory free: real routing is uneven and peaks higher than
balanced routing.
