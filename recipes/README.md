# Recipes

Configs for frontier models, one folder per model. Most folders hold a layered RL recipe at 256k
context (below). Some folders also hold, or only hold, self-contained configs (see
[Self-contained configs](#self-contained-configs)).

The layered RL recipes have three layers:

1. `<model>/rl.toml`: the model definition: the model, its optimizer, and its inference engine
   settings that do not depend on hardware (context length, parsers, prefix caching, router replay).
2. `<model>/gpu/h200.toml`: the compute overlay (`b300.toml` for B300, `h100.toml` for H100 80GB; its header gives the node counts): trainer nodes and parallelism, fit to one
   262144-token sample per micro batch, and one inference node per trainer node. The trainer keeps
   the defaults: optimizer offload, full activation checkpointing and activation offloading.
3. `data/`: one representative data overlay per domain, updated as models get stronger: `swe.toml`
   (ScaleSWE with the bash harness, bash + edit tools, eval on SWE-bench Verified), `tool-use.toml`
   (general-agent tiers 3-4 with native tool calls, eval on 128 held-out tasks) and `math.toml`
   (INTELLECT-3 math, eval on AIME 2025, for functional tests).

```bash
uv run rl @ recipes/<model>/rl.toml @ recipes/<model>/gpu/h200.toml @ recipes/data/swe.toml \
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
| [DeepSeek-V4-Flash](deepseek-v4-flash-0731/rl.toml) | 4 | | | | not tested in this layout; FP8 trainer and inference, also on B300 (MXFP8) |
| [Nemotron 3.5 Super](nemotron-3.5-super/rl.toml) | 4 | | | | not tested in this layout |
| [Inkling 2 Small](inkling-2-small/rl.toml) | 6 | 41.7k tok/s | 870 tok/s | 107.6 GiB allocated | fake 16k samples (no context parallelism for Inkling 2), FP8 routed experts, full offload, Sign-SGD; RL not tested in this layout |

"Reserved" includes PyTorch's allocator cache, so real use is lower; "allocated" is the peak of
live tensors (`perf/peak_allocated_memory`). Both include memory outside the allocator (NCCL, CUDA
context). The inference configs are not fitted yet, and no recipe has run on real weights and data yet.
The models not tested in this layout take their settings from the configs named in their overlay
headers.

H100 80GB overlays exist for Qwen3.6-35B-A3B, Muse Glimmer 30B and Qwen3.8 27B. They keep the H200
node counts and inference layouts and use cp 8 with activation offloading to fit in 80 GB. Only
Qwen3.6-35B-A3B has run (math, 5 steps): 61.9 GiB peak allocated, 2.0-2.3k tok/s per GPU.

## Self-contained configs

These configs run without overlays. Each header or README gives its launch command.

- [`qwen3-30b-a3b/`](qwen3-30b-a3b/README.md): `Qwen3-30B-A3B` RL on math, SWE and agentic tool use,
  and SFT on H200. 2 train + 2 infer nodes.
- [`glm-4.5-air/`](glm-4.5-air/README.md): `GLM-4.5-Air` RL on search, SWE and terminal at 131k
  context, a two-node SWE budget variant, and an IPO ablation.
- [`glm-5.3/`](glm-5.3/README.md): the GLM-5 family at scale: 16 trainer nodes, P/D-disaggregated FP8
  inference, an llm-d router + Mooncake KV-offload variant, `infer/` pre-flights and H200 SFT.
- [`glm-5.3-flash/swe.toml`](glm-5.3-flash/swe.toml): SWE RL at 128k on 2 trainer nodes + 4 replicas.
- [`inkling-2-small/`](inkling-2-small): `math-rl.toml` and `swe-rl.toml`, 6 trainer nodes + 2
  replicas at 8k and 16k.
- [`inkling-2-mini/math-rl.toml`](inkling-2-mini/math-rl.toml): math RL on one node, 4 trainer GPUs + 4
  single-GPU replicas at 8k.
- [`deepseek-v4-flash-0731/`](deepseek-v4-flash-0731): `sft.toml`, the vLLM serving pre-flight
  `inference.toml`, P/D inference `inference_pd.toml`, and the DSpark draft overlay `dspark.toml`.
- [`minimax-m2.5/swe.toml`](minimax-m2.5/swe.toml): `MiniMax-M2.5` SWE RL.
- [`intellect-3.1/reproduce.toml`](intellect-3.1/reproduce.toml): reproduce the `INTELLECT-3.1` run, 4 train + 12
  infer nodes.
- [`qwen3-235b-a22b/`](qwen3-235b-a22b/README.md): `Qwen3-235B-A22B` SFT on 4 trainer nodes at 16k,
  a max trainer throughput reference.

## ScaleSWE RL

Real weights, `data/swe.toml` (batch 128, group 8), steady steps after warm-up:

| Model | Trainer + inference | Step time | Trainer waiting | Decode per inference node | Peak allocated | Mismatch KL |
|---|---|---|---|---|---|---|
| Muse Glimmer 30B | 1 (cp 4) + 1 (TP2 × DP4) | 12-14 min | ~65% | 13.5k tok/s | 48 GiB | 0.0003 |
| Muse Glimmer 30B | 1 (cp 4) + 2 (TP2 × DP4) | ~7 min (3-16) | 0-60% | 14k tok/s | 60 GiB | 0.0003 |
| GLM-5.3 Flash | 6 (cp 8) + 6 (DEP8), 600 episodes in flight | 12.5-17 min | ~80% | 1.2-1.5k tok/s | 91 GiB | 0.006 (run without router replay; ~0.0004 with it, offline on the same rollouts) |
| Qwen3.8 Flash Next | 3 (cp 8) + 3 (DEP8), 768 episodes in flight | 15-35 min (median ~21) | ~85% | 1.5-4k tok/s | 98 GiB | 0.0035 |
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


Inkling 2 Small (TP8 + EP, online FP8 experts, 32k window, one node, synthetic agentic load):

| max_num_seqs / batched tokens / GPU memory | KV cache | 128 sessions | 192 sessions |
|---|---|---|---|
| 32 / 2,048 / 0.72 (source config) | 242k tokens | 1.4k tok/s, 23 s | – |
| 256 / 8,192 / 0.85 | 97k tokens | 2.4k tok/s, 10 s | collapses (1.0k at 256) |
| **128 / 4,096 / 0.85** | 201k tokens | 2.5k tok/s, 10 s | 2.5k tok/s, 16 s |

## Fit a trainer

`fake.toml` runs the trainer alone on fake 256k samples with random weights and balanced routing:

```bash
uv run rl @ recipes/<model>/rl.toml @ recipes/<model>/gpu/h200.toml @ recipes/fake.toml \
    --slurm.partition <partition> --output-dir <shared-dir>
```

MoE models add `--trainer.model.debug.force-balanced-routing`. Multi-node overlays add
`--trainer.data.fake.batch-size <N>` (one micro batch per data-parallel rank); each overlay's header
has the full command. Read tok/s and `perf/peak_allocated_memory` of steps 2+ in the trainer's `metrics.jsonl`; step 1
includes `torch.compile`. `perf/peak_memory` counts the allocator's cache and reads near-full whenever
memory is tight. Keep about 10% of GPU memory free: real routing is uneven and peaks higher than
balanced routing.
