# Scaling

This page covers how to scale `prime-rl` from a single GPU to a 1000-GPU cluster: single-node and multi-node deployments, FSDP / expert parallelism / context parallelism, and throughput benchmarking. See [Training](training.md) for detailed documentation of the trainer configuration and [Inference](inference.md) for the inference configuration.

## Table of Contents

- [Single-Node vs. Multi-Node Deployment](#single-node-vs-multi-node-deployment)
  - [Single-Node](#single-node)
    - [RL Placement](#rl-placement)
    - [SFT and Torchrun](#sft-and-torchrun)
  - [Multi-Node](#multi-node)
- [Parallelism Knobs](#parallelism-knobs)
  - [FSDP](#fsdp)
  - [Expert Parallelism](#expert-parallelism)
  - [Context Parallelism](#context-parallelism)
  - [Activation Checkpointing and Offloading](#activation-checkpointing-and-offloading)
  - [Optimizer Offloading](#optimizer-offloading)
  - [LM Head Chunking](#lm-head-chunking)
- [Memory-Tight Recipe](#memory-tight-recipe)
- [SLURM](#slurm)
  - [Activation](#activation)
  - [`[deployment]` Block](#deployment-block)
  - [Examples](#examples)
  - [Custom Templates](#custom-templates)
- [Benchmarking](#benchmarking)

## Single-Node vs. Multi-Node Deployment

The `rl`, `sft`, and `inference` entrypoints all accept a `[deployment]` block (`type = "single_node"` or `"multi_node"`) that picks how the trainer / orchestrator / inference processes are placed across hardware. **Single-node** runs locally; **multi-node** currently goes through [SLURM](#slurm) — the launcher writes an sbatch script that places inference replicas, the orchestrator, and the trainer with the right rendezvous endpoints, IPs, ports, and shared-filesystem paths wired in.

### Single-Node

#### RL Placement

`rl` defaults to 1 trainer GPU and 1 inference GPU. To give inference 6 GPUs with data parallelism and the trainer the remaining 2 on an 8-GPU node:

```bash
uv run rl @ rl.toml \
  --deployment.num-infer-gpus 6 \
  --deployment.num-train-gpus 2 \
  --inference.parallel.dp 6
```

The launcher allocates GPUs in order from `CUDA_VISIBLE_DEVICES` (or all visible GPUs): inference first, trainer next. To target a specific physical subset, pin `CUDA_VISIBLE_DEVICES` before launching.

For quick A/B ablations on the same node, run two RL instances side-by-side in separate tmux sessions, each pinned to half the GPUs and a separate inference port:

```bash
# session 1, GPUs 0–1, default port 8000
bash scripts/tmux.sh -s exp1 -o outputs/exp1
CUDA_VISIBLE_DEVICES=0,1 uv run rl @ rl.toml --output-dir outputs/exp1

# session 2, GPUs 2–3, port 8001
bash scripts/tmux.sh -s exp2 -o outputs/exp2
CUDA_VISIBLE_DEVICES=2,3 uv run rl @ rl.toml \
  --inference.server.port 8001 \
  --orchestrator.model.client.base-url http://localhost:8001/v1 \
  --output-dir outputs/exp2
```

#### SFT and Torchrun

`uv run sft` handles distributed launch internally. To scale from 1 to N GPUs, set the deployment GPU count (or just let it pick up `WORLD_SIZE`). For non-default layouts, the manual equivalent is:

```bash
uv run torchrun \
  --nproc-per-node 8 \
  --local-ranks-filter 0 \
  src/prime_rl/trainer/sft/train.py @ sft.toml
```

`--local-ranks-filter 0` keeps console output to rank 0 only; per-rank stdout/stderr is still captured in `<output_dir>/logs/trainer/torchrun/`.

### Multi-Node

Multi-node deployments (RL or SFT) are launched via [SLURM](#slurm) — set `[deployment] type = "multi_node"` plus the matching `[slurm]` block, and the launcher writes the sbatch script that places inference, orchestrator, and trainer across the requested nodes with the inter-process wiring set up correctly. See [SLURM § Examples](#examples) for full configs.

## Parallelism Knobs

### FSDP

FSDP2 is the default model sharding strategy. By default the trainer fully shards parameters, gradients, and optimizer state across the data-parallel mesh. Tweakable knobs:

| Knob | Effect |
|---|---|
| `trainer.model.dp_replicate` | Number of dimensions to **replicate** instead of shard. Set to 2 to run 2-way DP replication × FSDP sharding within each replica — useful for very large clusters where pure FSDP communication dominates. |
| `trainer.model.reshard_after_forward` | If `true` (default), parameters are resharded after the forward pass to free memory; the backward pass re-gathers. Set `false` to keep params resident — faster but more memory. |
| `trainer.model.fsdp_cpu_offload` | Offload params + grads + optimizer state to CPU. Big memory win, large throughput hit. |
| `trainer.model.optim_cpu_offload` | Offload only optimizer state. Mid-ground — small throughput cost, decent memory savings, especially at low GPU count. |

### Expert Parallelism

EP shards MoE expert weights across the EP mesh, dramatically reducing the FSDP communication volume per layer and improving the training throughput. EP is only available with the custom model implementation (`model.impl = "custom"` or `"auto"` for supported families).

`ep` defaults to `"auto"`, which resolves at startup to the largest valid EP degree up to 8. It loads the model config to read `num_experts`, then picks the biggest divisor of `num_experts` that also divides the FSDP island size (`world_size // dp_replicate`), is a multiple of `cp`, and is <= 8. For non-MoE models, resolves to 1 (no-op). Set `ep` to an explicit integer to override:

```toml
[trainer.model]
impl = "custom"
ep = 8                     # explicit EP degree; must divide num_experts
ep_comm_backend = "torch"  # or "deepep"
```

`ep_comm_backend = "deepep"` uses DeepEP's custom dispatch/combine kernels for speed, with two extra knobs (`deepep_num_sms`, `deepep_token_chunk_size`) — tune on your hardware.

### Context Parallelism

CP shards a single sequence across multiple GPUs along the token dimension — for long-context sequences. We reccomend using `ulysses` style CP for most of the models to get the most throughput. Some models (e.g. GLM-5) only support `ring` style CP. Wrong setting will be rejected on validation.

`ulysses` head-shards Q/K/V, so the CP degree must divide `num_attention_heads`. GQA models with fewer KV heads than the CP degree (e.g. NemotronH: 32 query heads, 2 KV heads) are supported via KV-head replication; the CP degree must then be a multiple of `num_key_value_heads`. Hybrid Mamba layers head-shard independently (`cp_mamba`), which requires the CP degree to divide `mamba_num_heads` and `n_groups`.

```toml
[trainer.model]
impl = "custom"
attn = "auto"                # auto = FA3 on Hopper, FA4 on Blackwell; or flash_attention_2/3/4
cp = 2                       # CP degree
cp_style = "ulysses"         # "ring"
```

### Activation Checkpointing and Offloading

| Knob | Memory ↓ | Throughput ↓ |
|---|---|---|
| `trainer.model.ac` | large | ~25% |
| `trainer.model.ac.mode = "selective"` | medium | small | 
| `trainer.model.ac_offloading` | extra | a bit more |

AC and AC offloading are enabled by default (full mode). For the best memory/throughput tradeoff, switch to selective AC (custom impl only):

```toml
[trainer.model.ac]
mode = "selective"
targets = ["norm", "attn_proj"]  # see Reference for the full list per architecture
```

`ac_offloading` is also on by default with `max_inflight_activations = 5`. We've observed this feature to be very effective, lowering the peak memory usage by 30-40% in some cases, while only lossing ~3-5% of throughput. To disable either, set `model.ac = "None"` or `model.ac_offloading = "None"`.

### Optimizer Offloading

Offloading optimizer states to CPU is enabled by default (`optim_cpu_offload = true`) — a near-free memory win at low GPU counts:

```toml
[trainer.model]
optim_cpu_offload = true   # already the default
```

Mutually exclusive with `fsdp_cpu_offload`. Also incompatible with `trainer.max_concurrent_runs > 1` (multi-tenant training) — set `optim_cpu_offload = false` for multi-run. Muon doesn't support `fsdp_cpu_offload` but does support `optim_cpu_offload`.

The optimizer step is performed per-transformer-layer with stream-overlapped H2D/D2H transfers: while layer *i* computes, layer *i+1* is prefetched and layer *i-1* is evicted. The prefetch waits for the eviction to complete, so at most two layers' optimizer states are on GPU at any time. This reduces peak GPU optimizer-state memory from the full model's states to about two layers' worth, preventing OOM when `weight + grad + all_opt_states > VRAM`.

#### Experimental gradient offloading

`grad_cpu_offload` extends optimizer offloading to FSDP2's sharded gradients:

```toml
[trainer.model]
optim_cpu_offload = true
grad_cpu_offload = true
```

It is opt-in and requires `optim_cpu_offload`. Each finalized gradient shard is copied into a persistent pinned CPU accumulator during backward and its CUDA `DTensor.grad` is released. Intermediate accumulation microbatches can continue copying while the next forward runs, with per-buffer synchronization preventing staging reuse before its CPU add completes. The worker collects the final L2 norm and zero count while those copies arrive. Gradient scaling and clipping are represented by one deferred factor that is applied when each chunk rides the optimizer state's existing H2D stream, so scaling does not require another CPU scan or GPU round trip.

The motivating 32×H200 budget can be estimated from the measured 64-rank peak. A 134 GiB peak at 64 ranks consists of about 114 GiB of non-gradient peak plus about 20 GiB of resident BF16 gradient shards. Halving the shard count doubles gradient bytes per rank: resident BF16 gradients imply roughly 154 GiB at 32 ranks, and FP32 gradients roughly 194 GiB. Both exceed a 141 GiB H200. Moving gradients only after `backward()` does not change that peak because the full set has already accumulated; eviction must overlap backward to approach the roughly 114 GiB non-gradient target.

FSDP2 assigns `param.grad` as a sharded `DTensor` only after reduce-scatter, post-division, and restoration to the parameter dtype. Ordinary autograd post-accumulate timing therefore cannot observe that finalized shard. PyTorch 2.11 manually replays registered post-accumulate hooks after assignment, but that replay is skipped by compiled autograd, and generic helpers such as torchao's do not provide CPU accumulation or the chunked step-side return path. The POC does not depend on that version-specific replay. Conversely, a trainer-level post-backward copy runs too late to lower backward peak.

In the locked PyTorch 2.11 implementation, `FSDPModule.set_all_reduce_hook` runs after reduce-scatter/all-reduce but before post-division, dtype restoration, and `DTensor.grad` assignment. The implementation therefore uses the lag-1 `full_backward_hook` fallback and gates its D2H stream on the FSDP post-reduce event. Recheck these semantics before changing the Torch pin; a future post-assignment all-reduce hook can replace the fallback. The hooks wrap FSDP outside compiled transformer blocks, so the fake-model compile check does not introduce a layer graph break.

Pinned accumulators are allocated once on the first gradient of each dtype and shape. A buffer's accumulation staging allocation is deferred until its second microbatch. The trainer logs total pinned allocation and the number of gradients evicted before backward returned. This lazy allocation also preserves BF16/FP32 router and FP8-produced gradient dtypes rather than guessing from parameter storage. Qwen3-4B with FP32 gradients and two-way sharding uses 7.49 GiB per rank for accumulators and, with accumulation greater than one, another 7.49 GiB for staging.

The local POC harness covers dense FSDP, accumulation greater than one, first-step AdamW state creation, BF16 and FP32 reduction, clipping parity, and compiled layers:

```bash
uv run torchrun --standalone --nproc-per-node=2 scripts/grad_cpu_offload_poc.py \
  --accumulation 3 --reduce-dtype bfloat16 --compile \
  --snapshot-dir /tmp/grad-offload-snapshots
```

Before enabling this on a large run, complete the cluster compatibility matrix:

| Configuration | POC status |
| --- | --- |
| Dense FSDP, BF16 reduce, accumulation > 1 | validated on 2 GPUs |
| Dense FSDP, FP32 reduce | validated on 2 GPUs |
| Compiled transformer blocks (`fullgraph=False`) | validated on 2 GPUs |
| Native HSDP / `dp_replicate > 1` | validated on a 2×1 mesh; larger meshes require cluster validation |
| FP32 router FSDP unit | validated on 2 GPUs |
| Expert parallel parameters | requires cluster validation |
| FP8/MXFP8 parameter paths | requires cluster validation |
| Full and weights-only checkpoint save/resume | requires cluster validation |

Acceptance for a production rollout is update and gradient-norm parity with accumulation greater than one across that matrix, CUDA snapshots showing no full resident gradient set, peak near 114 GiB for the 32-rank target, CPU RAM matching the one-time allocation log, and less than 5% steady-state step-time regression.

The optimized POC still does not meet the `<5%` performance gate on a two-GPU Qwen3-4B SFT benchmark with random initialization, compiled layers, full activation checkpointing, FP32 reduction, and two accumulated microbatches. At sequence length 1024, mean step time increased from 3.585 s to 4.198 s (+17.1%) and throughput fell 13.3%. At sequence length 4096, mean step time increased from 4.342 s to 5.185 s (+19.4%) and throughput fell 15.9%. This improves the original POC's +30.5% and +40.9% regressions by overlapping accumulation copies, collecting statistics on the worker, and deferring scaling to H2D prefetch. The first-step reserved-memory reading improved by about 5.4–6.0 GiB per GPU, but allocator reservation grows as streamed optimizer and gradient chunks populate the CUDA cache, so the benchmark's cumulative `max_memory_reserved` is not sufficient evidence for the target peak. Production work should next use a bounded staging pool or fused D2H accumulation and repeat large-cluster snapshots and timing on the target H200 topology.

### LM Head Chunking

The vanilla LM head materializes a `[batch * seq, vocab]` logits tensor on every step — a major memory tax when the vocabulary is large (often >100K). `fused_lm_head_token_chunk_size` swaps in a custom fused linear + logprob/entropy kernel that streams through `chunk_size` tokens at a time, avoiding the materialization. It defaults to `1024` for RL training:

```toml
[trainer.model]
fused_lm_head_token_chunk_size = 1024       # default
# fused_lm_head_token_chunk_size = "disabled"  # vanilla LM head
```

Drop the chunk size further when peak memory is still tight (e.g. with very long sequences); raise it to amortize kernel-launch overhead. SFT training silently disables this (not supported yet). Only available with `model.impl = "custom"`.

## Memory-Tight Recipe

The kitchen-sink config for fitting large MoE on limited GPUs at acceptable throughput. AC, AC offloading, compile, fused LM head chunking, optimizer offload, and EP auto-resolution are on by default — only CP needs to be set explicitly (and EP overridden if auto-resolution is not desired):

```toml
[trainer.model]
impl = "custom"
ep = 8
cp = 2

[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 1
```

The defaults already cover: fused LM head chunking (`1024`), `torch.compile` (fullgraph=False), AC (full mode), AC offloading (`max_inflight_activations=5`), and optimizer CPU offload. Walks through every memory lever in order: FSDP+EP shard the weights, CP shards the activations along the token dim, AC + AC offloading shrink the activation footprint, fused LM head chunks the loss, `torch.compile` reduces fragmentation, optim offload moves Adam state off GPU. Apply selectively — each knob has a throughput cost.

## SLURM

The `rl`, `sft`, and `inference` entrypoints all submit to SLURM when a `[slurm]` table is present — there's no separate entrypoint.

> **The prime-rl checkout and its `uv` venv must live on a shared filesystem** visible to every node. The generated sbatch script runs a single `uv sync --all-extras --all-packages` on the batch node (not once per node), so all ranks share that one environment — a node-local venv would leave the other nodes stale.

### Activation

A SLURM config is usually a thin overlay that adds `[slurm]` (and `[deployment]` for multi-node) on top of a base config. Configs are composed left-to-right via the `@` CLI syntax — see [Configuration § TOML Composition](configuration.md#toml-composition):

```toml
# my_slurm.toml
output_dir = "/shared/outputs/my-rl"

[slurm]
job_name = "my-rl-run"
```

Launch:

```bash
uv run rl @ base_rl.toml @ my_slurm.toml             # submits via sbatch
uv run rl @ base_rl.toml @ my_slurm.toml --dry-run   # writes the sbatch script + resolved config, exits
```

### `[deployment]` Block

`[deployment]` is a discriminated union picked by `type` — `single_node` or `multi_node` for RL/SFT, with an extra disaggregated variant for inference. RL multi-node:

```toml
[deployment]
type = "multi_node"
num_train_nodes = 2
num_infer_nodes = 1              # optional when inference.deployment defines the node topology
gpus_per_node = 8                # default
nodes_per_fsdp_group = 1         # optional — controls FSDP island size
```

SFT multi-node:

```toml
[deployment]
type = "multi_node"
num_nodes = 2
gpus_per_node = 8
```

### Examples

Full multi-node configs ship under [`examples/advanced/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced):

- [`nemotron-3-super/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/nemotron-3-super/swe.toml) — 4 trainer + 1 inference node RL on a 120B hybrid-Mamba MoE with NCCL weight broadcast.
- [`glm-5.2/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.2) — large-scale and P/D-disaggregated inference across the GLM-5 family.

For inference-only multi-node, set `[deployment] type = "multi_node"` on an inference TOML — each node runs an independent vLLM replica (TP and DP must fit within one node), fronted by a single global router on node 0. Point clients at the router URL the launcher prints.

### Custom Templates

For unusual partitions, module loads, or environment setup, supply your own Jinja2 template:

```bash
uv run rl @ my_config.toml --slurm.template-path path/to/my_template.sbatch.j2
```

The default templates live under [`src/prime_rl/templates/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/templates) — copy one as a starting point.

## Benchmarking

Every entrypoint supports a `--bench` flag that runs a few warm-up + measurement steps with fake data and prints a rich-formatted throughput / MFU table:

```bash
# SFT trainer alone
uv run sft @ sft.toml --bench
uv run sft ... --data.type fake --data.length variable --bench   # variable-length fake data

# RL trainer alone (no inference involved)
uv run trainer @ train.toml --data.fake --bench

# Inference alone — start the server normally, then bench the orchestrator
uv run inference @ infer.toml
uv run orchestrator @ orch.toml --bench

# Full RL stack (trainer with fake data, inference with real data from orchestrator)
uv run rl @ rl.toml --bench
```

Persist results with `--bench.output-json`. Use this to compare parallelism configs before committing a multi-day run.
