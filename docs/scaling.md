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
  --inference.vllm.data-parallel-size 6
```

The launcher allocates GPUs in order from `CUDA_VISIBLE_DEVICES` (or all visible GPUs): inference first, trainer next. To target a specific physical subset, pin `CUDA_VISIBLE_DEVICES` before launching.

For quick A/B ablations on the same node, run two RL instances side-by-side in separate shells, each pinned to half the GPUs and a separate inference port, and track both in one dashboard:

```bash
# shell 1, GPUs 0–1, default port 8000
CUDA_VISIBLE_DEVICES=0,1 uv run rl @ rl.toml --run.name exp1

# shell 2, GPUs 2–3, port 8001
CUDA_VISIBLE_DEVICES=2,3 uv run rl @ rl.toml \
  --inference.server.port 8001 \
  --orchestrator.model.client.base-url http://localhost:8001/v1 \
  --run.name exp2

# watch both side by side
uv run dashboard outputs
```

#### SFT and Torchrun

`uv run sft` handles distributed launch internally. To scale from 1 to N GPUs, set the deployment GPU count (or just let it pick up `WORLD_SIZE`). For non-default layouts, the manual equivalent is:

```bash
uv run torchrun \
  --nproc-per-node 8 \
  --local-ranks-filter 0 \
  src/prime_rl/trainer/sft/train.py @ sft.toml
```

`--local-ranks-filter 0` keeps console output to rank 0 only; per-rank stdout/stderr is still captured in `<run_dir>/logs/latest/trainer/torchrun/`.

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
| `trainer.model.optim_cpu_offload` | Offload optimizer state to CPU between steps. Enabled by default. |
| `trainer.model.full_offload` | Offload gradients, FP32 masters, and optimizer state and run the optimizer (AdamW or SignSGD) on CPU during backward. Disabled by default. |

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

State-only optimizer offload remains enabled by default with `model.optim_cpu_offload = true`. For full offload, set `model.optim_cpu_offload = false` and `model.full_offload = true`; this keeps BF16 compute weights on GPU and runs CPU optimizer chunks as gradients become ready during backward. Full offload only supports AdamW and SignSGD (`optim.type = "sign_sgd"`) and disables gradient clipping. SignSGD is stateless, so it halves the host RAM footprint versus AdamW (8 instead of 16 bytes per parameter: FP32 master + FP32 accumulated gradient, no moments).

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
uv run rl @ base_rl.toml @ my_slurm.toml --dry-run   # writes launcher/rl.sbatch + resolved config, exits
```

Every SLURM entrypoint stores its generated scripts and coordination files under `<run_dir>/launcher/`. SLURM stdout and stderr go to `launcher/logs/`. Local launches do not create `launcher/`.

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
num_train_nodes = 2
num_infer_nodes = 1  # required only for online evals
gpus_per_node = 8
```

### Examples

Full multi-node configs ship under [`examples/advanced/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced):

- [`nemotron-3-super/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/nemotron-3-super/swe.toml) — 4 trainer + 1 inference node RL on a 120B hybrid-Mamba MoE with NCCL weight broadcast.
- [`glm-5.2/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.2) — large-scale and P/D-disaggregated inference across the GLM-5 family.

For inference-only multi-node, set `[deployment] type = "multi_node"` on an inference TOML — each node runs an independent vLLM replica (TP and DP must fit within one node), fronted by a single global router on node 0. Point clients at the router URL the launcher prints.

### NIXL weight broadcast

Set `[weight_broadcast] type = "nixl"` to use receiver-driven NIXL weight transfer. Before the first SLURM run, install the NIXL/UCX build and the ModelExpress service binaries on the shared filesystem:

```bash
bash scripts/install_nixl_from_source.sh
uv pip install --reinstall --no-deps deps/nixl_cu12-*.whl
bash scripts/install_modelexpress.sh
```

The generated job starts a job-scoped ModelExpress server and Redis backend on the trainer head node and passes that address to every component. To use an existing service, set `slurm.launch_modelexpress = false` and configure `weight_broadcast.host` and `weight_broadcast.port`.

The launcher requires the CUDA and InfiniBand transports from `third_party/ucx`. Each NIXL process selects the active InfiniBand port nearest its GPU; an explicitly configured `UCX_NET_DEVICES` takes precedence. Inference ranks start their pulls at different trainer ranks so concurrent workers distribute traffic across all available source rails.

ModelExpress exchanges peer metadata during startup. Weight updates reuse prepared NIXL requests, post every trainer-rank read in a transfer group concurrently, and use versioned NIXL notifications for source readiness and buffer credits.

By default, the trainer and inference worker each allocate one transfer arena. Set `weight_broadcast.overlap_transfer_and_replay = true` to allocate two arenas on both sides and replay one weight group while receiving the next. The additional arena is the size of the largest transfer group per GPU; allocation errors are reported instead of silently disabling overlap.

### Custom Templates

For unusual partitions, module loads, or environment setup, supply your own Jinja2 template:

```bash
uv run rl @ my_config.toml --slurm.template-path path/to/my_template.sbatch.j2
```

The default templates live under [`src/prime_rl/templates/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/templates) — copy one as a starting point.

## Benchmarking

To benchmark a parallelism config before committing a multi-day run, run a short training with fake data and a step cap, and read throughput / MFU / step time / peak memory from the logs:

```bash
# SFT trainer alone
uv run sft @ sft.toml --data.type fake --max-steps 4

# RL trainer alone (no inference involved)
uv run trainer @ train.toml --data.fake --max-steps 4
```

Every step logs `Throughput`, `MFU`, and `Peak Mem.` to the console. For machine-readable numbers, the file monitor writes `metrics.jsonl` to the run's output directory by default (`monitors.file`); aggregate `perf/throughput`, `perf/mfu`, `time/step`, and `perf/peak_memory` from the run's `metrics.jsonl` — skip the first step, it is warmup. [`benchmarks/scripts/run_single_benchmark.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/benchmarks/scripts/run_single_benchmark.py) does exactly this and is what the CI benchmark matrix runs.
