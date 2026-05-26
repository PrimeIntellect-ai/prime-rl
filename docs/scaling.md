# Scaling

This page covers how to scale `prime-rl` from a single GPU to a 1000-GPU cluster: single-node and multi-node deployments, FSDP / expert parallelism / context parallelism, and throughput benchmarking. For knobs that fit on one box, see [Training](training.md) first. For prefill/decode disaggregated inference, see [Advanced](advanced.md#disaggregated-prefilldecode-inference).

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

The launcher allocates GPUs in order from `CUDA_VISIBLE_DEVICES` (or all visible GPUs): inference first, trainer next, teacher last. To target a specific physical subset, pin `CUDA_VISIBLE_DEVICES` before launching.

For quick A/B ablations on the same node, run two RL instances side-by-side in separate tmux sessions, each pinned to half the GPUs and a separate inference port:

```bash
# session 1, GPUs 0–1, default port 8000
bash scripts/tmux.sh -s exp1 -o outputs/exp1
CUDA_VISIBLE_DEVICES=0,1 uv run rl @ rl.toml --output-dir outputs/exp1

# session 2, GPUs 2–3, port 8001
bash scripts/tmux.sh -s exp2 -o outputs/exp2
CUDA_VISIBLE_DEVICES=2,3 uv run rl @ rl.toml \
  --inference.server.port 8001 \
  --orchestrator.client.base-url http://localhost:8001/v1 \
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

EP shards MoE expert weights across the EP mesh, dramatically reducing the FSDP communication volume per layer. EP is only available with the custom model implementation (`model.impl = "custom"` or `"auto"` for supported families).

```toml
[trainer.model]
impl = "custom"
ep = 8                     # EP degree; must divide num_experts
ep_comm_backend = "torch"  # or "deepep"
```

`ep_comm_backend = "deepep"` uses DeepEP's custom dispatch/combine kernels for speed, with two extra knobs (`deepep_num_sms`, `deepep_token_chunk_size`) — tune on your hardware. See [Reference § `trainer.model`](reference.md#trainer-model) for the full set.

### Context Parallelism

CP shards a single sequence across multiple GPUs along the token dimension — for long-context sequences. Only available with the custom impl and flash-attention.

```toml
[trainer.model]
impl = "custom"
attn = "flash_attention_2"   # or fa3 / fa4
cp = 2                       # CP degree
cp_style = "ring"            # "ulysses" for non-FA kernels
```

### Activation Checkpointing and Offloading

| Knob | Memory ↓ | Throughput ↓ |
|---|---|---|
| `trainer.model.ac` | large | ~25% |
| `trainer.model.ac.mode = "selective"` | medium | small | 
| `trainer.model.ac_offloading` | extra | a bit more |

Enable selective AC (custom impl only) for the best memory/throughput tradeoff:

```toml
[trainer.model.ac]
mode = "selective"
targets = ["norm", "attn_proj"]  # see Reference for the full list per architecture
```

### Optimizer Offloading

Offloading optimizer states to CPU is a near-free memory win at low GPU counts:

```toml
[trainer.optim]
# any optimizer type
type = "adamw"

[trainer.model]
optim_cpu_offload = true
```

Mutually exclusive with `fsdp_cpu_offload`. Also incompatible with `trainer.max_concurrent_runs > 1` (multi-tenant training). Muon doesn't support `fsdp_cpu_offload` but does support `optim_cpu_offload`.

### LM Head Chunking

The vanilla LM head materializes a `[batch * seq, vocab]` logits tensor on every step — a major memory tax when the vocabulary is large (often >100K). `fused_lm_head_token_chunk_size` swaps in a custom fused linear + logprob/entropy kernel that streams through `chunk_size` tokens at a time, avoiding the materialization:

```toml
[trainer.model]
fused_lm_head_token_chunk_size = "auto"     # picks 8192 for RL
# or explicit:
# fused_lm_head_token_chunk_size = 1024     # smaller = lower memory, more launches
# fused_lm_head_token_chunk_size = "disabled"  # default; vanilla LM head
```

`auto` is a safe starting point for RL. Drop the chunk size further when peak memory is still tight (e.g. with very long sequences); raise it to amortize kernel-launch overhead. Only available with `model.impl = "custom"`, and currently RL-only — the SFT trainer rejects integer values.

## Memory-Tight Recipe

The kitchen-sink config for fitting large MoE on limited GPUs at acceptable throughput:

```toml
[trainer.model]
impl = "custom"
fused_lm_head_token_chunk_size = 1024
ep = 8
cp = 2
optim_cpu_offload = true

[trainer.model.compile]

[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 1
```

Walks through every memory lever in order: FSDP+EP shard the weights, CP shards the activations along the token dim, AC + AC offloading shrink the activation footprint, fused LM head chunks the loss, `torch.compile` reduces fragmentation, optim offload moves Adam state off GPU. Apply selectively — each knob has a throughput cost.

## SLURM

The `rl`, `sft`, and `inference` entrypoints all submit to SLURM when a `[slurm]` table is present — there's no separate entrypoint.

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

`[deployment]` is a discriminated union picked by `type` — `single_node` or `multi_node` for RL/SFT, with an extra disaggregated variant for inference. For the full `[slurm]` field reference, see [Reference § `slurm`](reference.md#rl-slurm). RL multi-node:

```toml
[deployment]
type = "multi_node"
num_train_nodes = 2
num_infer_nodes = 1
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

Full multi-node configs ship in [`examples/multinode/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/multinode):

- [`rl.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/multinode/rl.toml) — two-node RL run with NCCL weight broadcast on a 30B MoE student.
- [`sft.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/multinode/sft.toml) — two-node SFT against the same model.

For inference-only multi-node, set `[deployment] type = "multi_node"` on an inference TOML — each node runs an independent vLLM replica (TP and DP must fit within one node), and the launcher prints one URL per node. Front the URLs with a router or point clients at any of them.

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
