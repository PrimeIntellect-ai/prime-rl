# Scaling

This page covers how to scale `prime-rl` from a single GPU to a 1000-GPU cluster: single-node and multi-node deployments, FSDP / expert parallelism / context parallelism, and throughput benchmarking. For knobs that fit on one box, see [Training](training.md) first. For prefill/decode disaggregated inference, see [Advanced](advanced.md#disaggregated-prefilldecode-inference).

## Table of Contents

- [Single-node vs. multi-node deployment](#single-node-vs-multi-node-deployment)
  - [Single GPU](#single-gpu)
  - [Single-node multi-GPU](#single-node-multi-gpu)
    - [RL placement](#rl-placement)
    - [SFT and torchrun](#sft-and-torchrun)
  - [Multi-node](#multi-node)
- [Parallelism knobs](#parallelism-knobs)
  - [FSDP](#fsdp)
  - [Expert parallelism](#expert-parallelism)
  - [Context parallelism](#context-parallelism)
  - [Activation checkpointing and offloading](#activation-checkpointing-and-offloading)
  - [CPU optimizer offload](#cpu-optimizer-offload)
- [Memory-tight recipe](#memory-tight-recipe)
- [SLURM](#slurm)
  - [Activation](#activation)
  - [`[slurm]` and `[deployment]` reference](#slurm-and-deployment-reference)
  - [RL example](#rl-example)
  - [SFT and inference examples](#sft-and-inference-examples)
  - [Custom templates](#custom-templates)
- [Benchmarking](#benchmarking)

## Single-node vs. multi-node deployment

The `rl`, `sft`, and `inference` entrypoints all accept a `[deployment]` block (`type = "single_node"` or `"multi_node"`) that picks how the trainer / orchestrator / inference processes are placed across hardware. **Single-node** runs locally; **multi-node** currently goes through [SLURM](#slurm) — the launcher writes an sbatch script that places inference replicas, the orchestrator, and the trainer with the right rendezvous endpoints, IPs, ports, and shared-filesystem paths wired in.

> Manual multi-node launches (`uv run inference` on one set of nodes, `uv run orchestrator` on another, `uv run torchrun src/prime_rl/trainer/rl/train.py` on the trainer nodes) are technically possible — that's what the SLURM launcher does for you under the hood — but you'd be wiring rendezvous endpoints, inference IPs and API keys, the rollout/weight-broadcast paths, and the shared filesystem mounts by hand. We don't currently document that path.

### Single GPU

For SFT, single-GPU is the default — `uv run sft` runs without torchrun unless you ask for multiple processes.

For RL, the `uv run rl` launcher partitions visible GPUs strictly between inference and trainer (inference takes the first `num_infer_gpus`, trainer takes the next `num_train_gpus`), so it needs at least 2 visible GPUs. To smoke-test the full RL stack on a **single physical GPU**, launch the three processes manually in separate panes so they can each pin to the same GPU. Tighten the inference memory budget so the trainer has room:

```bash
bash scripts/tmux.sh

# inference pane
CUDA_VISIBLE_DEVICES=0 uv run inference @ infer.toml --gpu-memory-utilization 0.5

# orchestrator pane
uv run orchestrator @ orch.toml

# trainer pane
CUDA_VISIBLE_DEVICES=0 uv run trainer @ train.toml
```

Single-GPU RL is for debugging only — production RL needs 2+ GPUs.

### Single-node multi-GPU

#### RL placement

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

#### SFT and torchrun

`uv run sft` manages torchrun internally — you don't need to call torchrun yourself. To scale from 1 to N GPUs, set the deployment GPU count (or just let it pick up `WORLD_SIZE`). For non-default layouts, the manual equivalent is:

```bash
uv run torchrun \
  --nproc-per-node 8 \
  --local-ranks-filter 0 \
  src/prime_rl/trainer/sft/train.py @ sft.toml
```

`--local-ranks-filter 0` keeps console output to rank 0 only; per-rank stdout/stderr is still captured in `<output_dir>/logs/trainer/torchrun/`.

### Multi-node

Multi-node deployments (RL or SFT) are launched via [SLURM](#slurm) — set `[deployment] type = "multi_node"` plus the matching `[slurm]` block, and the launcher writes the sbatch script that places inference, orchestrator, and trainer across the requested nodes with the inter-process wiring set up correctly. See [SLURM § RL example](#rl-example) and [SLURM § SFT and inference examples](#sft-and-inference-examples) for full configs.

## Parallelism knobs

### FSDP

FSDP2 is the default model sharding strategy. By default the trainer fully shards parameters, gradients, and optimizer state across the data-parallel mesh. Tweakable knobs:

| Knob | Effect |
|---|---|
| `trainer.model.dp_replicate` | Number of dimensions to **replicate** instead of shard. Set to 2 to run 2-way DP replication × FSDP sharding within each replica — useful for very large clusters where pure FSDP communication dominates. |
| `trainer.model.reshard_after_forward` | If `true` (default), parameters are resharded after the forward pass to free memory; the backward pass re-gathers. Set `false` to keep params resident — faster but more memory. |
| `trainer.model.fsdp_cpu_offload` | Offload params + grads + optimizer state to CPU. Big memory win, large throughput hit. |
| `trainer.model.optim_cpu_offload` | Offload only optimizer state. Mid-ground — small throughput cost, decent memory savings, especially at low GPU count. |

### Expert parallelism

EP shards MoE expert weights across the EP mesh, dramatically reducing the FSDP communication volume per layer. EP is only available with the custom model implementation (`model.impl = "custom"` or `"auto"` for supported families).

```toml
[trainer.model]
impl = "custom"
ep = 8                     # EP degree; must divide num_experts
ep_comm_backend = "torch"  # or "deepep"
```

`ep_comm_backend = "deepep"` uses DeepEP's custom dispatch/combine kernels for speed, with two extra knobs (`deepep_num_sms`, `deepep_token_chunk_size`) — tune on your hardware. See [Reference § `trainer.model`](reference.md#trainer-model) for the full set.

### Context parallelism

CP shards a single sequence across multiple GPUs along the token dimension — necessary for sequences past ~32K tokens. Only available with the custom impl and flash-attention.

```toml
[trainer.model]
impl = "custom"
attn = "flash_attention_2"   # or fa3 / fa4
cp = 2                       # CP degree (typically 2, 4, or 8)
cp_style = "ring"            # "ulysses" for non-FA kernels
```

`cp = 2` or `cp = 4` works for most 128K-token training. Pushing past CP 8 typically isn't worth it — cross-node CP collectives become the bottleneck.

### Activation checkpointing and offloading

| Knob | Memory ↓ | Throughput ↓ |
|---|---|---|
| `trainer.model.ac` | large | ~25% |
| `trainer.model.ac.mode = "selective"` | medium | small | 
| `trainer.model.ac_offloading` | extra (offloads checkpoints to CPU) | a bit more |

Enable selective AC (custom impl only) for the best memory/throughput tradeoff:

```toml
[trainer.model.ac]
mode = "selective"
targets = ["norm", "attn_proj"]  # see Reference for the full list per architecture
```

### CPU optimizer offload

In RL, the trainer typically does many gradient-accumulation steps per optimizer step, so the offload cost is amortized. Offloading optimizer states to CPU is a near-free memory win at low GPU counts:

```toml
[trainer.optim]
# any optimizer type
type = "adamw"

[trainer.model]
optim_cpu_offload = true
```

Mutually exclusive with `fsdp_cpu_offload`. Also incompatible with `trainer.max_concurrent_runs > 1` (multi-tenant training). Muon doesn't support `fsdp_cpu_offload` but does support `optim_cpu_offload`.

## Memory-tight recipe

The kitchen-sink config for fitting large MoE on limited GPUs at acceptable throughput:

```toml
[trainer.model]
impl = "custom"
attn = "flash_attention_2"
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

A SLURM config is usually a thin overlay that adds `[slurm]` (and `[deployment]` for multi-node) on top of a base config. Configs are composed left-to-right via the `@` CLI syntax — see [Configuration § TOML files and composition](configuration.md#toml-files-and-composition):

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

The dry-run mode is invaluable — inspect `<output_dir>/job.sbatch` and the per-process TOMLs before burning a queue slot.

### `[slurm]` and `[deployment]` reference

| `[slurm]` field | Default | Description |
|---|---|---|
| `job_name` | `"prime-rl"` | `#SBATCH --job-name` |
| `project_dir` | `"."` | Project root on the cluster (used to source `.env`, activate `.venv`, run `uv sync`) |
| `partition` | `"cluster"` | `#SBATCH --partition` |
| `nodelist` / `exclude` | `None` | `--nodelist` / `--exclude` |
| `account` | `None` | `--account` |
| `time` | `None` | Wall-time limit |
| `pre_run_command` | `None` | Shell command on head node before launch (cleanup, `pkill`, etc.) |
| `template_path` | auto-selected | Override the Jinja2 template |

`[deployment]` is a discriminated union picked by `type` — `single_node` or `multi_node` for RL/SFT, with an extra disaggregated variant for inference. RL multi-node:

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

### RL example

A two-node RL run with NCCL weight broadcast and a 30B MoE student. Compose with the base config at launch time (`uv run rl @ base.toml @ my_slurm.toml`):

```toml
# my_slurm.toml
output_dir = "/shared/outputs/rl-math-moe"
max_steps = 500
seq_len = 2048

[slurm]
job_name = "hendrycks-math-rl-moe"

[deployment]
type = "multi_node"
num_train_nodes = 1
num_infer_nodes = 1

[weight_broadcast]
type = "nccl"          # synchronous; max_async_level forced to 1

[model]
name = "Qwen/Qwen3-30B-A3B-Thinking-2507"

[trainer.model]
impl = "custom"
attn = "flash_attention_3"
optim_cpu_offload = true

[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 5

[orchestrator]
batch_size = 512
group_size = 16

[[orchestrator.train.env]]
id = "math-env"
name = "hendrycks-math"
args = { dataset_name = "PrimeIntellect/Hendrycks-Math", dataset_subset = "default" }

[inference.parallel]
tp = 4
dp = 2
```

See [`examples/multinode/rl.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/multinode/rl.toml) for a complete worked example.

### SFT and inference examples

SFT multi-node MoE (compose with `uv run sft @ base_sft.toml @ my_slurm.toml`):

```toml
# my_slurm.toml
output_dir = "/shared/outputs/sft-moe-math"
max_steps = 500

[slurm]
job_name = "sft-moe-math"

[deployment]
type = "multi_node"
num_nodes = 2

[model]
name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
impl = "custom"

[data]
type = "sft"
name = "PrimeIntellect/INTELLECT-3-SFT-10K"
batch_size = 128
seq_len = 8192
```

Multi-node inference (each node runs an independent vLLM replica — TP and DP must fit within one node):

```toml
output_dir = "/shared/outputs/my-inference"

[model]
name = "PrimeIntellect/INTELLECT-3-RL-600"

[parallel]
tp = 4
dp = 2

[deployment]
type = "multi_node"
num_nodes = 4

[slurm]
job_name = "my-inference"
```

Submission prints one URL per node — point clients at any of them, or front them with a router.

### Custom templates

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
