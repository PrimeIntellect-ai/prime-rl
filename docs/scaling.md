# Scaling

This page covers how to scale `prime-rl` from a single GPU to a 1000-GPU cluster: single-node multi-GPU layouts, multi-node SLURM and Kubernetes deployments, FSDP / expert parallelism / context parallelism, disaggregated prefill/decode inference, and throughput benchmarking. For knobs that fit on one box, see [Training](training.md) first.

## Table of Contents

- [Choosing a layout](#choosing-a-layout)
- [Single GPU](#single-gpu)
- [Single-node multi-GPU](#single-node-multi-gpu)
  - [RL placement](#rl-placement)
  - [SFT and torchrun](#sft-and-torchrun)
- [Parallelism knobs](#parallelism-knobs)
  - [FSDP](#fsdp)
  - [Expert parallelism](#expert-parallelism)
  - [Context parallelism](#context-parallelism)
  - [Activation checkpointing and offloading](#activation-checkpointing-and-offloading)
  - [CPU optimizer offload](#cpu-optimizer-offload)
- [Memory-tight recipe](#memory-tight-recipe)
- [Multi-node (manual)](#multi-node-manual)
  - [RL training](#rl-training)
  - [SFT training](#sft-training)
  - [Multi-node inference](#multi-node-inference)
- [SLURM](#slurm)
  - [Activation](#activation)
  - [`[slurm]` and `[deployment]` reference](#slurm-and-deployment-reference)
  - [RL example](#rl-example)
  - [SFT and inference examples](#sft-and-inference-examples)
  - [Custom templates](#custom-templates)
- [Kubernetes](#kubernetes)
- [Disaggregated prefill/decode inference](#disaggregated-prefilldecode-inference)
- [Benchmarking](#benchmarking)

## Choosing a layout

| You have… | Use this layout |
|---|---|
| 1 node, 2–8 GPUs | `uv run rl` with `--deployment.num-infer-gpus N --deployment.num-train-gpus M` |
| 1 node, 8 GPUs, large MoE | Custom impl + EP + activation checkpointing |
| 2+ nodes, SLURM | `[slurm]` + `[deployment]` overlay (recommended) |
| 2+ nodes, no SLURM | Manual `uv run inference` + `uv run orchestrator` + `uv run torchrun src/.../train.py` |
| Kubernetes | The bundled Helm chart at `k8s/prime-rl` |

## Single GPU

The trainer and inference server can share a GPU for small models or smoke tests. Pin everything to one physical GPU via `CUDA_VISIBLE_DEVICES`, set both deployment counts to 1, and tighten the inference memory budget so the trainer has room:

```bash
bash scripts/tmux.sh

CUDA_VISIBLE_DEVICES=0 uv run rl @ configs/<task>/rl.toml \
  --deployment.num-infer-gpus 1 \
  --deployment.num-train-gpus 1 \
  --inference.gpu-memory-utilization 0.5
```

Or launch the three processes manually if you want full control over each pane:

```bash
# inference pane
uv run inference @ infer.toml --gpu-memory-utilization 0.5
# orchestrator pane
uv run orchestrator @ orch.toml
# trainer pane
uv run trainer @ train.toml
```

For SFT, single-GPU is the default — `uv run sft` runs without torchrun unless you ask for multiple processes.

## Single-node multi-GPU

### RL placement

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

### SFT and torchrun

`uv run sft` manages torchrun internally — you don't need to call torchrun yourself. To scale from 1 to N GPUs, set the deployment GPU count (or just let it pick up `WORLD_SIZE`). For non-default layouts, the manual equivalent is:

```bash
uv run torchrun \
  --nproc-per-node 8 \
  --local-ranks-filter 0 \
  src/prime_rl/trainer/sft/train.py @ sft.toml
```

`--local-ranks-filter 0` keeps console output to rank 0 only; per-rank stdout/stderr is still captured in `<output_dir>/logs/trainer/torchrun/`.

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

## Multi-node (manual)

When you don't have SLURM (or want fine-grained control), launch each process by hand. Multi-node RL currently requires a **shared filesystem** for the rollout transport and the weight broadcast.

### RL training

```bash
# On all nodes
export OUTPUT_DIR=/shared/outputs/my-run
export INFERENCE_SERVER_IP=10.0.0.1
export INFERENCE_SERVER_API_KEY=...
```

```bash
# Inference node
uv run inference @ infer.toml \
  --api-key $INFERENCE_SERVER_API_KEY \
  --parallel.tp 4 --parallel.dp 2

# Orchestrator (either node)
uv run orchestrator @ orch.toml \
  --client.base-url http://$INFERENCE_SERVER_IP:8000/v1 \
  --client.api-key-var INFERENCE_SERVER_API_KEY \
  --output-dir $OUTPUT_DIR

# Trainer node
uv run torchrun \
  --nproc-per-node 8 \
  --local-ranks-filter 0 \
  src/prime_rl/trainer/rl/train.py @ train.toml \
  --output-dir $OUTPUT_DIR
```

You can scale inference and trainer independently — multiple inference nodes (each running its own vLLM replica), one orchestrator, one or more trainer nodes. The orchestrator must be a single instance.

### SFT training

For multi-node SFT, point torchrun at a rendezvous endpoint:

```bash
# On all nodes
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=...   # only if default isn't routable
export NCCL_SOCKET_IFNAME=...

# Node 0
uv run torchrun \
  --nnodes 2 --node-rank 0 \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local-ranks-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ sft.toml

# Node 1 — same but --node-rank 1
```

If your nodes aren't colocated, set up a VPN (e.g. Tailscale) and use the VPN-resolvable IP for `MASTER_ADDR`.

### Multi-node inference

Multi-node vLLM uses native data parallelism — see the [vLLM docs](https://docs.vllm.ai/en/v0.10.0/serving/data_parallel_deployment.html). For TP=4, DP=4, two nodes:

```bash
# Node 0 — DP ranks 0,1
uv run inference \
  --parallel.tp 4 --parallel.dp 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address $DATA_PARALLEL_ADDRESS \
  --data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT

# Node 1 — DP ranks 2,3 (headless)
uv run inference \
  --parallel.tp 4 --parallel.dp 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address $DATA_PARALLEL_ADDRESS \
  --data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT \
  --data-parallel-start-rank 2 \
  --headless
```

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

## Kubernetes

For Kubernetes-managed clusters, `prime-rl` ships a Helm chart at [`k8s/prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/k8s/prime-rl). It deploys three StatefulSets (orchestrator, trainer, inference) sharing a single `ReadWriteMany` PVC mounted at `/data`.

```bash
# Deploy with an example values file
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml

# Or with custom overrides
helm install my-exp ./k8s/prime-rl --set trainer.replicas=3 --set inference.replicas=2
```

After deployment, `kubectl exec` into `<release>-trainer-0` and launch with `uv run trainer @ <config>` (or `uv run rl @ <config>`). All three pod groups discover each other via stable DNS hostnames (`<release>-{trainer,orchestrator,inference}-<idx>.<release>-{...}-headless.<ns>.svc.cluster.local`).

Environment variables provided to every pod:

- `$POD_NAME`, `$POD_IP` — standard K8s
- `$STATEFUL_REPLICAS` — total replicas for this component
- `$HEADLESS_SERVICE` — DNS suffix for peer discovery
- `$INFERENCE_URL` — first inference pod's URL (set in orchestrator and trainer pods)

For distributed trainer launches inside K8s, extract the rank from the pod name and feed it to torchrun:

```bash
RANK=$(echo $POD_NAME | grep -o '[0-9]*$')
torchrun \
  --nnodes=$STATEFUL_REPLICAS --node-rank=$RANK \
  --nproc-per-node=8 \
  --rdzv-endpoint=my-exp-trainer-0.$HEADLESS_SERVICE:29501 \
  src/prime_rl/trainer/rl/train.py @ /data/configs/train.toml
```

Common operations (logs, exec, scale, uninstall) are standard `kubectl`/`helm`. Auth (W&B, HF) is via K8s secrets — set `config.secrets.enabled=true` and `config.secrets.name=<your-secret>`.

## Disaggregated prefill/decode inference

For large MoE serving, splitting prefill and decode onto separate vLLM groups can substantially improve throughput. Pick the prefill:decode ratio based on workload shape:

| Workload | P:D ratio | Why |
|---|---|---|
| Agentic (SWE, Lean) | 3:1 | Long growing contexts → prefill-heavy |
| Non-agentic (math, chat) | 1:2 | Short prompts, long generations → decode-heavy |

Example config: [`examples/glm5_pd_disag/rl.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/glm5_pd_disag/rl.toml) — full RL run on `GLM-5` with P/D disaggregation behind a `vllm-router`, FP8 inference, and NCCL weight broadcast (see the [README](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/glm5_pd_disag) for the launch story).

Monitor live queue depths to detect imbalance:

```bash
curl -s http://<prefill_node>:8100/metrics | grep num_requests_waiting
curl -s http://<decode_node>:8200/metrics | grep num_requests_waiting
```

If prefill queues and decode is idle, add prefill nodes (and vice versa).

**UCX 1.19 requirement.** NVSHMEM needs UCX ≥ 1.19 for multi-GPU CUDA. Most clusters ship UCX 1.17 via HPC-X, which manifests as `cuStreamCreate: invalid device context` errors during DeepEP internode dispatch. Check with `/opt/hpcx/ucx/bin/ucx_info -v` and, if needed, build from source:

```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
```

The script writes UCX 1.19 to `third_party/ucx/`; the bundled sbatch templates prepend it to `LD_LIBRARY_PATH` so it overrides the system version.

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
