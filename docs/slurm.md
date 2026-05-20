# SLURM

The `rl`, `sft`, and `inference` entrypoints all have built-in SLURM support. Adding a `[slurm]` section to your config switches from local execution to SLURM job submission — no separate entrypoint needed.

## Quick Start

```bash
# Local run
uv run rl @ examples/reverse_text/rl.toml

# SLURM run (same entrypoint, just add [slurm] to the config)
uv run rl @ examples/reverse_text/slurm_rl.toml
```

The SLURM config is a thin overlay that inherits from a base config and adds `[slurm]` + `[deployment]` sections:

```toml
# examples/reverse_text/slurm_rl.toml
toml_files = ["rl.toml"]

output_dir = "outputs/reverse-text-rl"

[slurm]
job_name = "reverse-text-rl"
```

## How it works

When `[slurm]` is present, the entrypoint:

1. Resolves the full config
2. Renders a SLURM batch script from a Jinja2 template
3. Writes the script and resolved config to `{output_dir}/`
4. Submits via `sbatch` (or prints the script with `--slurm.dry-run`)

For **single-node** jobs, the entire config is dumped to a TOML file and the template simply runs `uv run rl @` or `uv run sft @` on the allocated node.

For **multi-node** jobs, sub-configs are written separately and `srun` dispatches processes across nodes.

## Running inside an existing Isambard allocation

When an agent is already inside a researcher-held Isambard allocation, do not
add `[slurm]` submission settings just to launch a canary or smoke test. The
allocation wrapper provides the SLURM context; the PRIME config should describe
the experiment topology, not request a new job.

From the next allocation that uses the `mnode` wrapper, expect these values to
be pre-baked into the shell environment:

| Variable | Expected value |
|---|---|
| `MASTER_ADDR` | head node hostname |
| `MASTER_PORT` | `29500` |
| `NNODES` | `$SLURM_NNODES` |
| `NPROC_PER_NODE` | `4` on GH200 nodes |
| `GPUSTAT_DIR` | `$PROJECTDIR/joanv.a6r/.gpustat/$SLURM_JOB_ID` |
| `CUDA_HOME` | NVHPC CUDA 13.1 from `.env` |

The wrapper also sources the CUDA forward-compat environment from `.env`, so
`nvidia-smi` should report CUDA 13.1 through the compat `libcuda` shim and
PyTorch should report the CUDA 13 runtime. Verify this at the start of a fresh
allocation before launching GPU-bearing work:

```bash
nvidia-smi | grep CUDA
nvcc --version
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Keep CUDA/NCCL wheel versions in `uv.lock`; `uv pip install` is only a
temporary live-environment override and plain `uv run` will sync back to the
lock. If the lock pins `nvidia-nccl-cu13==2.30.4`, this probe should print
`23004`:

```bash
source .env
uv run python -c "import ctypes; lib=ctypes.CDLL('libnccl.so.2'); v=ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); print(v.value)"
```

For that exact `ctypes.CDLL('libnccl.so.2')` probe to work, `.env` must prepend
the virtualenv NCCL wheel library directory to `LD_LIBRARY_PATH`, for example
`${PWD}/.venv/lib/python3.12/site-packages/nvidia/nccl/lib`.

For distributed launches started from the allocation shell, prefer the wrapper
environment over rediscovering rendezvous values:

```bash
srun --overlap --jobid=$SLURM_JOB_ID --nodes=$NNODES --ntasks-per-node=4 \
  torchrun --nnodes=$NNODES --nproc-per-node=$NPROC_PER_NODE \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT train.py
```

Be precise about wrapper units: `launch-script-mnode 4` means 4 nodes, not 4
GPUs. For single-node work, use the single-node/GPU wrapper instead.

Each `srun` consumes one per-job step id until the job ends. Never put `srun`
inside `watch`, polling loops, or repeated liveness probes. The mnode wrapper's
gpustat daemon is the intended low-step-cost monitor: it uses one persistent
step for the allocation and writes the latest per-node GPU state into
`GPUSTAT_DIR`. Treat those files as live snapshots, not historical utilization
logs; use W&B or explicit metric logging for time series.

## Configuration

### `[slurm]` — Job submission (shared between RL and SFT)

| Field | Description | Default |
|---|---|---|
| `job_name` | SLURM job name | `"prime-rl"` |
| `project_dir` | Path to the project root on the cluster | `"."` |
| `template_path` | Path to a custom Jinja2 template | auto-selected |
| `partition` | SLURM partition | `"cluster"` |
| `nodelist` | Comma-separated list of specific nodes to run on (`--nodelist`) | `None` |
| `exclude` | Comma-separated list of nodes to exclude (`--exclude`) | `None` |
| `account` | SLURM account to charge (`--account`) | `None` |
| `time` | Maximum wall time, e.g. `"24:00:00"` (`--time`) | `None` |
| `pre_run_command` | Shell command to run on head node after env setup, before starting the job (e.g. cleanup) | `None` |

### `[deployment]` — Node and GPU allocation

**RL** uses a discriminated union with `type = "single_node"` (default) or `type = "multi_node"`:

| Field | single_node | multi_node |
|---|---|---|
| `gpus_per_node` | Number of GPUs per node (default: 8) | Same |
| `num_train_gpus` | Training GPUs | — |
| `num_infer_gpus` | Inference GPUs | — |
| `num_train_nodes` | — | Training nodes |
| `num_infer_nodes` | — | Inference nodes |
| `nodes_per_fsdp_group` | — | Nodes per FSDP island (optional) |

**SFT** follows the same pattern but only has training nodes:

| Field | single_node | multi_node |
|---|---|---|
| `gpus_per_node` | Number of GPUs per node (default: 8) | Same |
| `num_gpus` | Number of GPUs (default: 1) | — |
| `num_nodes` | — | Training nodes (default: 2) |
| `nodes_per_fsdp_group` | — | Nodes per FSDP island (optional) |

**Inference** runs independent vLLM replicas per node:

| Field | single_node | multi_node |
|---|---|---|
| `gpus_per_node` | Number of GPUs per node (default: 8) | Same |
| `num_nodes` | — | Number of inference nodes (default: 1) |

The SLURM template is auto-selected based on `deployment.type`. You can override it with `slurm.template_path`.

### Constraints

- `output_dir` should be explicitly set when using SLURM (defaults to `"outputs"`)
- Multi-node deployment requires `[slurm]` to be set

---

## RL Examples

### Single-node SLURM

The simplest case: run on a single allocated node. No `[deployment]` needed — defaults to `single_node`.

```toml
output_dir = "/shared/outputs/my-rl-run"

[slurm]
job_name = "my-rl-run"
```

### Multi-node SLURM (Hendrycks Math)

```toml
output_dir = "outputs/rl-math-moe"
max_steps = 500
seq_len = 2048

[slurm]
job_name = "hendrycks-math-rl-moe"

[deployment]
type = "multi_node"
num_train_nodes = 1
num_infer_nodes = 1

[weight_broadcast]
type = "nccl"

[model]
name = "Qwen/Qwen3-30B-A3B-Thinking-2507"

[trainer.model]
impl = "custom"
attn = "flash_attention_3"
optim_cpu_offload = true

[trainer.model.ac_offloading]
max_inflight_activations = 5

[trainer.model.ac]
freq = 1

[orchestrator]
batch_size = 512
rollouts_per_example = 16

[orchestrator.sampling]
max_tokens = 2048

[[orchestrator.env]]
id = "math-env"
name = "hendrycks-math"
args = { dataset_name = "PrimeIntellect/Hendrycks-Math", dataset_subset = "default" }

[inference.parallel]
tp = 4
dp = 2
```

See [`examples/hendrycks_math/rl.toml`](../examples/hendrycks_math/rl.toml) for the full example.

---

## SFT Examples

### Single-node SLURM

```toml
output_dir = "/shared/outputs/my-sft-run"

[slurm]
job_name = "my-sft-run"
```

### Multi-node SLURM (MoE SFT)

```toml
output_dir = "outputs/sft-moe-math"
max_steps = 500

[slurm]
job_name = "sft-moe-math"

[deployment]
type = "multi_node"
num_nodes = 2

[model]
name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
impl = "custom"
attn = "flash_attention_3"
optim_cpu_offload = true

[model.ac_offloading]
max_inflight_activations = 5

[model.ac]
freq = 1

[data]
type = "sft"
name = "PrimeIntellect/INTELLECT-3-SFT-10K"
subsets = ["default"]
splits = ["math"]
batch_size = 128
seq_len = 8192
```

See [`examples/hendrycks_math/sft.toml`](../examples/hendrycks_math/sft.toml) for the full example.

---

## Inference Examples

### Single-node SLURM

Run a vLLM server on a single allocated node:

```toml
output_dir = "/shared/outputs/my-inference"

[model]
name = "Qwen/Qwen3-8B"

[parallel]
tp = 8

[slurm]
job_name = "my-inference"
```

```bash
uv run inference @ inference_slurm.toml
```

### Multi-node SLURM

Each node runs an independent vLLM replica. TP and DP must fit within a single node — there is no cross-node parallelism.

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

After submission, the SLURM template prints the inference URLs for all nodes (one per node).

### Dry run

Use `dry_run = true` to generate the sbatch script without submitting:

```bash
uv run inference @ config.toml --dry-run true
```

---

## Custom SLURM Templates

The default templates handle standard setups with InfiniBand detection, environment setup, and `srun`-based process dispatch. For advanced use cases (custom partitions, account settings, module loads, etc.), provide your own Jinja2 template:

```bash
uv run rl @ my_config.toml --slurm.template-path path/to/my_template.sbatch.j2
```

See [`src/prime_rl/templates/`](../src/prime_rl/templates/) for the default templates as a starting point.

## Monitoring

After submission, logs are available at:

```bash
# All deployment types (trainer.log and inference.log are symlinks for multi-node)
tail -F {output_dir}/logs/trainer.log
tail -F {output_dir}/logs/orchestrator.log
tail -F {output_dir}/logs/inference.log

# Multi-node: per-node logs
tail -F {output_dir}/logs/trainer/node_*.log
tail -F {output_dir}/logs/inference/node_*.log

# Multi-node inference: per-replica router logs
tail -F {output_dir}/logs/inference/router_*.log
```

For convenience, a tmux launcher sets up a session with all log streams:

```bash
bash scripts/tmux.sh my-rl-job /shared/outputs/my-rl-job
```
