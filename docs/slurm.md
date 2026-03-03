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

## Configuration

### `[slurm]` — Job submission (shared between RL and SFT)

| Field | Description | Default |
|---|---|---|
| `job_name` | SLURM job name | `"prime-rl"` |
| `project_dir` | Path to the project root on the cluster | `"."` |
| `template_path` | Path to a custom Jinja2 template | auto-selected |
| `partition` | SLURM partition | `"cluster"` |

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
# Single-node
tail -F {output_dir}/logs/trainer/rank_0.log

# Multi-node RL
tail -F {output_dir}/slurm/latest_train_node_rank_0.log
tail -F {output_dir}/slurm/latest_infer_node_rank_0.log
tail -F {output_dir}/slurm/latest_orchestrator.log

# Inference (single or multi-node)
tail -F {output_dir}/slurm/latest_infer_node_rank_0.log
```

For convenience, a tmux launcher sets up a session with all log streams:

```bash
bash scripts/slurm_tmux.sh my-rl-job /shared/outputs/my-rl-job
```
