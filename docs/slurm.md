# SLURM

## RL

For RL on SLURM clusters, use the `rl_slurm` entrypoint. It resolves the full config (trainer, orchestrator, inference), dumps sub-configs as TOML files, renders a SLURM batch script from a Jinja2 template, and submits it with `sbatch`.

```bash
uv run rl_slurm @ examples/slurm/hendrycks_math.toml
```

This will:

1. Write resolved sub-configs to `{output_dir}/configs/` (trainer.toml, orchestrator.toml, inference.toml)
2. Render the SLURM script to `{output_dir}/rl.sh`
3. Submit the job via `sbatch`

To only generate the script without submitting, use `--dry-run`:

```bash
uv run rl_slurm @ examples/slurm/hendrycks_math.toml --dry-run
```

## Configuration

The SLURM config extends the standard RL config with the following fields:

| Field | Description |
|---|---|
| `job_name` | SLURM job name |
| `output_dir` | Directory for outputs, sub-configs, and logs |
| `project_dir` | Path to the project root on the cluster (defaults to cwd) |
| `num_train_nodes` | Number of nodes for training |
| `num_infer_nodes` | Number of nodes for inference |
| `gpus_per_node` | Number of GPUs per node (default: 8) |
| `nodes_per_fsdp_group` | Number of train nodes per FSDP island (optional, auto-sets `trainer.dp_replicate`) |
| `template_path` | Path to a custom Jinja2 template (optional) |
| `dry_run` | Only generate the script without submitting (default: false) |

All standard RL config fields (model, trainer, orchestrator, inference, wandb, ckpt, etc.) are available and will be resolved into the sub-configs.

### Dense model (Hendrycks Math)

```toml
job_name = "hendrycks-math"

num_train_nodes = 1
num_infer_nodes = 1

max_steps = 500
seq_len = 2048

[wandb]
project = "hendrycks-math"
name = "hendrycks-math"

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[orchestrator]
batch_size = 128
rollouts_per_example = 16

[orchestrator.sampling]
max_tokens = 2048

[[orchestrator.env]]
id = "math-env"
name = "hendrycks-math"
args = { dataset_name = "PrimeIntellect/Hendrycks-Math", dataset_subset = "default" }

[inference.parallel]
tp = 1
dp = 8
```

### MoE model (Hendrycks Math)

For MoE models like `Qwen3-30B-A3B`, the config is designed to minimize memory usage on a single training node. It enables activation checkpointing with offloading, optimizer CPU offload, and NCCL weight broadcast:

```toml
job_name = "hendrycks-math-moe"

num_train_nodes = 1
num_infer_nodes = 1

max_steps = 500
seq_len = 2048

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

### Custom SLURM Templates

The default RL template handles a standard multi-node setup with NCCL weight broadcast, InfiniBand detection, and `srun`-based process dispatch. For more advanced use cases (custom partitions, account settings, module loads, different networking setups, etc.), provide your own Jinja2 template:

```bash
uv run rl_slurm \
    @ my_config.toml \
    --slurm.template-path path/to/my_template.sh.j2
```

The template receives the following variables: `job_name`, `project_dir`, `output_dir`, `config_dir`, `num_train_nodes`, `num_infer_nodes`, `gpus_per_node`, `hf_hub_offline`. See `src/prime_rl/slurm/rl_slurm.sh.j2` for the default template as a starting point.

### Monitoring

After submission, the logs are available at:

```bash
tail -f {output_dir}/slurm/latest_train_node_rank_0.log
tail -f {output_dir}/slurm/latest_infer_node_rank_0.log
tail -f {output_dir}/slurm/latest_orchestrator.log
```

For convenience, a tmux launcher is provided that sets up a session with all three log streams:

```bash
bash scripts/slurm_tmux.sh my-rl-job /shared/outputs/my-rl-job
```

This creates a tmux session `slurm-my-rl-job` with two windows:

- **Window 0 (Terminal)**: a plain shell for running commands
- **Window 1 (Logs)**: three vertical panes tailing trainer, orchestrator, and inference logs

The trainer and inference panes use glob patterns (`latest_train_node_rank_*.log`, `latest_infer_node_rank_*.log`) to follow logs from all node ranks. Re-running the same command attaches to the existing session.

---

## SFT

For SFT on SLURM, use the `sft_slurm` entrypoint. It works the same way as `rl_slurm` but only needs a trainer config — no inference or orchestrator nodes.

```bash
uv run sft_slurm @ examples/slurm/sft_moe.toml
```

See [`examples/slurm/sft_moe.toml`](../examples/slurm/sft_moe.toml) for a MoE example with activation checkpointing, CPU offload, and compilation. Use `--dry-run` to generate the script without submitting.
