# SLURM

For SLURM clusters, use the `rl_slurm` entrypoint. It resolves the full config (trainer, orchestrator, inference), dumps sub-configs as TOML files, renders a SLURM batch script from a Jinja2 template, and submits it with `sbatch`.

```bash
uv run python -m prime_rl.slurm.rl @ examples/slurm/reverse_text.toml
```

This will:

1. Write resolved sub-configs to `{output_dir}/configs/` (trainer.toml, orchestrator.toml, inference.toml)
2. Render the SLURM script to `{output_dir}/rl.sh`
3. Submit the job via `sbatch`

To only generate the script without submitting, use `--dry-run`:

```bash
uv run python -m prime_rl.slurm.rl @ examples/slurm/reverse_text.toml --dry-run
```

## Configuration

The SLURM config extends the standard RL config with the following fields:

| Field | Description |
|---|---|
| `job_name` | SLURM job name |
| `output_dir` | Directory for outputs, sub-configs, and logs |
| `base_dir` | Path to the project root on the cluster (defaults to cwd) |
| `num_train_nodes` | Number of nodes for training |
| `num_infer_nodes` | Number of nodes for inference |
| `slurm_template` | Path to a custom Jinja2 template (optional) |
| `dry_run` | Only generate the script without submitting (default: false) |

All standard RL config fields (model, trainer, orchestrator, inference, wandb, ckpt, etc.) are available and will be resolved into the sub-configs.

A minimal config looks like:

```toml
job_name = "my-rl-job"
output_dir = "/shared/outputs/my-rl-job"
num_train_nodes = 1
num_infer_nodes = 1

[model]
name = "Qwen/Qwen3-4B"

[orchestrator]
batch_size = 16

[[orchestrator.env]]
id = "math"

[inference]
```

## Custom SLURM Templates

The default template handles a standard multi-node setup with NCCL weight broadcast, InfiniBand detection, and `srun`-based process dispatch. For more advanced use cases (custom partitions, account settings, module loads, different networking setups, etc.), provide your own Jinja2 template:

```bash
uv run python -m prime_rl.slurm.rl \
    @ my_config.toml \
    --slurm-template path/to/my_template.sh.j2
```

The template receives the following variables: `job_name`, `base_dir`, `output_dir`, `config_dir`, `num_train_nodes`, `num_infer_nodes`. See `src/prime_rl/slurm/rl_slurm.sh.j2` for the default template as a starting point.

## Monitoring

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
