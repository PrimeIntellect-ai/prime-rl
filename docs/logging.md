# Logging

prime-rl uses [loguru](https://loguru.readthedocs.io/en/stable/) for logging with a global logger pattern. All logs are captured at the deployment level (stdout/stderr redirection for local, `tee` for SLURM) under `{output_dir}/logs/`. For RL training, we recommend streaming logs into tmux panes (as set up by `tmux.sh`).

## Logger Architecture

### `setup_logger` and `get_logger`

We use a **singleton pattern** with a module-level global logger instance (`_LOGGER`).

```python
from prime_rl.utils.logger import setup_logger, get_logger

# At entrypoint - call ONCE
logger = setup_logger("info")

# Anywhere else in codebase
logger = get_logger()
logger.info("Hello world")
```

**How it works:**

1. **`get_logger()`** - Returns the global logger instance. Always works — if `setup_logger` hasn't been called yet, it initializes a default logger automatically. Safe to call from any module at any time.

2. **`setup_logger(log_level)`** - Configures (or reconfigures) the global logger:
   - Creates an isolated loguru `Logger` instance (not the default `loguru.logger`) to prevent third-party code from hijacking our logs
   - Adds a stdout handler with colorized output (or JSON output if `json_logging=True`)
   - Can be called multiple times — cleans up the previous logger before creating a new one

3. **`reset_logger()`** - Resets the global logger to `None`:
   - Used in subprocesses that inherit parent state (e.g., env workers)
   - Used in tests between test cases

## Log File Structure

Logs are captured at the deployment level — the entrypoint redirects subprocess stdout/stderr to files (local) or `tee` captures them (SLURM). The structure is consistent across deployment types: `logs/trainer.log` and `logs/inference.log` always exist, regardless of whether the run is local or multi-node SLURM.

### Local (single node)

```
{output_dir}/logs/
├── trainer.log                  # trainer stdout (rank 0 only)
├── orchestrator.log             # orchestrator stdout
├── inference.log                # vLLM inference server stdout
├── trainer/
│   └── torchrun/                # per-rank stdout/stderr (all ranks)
└── envs/
    ├── train/{env_name}/
    │   ├── env_server.log
    │   └── env_worker_{id}.log
    └── eval/{env_name}/
        └── ...
```

### SLURM multi-node

```
{output_dir}/logs/
├── trainer.log                  -> trainer/node_0.log (symlink)
├── inference.log                -> inference/node_0.log (symlink)
├── orchestrator.log             # orchestrator stdout
├── trainer/
│   ├── node_0.log               # per-node trainer output (rank 0 only)
│   ├── node_1.log
│   └── torchrun/                # per-rank stdout/stderr (all ranks)
├── inference/
│   ├── node_0.log               # per-node inference output
│   ├── node_1.log
│   └── router_0.log             # vllm-router per replica
└── envs/
    └── ...
```

Environment logs live under `logs/envs/train/{env_name}/` and `logs/envs/eval/{env_name}/`. Env log verbosity is controlled by `orchestrator.log.vf_level`.

Only rank 0 output is shown in `trainer.log`. Per-rank logs from all ranks are available under `logs/trainer/torchrun/{rdzv_id}/attempt_0/{rank}/{stdout,stderr}.log`, written by torchrun's `--log-dir`.

## tmux helper (`scripts/tmux.sh`)

`scripts/tmux.sh` sets up a tmux session for RL runs with **four panes**:

- **Trainer**: follows `{output_dir}/logs/trainer.log`
- **Orchestrator**: follows `{output_dir}/logs/orchestrator.log`
- **Envs**: follows `{output_dir}/logs/envs/*/*/*.log`
- **Inference**: follows `{output_dir}/logs/inference.log`
