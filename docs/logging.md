# Logging

prime-rl uses [loguru](https://loguru.readthedocs.io/en/stable/) for logging with a global logger pattern. Logs are written to both console and files under `{output_dir}/logs/`. For RL training, we recommend streaming file logs into tmux panes (as set up by `tmux.sh`).

## tmux helper (`scripts/tmux.sh`)

`scripts/tmux.sh` sets up a tmux session for RL runs with **three panes (one per subprocess)**:

- **Trainer**: run `uv run rl ...` here
- **Orchestrator**: follows `{output_dir}/logs/orchestrator.stdout`
- **Inference**: follows `{output_dir}/logs/inference.stdout`

## Logger Architecture

### `setup_logger` and `get_logger`

We use a **singleton pattern** with a module-level global logger instance (`_LOGGER`).

```python
from prime_rl.utils.logger import setup_logger, get_logger

# At entrypoint - call ONCE
logger = setup_logger("info", log_file=Path("output/logs/my.log"))

# Anywhere else in codebase
logger = get_logger()
logger.info("Hello world")
```

**How it works:**

1. **`setup_logger(log_level, log_file)`** - Initializes the global logger exactly once:
   - Creates an isolated loguru `Logger` instance (not the default `loguru.logger`) to prevent third-party code from hijacking our logs
   - Adds a stdout handler with colorized output
   - Optionally adds a file handler (deletes existing file first)
   - Raises `RuntimeError` if called twice

2. **`get_logger()`** - Returns the global logger instance:
   - Raises `RuntimeError` if `setup_logger` hasn't been called yet
   - Safe to call from any module after initialization

3. **`reset_logger()`** - Resets the global logger to `None`:
   - Used in subprocesses that inherit parent state (e.g., env workers)
   - Used in tests between test cases

## RL Log File Structure

For RL training, logs are organized by component:

| Component | Log Path | Description |
|-----------|----------|-------------|
| **RL (parent)** | `logs/rl.log` | Main process that spawns subprocesses |
| **Inference** | `logs/inference.stdout` | vLLM inference server stdout/stderr |
| **Orchestrator** | `logs/orchestrator.log` | Rollout generation, buffer, scheduling |
| **Trainer** | `logs/trainer/rank_{N}.log` | Training process (one file per GPU rank) |
| **Env Workers** | `logs/env_workers/{env_name}/worker_{N}.log` | Per-environment worker logs (optional) |

## Per-Environment Worker Logging

Environment workers run in **separate subprocesses** to isolate event loop lag. Worker logging is controlled at the orchestrator level via `orchestrator.log`:

```toml
[orchestrator.log]
level = "debug"           # Log level for prime-rl logger
vf_level = "info"         # Log level for verifiers library
env_worker_logs = true    # Enable file logging for env workers
```

When `env_worker_logs = true`, logs are written to:
```
output_dir/
└── logs/
    └── env_workers/
        ├── {env_name_1}.log
        ├── {env_name_2}.log
        └── ...
```

All workers for an environment share the same log file.

Set `env_worker_logs = false` to disable worker file logging (workers inherit parent process logging).

## Torchrun

For multi-node training with `torchrun`, all ranks log simultaneously. To filter to master rank only:

```bash
uv run torchrun \
  --local-ranks-filter 0 \
  --nproc-per-node 8 \
  ...
```

You can also use torchrun's native log redirection:

```bash
uv run torchrun \
  --local-ranks-filter 0 \
  --nproc-per-node 8 \
  --log-dir outputs/torchrun \
  --redirects 3 \
  --tee 3 \
  ...
```

This writes to `outputs/torchrun/{rdzv_id}/attempt_0/{rank}/{stdout,stderr}.log`.

