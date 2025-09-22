# Logging

## Loguru

We log to console and files using `loguru`. We share a single `setup_logger` utility function across all submodules which should be called *exactly once* at the very beginning of each entrypoint to configure the logger as a global instance. It can then be pulled using a `get_logger` utility function into any component that needs to log. For more details on loguru, see the [documentation](https://loguru.readthedocs.io/en/stable/).

## W&B

For most runs we recommend logging to W&B (`wandb`). Since it is disabled by default, you have to set up W&B using the `--wandb` config key.

### RL

Both the trainer and orchestrator can log to W&B as separate runs using the `--monitor.wandb` subconfig. You can set the project (`--monitor.wandb.project`, defaults to `prime-rl`), run name (`--monitor.wandb.name`, defaults to `None` which will make W&B generate a name randomly), run ID (`--monitor.wandb.id`, defaults to `None`), the log directory (`--monitor.wandb.dir`, defaults to `logs`) and whether or not to run in offline mode (`--monitor.wandb.offline`, defaults to `False`). 

First, start your inference server

```bash
uv run inference @ configs/reverse_text/infer.toml
```

Then, start the trainer and orchestrator with logging enabled.

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml --monitor.wandb.project example-project --monitor.wandb.name trainer
```

```bash
uv run orchestrator @ configs/reverse_text/orch.toml --monitor.wandb.project example-project --monitor.wandb.name orchestrator
```

Usually it will be more convenient to use the `rl` entrypoint. To setup W&B concisely, you can specify shared configs using the `--wandb` subconfig, e.g. the project (`--wandb.project`), run name (`--wandb.name`), directory (`--wandb.dir`) and offline mode (`--wandb.offline`). It will automatically share these configs to the trainer and orchestrator. For the run name, it will automatically suffix the specified name with `-trainer` and `-orchestrator` to clearly distinguish those runs.

```bash
uv run rl   \
  --trainer @ configs/reverse_text/train.toml  \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --wandb.project example-project \
  --wandb.name example-run
```

We support logging samples (e.g. prompt, completion, reward, advantage for selected rollouts) and distributions (e.g. reward, advantage, entropy distributions) as W&B tables using the `monitor.wandb.log-extras` subconfig. On the orchestrator you can log activate logging samples (`--monitor.wandb.log-extras.samples`) and distributions (`--monitor.wandb.log-extras.samples`). On the trainer you can only log distributions (`--monitor.wandb.log-extras.distributions`). On both, you can specify the logging step interval using `--monitor.wandb.log-extras.interval`. To log all extras on trainer and orchestrator every 10 steps, 

```bash
uv run rl   \
  --trainer @ configs/reverse_text/train.toml  \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --wandb.project example-project \
  --wandb.name example-run \
  --trainer.monitor.wandb.log-extras.distributions \
  --trainer.monitor.wandb.log-extras.interval 10 \
  --orchestrator.monitor.wandb.log-extras.samples \
  --orchestrator.monitor.wandb.log-extras.distributions \
  --orchestrator.monitor.wandb.log-extras.interval 10
```

### SFT