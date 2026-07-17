# Hendryck's Math

In this example, we demonstrate how to train `Qwen/Qwen3-4B-Instruct-2507` to answer math problems from the Hendryck's Math dataset.

> This example runs on 2 GPUs (1 for inference, 1 for training).

## Setup

The taskset is included through the research-environments workspace. After syncing the repository, verify it with:

```bash
uv run python -c "import math_env_v1"
```

Start the tmux session:

```bash
bash scripts/tmux.sh
```

## Baseline Evaluation

Start the inference server:

```bash
# In the `Inference` pane
uv run inference --model.name Qwen/Qwen3-4B-Instruct-2507
```

Evaluate the base model:

```bash
# In the `Trainer` pane
uv run vf-eval math-env-v1 \
  -a '{"dataset_name": "PrimeIntellect/Hendrycks-Math", "dataset_subset": "default"}' \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 20 \
  -t 2048
```

## RL Training

Train with the config file:

```bash
# In the `Trainer` pane
uv run rl @ configs/hendrycks_math/rl.toml \
  --wandb.project your-project-name \
  --wandb.name your-run-name
```
