# Hendrycks Sanity

This example runs the Hendrycks Sanity Check experiment proposed in [Defeating the Training-Inference Mismatch](https://arxiv.org/abs/2510.26788). The sanity check tests whether an RL algorithm can reliably improve a model on problems it can *already partially solve*. The dataset is filtered from MATH to only include problems where the base model (`DeepSeek-R1-Distill-Qwen-1.5B`) solves 20-80% of the time across 40 rollouts. A reliable algorithm should push training accuracy on this "perfectible" subset above 95%.

Because our trainer is asynchronous, we perform only one gradient step per batch (the inference engine generates the next batch while the trainer processes the current one).

> This example runs on 8 GPUs (4 for inference, 4 for training).

## Setup

Install the math environment:

```bash
prime env install primeintellect/math-env
```

Verify installation:

```bash
uv run python -c "import math_env"
```

## Training

### Local (8 GPUs)

```bash
uv run rl @ examples/hendrycks_sanity/rl.toml \
  --wandb.project your-project \
  --wandb.name your-run
```

### SLURM

```bash
uv run rl @ examples/hendrycks_sanity/slurm_rl.toml \
  --wandb.project your-project \
  --wandb.name your-run
```
