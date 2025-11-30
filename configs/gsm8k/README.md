# GSM8K

In this example, we demonstrate how to train `Qwen3-0.6B` to answer math problems from the GSM8K dataset.

> This example runs on 2 GPUs (1 for inference, 1 for training).

## Setup

Install the environment:

```bash
prime env install primeintellect/single-turn-math
```

Verify installation:

```bash
uv run python -c "import single_turn_math"
```

Start the tmux session:

```bash
bash scripts/tmux.sh
```

## Baseline Evaluation

Start the inference server:

```bash
# In the `Inference` pane
uv run inference --model.name PrimeIntellect/Qwen3-0.6B
```

Evaluate the base model:

```bash
# In the `Trainer` pane
uv run vf-eval single-turn-math \
  -a '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}' \
  -m PrimeIntellect/Qwen3-0.6B \
  -b http://localhost:8000/v1 \
  -n 20 \
  -t 2048
```

## RL Training

Train with the config file:

```bash
# In the `Trainer` pane
uv run rl @ configs/gsm8k/rl.toml \
  --wandb.project your-project-name \
  --wandb.name your-run-name \
  --ckpt
```

This will write weight checkpoints in `outputs/weights/step_*`. Upload the final checkpoint to HuggingFace:

```bash
uv run hf upload <user>/Qwen3-0.6B-GSM8K-RL outputs/weights/step_100
```

## Evaluation

Evaluate your trained model:

```bash
# In the `Inference` pane
uv run inference --model.name <user>/Qwen3-0.6B-GSM8K-RL
```

```bash
# In the `Trainer` pane
uv run vf-eval single-turn-math \
  -a '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}'
  -m <user>/Qwen3-0.6B-GSM8K-RL \
  -b http://localhost:8000/v1 \
  -n 20 \
  -t 2048
```