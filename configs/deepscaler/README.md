# DeepScaler

This is a reproduction of the [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) experiments. They use RL to train from `Deepseek-R1-Distill-Qwen-1.5B` to 43.1% on AIME2024, surpassing OpenAI’s o1-preview with just 1.5B parameters.

## Setup

Install the environment using the `prime` CLI.

```bash
# TODO: Change to `prime env install ...` later
uv pip install -e ~/prime-environments/environments/deepscaler
```

Verify that the environment is installed correctly.

```bash
uv run python -c "import deepscaler"
```

## Evals

They evaluate on a series of math benchmarks, including Math500, AIME24, AMC23, Minerva Math and Olympiad Math. We will focus on `math500` and `aime2024` for the reproduction, as these are already implemented as evaluation environments on the Environment Hub and also the most prominent benchmarks.

```bash
bash scripts/tmux.sh
```

```bash
# Run this in the `Inference` pane
uv run inference --model.name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 32768
```

```bash
uv run eval @ configs/deepscaler/eval.toml --rollouts-per-example 2,32
```

We get the following results:
- Evaluated math500 in 2184.55s (Avg@2=0.8280, Pass@1: 0.8269, Completion Length: 5473.66 (±7174.64, ∈[310.00, 32729.00]), Truncated: 3.1%)
- Evaluated aime2024 in 2388.36s (Avg@32=0.2750, Pass@16: 0.6800, Completion Length: 17216.25 (±10389.03, ∈[1166.00, 32701.00]), Truncated: 18.8%)

## Training

A key insight from the paper is that they train in **stages of increasing context length**. In stages 1, 2 and 3, they train with context lengths of 8192, 16384 and 24576 tokens, respectively. We match their training setup here.

### Stage 1

```bash
bash scripts/tmux.sh -s stage1 -o outputs/stage1
```

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ configs/deepscaler/stage1/rl/train.toml \
  --orchestrator @ configs/deepscaler/stage1/rl/orch.toml \
  --inference @ configs/deepscaler/stage1/rl/infer.toml \
  --output-dir outputs/stage1 \
  --wandb.project ... \
  --wandb.name ... \
  --log.level debug
```

### Stage 2

TBD.

### Stage 3

TBD.