# Math-Python

In this example, we train `Qwen3-8B` on math problems while having access to a Python REPL tool. We will RL against the [`math-python`](https://app.primeintellect.ai/dashboard/environments/will/math-python) environment.

## Setup

First, let's install the environment using the `prime` CLI.

```bash
prime env install will/math-python
```

Verify your installation by trying to import the environment.

```bash
uv run python -c "import math_python"
```

Start the pre-layouted `tmux` session which we will use to run all experiments and view logs conveniently

```bash
bash scripts/tmux.sh
```

Before training, we want to get a baseline score and test how well `Qwen3-8B` does out-of-the-box in the `math-python` environment so that we quantify our training effect. To do so, first start a local inference server to serve `Qwen3-8B` with the correct tool call parser and give the model the ability to automatically decide whether to use the tool or not.

```bash
# Run this in the `Inference` pane
uv run inference --model.name Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```
```bash
# Run this in the `Trainer` pane
uv run vf-eval math-python -m Qwen/Qwen3-8B -b http://localhost:8000/v1 -n 500 -r 1 -c -1 --max-tokens 8192
```

Reward: 0.704, Correct: 0.714, #Turns: 2.118, #Tool calls: 1.120, Errors: 0.102

Let's also test how the base model that we will use does out-of-the-box. It is the same model, but with a different chat template and tool call format which is inspired by the Qwen3-Coder chat template. We expect it to be worse out of the box.

```bash
uv run inference --model.name PrimeIntellect/Qwen3-8B --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

```bash
uv run vf-eval math-python -m PrimeIntellect/Qwen3-8B -b http://localhost:8000/v1 -n 500 -r 1 -c -1 --max-tokens 8192
```

Reward: 0.646, Correct: 0.660, #Turns: 2.284, #Tool calls: 1.288, Errors: 0.144

## RL

```bash
# Run this in the `Inference` pane
uv run inference @ configs/math-python/rl/infer.toml \
  --parallel.tp ... \
  --parallel.dp ...
```

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ configs/math-python/rl/train.toml \
  --orchestrator @ configs/math-python/rl/orch.toml \
  --trainer-gpu-ids ... \
  --wandb.project ... \
  --wandb.name ...
```

## Evals

### Qwen3-8B

Start the inference server

```bash
uv run inference \
  --model.name Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --parallel.dp ... \
  --parallel.tp ...
```

Run evals against the training environment and common math benchmarks with and without tool use.

```bash
uv run eval \
  --model.name Qwen/Qwen3-8B \
  --environment-ids math-python \
  --num-examples 500 \
  --rollouts-per-example 1
```

Run evals against the training environment and common math benchmarks with and without tool use.

```bash
uv run eval \
  --model.name Qwen/Qwen3-8B \
  --environment-ids math500,aime2024,aime2025 \
  --rollouts-per-example 2,32,32
```

### Qwen3-8B-Math-Python-RL

Start the inference server

```bash
uv run inference \
  --model.name mikasenghaas/Qwen3-8B-Math-Python-RL \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --parallel.dp ... \
  --parallel.tp ...
```

Run evals against the training environment and common math benchmarks with and without tool use.

```bash
# Training environment
uv run eval \
  --model.name mikasenghaas/Qwen3-8B-Math-Python-RL \
  --environment-ids math-python \
  --num-examples 500 \
  --rollouts-per-example 1
```

```bash
# Math benchmarks (w/o tool use)
uv run eval \
  --model.name mikasenghaas/Qwen3-8B-Math-Python-RL \
  --environment-ids math500,aime2024,aime2025 \
  --rollouts-per-example 2,32,32
```

```bash
# Math benchmarks (w/ tool use)
# TBD.
```

```txt
Evaluated math-python in 390.11s (Avg@1=0.8728, Completion Length: 2278.98 (±6523.75, ∈[238.00, 48567.00]), Truncated: 2.0%)
```