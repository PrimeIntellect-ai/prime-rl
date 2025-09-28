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

| Model | MATH 500 | AIME 2024 | AIME 2025 | MATH 500 (w/ tools) | AIME 2024 (w/ tools) | AIME 2025 (w/ tools) |
|-------|-----------|----------|----------|-----------------|---------------------|---------------------|
| Qwen3-8B | 95.0% (5470±5107) | 73.5% (15019±8507) | - | 82.6% (3431±4176) | 58.8% (11710±7330) | 52.5% (14104±8421) |
| Qwen3-8B-Math-Python-... | - | - | - | - | - | - |
| Qwen3-8B-Math-Python | - | - | - | - | - | - |

### Qwen3-8B

Start the inference server

```bash
uv run inference \
  --model.name Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 \
  --parallel.dp ... \
  --parallel.tp ...
```

Run evals against common math benchmarks with and without tool use.

```bash
uv run eval \
  --model.name Qwen/Qwen3-8B \
  --environment-ids math500,aime2024,aime2025 \
  --environment-args '{"math500": {"use_think": true}, "aime2024": {"use_think": true}, "aime2025": {"use_think": true}}' \
  --rollouts-per-example 1,16,16
```

```bash
uv run eval \
  --model.name Qwen/Qwen3-8B \
  --environment-ids math500,aime2024,aime2025 \
  --environment-args '{"math500": {"use_think": true, "use_tools": true}, "aime2024": {"use_think": true, "use_tools": true}, "aime2025": {"use_think": true, "use_tools": true}}' \
  --rollouts-per-example 1,16,16
```

<details>
<summary>Raw results</summary>
<pre><code>
Without tools:
Evaluated math500 in 1390.31s (Avg@1=0.9500, Pass@1: 0.9500, Completion Length: 5470.91 (±5107.79, ∈[952.00, 32694.00]), Truncated: 0.4%)
Evaluated aime2024 in 1615.29s (Avg@16=0.7354, Pass@8: 0.8257, Completion Length: 15019.11 (±8507.26, ∈[3465.00, 32678.00]), Truncated: 4.8%)

With tools:
Evaluated math500 in 1632.09s (Avg@1=0.8260, Pass@1: 0.8260, Completion Length: 3431.37 (±4176.18, ∈[229.00, 32198.00]), Truncated: 0.2%)
Evaluated aime2024 in 1576.71s (Avg@16=0.5875, Pass@8: 0.8630, Completion Length: 11710.74 (±7330.19, ∈[1136.00, 32346.00]), Truncated: 2.5%)
Evaluated aime2025 in 1603.97s (Avg@16=0.5250, Pass@8: 0.8120, Completion Length: 14104.06 (±8421.81, ∈[949.00, 32360.00]), Truncated: 4.2%)
</code></pre>
</details>

### Qwen3-8B-Math-Python

Start the inference server

```bash
uv run inference \
  --model.name ... \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 32768 \
  --parallel.dp ... \
  --parallel.tp ...
```

Run evals against common math benchmarks with and without tool use.

```bash
# Math benchmarks (w/o tool use)
uv run eval \
  --model.name ... \
  --environment-ids math500,aime2024,aime2025 \
  --environment-args '{"math500": {"use_think": true}, "aime2024": {"use_think": true}, "aime2025": {"use_think": true}}' \
  --rollouts-per-example 1,16,16
```

```bash
# Math benchmarks (w/ tool use)
uv run eval \
  --model.name ... \
  --environment-ids math500,aime2024,aime2025 \
  --environment-args '{"math500": {"use_think": true, "use_tools": true}, "aime2024": {"use_think": true, "use_tools": true}, "aime2025": {"use_think": true, "use_tools": true}}' \
  --rollouts-per-example 1,16,16
```

## Ablations

### Ablations 1

From commit `0r23ugcd`. Train at 8K context with `max_turns=3. Check out the [W&B project](https://wandb.ai/primeintellect/math-python?nw=lpxt5c3z2nr).

| Model | MATH 500 | AIME 2024 | AIME 2025 | MATH 500 (w/ tools) | AIME 2024 (w/ tools) | AIME 2025 (w/ tools) |
|-------|-----------|----------|----------|-----------------|---------------------|---------------------|
| Qwen/Qwen3-8B | 95.0% (5470±5107) | 73.5% (15019±8507) | - | 82.6% (3431±4176) | 58.8% (11710±7330) | 52.5% (14104±8421) |
| PrimeIntellect/Qwen3-8B |  |  |  | Err. | Err. | Err. |
| mikasenghaas/Qwen3-8B-Math-Python-v1-100 | 96.4% (4084±3903) | 73.3% (12494±7709) | 65.0% (15171±8457) | 91.6% (2901±5865) | 75.0% (8724±6630) | 61.6% (11006±6559) |
| mikasenghaas/Qwen3-8B-Math-Python-v1-200 | 95.8% (3498±3551) | 72.3% (11275±7183) | 62.1% (13529±8257) | 95.0% (3498±3551) | 66.6% (8446±6128) | 55.0% (10359±6871) |

<details>
<summary>Raw results (without tools)</summary>
<pre><code>
Base:
Evaluated math500 in 2117.75s (Avg@1=0.9680, Pass@1: 0.9680, Completion Length: 5320.54 (±4752.65, ∈[1027.00, 32711.00]), Truncated: 0.4%)
Evaluated aime2024 in 2169.68s (Avg@16=0.7604, Pass@8: 0.8670, Completion Length: 14531.11 (±8087.02, ∈[3890.00, 32678.00]), Truncated: 4.0%)
Evaluated aime2025 in 2072.47s (Avg@16=0.6813, Pass@8: 0.8097, Completion Length: 17723.54 (±9243.42, ∈[3583.00, 32678.00]), Truncated: 12.3%)

Step 100:
Evaluated math500 in 1586.42s (Avg@1=0.9640, Pass@1: 0.9640, Completion Length: 4084.66 (±3903.90, ∈[714.00, 30126.00]), Truncated: 0.0%)
Evaluated aime2024 in 1375.47s (Avg@16=0.7333, Pass@8: 0.8500, Completion Length: 12494.87 (±7709.14, ∈[2267.00, 32670.00]), Truncated: 1.9%)
Evaluated aime2025 in 1748.57s (Avg@16=0.6500, Pass@8: 0.8487, Completion Length: 15171.81 (±8457.21, ∈[2531.00, 32640.00]), Truncated: 3.3%)

Step 200:
Evaluated math500 in 1142.00s (Avg@1=0.9580, Pass@1: 0.9580, Completion Length: 3498.88 (±3551.77, ∈[645.00, 26636.00]), Truncated: 0.0%)
Evaluated aime2024 in 1414.09s (Avg@16=0.7229, Pass@8: 0.8197, Completion Length: 11275.69 (±7183.51, ∈[2103.00, 32601.00]), Truncated: 0.6%)
Evaluated aime2025 in 1527.87s (Avg@16=0.6208, Pass@8: 0.8233, Completion Length: 13529.70 (±8257.25, ∈[1982.00, 32640.00]), Truncated: 4.0%)
</code></pre>
</details>

<details>
<summary>Raw results (with tools)</summary>
<pre><code>
Base:
Runs out of context - doesn't know how to call tools right.

Step 100:
Evaluated math500 in 372.61s (Avg@2=0.9167, Pass@1: 0.9140, Completion Length: 2901.47 (±5865.64, ∈[401.00, 40366.00]), Truncated: 1.7%)
Evaluated aime2024 in 372.58s (Avg@2=0.7500, Pass@1: 0.7523, Completion Length: 8724.37 (±6630.50, ∈[1743.00, 40445.00]), Truncated: 1.7%)
Evaluated aime2025 in 227.39s (Avg@2=0.6167, Pass@1: 0.6197, Completion Length: 11006.38 (±6559.82, ∈[1795.00, 23014.00]), Truncated: 0.0%)

Step 200:
Evaluated math500 in 91.87s (Avg@2=0.9500, Pass@1: 0.9493, Completion Length: 1885.90 (±2085.23, ∈[265.00, 8447.00]), Truncated: 0.0%)
Evaluated aime2024 in 202.82s (Avg@2=0.6667, Pass@1: 0.6763, Completion Length: 8446.05 (±6128.38, ∈[1131.00, 21280.00]), Truncated: 0.0%)
Evaluated aime2025 in 246.84s (Avg@2=0.5500, Pass@1: 0.5427, Completion Length: 10359.68 (±6871.26, ∈[1292.00, 28027.00]), Truncated: 0.0%)
</code></pre>
</details>

### Ablations 2

From commit 9tchdk0w. Train at 8K context with no turn limit. Check out the [W&B project](https://wandb.ai/primeintellect/math-python/workspace?nw=71j0m1uason).

| Model | MATH 500 | AIME 2024 | AIME 2025 | MATH 500 (w/ tools) | AIME 2024 (w/ tools) | AIME 2025 (w/ tools) |
|-------|-----------|----------|----------|-----------------|---------------------|---------------------|
| Qwen/Qwen3-8B | 95.0% (5470±5107) | 73.5% (15019±8507) | - | 82.6% (3431±4176) | 58.8% (11710±7330) | 52.5% (14104±8421) |
| PrimeIntellect/Qwen3-8B | - | - | - | Err. | Err. | Err. |
| mikasenghaas/Qwen3-8B-Math-Python-v2-100 | 94.4% (3918±3897) | 73.5% (11800±7627) | 60.6% (14122±8764) | - | - | - |
| mikasenghaas/Qwen3-8B-Math-Python-v2-200 | - | - | - | - | - | - |

<details>
<summary>Raw results (without tools)</summary>
<pre><code>
Step 100:
Evaluated math500 in 5878.00s (Avg@1=0.9440, Pass@1: 0.9440, Completion Length: 3918.20 (±3897.48, ∈[731.00, 28060.00]), Truncated: 0.0%)
Evaluated aime2024 in 3827.36s (Avg@16=0.7354, Pass@8: 0.8793, Completion Length: 11800.68 (±7627.06, ∈[1880.00, 32678.00]), Truncated: 2.5%)
Evaluated aime2025 in 6583.08s (Avg@16=0.6062, Pass@8: 0.7807, Completion Length: 14122.16 (±8764.12, ∈[2022.00, 32678.00]), Truncated: 4.4%)

Step 200:
</code></pre>
</details>