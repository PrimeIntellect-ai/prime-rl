# math-python

## Setup

CLone `research-environments`

```bash
cd ~ && curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/research-environments/main/scripts/install.sh | bash && cd -
```

Install the environment as local packages

```bash
uv pip install -e ~/research-environments/environments/math_python
```

This will automatically install the environment, and a pinned verifiers commit (`71006c`) which includes necessary changes to the PythonEnv.

## Eval

Get a quick vibe-check of the model

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507
```

```bash
uv run --no-sync vf-eval math-python -n 16 -r 1 -v -m Qwen/Qwen3-4B-Instruct-2507 -b http://localhost:8000/v1 -a '{"max_turns": 10, "difficulty_key": "avg@8_qwen3_4b_thinking_2507", "min_avg_reward": 0.5}' -t 4096
```

## RL

Again, make sure to have installed the environment as local packages from research-environments

```bash
uv pip install -e ~/research-environments/environments/math_python
```

Then, run RL in debug mode (small batch size, limited turns, 2 GPUs, etc.)

```bash
uv run --no-sync rl @ configs/math_python/math_python.toml --log.level debug
```