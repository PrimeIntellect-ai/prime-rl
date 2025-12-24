# math-python

## Setup

Install the environment as local packages

```bash
uv pip install -e ~/research-environments/environments/math_env
uv pip install -e ~/research-environments/environments/code_env
uv pip install -e ~/research-environments/environments/science_env
uv pip install -e ~/research-environments/environments/logic_env
```

## RL

```bash
uv run --no-sync rl @ configs/env_mix/env_mix.toml --log.level debug
```