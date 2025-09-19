# Environments

PRIME-RL can train and evaluate in any [`verifiers`](https://github.com/willccbb/verifiers) environments. To train in a new environment, simply install it from the [Environment Hub](https://app.primeintellect.ai/dashboard/environments).

## Installation

You can explore the installation options using

```bash
prime env info <owner>/<name>
```

For example, to install an environment temporarily

```bash
uv run prime env install <owner>/<name>
```

To persist the environment installation in `pyproject.toml` and the lock file, do

```bash
uv add <name> --index https://hub.primeintellect.ai/<owner>/simple/ --optional vf 
```

To verify your installation try

```bash
uv run python -c "import <name>"
```

## Training

To train in the environment, we recommend to create `trainer`, `inference` and `orchestrator` config files similar to those found in the [`examples`](examples) directory, then set `id = <name>` in the `[environment]` section of your `orchestrator` config (along with any desired Environment-level args in `[environment.args]`).

## Evaluating

For quick testing of the environment against an API model, you can use `verifiers` eval entrypoint.

```bash
export OPENAI_API_KEY=...
uv run vf-eval <name>
```