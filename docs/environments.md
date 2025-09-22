# Environments

PRIME-RL can train and evaluate in any [`verifiers`](https://github.com/willccbb/verifiers) environments. To train in a new environment, simply install it from the [Environment Hub](https://app.primeintellect.ai/dashboard/environments).

## Installation

You can explore the installation options using

```bash
prime env info <owner>/<name>
```

To install an environment temporarily

```bash
uv run prime env install <owner>/<name>
```

To persist the environment installation in `pyproject.toml` and the lock file

```bash
uv add <name> --index https://hub.primeintellect.ai/<owner>/simple/ --optional vf 
```

To verify your installation

```bash
uv run python -c "import <name>"
```