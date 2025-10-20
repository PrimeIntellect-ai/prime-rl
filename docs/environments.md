# Environments

PRIME-RL can train and evaluate in any [`verifiers`](https://github.com/willccbb/verifiers) environments. To train in a new environment, simply install it from the [Environment Hub](https://app.primeintellect.ai/dashboard/environments) or install a local environment.

## Installation

You can explore the installation options using

```bash
prime env info <owner>/<name>
```

To install an environment temporarily

```bash
prime env install <owner>/<name>
# Or: uv pip install <name> --extra-index-url https://hub.primeintellect.ai/<owner>/simple/
```

To install a local environment

```bash
uv pip install -e path/to/env
```

To verify your installation

```bash
uv run python -c "import <name>"
```