<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<h3 align="center">
PRIME-RL: Decentralized RL Training at Scale
</h3>

---

## Installation

### Setup

**Quick Installation (Recommended)**

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/install.sh | bash
```

After, enter the project repository and optionally install pre-commit hooks (see below).

**Manual Installation**

1. Clone the repository

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Synchronize the environment

```bash
uv sync && uv sync --extra fa
```

4. Install pre-commit hooks

```bash
uv run pre-commit install
```

*Note: For now, development is only possible on CUDA-enabled devices. However, we build production-ready images for both CUDA (NVIDIA) and ROCM (AMD) GPUs that should work out of the box.*


### Validation

1. Check that environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run training debug mode 

```bash
uv run train @ configs/training/debug.toml
```

4. Check that you can run the orchestrator against an inference server

```bash
uv run infer @ configs/inference/debug.toml
```
```bash
uv run orchestrator @ configs/training/orchestrator/debug.toml
```

## 

## RL

To evaluate a base model or your model checkpoint, you have to use the `eval` endpoint which queries into an inference engine.

### Level: Easy

Train a tiny model (`willcb/Qwen2.5-0.5B-Reverse-SFT`) to learn to reverse a small chunk of text. Training is extremely cheap and quick to run because we allow a maximum context of 128 tokens and train on small chunks of text. With two small GPUs (e.g. RTX 3090/ 4090), this experiment should finish in less than 5 minutes.

First, start the inference server

```bash
uv run infer @ configs/inference/reverse_text.toml
```

Then, start the trainer which will spawn the orchestrator as a subprocess

```bash
CUDA_VISIBLE_DEVICES=1 uv run train @ configs/training/reverse_text.toml
```

### Level: Medium

Train a small model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) on high-school level math questions and a relatively small context of 2048 tokens. 

First, start the inference server

```bash
uv run infer @ configs/inference/simple_math.toml --parallel.dp 1
```

Then, start the trainer which will spawn the orchestrator as a subprocess

```bash
uv run train @ configs/training/simple_math.toml
```

*NB: If you have more than 2 GPUs available, the best way to speed up the run is to increase the DP size of the inference worker, i.e. adjusting the `--parallel.dp` argument.*

### Level: Expert

*TBD*

## Evals

*TBD*

## Synthetic Data

To generate synthetic data, simply spin up an inference server and write a script that uses an OAI client to query into the pool.

## Contributing

Before starting to work on your PR, make sure to install prec=

1. Install pre-commit hooks

```bash
uv run pre-commit install
```

2. Run tests

```bash
uv run pytest
```

*The above runs the full test suite*

3. Open PR


## Tests

Run the full test suite 

```bash
uv run pytest -v
```

To run unit tests, run

```bash
uv run pytest tests/unit -v
```

To run integration tests, run

```bash
uv run pytest tests/integration -v
```

To run CPU-only tests, use the inverse of the `gpu` marker:

```bash
uv run pytest -v -m "not gpu"
```

To run fast tests, use the inverse of the `slow` marker:

```bash
uv run pytest -v -m "not slow"
```

## Configs

We use `pydantic-settings` to configure `prime-rl`. To get an overview of the available configurations, run the following command:

```bash
uv run python src/zeroband/training/train.py --help
```

```bash
uv run python src/zeroband/inference/server.py --help
```

### Sources

We support the following sources for configuration, in this order of precedence:

1. **Command-line arguments**: You can pass (nested) arguments as `--key.subkey value` to the script. For example, to set the model name you can run `--model.name`

2. **Config files**: You can pass `.toml` config files (defined in the `configs` directory) using the `@` prefix. For example, to use the `debug.toml` config file, you can run `uv run python src/zeroband/inference/server.py @ configs/inference/debug.toml`. (*If you leave a space between the `@` and the config file, you will get shell path auto-completions.*)

3. **Environment variables**: You can set environment variables to override the config values. All environment variables must be prefixed with `PRIME_` and use the `__` delimiter to nest the keys. For example, to set the model name you can run `export PRIME_MODEL__NAME=Qwen/Qwen3-0.6B`.

4. **Defaults**: For almost all config arguments, we have a default value which will be used if no other source is provided.

In general we recommend setting configurations via config files to define reproducible experiments and use command-line arguments to override the config values to run variants of the same experiment. Environment variables are usually only used in production settings to communicate with the [Prime Protocol](https://github.com/PrimeIntellect-ai/protocol) worker. In most cases, you should not need to use environment variables.

The precendence order will be important if multiple sources try to configure the same argument. For example, in the following command, all sources will define a model name

```toml
# qwen8b.toml
[model]
name = "Qwen/Qwen3-8B"
```

```toml
# qwen14b.toml
[model]
name = "Qwen/Qwen-14B"
```

```bash
PRIME_MODEL__NAME=Qwen/Qwen3-4B uv run src/zeroband/inference/server.py @qwen8b.toml @qwen14b.toml --model.name Qwen/Qwen3-32B
```

In this example, the CLI argument `--model.name Qwen/Qwen3-32B` will take precendence and the script will use `Qwen/Qwen3-32B` as the model name. If the CLI argument wasn't set, then the second config file would take precedence and the script would use `Qwen/Qwen-14B` as the model name. If the second config file wasn't set, then the first config file would take precedence and the script would use `Qwen/Qwen3-8B` as the model name. Finally, if the first config file wasn't set, then the environment variable would take precedence and the script would use `Qwen/Qwen-4B` as the model name. If the environment variable wasn't set, then the default value would be used and the script would use `Qwen/Qwen3-0.6B` as the model name.


## Citation

*TBD*