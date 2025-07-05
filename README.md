<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<h3 align="center">
PRIME-RL: Decentralized RL Training at Scale
</h3>

---

## Installation

**Quick Installation (Recommended)**

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/install.sh | bash
```

After, enter the project directory and, optionally, install pre-commit hooks (see below).

**Manual Installation**

1. Clone the repository

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Synchronize the environment

```bash
uv sync && uv sync --extra fa
```

4. Optionally, install [pre-commit](https://pre-commit.com) hooks

```bash
uv run pre-commit install
```

*For now, development is only possible on CUDA-enabled devices. However, we build production-ready images for both CUDA (NVIDIA) and ROCM (AMD) GPUs that should work out of the box.*

These are some useful checks to do to test the environment setup.

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

## Entrypoints

### RL

**Level: Easy**

Train a tiny model (`willcb/Qwen2.5-0.5B-Reverse-SFT`) to learn to reverse a small chunk of text. Training is extremely cheap and quick to run because we allow a maximum context of 128 tokens and train on small chunks of text. With two small GPUs (e.g. RTX 3090/ 4090), this experiment should finish in less than 5 minutes.

First, start the inference server

```bash
uv run infer @ configs/inference/reverse_text.toml
```

Then, start the trainer which will spawn the orchestrator as a subprocess

```bash
CUDA_VISIBLE_DEVICES=1 uv run train @ configs/training/reverse_text.toml
```

**Level: Medium**

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

**Level: Expert**

*TBD*

### Evals

*TBD*

### Synthetic Data

*TBD*

### Benchmark

*TBD*

## Citation

*TBD*