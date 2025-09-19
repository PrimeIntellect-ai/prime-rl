<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
PRIME-RL: Decentralized RL Training at Scale
</h3>

---

</br>
<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yaml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yaml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/cpu_tests.yaml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/cpu_tests.yaml/badge.svg" alt="Test" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/gpu_tests.yaml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/gpu_tests.yaml/badge.svg" alt="Test" />
  </a>
</p>

## Overview

PRIME-RL is a framework for large-scale reinforcement learning. It is designed to be easy-to-use and hackable, yet capable of scaling to 1000+ GPUs. Beyond that, here is why we think you might like it:

1. Integrates natively with [`verifiers`](https://github.com/willccbb/verifiers) environments via the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars)
2. Supports end-to-end post-training, including [SFT](#sft) and [RL](#rl) training and [Evals](#evals)
3. Rayless multi-node deployment with [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training and [vLLM](https://github.com/vllm-project/vllm) inference backend
4. Designed for asynchronous training in decentralized settings
5. Hackable, modular and extensible by nature

## Setup

> *We develop and test on NVIDIA RTX 3090/4090/5090, A100, H100, H200, and B200. If your setup fails, please create an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

### Prerequisites

> Support for AMD GPUs is on our [roadmap](https://github.com/PrimeIntellect-ai/prime-rl/issues).

Currently, you **need at least one NVIDIA GPU to use PRIME-RL**. If you don't already have access, we recommend our [compute platform](https://app.primeintellect.ai) for everything from renting GPUs with low on-demand rates for developing, debugging and small ablations, to [reserving 1000+ GPU clusters](https://app.primeintellect.ai/dashboard/quotes) for large-scale training.

### Quick Setup

Set up PRIME-RL in a single command.

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>
Manual Setup
</summary>
<br>

1. Clone the repository

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install dependencies from the lock file

```bash
uv sync && uv sync --all-extras
```

</details>

<details>
<summary>
Validate your environment setup
</summary>
<br>

1. Check that the environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run SFT trainer in debug mode (*this requires 1 GPU*)

```bash
uv run sft @ examples/debug/sft.toml
```

4. Check that you can run the RL trainer in debug mode (*this requires 1 GPU*)

```bash
uv run trainer @ examples/debug/rl/train.toml
```

5. Check that you can run the inference server (*this requires 1 GPU*)

```bash
uv run inference @ examples/debug/rl/infer.toml
```

*Keep the inference server running in the background for the next steps.*

5.1. Check that you can run the orchestrator against the inference server

```bash
uv run orchestrator @ examples/debug/rl/orch.toml
```

5.2. Check that you can run evals against the inference server

```bash
uv run eval @ examples/debug/eval.toml
```

</details>

### Additional Setup

1. If you want to log your runs to [W&B](https://wandb.ai), log in

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

2. If you require gated/ private models or datasets from [HuggingFace](https://huggingface.co), log in

```bash
uv run huggingface-cli login
# Or set `export HF_TOKEN=...`
```

## End-to-End Example

We provide end-to-end training examples in the [`examples`](examples) directory. We want to highlight one here:

TBD.

## Docs

Check out these docs to learn more about how to use PRIME-RL.

- [SFT](docs/sft.md)
- [RL](docs/rl.md)
- [Evals](docs/evals.md)
- And more, such as 

## Troubleshooting

- Especially for large training workloads with large batch sizes, you may find yourself getting API timeout errors because your OS limits the number of open files. If this is the case, you can increase the maximum number of open files with

  ```bash
  ulimit -n 32000
  ```

## Contributing

We warmly welcome community contributions! We use [issues](https://github.com/PrimeIntellect-ai/prime-rl/issues) to track bugs, feature requests, and share our internal roadmap. If you encounter bugs, have pain points during development, or have ideas for new features, please open an issue.

Contributions are welcome via PR. Please follow these guidelines:
1. Install the [pre-commit](https://pre-commit.com) hooks to ensure your code is formatted correctly.
  ```bash
  uv run pre-commit install
  ```
2. Please keep your PR in "Draft" until it is ready for review.
3. If your PR resolves an issue, please link the issue in the PR description


## License

This project is licensed under the Apache 2.0 license, as found in the [License](LICENSE) file.

## Citation

If you find our work useful, feel free to cite it using

```tex
@misc{primeintellect2025prime-rl,
  author = {Prime Intellect},
  title = {PRIME-RL},
  url = {https://github.com/PrimeIntellect-ai/prime-rl},
  year = {2025}
}
```
