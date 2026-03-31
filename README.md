<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
PRIME-RL: Async RL Training at Scale
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

PRIME-RL is a framework for large-scale reinforcement learning. It is designed to be easy to use and hackable, yet capable of scaling to 1000+ GPUs. Here is what we think sets it apart:

1. Fully asynchronous RL for high-throughput agentic training at scale.
2. Performant: built to train 1T+ MoE models on 1000+ GPUs with [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for training and [vLLM](https://github.com/vllm-project/vllm) for inference, with FP8 inference, PD disaggregation, EP and CP parallelism, and more.
3. Native integration with [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environments through the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), including built-in support for SWE and agentic environments.
4. End-to-end post-training: SFT, RL training, and evals.
5. Multi-node deployment with Slurm and Kubernetes support.
6. Multimodal support for VLMs such as Qwen3-VL.
7. Hackable, modular, and extensible by design.


## Models support


The trainer works with both Hugging Face and Prime custom `ModelForCausalLM` out of the box. For selected families (especially large MoE) we also ship highly optimized training code under `src/prime_rl/trainer/models/`, including expert parallelism (EP) for MoE layers and context parallelism (CP) for long sequences (see the table), and additional kernels like [quack-kernels](https://github.com/quack-kernels/quack-kernels).

With `[model] impl = "auto"` (the default), the trainer selects that custom stack when the Hugging Face config type is registered.

| Family | Example IDs | MoE | EP | CP |
|--------|-------------|-----|----|-----|
| GLM-5 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8` | yes | ✅ | ❌ |
| Qwen3 MoE (`qwen3_moe`) | `Qwen/Qwen3-30B-A3B`, … | yes | ✅ | ✅ |
| Qwen3.5 MoE (`qwen3_5_moe`) | `Qwen/Qwen3.5-35B-A3B`, … | yes | ✅ | ✅ |
| Qwen3 / Qwen3.5 VLMs | [multimodal.md](docs/multimodal.md) (`qwen3_vl`, `qwen3_5`, `qwen3_5_moe`) | MoE only on MoE VLMs | MoE only | ✅ |
| MiniMax M2 (`minimax_m2`) | `MiniMax/MiniMax-M2` | yes | ✅ | ✅ |
| Nemotron H (`nemotron_h`) | `nvidia/Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Super-120B-A12B`, … | yes | ✅ | ❌ |
| Trinity (`afmoe`) | `arcee-ai/Trinity-Mini`, … | yes | ✅ | ✅ |
| GLM-4 · GLM-4.5 MoE · INTELLECT-3 (`glm4_moe`) | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | yes | ✅ | ✅ |
| GPT-OSS (HF, MoE) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | yes | ❌ | ✅ |
| Other HF causal LMs | Qwen3 dense, Mistral, … (`impl = "hf"`) | varies | ❌ | ✅ |


## Setup

> *We develop and test on NVIDIA RTX 3090/4090/5090, A100, H100, H200, and B200. If your setup fails, please create an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

### Prerequisites

Currently, you **need at least one NVIDIA GPU to use PRIME-RL**. If you don't already have access to one, we recommend our [compute platform](https://app.primeintellect.ai) for everything from renting on-demand single GPUs for developing, debugging and small ablations, to [reserving 1000+ GPU clusters](https://app.primeintellect.ai/dashboard/quotes) for production-scale training.

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
uv sync --all-extras
```

3.1. Optional: Install Flash Attention 3 (on Hopper GPUs only, for flash_attention_3 attention backend)

> *NOTE*: This step will take a while, as it builds the Flash Attention 3 extension from source, as it has no wheels prebuilt.
> *NOTE*: After this step, you can't run `uv sync --all-extras` or `uv run` as it will uninstall the package, you can avoid it by running `uv sync --inexact` or `uv run --no-sync`

```bash
uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation
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

3. Check that you can run SFT trainer  (*this requires 1 GPU*)

```bash
uv run sft @ configs/debug/sft/train.toml
```

4. Check that you can run the RL trainer (*this requires 1 GPU*)

```bash
uv run trainer @ configs/debug/rl/train.toml
```

5. Check that you can run the inference server (*this requires 1 GPU*)

```bash
uv run inference @ configs/debug/infer.toml
```

*Keep the inference server running in the background for the next steps.*

5.1. Check that you can run the orchestrator against the inference server

```bash
uv run orchestrator @ configs/debug/orch.toml
```

5.2. Check that you can run evals against the inference server

```bash
uv run eval @ configs/debug/eval.toml
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
uv run hf auth login
# Or set `export HF_TOKEN=...`
```

## Training Examples
We provide end-to-end training examples in the [`examples`](examples) directory to highlight features of the framework and guide you through the process of training your own models.

### Basic Training: 1 to 8 GPUs

Follow this guide to learn the basics of Prime-RL. You can train your own models on 1 to 8 GPUs. Ideal for getting started and exploring the capabilities of the framework. These guides cover most use cases -- single-turn, multi-turn, tool calling, etc. -- on toy environments and small models.

1. [**Reverse Text**](examples/reverse_text/README.md): Train `Qwen3-0.6B` to reverse a small chunk of text. Demonstrates tiny-scale single-turn SFT and RL training. Can be trained on a single consumer GPU in a few minutes, and is ideal for getting started.
2. [**Wordle**](examples/wordle/README.md): Train `Qwen3-1.7B` to play Wordle. A fun example of multi-turn SFT and RL training. Can be trained on a 2-4 H100 GPUs in a few hours. Ideal for exploring the multi-turn training capabilities of the framework.
3. [**Alphabet Sort**](examples/alphabet_sort/README.md): Train `Qwen3-4B-Instruct-2507` to sort names alphabetically. Demonstrates multi-turn RL training via LoRA without SFT warmup. Can be trained on a single H100 GPU in just over an hour. Ideal for exploring LoRA-based training.
4. [**Wiki Search**](examples/wiki_search/README.md): Train `Qwen3-4B-Instruct-2507` to answer trivia questions by searching through a Wikipedia. Demonstrates multi-turn with web search tool use.
5. [**Hendrycks Sanity**](examples/hendrycks_sanity/README.md): Run a sanity check experiment on `DeepSeek-R1-Distill-Qwen-1.5B` using a filtered subset of MATH where the model already partially solves 20-80% of problems. Useful for algorithm ablations.

### Advanced Training: 32 - 2048 GPUs:

Follow this guide to train large models on hard reasoning and agentic / swe environments.
These guides are designed to be run from a Slurm cluster but can also be adapted to k8s deployments.

1. [**Qwen 3 30B - A3B Math**](examples/qwen30b_math/README.md): Train `Qwen3-30B-A3B` to solve hard math problems.
2. [**Qwen 3 30B - A3B SWE**](examples/qwen30b_swe/README.md): Train `Qwen3-30B-A3B` to solve hard SWE problems.
3. [**Intellect-3.1**](examples/Intellect-3.1/README.md): Reproduce our `INTELLECT-3.1` training run.
4. [**MiniMax-M2.5 SWE**](examples/minimax_m2.5_swe/README.md): Train `MiniMax-M2.5` on agentic SWE tasks.
5. [**High-throughput GLM-5**](examples/glm5_pd_disag/README.md): Train `GLM-5` with PD disaggregation and FP8 inference on SWE.

## Docs

Check out the [docs](docs) directory for in-depth guides on how to use PRIME-RL.

- [**Entrypoints**](docs/entrypoints.md) - Overview of the main components (orchestrator, trainer, inference) and how to run SFT, RL, and evals
- [**Configs**](docs/configs.md) - Configuration system using TOML files, CLI arguments, and environment variables
- [**Environments**](docs/environments.md) - Installing and using verifiers environments from the Environments Hub
- [**Async Training**](docs/async.md) - Understanding asynchronous off-policy training and step semantics
- [**Logging**](docs/logging.md) - Logging with loguru, torchrun, and Weights & Biases
- [**Checkpointing**](docs/checkpointing.md) - Saving and resuming training from checkpoints
- [**Benchmarking**](docs/benchmarking.md) - Performance benchmarking and throughput measurement
- [**Deployment**](docs/deployment.md) - Training deployment on single-GPU, multi-GPU, and multi-node clusters
- [**Memory Usage**](docs/memory_usage.md) - Techniques for reducing memory usage (activation checkpointing, offloading, EP, CP, LoRA, etc.)
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues and their solutions
- [**Multimodal**](docs/multimodal.md) - Training VLMs like Qwen3-VL

## Contributing

We warmly welcome community contributions! We use [issues](https://github.com/PrimeIntellect-ai/prime-rl/issues) to track bugs, feature requests, and share our internal roadmap. If you encounter bugs, have pain points during development, or have ideas for new features, please open an issue.

Contributions are welcome via PR. Please follow these guidelines:
1. Install the [pre-commit hooks](#pre-commit-hooks) to ensure your code is formatted correctly.
2. Please keep your PR in "Draft" until it is ready for review.
3. If your PR resolves an issue, please link the issue in the PR description
4. If you can, try running the [test suite](#tests) locally to ensure your changes are working as expected.

### Pre-Commit Hooks

Please install the [pre-commit](https://pre-commit.com) hooks to ensure your code is formatted correctly.

```bash
uv run pre-commit install
```

### Tests

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
