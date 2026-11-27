<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
prime-rl: Async RL Training at Scale
</h3>

---

</br>
<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/tests.yml">
    <img src="https://github.com/PrimeIntellect-ai/prime-rl/actions/workflows/tests.yml/badge.svg" alt="Tests" />
  </a>
</p>

## Overview

prime-rl is a framework for large-scale reinforcement learning. It is designed to be easy to use and hackable, yet capable of scaling to 1000+ GPUs. Here is what we think sets it apart:

1. Fully asynchronous RL for high-throughput agentic training at scale.
2. Performant: built to train 1T+ MoE models on 1000+ GPUs with [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for training and [vLLM](https://github.com/vllm-project/vllm) for inference, with FP8 inference, PD disaggregation, EP and CP parallelism, and more.
3. Native integration with [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environments through the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), including built-in support for SWE and agentic environments.
4. End-to-end post-training: SFT, RL training, and evals.
5. Multi-node deployment with Slurm support.
6. Multimodal support for Qwen3.5 and Qwen3.8 VLMs, with [image RL and SFT examples](examples/extra/vlm/README.md).
7. Hackable, modular, and extensible by design.
8. One-line SLURM deployment for frontier models — e.g. [`GLM-5` FP8 with P/D disaggregation, the `llm-d` router, and Mooncake KV offload](recipes/glm-5.3/).


## Models support


The trainer runs the PrimeRL model implementations under `src/prime_rl/trainer/models/`: dense Llama, Qwen3 and Qwen3.5, plus the families in the table below, with expert parallelism (EP) for MoE layers, context parallelism (CP) for long sequences, and additional kernels like [quack-kernels](https://github.com/quack-kernels/quack-kernels). Other architectures fail at trainer setup.

| Family | Example IDs | MoE | EP | CP |
|--------|-------------|-----|----|-----|
| GLM-5 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8` | yes | ✅ | ✅ |
| GLM-5.3 Flash, text only (`glm5_next`) | `zai-org/GLM-5.3-Flash-BF16` | yes | ✅ | ✅ |
| Qwen3 MoE (`qwen3_moe`) | `Qwen/Qwen3-30B-A3B`, … | yes | ✅ | ✅ |
| Qwen3.5 MoE (`qwen3_5_moe`) | `Qwen/Qwen3.5-35B-A3B`, … | yes | ✅ | ✅ |
| Qwen3.5 VLMs | see [advanced.md](docs/advanced.md#multimodal-training) (`qwen3_5`, `qwen3_5_moe`) | MoE only on MoE VLMs | MoE only | ✅ |
| Qwen3.8 Flash Next VLM (`qwen4_exp`) | `Qwen/Qwen3.8-Flash-Next` | yes | ✅ | ✅ |
| Poolside Laguna (`laguna`) | `poolside/Laguna-XS.2` | yes | ✅ | ✅ |
| MiniMax M2 (`minimax_m2`) | `MiniMax/MiniMax-M2` | yes | ✅ | ✅ |
| Muse Glimmer, text only (`muse_glimmer`) | `meta-models/Muse-Glimmer-30B` | no | — | ✅ |
| Nemotron H (`nemotron_h`) | `nvidia/Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Super-120B-A12B`, … | yes | ✅ | ✅ |
| Nemotron H Omni (`nemotron_h_omni`, text only) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | yes | ✅ | ✅ |
| Trinity (`afmoe`) | `arcee-ai/Trinity-Mini`, … | yes | ✅ | ✅ |
| GLM-4 · GLM-4.5 MoE · INTELLECT-3 (`glm4_moe`) | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | yes | ✅ | ✅ |
| GPT-OSS (`gpt_oss`) | `unsloth/gpt-oss-20b-BF16`, … | yes | ✅ | ✅ |

## Setup

> *We develop and test on NVIDIA RTX 3090/4090/5090, A100, H100, H200, and B200. If your setup fails, please create an [issue](https://github.com/PrimeIntellect-ai/prime-rl/issues).*

### Prerequisites

Currently, you **need at least one NVIDIA GPU to use prime-rl**. If you don't already have access to one, we recommend our [compute platform](https://app.primeintellect.ai) for everything from renting on-demand single GPUs for developing, debugging and small ablations, to [reserving 1000+ GPU clusters](https://app.primeintellect.ai/dashboard/quotes) for production-scale training.

> *NOTE*: Intel GPUs (XPU) run prime-rl from [`Dockerfile.xpu`](Dockerfile.xpu), whose header has the build and `docker run` commands.

### Quick Setup

Set up prime-rl in a single command.

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

2. Initialize submodules

```bash
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
```

3. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

4. Install dependencies from the lock file

```bash
uv sync --all-extras
```

> *NOTE*: Environments are opt-in uv workspace members — `uv sync --all-extras` does not install them. To train on them, install all with `uv sync --all-extras --all-packages`, or a subset with `uv sync --package prime-rl --package <env>`.

> *NOTE*: The `gpu` extra is the union of `train` (the trainer stack) and `inference` (vLLM and the transports that broadcast weights to it). SFT and pretraining without online evals only need `uv sync --extra train`; RL and online evals need both. FP8 training, MegaMoE and the `cudnn_flashmla` sparse-attention backend use kernels bundled in the vLLM wheel, so they need `inference` as well.

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
uv run sft @ dev/fake/sft.toml
```

4. Check that you can run the inference server (*this requires 1 GPU*)

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B
```

5. Check that the full RL stack (inference + orchestrator + trainer) runs end-to-end (*this requires 2 GPUs*)

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
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

## Examples

The [`examples`](examples) directory holds end-to-end walkthroughs on 1 to 8 GPUs. Each one has a README that takes you from a baseline eval through training to a final eval. Start here to learn prime-rl.

1. [**Reverse Text**](examples/basic/reverse-text/README.md): Train `Qwen3-0.6B` to reverse a small chunk of text. Tiny single-turn SFT and RL on a single consumer GPU in a few minutes. Start here.
2. [**Wordle**](examples/basic/wordle/README.md): Train `Qwen3-1.7B` to play Wordle. Multi-turn SFT and RL on 2-4 H100s in a few hours.
3. [**Alphabet Sort**](examples/basic/alphabet-sort/README.md): Train `Qwen3-4B-Instruct-2507` to sort names alphabetically. Multi-turn LoRA RL without SFT warmup on one H100.
4. [**Wiki Search**](examples/basic/wiki-search/README.md): Train `Qwen3-4B-Instruct-2507` to answer trivia by searching Wikipedia. Multi-turn tool use.
5. [**Hendrycks Sanity**](examples/basic/hendrycks-sanity/README.md): Train `Qwen3-1.7B` on a MATH subset that `DeepSeek-R1-Distill-Qwen-1.5B` partially solves (20-80%). Useful for algorithm ablations.
6. [**VLM**](examples/extra/vlm/README.md): Image RL for Qwen3.5 and Qwen3.8, and LoRA SFT for dense and MoE models.
7. [**Dynamo**](examples/extra/dynamo/README.md): Train with external Dynamo inference and NCCL weight updates.

## Recipes

The [`recipes`](recipes) directory holds the configs for frontier RL, one folder per model. They run multi-node on a Slurm cluster. Most recipes compose a model config, a GPU overlay and a data overlay; see the [recipes README](recipes/README.md) for the layout, trainer fits and RL results.

| Model | Recipe |
|---|---|
| `DeepSeek-V4-Flash` | [`deepseek-v4-flash-0731`](recipes/deepseek-v4-flash-0731/) |
| `GLM-4.5-Air` | [`glm-4.5-air`](recipes/glm-4.5-air/) |
| `GLM-5.3` | [`glm-5.3`](recipes/glm-5.3/) |
| `GLM-5.3-Flash` | [`glm-5.3-flash`](recipes/glm-5.3-flash/) |
| `INTELLECT-3.1` | [`intellect-3.1`](recipes/intellect-3.1/) |
| `Inkling-2-Small` | [`inkling-2-small`](recipes/inkling-2-small/) |
| `MiniMax-M2.5` | [`minimax-m2.5`](recipes/minimax-m2.5/) |
| `Muse-Glimmer-30B` | [`muse-glimmer-30b`](recipes/muse-glimmer-30b/) |
| `Nemotron-3-Super-120B-A12B` | [`nemotron-3-super`](recipes/nemotron-3-super/) |
| `Nemotron-3.5-Super-VL-120B-A12B` | [`nemotron-3.5-super`](recipes/nemotron-3.5-super/) |
| `Qwen3-30B-A3B` | [`qwen3-30b-a3b`](recipes/qwen3-30b-a3b/) |
| `Qwen3-235B-A22B` | [`qwen3-235b-a22b`](recipes/qwen3-235b-a22b/) |
| `Qwen3.6-35B-A3B` | [`qwen3.6-35b-a3b`](recipes/qwen3.6-35b-a3b/) |
| `Qwen3.8-27B` | [`qwen3.8-27b`](recipes/qwen3.8-27b/) |
| `Qwen3.8-Flash-Next` | [`qwen3.8-flash-next`](recipes/qwen3.8-flash-next/) |

## Docs

Check out the [docs](docs) directory for in-depth guides on how to use prime-rl.

- [**Overview**](docs/overview.md) - Architecture, install, and a copy-pasteable end-to-end RL run
- [**Configuration**](docs/configuration.md) - TOML composition, CLI overrides, env vars, validation
- [**Training**](docs/training.md) - RL, SFT, checkpointing, observability, rules of thumb
- [**Eval**](docs/eval.md) - Standalone evals against any inference server, online evals, resume, platform upload
- [**Scaling**](docs/scaling.md) - Single-GPU through multi-node, FSDP/EP/CP, SLURM, benchmarking
- [**Algorithms**](docs/algorithms.md) - Async training and staleness, the AIPO loss, advantage and filter plugins, trajectory merging
- [**Advanced**](docs/advanced.md) - Custom modeling, multimodal training, LoRA
- [**Development**](docs/development.md) - Test suite, pre-commit hooks, adding a new model

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

```bash
uv run pytest -v                    # everything
uv run pytest tests/unit -v         # unit only
uv run pytest tests/integration -v  # integration only
uv run pytest -v -m "not gpu"       # CPU-only (inverse of the gpu marker)
```

## License

This project is licensed under the Apache 2.0 license, as found in the [License](LICENSE) file.

## Citation

If you find our work useful, feel free to cite it using

```tex
@misc{primeintellect2025prime-rl,
  author = {Prime Intellect},
  title = {prime-rl},
  url = {https://github.com/PrimeIntellect-ai/prime-rl},
  year = {2025}
}
```
