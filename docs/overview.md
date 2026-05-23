# Overview

`prime-rl` is a framework for large-scale, asynchronous reinforcement learning of large language models. It is designed to be easy to use and hackable, yet capable of training 1T+-parameter MoE models on 1000+ GPU clusters.

## Architecture

A `prime-rl` RL run is three cooperating processes:

![Architecture](assets/architecture.png)

- **Inference** — A vLLM-backed server (or fleet) that holds the current policy and serves OpenAI-compatible completions. Scales from a single co-located GPU to multi-node fleets with tensor + data parallelism, FP8 inference, and prefill/decode disaggregation for high-throughput long-context serving. Updated in place via a custom `update_weights` endpoint, with NCCL or filesystem transports.
- **Orchestrator** — A lightweight CPU process that samples prompts from one or more [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments, drives multi-turn rollouts against the inference fleet (tool use, browsers, sandboxes, long horizons) without re-tokenizing across turns, computes advantages, packs the rollouts into training batches, and relays new weights back to inference.
- **Trainer** — A torchrun-launched FSDP2 process group that consumes packed rollouts and steps the optimizer. For MoE families we ship optimized custom modeling code with expert parallelism (EP) — including DeepEP kernels — and context parallelism (CP) for long-sequence training. Plus selective activation checkpointing, FP8 training on Hopper+, LoRA, and a multi-run manager that hosts many concurrent adapters in one trainer process.

The three processes communicate through configurable transports — by default the trainer↔orchestrator rollout link uses the local filesystem, and weight broadcast uses the filesystem (or NCCL for synchronous setups). Swap to ZMQ for multi-host setups without shared storage. See [Scaling](scaling.md) for the deployment options.

Training is **asynchronous by default**: inference is allowed to run ahead of training by up to `max_async_level` steps, which hides the weight-broadcast latency behind ongoing rollouts. The loss is an off-policy-aware variant of [AIPO](https://arxiv.org/abs/2505.24034); see [Algorithms](algorithms.md) for the details.

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

The script clones the repo, initializes the `verifiers` / `renderers` / `research-environments` submodules, installs `uv`, and runs `uv sync --all-extras`. For manual setup, MoE-only installs (DeepGEMM / DeepEP / NIXL), or troubleshooting, see the [README](https://github.com/PrimeIntellect-ai/prime-rl#setup).

You need at least one NVIDIA GPU (RTX 3090/4090/5090, A100, H100, H200, or B200). Single-GPU runs are supported for debugging; production RL is typically 1× inference node + 1+ trainer nodes.

## Quick run

Train an SFT-warmed `Qwen3-0.6B` on the `reverse-text` task — the env is bundled with the `verifiers` submodule so no separate install is needed. This config ships in the repo and runs on two GPUs (one for inference, one for the trainer):

```bash
uv run rl @ examples/reverse_text/rl.toml \
  --wandb.project your-project \
  --wandb.name reverse-text-smoke \
  --ckpt
```

The `rl` entrypoint reads `examples/reverse_text/rl.toml`, splits it into per-process sub-configs, picks GPU 0 for inference and GPU 1 for the trainer, launches all three processes, and tees their stdout into `outputs/logs/{trainer,orchestrator,inference}.log`. Within a minute the trainer should log `step 1` and a reward sample; after 20 steps the run completes and final HF-compatible weights land at `outputs/weights/step_20`.

For multi-GPU, multi-node, SLURM, and Kubernetes layouts, see [Scaling](scaling.md).

## Where to go next

- **[Configuration](configuration.md)** — How TOML files, `@` composition, CLI overrides, and env vars combine; the precedence rules; worked examples.
- **[Training](training.md)** — End-to-end recipes for RL, SFT, and evals; checkpointing and resume; observability (logs, W&B, Prometheus, platform monitoring); rules of thumb and common issues.
- **[Scaling](scaling.md)** — Single-GPU through 1000+ GPU; FSDP / EP / CP knobs; SLURM and Kubernetes guides; disaggregated prefill/decode inference; benchmarking.
- **[Algorithms](algorithms.md)** — Async / off-policy semantics; the AIPO loss; built-in and custom losses, advantages, and filters; multi-turn trajectory merging.
- **[Advanced](advanced.md)** — MoE training (EP backends, custom impls); VLMs; LoRA and the multi-run manager; small-scale MoE testing; environments deep-dive.
- **[Reference](reference.md)** — Auto-generated field-by-field reference for every entrypoint config.
- **[FAQs](faqs.md)** — Quick answers to recurring questions.
