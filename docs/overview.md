# Overview

`prime-rl` is a framework for large-scale, asynchronous reinforcement learning of large language models. It is designed to be easy to use and hackable, yet capable of scaling to 1000+ GPU clusters. Models are trained with PyTorch FSDP2 (with optional expert and context parallelism), rollouts are generated with vLLM, and the two halves talk to each other through a thin orchestrator process that owns dataset sampling, advantage computation, and weight broadcasting.

Use `prime-rl` when you want to:

- Train an open-weights LLM with RL on one of the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars) tasks, or your own [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment.
- Post-train with SFT, then continue with RL, using the same model loader, checkpoint format, and chat template plumbing for both phases.
- Scale across multiple nodes — SLURM, Kubernetes, or hand-launched — without rewriting your config.
- Run agentic multi-turn rollouts (tool use, browser environments, long horizons) without re-tokenizing across turns.

## Architecture

A `prime-rl` RL run is three cooperating processes:

![Architecture](assets/architecture.png)

- **Inference** — A vLLM server (or fleet) that holds the current policy weights and serves OpenAI-compatible completions. Updated in place via a custom `update_weights` endpoint after each trainer step.
- **Orchestrator** — A lightweight CPU process that samples prompts, drives `verifiers` environments to generate rollouts against the inference server, packs them into training batches, ships them to the trainer, and relays new weights back to inference.
- **Trainer** — A torchrun-launched FSDP2 process group that consumes packed rollouts, computes the loss, steps the optimizer, and writes the new policy to the weight broadcast transport.

The three processes communicate through configurable transports — by default the trainer↔orchestrator rollout link uses the local filesystem, and weight broadcast uses the filesystem (or NCCL for synchronous setups). Swap to ZMQ for multi-host setups without shared storage. See [Scaling](scaling.md) for the deployment options.

Training is **asynchronous by default**: inference is allowed to run ahead of training by up to `max_async_level` steps, which hides the weight-broadcast latency behind ongoing rollouts. The loss is an off-policy-aware variant of [AIPO](https://arxiv.org/abs/2505.24034); see [Algorithms](algorithms.md) for the details.

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

The script clones the repo, initializes the `verifiers` / `renderers` / `research-environments` submodules, installs `uv`, and runs `uv sync --all-extras`. For manual setup, MoE-only installs (DeepGEMM / DeepEP / NIXL), or troubleshooting, see the [README](https://github.com/PrimeIntellect-ai/prime-rl#setup).

You need at least one NVIDIA GPU (RTX 3090/4090/5090, A100, H100, H200, or B200). Single-GPU runs are supported for debugging; production RL is typically 1× inference node + 1+ trainer nodes.

## Quick run

Train `Qwen3-0.6B` on GSM8K with one trainer GPU and one inference GPU. This config ships in the repo:

```bash
# 1. Install the verifiers environment from the Environments Hub.
prime env install primeintellect/math-env

# 2. Set up a four-pane tmux session that tails each process's logs.
bash scripts/tmux.sh

# 3. From the `Trainer` pane, launch all three processes co-located on this node.
uv run rl @ configs/gsm8k/rl.toml \
  --wandb.project your-project \
  --wandb.name gsm8k-smoke \
  --ckpt
```

The `rl` entrypoint reads `configs/gsm8k/rl.toml`, splits it into per-process sub-configs, picks GPU 0 for inference and GPU 1 for the trainer, launches all three processes, and tees their stdout into `outputs/logs/{trainer,orchestrator,inference}.log`. Watch the tmux panes — within a minute the trainer should log `step 1` and a reward sample.

After 100 steps the run completes. Final HF-compatible weights land at `outputs/weights/step_100`.

For a CPU-only smoke check (no real training, no GPU), use the SFT fake-data config:

```bash
uv run sft @ configs/debug/sft/train.toml
```

For multi-GPU, multi-node, SLURM, and Kubernetes layouts, see [Scaling](scaling.md).

## Where to go next

- **[Configuration](configuration.md)** — How TOML files, `@` composition, CLI overrides, and env vars combine; the precedence rules; worked examples.
- **[Training](training.md)** — End-to-end recipes for RL, SFT, and evals; checkpointing and resume; observability (logs, W&B, Prometheus, platform monitoring); rules of thumb and common issues.
- **[Scaling](scaling.md)** — Single-GPU through 1000+ GPU; FSDP / EP / CP knobs; SLURM and Kubernetes guides; disaggregated prefill/decode inference; benchmarking.
- **[Algorithms](algorithms.md)** — Async / off-policy semantics; the AIPO loss; built-in and custom losses, advantages, and filters; multi-turn trajectory merging.
- **[Advanced](advanced.md)** — MoE training (EP backends, custom impls); VLMs; LoRA and the multi-run manager; small-scale MoE testing; environments deep-dive.
- **[Reference](reference.md)** — Auto-generated field-by-field reference for every entrypoint config.
- **[FAQs](faqs.md)** — Quick answers to recurring questions.
