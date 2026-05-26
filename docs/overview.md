# Overview

`prime-rl` is a framework for large-scale, asynchronous reinforcement learning of large language models. It is designed to be easy to use and hackable, yet capable of training 1T+-parameter MoE models on 1000+ GPU clusters.

## Architecture

A `prime-rl` RL run is three cooperating processes:

![Architecture](assets/architecture.png)

- **Inference** — vLLM-backed server (or fleet) holding the current policy. The orchestrator drives rollouts through the token-in `/v1/generate` route via the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package (OpenAI-compatible chat/completions routes are also exposed for external clients). Supports data + tensor + expert parallelism (with `deepep` and `flashinfer` all-to-all backends and EPLB), FP8 inference, prefill/decode disaggregation behind a `vllm-router`, CPU KV-cache offload, and *router replay* (the routed-expert mask is returned to the trainer for FP8 MoE numerical parity). Weights are pushed in place through a custom `update_weights` endpoint over filesystem or NCCL transports.
- **Orchestrator** — Lightweight CPU process that owns the data plane across many [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) training and eval environments. Each env runs in an isolated subprocess with a variable-size pool of env workers for scalability. The orchestrator drives multi-turn rollouts against the inference fleet (tool use, browsers, sandboxes, long horizons) without re-tokenizing across turns, computes advantages, packs the rollouts into training batches, and relays new weights from trainer to inference.
- **Trainer** — FSDP2 process group that consumes packed rollouts and steps the optimizer. We ship optimized custom modeling code for many MoE / dense / VLM families that unlocks advanced trainer parallelism — expert parallelism (EP, with DeepEP kernels) and context parallelism (CP) for long-sequence training — plus selective activation checkpointing, FP8 training on Hopper+, LoRA, and multi-tenant training (many concurrent LoRA tenants sharing one trainer + inference deployment).

The three processes communicate through configurable transports — by default the trainer↔orchestrator rollout link uses the local filesystem, and weight broadcast uses the filesystem (or NCCL for synchronous setups). Swap to ZMQ for multi-host setups without shared storage. See [Scaling](scaling.md) for the deployment options.

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

The script clones the repo, initializes the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) / [`renderers`](https://github.com/PrimeIntellect-ai/renderers) / [`research-environments`](https://github.com/PrimeIntellect-ai/research-environments) submodules, installs `uv`, and runs `uv sync --all-extras`. For manual setup, MoE-only installs (DeepGEMM / DeepEP / NIXL), or troubleshooting, see the [README](https://github.com/PrimeIntellect-ai/prime-rl#setup).

You need at least one NVIDIA GPU (RTX 3090/4090/5090, A100, H100, H200, or B200). Single-GPU runs are supported for debugging; production RL is typically 1× inference node + 1+ trainer nodes.

## Quick Run

Train an SFT-warmed `Qwen3-0.6B` on the `reverse-text` task — the env is bundled with the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) submodule so no separate install is needed. This config ships in the repo and runs on two GPUs (one for inference, one for the trainer):

```bash
uv run rl @ examples/reverse_text/rl.toml
```

The `rl` entrypoint reads `examples/reverse_text/rl.toml`, splits it into per-process sub-configs, picks GPU 0 for inference and GPU 1 for the trainer, launches all three processes, and tees their stdout into `outputs/logs/{trainer,orchestrator,inference}.log`. Within a minute the trainer should log `step 1` and a reward sample; after 20 steps the run completes and final HF-compatible weights land at `outputs/weights/step_20`.

## Documentation

- **[Configuration](configuration.md)** — TOML composition, CLI overrides, dry-run.
- **[Training](training.md)** — Launch and observe RL and SFT runs.
- **[Scaling](scaling.md)** — Single-GPU through multi-node clusters via FSDP / EP / CP and SLURM.
- **[Algorithms](algorithms.md)** — Async semantics, loss / advantage / filter plugins, trajectory merging.
- **[Advanced](advanced.md)** — Custom modeling, multimodal, LoRA, multi-tenant, P/D inference.
- **[Development](development.md)** — Test suite, pre-commit hooks, adding a new architecture.
- **[Reference](reference.md)** — Auto-generated config field reference.
