---
name: entrypoints
description: All available prime-rl entrypoints — what they do, how to launch them, and which config class they use. Use when running commands, launching training, or starting servers.
---

# Entrypoints

All entrypoints are run via `uv run <command>` and accept TOML configs via `@ path/to/config.toml` with CLI overrides. See the `config` skill for config system details.

## `rl` — RL training

Orchestrates the complete RL loop: launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml # with SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run # generate scripts without running
```

- **Config:** `RLConfig` (`src/prime_rl/configs/rl.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/rl.py`
- **SLURM:** yes — single-node and multi-node

## `sft` — SFT training

Trains a model on labeled data. Uses torchrun for distributed training.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm # with SLURM
uv run sft @ examples/reverse_text/sft.toml --dry-run # generate scripts without running
```

The entrypoint launches torchrun internally — no need to call torchrun directly.

- **Config:** `SFTConfig` (`src/prime_rl/configs/sft.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/sft.py`
- **SLURM:** yes — single-node and multi-node

## `inference` — Standalone inference server

Launches an OpenAI-compatible inference server. vLLM is the default backend; SGLang is available with `--backend sglang` or `inference.backend = "sglang"`, and Dynamo+vLLM is available with `--backend dynamo` or `inference.backend = "dynamo"`.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Always use the `inference` entrypoint — never `vllm serve` directly.

vLLM custom endpoints beyond standard OpenAI API:
- `/v1/chat/completions/tokens` — accepts token IDs as prompt input
- `/update_weights` — hot-reload model weights from the trainer
- `/load_lora_adapter` — load LoRA adapters at runtime
- `/init_broadcaster` — initialize weight broadcast for distributed training

SGLang custom endpoints used by prime-rl:
- `/update_weights_from_disk` — filesystem weight reload
- `/init_broadcaster` — initialize SGLang NCCL receiver
- `/update_weights` — receive an NCCL weight broadcast

Dynamo launches a public prime-rl OpenAI proxy on `server.port`; the native Dynamo frontend runs on an internal port. Dynamo custom endpoints used by prime-rl run on the worker system server (`inference.dynamo.system_port`, default `8081`):
- `/engine/update_weights` — hot-reload model weights from the trainer
- `/engine/init_broadcaster` — initialize weight broadcast for distributed training
- `/engine/pause` and `/engine/resume` — pause/resume vLLM generation
- `/engine/chat_completions` — proxy target for OpenAI chat completions with prompt/completion token IDs and logprobs

Check health with:
```bash
curl http://<ip>:<port>/health
```

Check served models with:
```bash
curl http://<ip>:<port>/v1/models
```

Test chat completions with:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- **Config:** `InferenceConfig` (`src/prime_rl/configs/inference.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/inference.py`
- **SLURM:** yes — single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | SLURM | Typical use |
|---------|---------|-------|-------------|
| `rl` | Full RL pipeline | yes | Production RL training |
| `sft` | Supervised fine-tuning | yes | SFT training |
| `inference` | vLLM server | yes | Standalone inference or debugging |

## Key directories

- `src/prime_rl/entrypoints/` — top-level entrypoints (`rl`, `sft`, `inference`)
- `src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal configs for quick testing
- `examples/` — full example configs for various tasks
