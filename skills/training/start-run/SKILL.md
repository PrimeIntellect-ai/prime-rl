---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

When running from a fresh git worktree, initialize its pinned dependencies first with
`git submodule update --init --recursive`. If reusing a synced environment from another
checkout, set `UV_PROJECT_ENVIRONMENT` to that environment and prepend the worktree's
`src`, `packages/prime-rl-configs/src`, `deps/renderers`, `deps/verifiers`, and
`deps/pydantic-config/src` directories to `PYTHONPATH`. Verify representative imports
with `inspect.getfile` before launching so the run cannot silently use another checkout.

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbConfig | None`): bare `--wandb` enables defaults; `--wandb @ wandb.toml` enables from a file; `--no-wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.
- Experimental gradient offload uses `model.optim_cpu_offload = { gradients = true }`.
  Budget pinned CPU RAM for an accumulator plus
  a same-sized staging set when gradient accumulation is greater than one.
- The experimental CPU optimizer step uses
  `model.optim_cpu_offload = { gradients = true, step_on_cpu = true }`. It keeps a
  persistent BF16 compute model on GPU, stores FP32 master weights and gradients
  in pinned CPU RAM, uses fused CPU AdamW, and refreshes the BF16 weights once
  per optimizer step. With
  gradient accumulation, budget two FP32 gradient-sized buffers. It currently
  supports AdamW and weights-only checkpoints; resumable optimizer checkpoints
  are rejected.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
uv run rl @ examples/basic/reverse-text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym_v1'))"`).
  If a local env exists under `deps/research-environments/environments/` or
  `deps/verifiers/environments/` but does not import, install the env workspace
  members with `uv sync --all-packages` (all) or `uv sync --package prime-rl
  --package <env>` (one) — they're auto-discovered, no `pyproject.toml` edit needed.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/basic/reverse-text/sft.toml
uv run sft @ examples/basic/reverse-text/sft.toml --slurm
uv run sft @ examples/basic/reverse-text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly. It starts a `vllm-router` on `server.port` (default 8000, the client-facing URL) fronting the engine on `backend_port` (default 8100); admin endpoints must target the engine port directly.

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Smoke checks:

```bash
curl http://<host>:<port>/health
curl http://<host>:<port>/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- Config: `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- Entrypoint: `src/prime_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse-text/`)
