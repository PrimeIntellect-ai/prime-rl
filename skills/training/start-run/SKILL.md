---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

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

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml   # SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym_v1'))"`).
  If a local env exists under `deps/research-environments/environments/` but does not
  import, add it to the root `pyproject.toml` env extra, workspace members, and
  `[tool.uv.sources]`, then run `uv sync --all-extras`.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm
uv run sft @ examples/reverse_text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly.

```bash
uv run inference @ configs/debug/infer.toml
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

### Multi-node SLURM bring-up gotchas

- **Cache paths must be user-scoped.** On shared clusters `/tmp/.cache/*` may be
  owned by another user; set `TRITON_CACHE_DIR` / `VLLM_CACHE_ROOT` /
  `DG_JIT_CACHE_DIR` under a per-user path (e.g. `/tmp/<user>-cache/...`) in
  `[env_vars]`. Triton JIT failures from EACCES are fatal mid-run.
- **Node starts must be near-simultaneous for DP groups.** vLLM's DP
  coordinator handshake has a hard 5-minute deadline
  (`HANDSHAKE_TIMEOUT_MINS`, not env-tunable): if setup staggers nodes by more
  than ~5 min, the early nodes' engines die waiting for the last node's
  front-end. The sbatch template syncs the venv once at the batch level and
  launches all nodes with a single srun for this reason.
- **Don't let ranks re-sync uv concurrently.** Per-rank `uv run` without
  `--no-sync` re-resolves the project; on a shared checkout the ranks race on
  `~/.cache/uv` ("Text file busy" → exit 2 kills the job) and add ~30-60 s of
  stagger per rank. The launch helper uses `uv run --no-sync`.
- **Per-role vLLM overrides accept nested dicts.** `decode_vllm_overrides` /
  `prefill_vllm_overrides` are JSON-serialized into the per-rank engine args
  and applied after the base config (last key wins), so e.g.
  `decode_vllm_overrides = { gpu_memory_utilization = 0.9, attention_config = { hisparse_config = { host_pool_gib = 256 } } }`
  works as-is (see `configs/glm51_rlm_hisparse/`).
- **HiSparse + llm-d routing**: set `non_cached_tokens = 1` (always route
  prefills to prefill nodes). Decode-local prefill concurrent with NIXL
  context arrivals can hit a device-side fault in HiSparse's mixed-batch
  path (open follow-up in the HiSparse PR); do not enable the short-suffix
  shortcut (e.g. 128) until that lands. Note `non_cached_tokens = 0`
  disables P/D splitting entirely (everything decodes locally) — never use
  0 with HiSparse. MTP on HiSparse is rolled back pending a sanitizer-clean
  wheel; do not set `speculative_config` with the pinned HiSparse wheel.

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
- `examples/` — full example configs (e.g. `reverse_text/`)
