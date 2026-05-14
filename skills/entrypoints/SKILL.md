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

- **Config:** `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
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

- **Config:** `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/sft.py`
- **SLURM:** yes — single-node and multi-node

## `inference` — Standalone inference server

Launches a vLLM-based inference server with OpenAI-compatible API.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Always use the `inference` entrypoint — never `vllm serve` directly.

Custom endpoints beyond standard OpenAI API:
- `/v1/chat/completions/tokens` — accepts token IDs as prompt input
- `/update_weights` — hot-reload model weights from the trainer
- `/load_lora_adapter` — load LoRA adapters at runtime
- `/init_broadcaster` — initialize weight broadcast for distributed training

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

- **Config:** `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/inference.py`
- **SLURM:** yes — single-node, multi-node, and disaggregated deployments

## `sweep` — Hyperparameter sweeps

Materializes and launches hyperparameter sweep trials for `rl` or `sft` target configs.

```bash
uv run sweep @ examples/sweep/grid_local.toml
uv run sweep @ examples/sweep/grid_local.toml --dry-run
uv run sweep @ examples/sweep/optuna_local.toml # requires uv sync --extra hpo
```

Sweeps support grid/random/Optuna strategies over dotted target config paths. Standard trials write `overrides.toml`, `resolved.toml`, `command.txt`, and `status.json` under the study output directory, then launch the target entrypoint with the configured base files plus the generated `overrides.toml` unless `--dry-run` is set.

Local `max_parallel > 1` requires explicit disjoint GPU groups under `[scheduler.gpu_assignment]`. SLURM sweeps default to asynchronous submission and exit after queueing jobs; set `scheduler.synchronous = true` when early stopping, Optuna, or Optuna pruners need per-trial completion feedback. Set `scheduler.max_parallel > 1` (synchronous mode only) to drive multiple concurrent SLURM jobs from a single Optuna study — TPE auto-enables `constant_liar` so concurrent asks don't collide on the same region; pruners are rejected under parallel SLURM since Optuna trial objects are not thread-safe.

`multi_run_lora` sweeps materialize each trial under `shared/run_<trial_id>/`, write the per-run orchestrator config to `control/orch.toml`, and keep `orchestrator.output_dir` pinned to that run directory. The materializer validates the shared `RLConfig` first, requires `trainer.model.lora`, checks `scheduler.max_concurrent_runs <= trainer.max_concurrent_runs`, then layers allowlisted per-run orchestrator overrides and validates the final `OrchestratorConfig` so multi-run LoRA ranks can be below the trainer's max rank. Their replay command uses the shared trainer configs, the generated `shared/_output_override.toml`, and `--runs-dir <run_dir>` so it goes through the same parser as `rl-multi-run`.
Before each `multi_run_lora` launch, the sweep controller writes `control/evicted.txt` in inactive `shared/run_*` directories so the trainer's directory scan ignores stale runs from previous waves or older studies. The Optuna wave driver also wipes trainer-owned subdirs (`checkpoints/`, `weights/`, `broadcasts/`, `rollouts/`, `run_default/`) inside `shared_dir` between waves via `_reset_trainer_state_for_wave`. If the shared `RLConfig` sets `ckpt.output_dir`, that directory's `checkpoints/` is reset too, resolved once at the top of the wave loop. Without this, a fresh wave-N+1 trainer would silently resume from the previous wave's checkpoints or collide on `step_*` writes; per-trial `run_<id>` directories are intentionally preserved so the sweep controller's eviction logic stays in charge of them.
In shared W&B mode, the `rl-multi-run` launcher process itself is the shared-run primary: it calls `init_wandb_shared_primary` before spawning subprocesses (label `launcher`, `x_primary=True`, `x_update_finish_state=True`) and `finish()`es the run in a `try/finally`. Trainer and per-trial orchestrators stay non-primary (always `WANDB_SHARED_PRIMARY=0`, with explicit per-trial labels `orchestrator-<run_id>`); they attach to the run the launcher created. This binds run finish state to the supervisor's lifetime, so a trainer exit at `max_steps` or a pruned orchestrator can no longer mark the shared run finished while sibling orchestrators are still emitting final-eval logs. Single-run still relies on the `WANDB_SHARED_LABEL=="orchestrator"` fallback in `WandbMonitor`.
`rl-multi-run --runs-dir` takes colon-separated run directories. Empty entries (`a::b`, leading/trailing `:`) and duplicate resolved directories are rejected before config parsing because each run needs a distinct `control/orch.toml` and `control/exit_code`. The launcher writes each run's `control/exit_code` as soon as that orchestrator process exits, then rewrites all exit codes during final cleanup; Optuna wave pruning relies on those early files to avoid pruning an already-completed run while sibling orchestrators are still active. A non-zero orchestrator exit must also leave `control/evicted.txt` in place, without overwriting an existing pruning reason, so the shared trainer stops discovering a crashed run and the rest of the wave can drain. Once every orchestrator has stopped and at least one exited non-zero, the launcher should tear down the trainer instead of waiting forever for batches that can no longer arrive.
If the sweep controller cannot spawn `rl-multi-run` at all (`OSError`), it retries the static wave or Optuna wave according to `retry_budget`; only the final failed launch attempt marks the affected run statuses with `failure_stage = "launch"`. Runtime non-zero exits are still reconciled from per-run `control/exit_code` instead of retrying the wave.

- **Config:** `SweepConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sweep.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/sweep.py`
- **SLURM:** yes, through the target `rl`/`sft` config's existing `[slurm]` support

## Summary

| Command | Purpose | SLURM | Typical use |
|---------|---------|-------|-------------|
| `rl` | Full RL pipeline | yes | Production RL training |
| `sft` | Supervised fine-tuning | yes | SFT training |
| `inference` | vLLM server | yes | Standalone inference or debugging |
| `sweep` | Hyperparameter sweeps | yes | Launching multiple RL/SFT variants |
| `rl-multi-run` | Shared-trainer LoRA wave launcher | no | Internal target for `multi_run_lora` sweeps |

## Key directories

- `src/prime_rl/entrypoints/` — top-level entrypoints (`rl`, `sft`, `inference`, `sweep`)
- `packages/prime-rl-configs/src/prime_rl/configs/` — config classes
- `configs/debug/` — minimal configs for quick testing
- `examples/` — full example configs for various tasks
