---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, `inference`, and `evals` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

SLURM launches write generated scripts and coordination files under `<run_dir>/launcher/`, with batch logs under `launcher/logs/`. Local launches do not create this directory. Every launch writes configs and `command.txt` under `configs/attempt_<n>/`. `configs/latest` points to the current attempt. The command uses shell-safe quoting.

## Run directories

`output_dir` (default `outputs`) groups related runs; each run writes all its artifacts (logs, configs, checkpoints, broadcasts, rollouts) to its own run directory `<output_dir>/<run_name>`. `run.name` auto-generates as `<envs>--<model>--<short-id>` (SFT: `<dataset>--<model>--<short-id>`), so every launch gets a fresh, readable run directory; `run.dir` overrides the directory leaf when it should differ from the name. Pass `--run.name <name>` to make the run directory predictable — required to resume the run later (`--resume`, or `--resume.step N`, reuses the named run directory; without `[ckpt]` it loads but saves no new checkpoints). Launching into a run directory that already contains artifacts fails unless resuming or `--clean` is set (which wipes only that run directory).

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbMonitorConfig | None`): bare `--monitors.wandb` enables defaults; `--monitors.wandb @ wandb.toml` enables from a file; `--no-monitors.wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.
- State-only optimizer offload remains enabled by default with `model.optim_cpu_offload = true`.
- Runtime fusions default to `model.fusions.enabled = ["gate_up", "qkv"]` and are
  skipped for LoRA. To run an unfused comparison, explicitly pass
  `--model.fusions.enabled '[]'` (prefix with `trainer.` for `rl`). Checkpoints use
  canonical projection names, so a resume can change this list. Muon requires the
  pinned Dion revision's `matrix_partitions` support; after changing the pin, sync
  before submitting GPU jobs and use `uv run --no-sync` inside verification jobs
  to keep their environment fixed.
  For BF16 gradient comparisons, repeat the unfused calculation too: expert bias
  broadcasts use CUDA reductions that can differ between identical runs. Record
  this noise separately from matrix-weight gradients and optimizer equivalence.
  Check optimizer resumes with identical supplied gradients when asserting exact
  next updates. Dion calls both Muon and AdamW's first moment `momentum`; their
  decay coefficients are `mu` and `beta1`, respectively.
  Distributed offload diagnostics need both Gloo (CPU) and NCCL (CUDA) backends
  to gather CPU-resident DTensor state, as the trainer's process group provides.
- For gradients, FP32 masters, optimizer state, and optimizer-in-backward CPU execution, set
  `model.optim_cpu_offload = false` and `model.full_offload = true`. This mode uses the native
  CPU optimizer kernel, only supports AdamW and SignSGD (SignSGD is stateless and
  halves the host RAM footprint), and disables gradient clipping. Use a
  `[model.full_offload]` table only to select the Torch debugging backend or disable NUMA binding.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
uv run rl @ examples/basic/reverse-text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- Fake-data runs generate random inference logprobs, so their mismatch KL does
  not measure trainer/inference agreement. Use a live inference run for that check.
- SLURM: single- and multi-node
- Multi-node SLURM stops after `.trainer.done` for trainer-only fake-data runs. Runs with inference stop after both `.trainer.done` and `.orchestrator.done`.
- NIXL on SLURM: install NIXL and ModelExpress with the provided scripts. The job starts ModelExpress and Redis unless `slurm.launch_modelexpress = false`.
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym'))"`).
  If a local env exists under `deps/prime-envs/environments/` or
  `deps/verifiers/environments/` but does not import, install the env workspace
  members with `uv sync --all-extras --all-packages` (all) or `uv sync --all-extras
  --package prime-rl --package <env>` (one) — they're auto-discovered, no
  `pyproject.toml` edit needed. Keep `--all-extras` for training so a targeted
  package sync does not prune accelerator dependencies from the environment.

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
- Multi-node online evals use one SLURM job with `num_train_nodes + num_infer_nodes` nodes. The generated `launcher/sft.sbatch` assigns inference nodes first, then trainer nodes.

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly. It starts a `vllm-router` on `server.port` (default 8000, the client-facing URL) fronting the engine on `backend_port` (default 8100); admin endpoints must target the engine port directly.

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B
uv run inference --vllm.model Qwen/Qwen3-0.6B --vllm.enforce-eager
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

## `evals` — multi-env evals

Runs the configured eval sources against a live inference server. Standalone (no `[online]` block): one epoch of every source against the served weights, then exit. Add `[ckpt]` (`interval` counts completed task groups in the eval-source order) to make the run interruptible, then use `--resume`, `--resume.step N`, or `--resume.dir path/to/checkpoints/step_N`; for standalone evals, checkpoint step N means resume from task cursor N. Checkpoints store only the cursor, not generated episode records; partially completed groups and groups beyond the durable cursor are retried. Resume loads without `[ckpt]` but does not save new checkpoints. Checkpoint/resume is rejected with `[online]` because that process is coupled to the trainer's live broadcast handshake. With `[online]` (`broadcasts_dir`, `max_steps`, `resume_step`): watch the broadcasts dir for stable `step_{n}` weight broadcasts and evaluate each — the `sft` launcher writes this config for online evals. By default a newer checkpoint cancels unfinished episodes from the prior eval. Set `eval.cancel_on_new_checkpoint = false` to drain every epoch. The trainer can idle while it waits for slow evals. Launcher-managed SFT evals use NCCL weight broadcast by default, including multi-node SLURM deployments. LoRA and external inference use filesystem broadcast.

```bash
uv run inference --vllm.model Qwen/Qwen3-4B   # start inference separately
uv run evals @ eval.toml
```

Minimal standalone `eval.toml`:

```toml
model = "Qwen/Qwen3-4B"

[eval.client]
base_url = "http://localhost:8000/v1"

[eval.concurrency]  # adaptive; same controller as [orchestrator.concurrency]
min_inflight = 8
max_inflight = 128

[[eval.source]]
num_examples = 32   # always cap eval size for smokes
group_size = 4
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[ckpt]               # optional: make a standalone eval resumable
interval = 10        # completed task groups between cursor saves
```

- Env servers: spawned by the evals process, one per source without an explicit `serve.address`, at `tcp://127.0.0.1:<eval.env_server_base_port + index>`; logs at `{output_dir}/logs/latest/envs/eval/{name}.log`.
- External inference APIs (no vLLM `/metrics`, e.g. Prime Inference) have no load signal for adaptive concurrency: the startup `/metrics` probe fails fast unless the band is pinned (`min_inflight = max_inflight`). Full example: `examples/evals/swe.toml` (SWE-bench Verified + Terminal-Bench 2 on Prime Inference, `agent.timeout.rollout = 3600`).
- Config: `EvalsConfig` (`packages/prime-rl-configs/src/prime_rl/configs/evals.py`)
- Entrypoint: `src/prime_rl/entrypoints/evals.py` (implementation: `src/prime_rl/evals/evals.py`)

## Exporting checkpoints

Trainer checkpoints are DCP-sharded (`<run_dir>/checkpoints/step_{n}/trainer`). Convert to HF safetensors with `uv run python tools/convert_dcp_to_bf16.py <run_dir>/checkpoints/step_{n}` (writes `<ckpt_dir>/weights`, serveable via `uv run inference --vllm.model <dir>`; model config auto-read from the run’s `configs/latest/resolved/trainer.json`/`sft.json`; multi-rank via `torchrun --nproc-per-node N`; full fine-tunes only, LoRA rejected). Quantize a bf16 HF dir to blockwise FP8 with `tools/convert_bf16_to_fp8.py <dir>` (vLLM-native format), or straight from a checkpoint with `tools/convert_dcp_to_fp8.py <ckpt_dir>` (rank-parallel, writes only `<ckpt_dir>/weights-FP8`, no bf16 on disk); dequantize fp8-only releases with `tools/convert_fp8_to_bf16.py <dir>`. Caveat: on SM120 GPUs (RTX PRO 6000) vLLM 0.26 picks `CutlassFp8BlockScaledMMKernel` for blockwise-fp8 checkpoints and it silently degrades outputs — serve with `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel,MarlinFP8ScaledMMLinearKernel` to fall back to the Triton kernel.

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |
| `evals` | Multi-env evals | Standalone evals / SFT online evals |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse-text/`)

## Dashboard

Interactive launches auto-start one shared dashboard daemon per user (process title
`PRL::Dashboard`) and end startup with a `Dashboard · <url>` banner. Relay
that URL to the researcher. Discovery: `~/.cache/prime-rl/dashboard/daemon.json` holds
the live `url` (the port can differ from 7788 when it was taken). `--no-dashboard`
opts a run out.
