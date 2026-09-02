---
name: start-run
description: How to launch prime-rl runs — the `rl`, `sft`, `inference`, `eval`, and `online-eval` entrypoints, their config classes, CLI shorthands, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
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

## `eval` — multi-env evals

Runs one epoch of every configured eval source against a live inference server, then exits. The evals process spawns one env server per source (unless the source sets `serve.address`), sizes concurrency with the same adaptive controller as `[orchestrator.concurrency]`, streams every episode through the monitors (file monitor + trace stream by default; W&B and the Prime platform on request), and checkpoints the task cursor after every completed group so an interrupted run resumes with `--resume`.

```bash
uv run eval gsm8k -n 32 -r 4 -c 8                                # Prime Inference (default client + model), pinned band
uv run inference --vllm.model Qwen/Qwen3-4B                      # or a local vLLM server ...
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1   # ... adaptive band
uv run eval @ eval.toml --run.name my-eval                        # multi-source TOML
uv run eval @ eval.toml --run.name my-eval --resume               # resume the interrupted run
```

Shorthands (single-source runs): `<taskset-id>` names the run's only source, `--env.<field> <value>` sets a field of that source's env block (`--env.agent.harness.id bash`, `--env.taskset.tasks '["fix-git"]'`), `-n` `num_examples`, `-r` `group_size`, `-m` `model`, `-c N` pins the concurrency band (`concurrency.min_inflight = max_inflight = N`). The shorthands cannot be combined with a TOML that defines `[[source]]` blocks. `uv run eval -h` lists them.

Minimal multi-source `eval.toml`:

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32   # always cap eval size for smokes
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]       # adaptive against vLLM; pin with min_inflight = max_inflight
max_inflight = 128

[[source]]
env.taskset.id = "gsm8k"
env.agent.harness.id = "bash"

[[source]]
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[monitors.prime]    # optional: upload each source's epoch as a platform evaluation
```

- Run dir: `output_dir / run.name` (auto `<envs>--<model>--<short-id>`); `configs/attempt_N/`, `logs/attempt_N/{eval.log,envs/eval/<name>.log}`, `monitors/file/`, `checkpoints/step_<cursor>/evals/progress.pt` (only the newest kept). `--clean` wipes a used run dir, `--dry-run` writes the config and exits, `--no-dashboard` skips the dashboard daemon.
- Resume: cursor checkpoints are on by default (`[ckpt]`, `interval` counts completed task groups, `keep_last` prunes older cursors). Relaunch with the same `--run.name` and `--resume` (or `--resume.step N` / `--resume.dir path/to/checkpoints/step_N`) to skip the completed prefix; partially completed groups are retried. Disable saving with `--no-ckpt`.
- Env servers: spawned by the evals process at `tcp://127.0.0.1:<env_server_base_port + index>`; logs at `{run_dir}/logs/latest/envs/eval/{name}.log`.
- Defaults: model `deepseek/deepseek-v4-flash` on Prime Inference (`PRIME_API_KEY`, else the `prime login` config). External inference APIs (no vLLM `/metrics`) have no load signal for adaptive concurrency: the startup `/metrics` probe fails fast unless the band is pinned (`-c N`).
- Console: the launcher prints the start line, log paths and dashboard URL, then only per-env results and warnings; `logs/latest/eval.log` has everything. Examples: `examples/eval/` (README lists them: gsm8k, wordle, wiki-search, terminal-bench-2, best-of-n, agentic-judge, rlm-docker, swe).
- Platform: `--monitors.prime` (needs `PRIME_API_KEY` or `prime login`) creates one evaluation per source on app.primeintellect.ai once its epoch finishes and logs the URL.
- Config: `EvalConfig` (`packages/prime-rl-configs/src/prime_rl/configs/eval.py`)
- Entrypoint: `src/prime_rl/entrypoints/eval.py` (implementation: `src/prime_rl/eval/eval.py`, shared engine `src/prime_rl/eval/runner.py`)

## `online-eval` — SFT online evals

Spawned by the `sft` launcher next to the trainer when the SFT config has an `[eval]` block: watches `broadcasts/step_{n}` for the trainer's weight broadcasts, moves the inference server onto each one, and runs the due sources against the updated weights (config `OnlineEvalConfig` = the SFT `[eval]` block + `model`, `broadcasts_dir`, `max_steps`, `resume_step`, `weight_broadcast`). Launcher-managed SFT evals use NCCL weight broadcast by default; LoRA and external inference use filesystem broadcast. By default a newer checkpoint cancels unfinished episodes from the prior eval; set `eval.cancel_on_new_checkpoint = false` to drain every epoch. The trainer can idle while it waits for slow evals. With `weight_broadcast.type = "filesystem"` the process also runs standalone (`uv run online-eval @ online_eval.toml`) against any trainer that writes `broadcasts/step_{n}` with the broadcast markers. No cursor checkpoints: the process is coupled to the trainer's live broadcast lifecycle. Logs at `{run_dir}/logs/latest/online-eval.log`.

- Config: `OnlineEvalConfig` (`packages/prime-rl-configs/src/prime_rl/configs/eval.py`)
- Entrypoint: `src/prime_rl/entrypoints/online_eval.py` (implementation: `src/prime_rl/eval/online.py`)

## Exporting checkpoints

Trainer checkpoints are DCP-sharded (`<run_dir>/checkpoints/step_{n}/trainer`). Convert to HF safetensors with `uv run python tools/convert_dcp_to_bf16.py <run_dir>/checkpoints/step_{n}` (writes `<ckpt_dir>/weights`, serveable via `uv run inference --vllm.model <dir>`; model config auto-read from the run’s `configs/latest/resolved/trainer.json`/`sft.json`; multi-rank via `torchrun --nproc-per-node N`; full fine-tunes only, LoRA rejected). Quantize a bf16 HF dir to blockwise FP8 with `tools/convert_bf16_to_fp8.py <dir>` (vLLM-native format), or straight from a checkpoint with `tools/convert_dcp_to_fp8.py <ckpt_dir>` (rank-parallel, writes only `<ckpt_dir>/weights-FP8`, no bf16 on disk); dequantize fp8-only releases with `tools/convert_fp8_to_bf16.py <dir>`. Caveat: on SM120 GPUs (RTX PRO 6000) vLLM 0.26 picks `CutlassFp8BlockScaledMMKernel` for blockwise-fp8 checkpoints and it silently degrades outputs — serve with `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel,MarlinFP8ScaledMMLinearKernel` to fall back to the Triton kernel.

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |
| `eval` | Multi-env evals against a live server | Offline evals |
| `online-eval` | Evals per weight broadcast | SFT online evals (launcher-spawned) |

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
