# Sweeps

The `sweep` entrypoint materializes and launches hyperparameter studies for `rl`
or `sft` configs. It supports grid search, seeded random search, Optuna ask/tell
optimization, local execution, SLURM submission, and shared-trainer LoRA sweeps
for RL.

## Quick Start

Run a sweep from a sweep TOML:

```bash
uv run sweep @ examples/sweep/grid_local.toml
```

Validate and materialize trial artifacts without launching anything:

```bash
uv run sweep @ examples/sweep/grid_local.toml --dry-run
```

Optuna support is optional. Install the HPO extra before using
`strategy.type = "optuna"`:

```bash
uv sync --extra hpo
```

## Sweep Config

A sweep config names the target entrypoint, base target configs, study output
directory, search strategy, scheduler, optional objective, and parameter space.

```toml
name = "reverse-text-lr"
entrypoint = "rl"
base = ["examples/reverse_text/rl.toml"]
output_dir = "outputs/studies/reverse-text-lr"

[strategy]
type = "grid"

[scheduler]
type = "local"
max_parallel = 1

[objective]
metric = "reward/reverse-text/mean"
direction = "maximize"

[parameters."trainer.optim.lr"]
values = [1e-6, 3e-6, 1e-5]
```

Parameter keys are dotted paths into the target `rl` or `sft` config. The sweep
controller writes each trial's generated overrides to `overrides.toml`, validates
the resolved target config, and launches the target entrypoint with the original
base files plus those overrides.

## Strategies

### Grid

Grid search exhaustively evaluates every combination of `values` entries. Grid
parameters must use explicit choices:

```toml
[strategy]
type = "grid"

[parameters."trainer.optim.lr"]
values = [1e-6, 3e-6, 1e-5]

[parameters."orchestrator.train.sampling.temperature"]
values = [0.7, 1.0]
```

### Random

Random search draws independent samples from the declared distributions. Set a
seed when you need reproducible trial IDs and resume behavior:

```toml
[strategy]
type = "random"
num_trials = 8
seed = 42

[parameters."trainer.optim.lr"]
distribution = "log_uniform"
min = 1e-7
max = 1e-4
```

Supported parameter distributions are `choice`, `uniform`, `log_uniform`, and
`int_uniform`.

### Optuna

Optuna proposes one trial at a time for local sweeps, or one wave at a time for
`multi_run_lora` sweeps. Optuna requires an `[objective]`.

```toml
[strategy]
type = "optuna"
num_trials = 12
seed = 42
sampler = "tpe"

[objective]
metric = "reward/reverse-text/mean"
direction = "maximize"
```

Use `strategy.storage` to persist the Optuna study across resume:

```toml
[strategy]
type = "optuna"
num_trials = 12
storage = "sqlite:///outputs/studies/reverse-text-optuna/optuna.db"
study_name = "reverse-text-optuna"
```

Optuna pruners read intermediate metrics from each trial's local
`metrics.jsonl` sidecar:

```toml
[strategy.pruner]
type = "median"
n_startup_trials = 2
n_warmup_steps = 1
interval_steps = 1
```

Supported pruners are `none`, `median`, `asha`, and `hyperband`.

## Schedulers

### Local

The local scheduler runs target commands as subprocesses on the current machine.
With `max_parallel = 1`, the controller runs trials sequentially.

```toml
[scheduler]
type = "local"
max_parallel = 1
```

Parallel local sweeps require explicit disjoint GPU groups. Each worker gets a
`CUDA_VISIBLE_DEVICES` value from one group:

```toml
[scheduler]
type = "local"
max_parallel = 2

[scheduler.gpu_assignment]
mode = "static"
visible_devices = [[0, 1], [2, 3]]
```

The validator rejects `max_parallel > 1` without GPU assignment so parallel
workers cannot silently colocate trainer and inference stacks on the same GPUs.

### SLURM

The SLURM scheduler submits one target job per trial through the target
entrypoint's existing `[slurm]` support, then exits.

```toml
[scheduler]
type = "slurm"
```

The base target config must include a valid `[slurm]` block, usually by composing
the normal run config with a SLURM overlay:

```toml
base = [
  "examples/reverse_text/rl.toml",
  "examples/reverse_text/slurm_rl.toml",
]
```

SLURM sweeps are asynchronous after submission, so early stopping and Optuna are
not supported with `scheduler.type = "slurm"`.

### Shared-Trainer LoRA

The `multi_run_lora` scheduler is RL-only. It launches one shared trainer and
one orchestrator per trial. Each orchestrator trains a separate LoRA adapter slot
through the trainer's `MultiRunManager`.

```toml
entrypoint = "rl"
base = ["examples/reverse_text/rl_multi_run_lora_disagg.toml"]

[scheduler]
type = "multi_run_lora"
max_concurrent_runs = 3
shared = ["examples/reverse_text/rl_multi_run_lora_disagg.toml"]
```

Requirements:

- The shared config must enable `trainer.model.lora`.
- `trainer.max_concurrent_runs` must be at least
  `scheduler.max_concurrent_runs`.
- Only per-run orchestrator fields may vary. Trainer, model, deployment, and
  inference fields are shared by the wave and cannot be swept inside one
  `multi_run_lora` launch.

Static grid/random `multi_run_lora` sweeps launch the whole wave at once.
Optuna `multi_run_lora` sweeps run in waves of `scheduler.max_concurrent_runs`;
the controller tells Optuna each wave's results before asking for the next wave.

## Objectives and Metrics

Set `[objective]` when the sweep should rank trials, perform early stopping, or
run Optuna:

```toml
[objective]
metric = "reward/reverse-text/mean"
direction = "maximize"
```

During sweep runs, the launcher sets `PRIME_RL_SWEEP_METRICS_JSONL` for each
trial subprocess. `FileMonitor` writes step-indexed metrics to
`<run_dir>/metrics.jsonl`. The controller reads the latest valid-step row for
final objective attribution and polls the same file for Optuna pruning.

If the sidecar has no usable objective value, the controller falls back to
legacy `run-*/final_summary.json` files. A clean process exit without a finite
objective is recorded as a failed trial with `failure_stage = "objective"`.

## Early Stopping

Early stopping applies after completed trials. It can stop future local trials
or future Optuna `multi_run_lora` waves, but it does not cancel already-running
siblings.

```toml
[early_stopping]
type = "patience"
patience = 3
min_trials = 5
```

Threshold stopping is also available:

```toml
[early_stopping]
type = "threshold"
threshold = 0.7
min_trials = 2
```

Early stopping requires an objective and is not supported with the SLURM
scheduler. Static `multi_run_lora` launches the whole wave at once, so use
Optuna `multi_run_lora` when you need wave-by-wave stopping.

## Artifacts and Resume

Each standard trial gets a stable directory under the study output directory:

```text
outputs/studies/reverse-text-lr/
  study.toml
  manifest.json
  trials/
    0000-a1b2c3d4/
      overrides.toml
      resolved.toml
      command.txt
      status.json
      run/
        metrics.jsonl
```

The manifest records trial metadata, commands, resolved-config checksums, base
file checksums, git metadata, and objective summaries. Trial IDs have the form
`<index>-<hash8>`, where the hash is derived from the flat parameter override
dict.

Use `resume = true` to reuse completed grid/random trials. Resume fails closed
if the previous manifest is malformed, the target entrypoint changes, the
objective changes, parameter order changes, base file checksums drift, or a
terminal status cannot be trusted.

Optuna resume requires persistent `strategy.storage`. On resume, the controller
reconciles leftover RUNNING Optuna trials against `status.json` so Optuna
storage, the manifest, and trial artifacts agree.

`multi_run_lora` resume against a still-running shared trainer is not supported.

## Failure Handling

Two fields control failure behavior:

```toml
continue_on_failure = true
retry_budget = 1
```

`retry_budget` applies to launch failures and failed trial processes where the
scheduler can safely retry. If `continue_on_failure = false`, the controller
stops launching new work after the first failed materialization, runtime failure,
or missing objective, then exits non-zero after writing status and manifest
updates.

For `multi_run_lora`, `rl-multi-run` writes each orchestrator's return code to
`<run_dir>/control/exit_code`. The controller reconciles per-trial state from
those files instead of marking every trial failed from one aggregate launcher
return code.

## Examples

- `examples/sweep/grid_local.toml` — grid search with local sequential execution.
- `examples/sweep/random_local.toml` — seeded random search.
- `examples/sweep/parallel_local.toml` — local parallel trials with explicit GPU groups.
- `examples/sweep/optuna_local.toml` — Optuna TPE ask/tell loop.
- `examples/sweep/optuna_pruner_median.toml` — Optuna median pruning.
- `examples/sweep/optuna_pruner_asha.toml` — Optuna ASHA pruning.
- `examples/sweep/optuna_pruner_hyperband.toml` — Optuna Hyperband pruning.
- `examples/sweep/slurm.toml` — one target SLURM job per trial.
- `examples/sweep/grid_multi_run_lora.toml` — static shared-trainer LoRA sweep.
- `examples/sweep/optuna_multi_run_lora.toml` — Optuna waves over shared-trainer LoRA runs.
