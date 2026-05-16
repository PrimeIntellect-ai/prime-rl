---
name: config
description: How the prime-rl config system works — TOML files, CLI, config composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Config

prime-rl uses `pydantic_config` (combines `tyro` and `pydantic`) for configuration. 

## Use configs

Every entrypoint accepts TOML files via `@` syntax and CLI overrides to configure it.

```bash
# Configure RL training with a TOML file
uv run rl @ examples/reverse_text/rl.toml

# Override specific fields via CLI
uv run rl @ examples/reverse_text/rl.toml --max-steps 50
```

Config resolve in the following order:

1. CLI arguments
2. Config files (merged left-to-right)
3. Class defaults (lowest)

## Compose configs

Multiple config files are merged left-to-right (later files override earlier ones):

```bash
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml
```

Nested configs can be loaded for specific sections:

```bash
uv run rl --model @ model.toml --data @ data.toml
```

Mixed composition works too:

```bash
uv run rl @ base.toml --trainer @ trainer_override.toml --trainer.lr 1e-3
```

Merging is deep — unset fields in the override are preserved from the base config.

## Inspect & validate configs

Use `--help` to see all available fields and their defaults. When combined with a config file, defaults reflect the TOML values:

```bash
uv run rl --help                                  # shows class defaults
uv run rl @ examples/reverse_text/rl.toml --help  # shows defaults from TOML
```

Use `--dry-run` to validate and dump the fully resolved config:

```bash
uv run rl @ examples/reverse_text/rl.toml --dry-run --output-dir /tmp/test
# Writes resolved TOML to /tmp/test/configs
```

Sweep configs also support `--dry-run`. A sweep dry-run validates the sweep and target trial configs, writes study/trial artifacts, and does not launch target runs:

```bash
uv run sweep @ path/to/sweep.toml --dry-run
```

## Naming

CLI uses kebab-case (`--model.max-model-len`), TOML uses snake_case (`max_model_len`). Both refer to the same field.

## General rules

- **Fail early**: incompatible option combinations (e.g. CP requires flash attention, NCCL broadcast requires async level 1) should raise in `model_validator` at config resolution time, not at runtime. When adding new constraints, add a validator to the config class.
- **Deprecation**: when renaming or removing config fields, emit a deprecation warning with a clear migration path (e.g. "field X is deprecated, use Y instead"). Do not silently drop fields — help users update their configs.

## Important patterns

### Boolean fields

```bash
uv run inference --model.enforce-eager          # sets to true
uv run inference --model.no-enforce-eager       # sets to false
```

In TOML, booleans must be explicit:

```toml
[model]
enforce_eager = true
```

### None fields

TOML has no null type. Use the string `"None"`:

```toml
max_model_len = "None"
```

On the CLI, pass `None` as a plain string:

```bash
uv run inference --model.max-model-len None
```

### List fields

In TOML, use `[[double brackets]]` (array of tables) for lists of objects:

```toml
[[orchestrator.env]]
id = "reverse-text"

[[orchestrator.env]]
id = "math-env"
```

On the CLI, list items are indexed: `--env.0.id reverse-text --env.1.id math-env`.

### Dict fields

In TOML, use a section:

```toml
[vllm_extra]
key1 = "value1"
key2 = 123
```

On the CLI, pass as a JSON string:

```bash
uv run inference --vllm-extra '{"key1": "value1", "key2": 123}'
```

### Sweep parameter paths

Sweep configs name a target entrypoint, one or more base config files, an output directory, a strategy, a scheduler, optional objective/stopping rules, and dotted target config paths under `[parameters]`. The sweep launcher converts parameters to generated override TOML files and then validates the target `rl` or `sft` config normally:

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
```

Use `strategy.type = "grid"` for exhaustive choice combinations, `random` for seeded independent samples, and `optuna` for adaptive ask/tell studies. Optuna requires the `hpo` extra (`uv sync --extra hpo`), an `[objective]`, and local, synchronous SLURM, or `multi_run_lora` scheduling. Persistent Optuna resume requires `strategy.storage`, usually a SQLite URL such as `sqlite:///outputs/studies/name/optuna.db`.

Schedulers are `local`, `slurm`, and `multi_run_lora`. Local parallel sweeps require explicit disjoint GPU groups:

```toml
[scheduler]
type = "local"
max_parallel = 2

[scheduler.gpu_assignment]
mode = "static"
visible_devices = [[0, 1], [2, 3]]
```

SLURM sweeps submit through each target config's existing `[slurm]` support. The default asynchronous mode exits after submission and cannot use early stopping or Optuna; set `scheduler.synchronous = true` to submit each trial with blocking SLURM behavior so early stopping, Optuna, and Optuna pruners can observe per-trial outcomes. Combine `synchronous = true` with `max_parallel > 1` to drive up to N concurrent in-flight SLURM jobs from a single Optuna study — TPE automatically opts into `constant_liar` so concurrent asks diversify; pruners are not supported under parallel SLURM. `multi_run_lora` is RL-only and uses one shared trainer plus one orchestrator per trial; its `shared` config list describes the shared RL stack, while `base` is still the target config list used for study materialization.

```toml
[parameters."trainer.optim.lr"]
values = [1e-5, 3e-5]

[parameters."orchestrator.train.sampling.temperature"]
values = [0.7, 1.0]
```

Do not include both a parent path and one of its sub-paths in the same sweep, such as `optim` and `optim.lr`; the generated override TOML can only use one shape for a path.
Every path segment must be non-empty, including keys inside structured parent-table choices: avoid leading dots, trailing dots, empty nested keys, and doubled dots such as `optim..lr`.
Do not sweep `output_dir` or nested `*.output_dir` fields, including inside structured choice values such as `trainer = { ckpt = { output_dir = ... } }`; the sweep materializer owns per-trial output directories so metrics, status, and resume artifacts stay isolated.
When sweep W&B injection is enabled (the default), do not sweep `wandb`, `wandb.group`, `wandb.name`, `wandb.tags`, or nested shared W&B fields such as `trainer.wandb.name` and `orchestrator.wandb.project`, including when those fields are hidden inside a parent structured choice value; the sweep materializer owns per-trial identity and the RL shared W&B auto-setup propagates those fields into trainer/orchestrator configs. Set sweep `wandb = None` first if you need to manage target W&B fields yourself. Top-level non-identity fields such as `wandb.project`, and non-shared nested extras such as `orchestrator.wandb.log_extras.interval`, remain sweepable.
Sweep values that affect search space, scheduling, retry counts, or stopping logic must be real finite numbers, not booleans. Choice values must be TOML-serializable and are checked recursively for `nan`/`inf`, `None`, unsupported Python objects, and non-string dict keys; use the string `"None"` for nullable target fields. Boolean choice values are only valid when the resolved target field remains boolean; materialization rejects cases where Pydantic would coerce `true`/`false` into a numeric or string target, including booleans nested inside structured parent-table choices. Uniform/log-uniform bounds must be finite, and uniform ranges must also be finite after subtraction so random sampling cannot produce `inf`; int-uniform bounds/step reject booleans. Scheduler concurrency, GPU indices, retry budget, Optuna numeric fields, pruner numeric fields, and early-stopping numeric fields reject booleans. Optuna `poll_interval_seconds` and threshold early stopping also reject `nan`/`inf`; Optuna `seed` must be in NumPy's accepted range (`0 <= seed <= 2**32 - 1`); Optuna `storage`, when set, must be a non-blank SQLAlchemy URL. ASHA `min_resource` must be `auto` or at least 1, and Hyperband integer `max_resource` must be at least `min_resource`. Optuna choice values are stricter than grid/random choices because Optuna storage must round-trip them: use only `bool`, `int`, finite `float`, or `str`, and do not mix duplicate or equality-colliding values such as `True` with `1` or `1` with `1.0`.
Sweep objectives must name a non-blank `objective.metric`; blank names are rejected at config validation instead of producing every clean trial as a missing-objective failure.

For `multi_run_lora`, parameters are limited to per-run orchestrator fields that the materializer can represent as TOML tables or exact scalar/list replacements. Do not use sub-paths under list fields such as `orchestrator.train.env.*`, `orchestrator.eval.env.*`, or `orchestrator.buffer.hash_keys.*`, including when those sub-paths are hidden inside structured choice values; swap the whole list only when the allowlist exposes that exact field. Exact allowlisted fields must also use values with the right shape: scalar fields cannot use structured values, `orchestrator.buffer.hash_keys` must use non-empty `list[str]` values, whole `*.sampling` or `*.sampling.extra_body` replacements must use dict/table values, and structured whole-`*.sampling` choices must still keep nested exact fields at their scalar/list/table shape. Do not set both `*.sampling.max_completion_tokens` and the deprecated `*.sampling.max_tokens` alias, including inside a structured whole-`*.sampling` choice. Fields coupled to the shared trainer, such as `orchestrator.max_steps` and `orchestrator.max_async_level`, are not per-run sweep fields. The shared `RLConfig` must enable `trainer.model.lora`, and `scheduler.max_concurrent_runs` must be no greater than `trainer.max_concurrent_runs`. Per-run LoRA rank/alpha overrides are validated against the shared trainer LoRA config after the shared `RLConfig` resolves. `orchestrator.batch_size` and `orchestrator.token_batch_size` are mutually exclusive sweep parameters; varying rollout `batch_size` or `oversampling_factor` clears inherited `token_batch_size`, and varying `token_batch_size` clears inherited rollout-only `batch_size` and `oversampling_factor`. `orchestrator.oversampling_factor` cannot be combined with `orchestrator.token_batch_size` because oversampling only applies to rollout batching. When `orchestrator.batch_size`, `orchestrator.token_batch_size`, or `orchestrator.oversampling_factor` varies and the shared TOML did not explicitly set `orchestrator.max_inflight_rollouts`, the materializer drops the auto-resolved shared value; rollout batching recomputes it, while token batching raises the normal requirement to set max-inflight explicitly. Group-level train/eval defaults such as `*.sampling`, `*.sampling.*`, `*.num_workers`, `*.max_retries`, eval `num_examples`, eval `rollouts_per_example`, and eval `interval` must also be allowed to re-propagate into env entries unless that env explicitly set the corresponding field in the shared TOML; deprecated shared aliases like `[[orchestrator.env]]`, `[orchestrator.sampling]`, and `max_tokens` still count as explicit shared settings while they are supported by the config loader.

When resuming Optuna sweeps with persistent storage, the controller reconciles leftover RUNNING Optuna trials from the previous process. Completed/pruned status files are replayed to Optuna; stale running or missing-objective status files are marked failed so `status.json`, the manifest, and Optuna storage agree. Replayed pruned statuses are normalized to `objective = None`. Terminal Optuna storage trials must also agree with their terminal `status.json` state: completed trials need the same finite objective value, failed/pruned trials must not carry a finite objective, and each `status.json` id must match its manifest variant id. Newly pruned trials should still carry terminal bookkeeping (`finished_at` and the subprocess/per-run return code) while keeping `objective = None`. Grid/random, static `multi_run_lora`, and Optuna trials that fail target-config materialization are written as failed manifest/status artifacts, excluded from scheduler launches, and must clear stale generated resolved configs (`resolved.toml`, plus `control/orch.toml` for `multi_run_lora`) from any reused trial directory. With `continue_on_failure=false`, grid/random and static `multi_run_lora` materialization stops after the first failed trial and does not launch scheduler work after that preflight failure. In Optuna `multi_run_lora`, if `continue_on_failure=false` aborts a wave after one trial fails materialization, already-materialized siblings that were asked but never launched are marked failed with `failure_stage = "scheduler"` and an explanatory error before being told failed to Optuna. A clean process exit without a finite objective is recorded as `state = "failed"` with `failure_stage = "objective"` in local grid/random, static `multi_run_lora`, single-trial Optuna, and `multi_run_lora` Optuna flows. Resume requires the previous manifest to be valid JSON, be a JSON object whose `variants` are JSON object entries, the strategy (except increasing `strategy.num_trials`), search parameters, parameter order, and objective to match the previous manifest, exactly one manifest variant entry with an existing `status.json` and `resolved_checksum` for every existing Optuna storage trial, no manifest variants missing from storage, manifest `overrides` and id hashes that match Optuna storage parameters, and matching base-config checksums; previous `TrialState.FAIL` entries are carried into the resumed failure count.
When resuming grid/random sweeps, terminal `status.json` files are only skipped if the previous manifest is valid JSON, is a JSON object with the same entrypoint and scheduler type, compatible strategy (except increasing `strategy.num_trials`), same search parameters/objective and parameter order, unique well-formed variant IDs, no manifest variants outside the regenerated trial set, matching `status.json` ids, valid object-shaped `status.json` files, and still carries resolved/base checksums for the same base file list. Parameter order matters because grid/random generation consumes parameters in order, and the manifest stores `parameter_order` explicitly because JSON key sorting cannot preserve it. If the manifest is missing or incomplete, status files are malformed, or a status file is not a JSON object, resume fails closed instead of silently trusting stale completed/submitted statuses. A rejected resume drift must not overwrite the previous trial's `overrides.toml`, `resolved.toml`, or `command.txt`. Legacy completed trials with missing objectives are still counted as objective failures on resume; if an Optuna resume finds a leftover RUNNING storage trial whose status already recorded clean completion without a finite objective, keep `returncode = 0` and mark `failure_stage = "objective"` rather than turning it into a launcher failure.
For local grid/random and Optuna sweeps, `continue_on_failure = false` stops launching new trials after the first runtime failure or missing objective, but the controller should still write per-trial `status.json` updates and the manifest summary for any completed objective-bearing trials before exiting non-zero. Static `multi_run_lora` launches the whole wave at once, so it cannot stop already-running siblings and does not support early stopping; Optuna `multi_run_lora` can early-stop between waves. Static `multi_run_lora` should still reconcile all per-run exit codes and write the objective summary before exiting non-zero.
Before launching a fresh attempt, the sweep runtime clears attempt-scoped objective artifacts (`metrics.jsonl` plus legacy `run-*/final_summary.json`). `multi_run_lora` also clears stale `control/exit_code` and `control/evicted.txt` before invoking `rl-multi-run`, so old pruning or exit signals cannot affect a fresh wave.
Because the shared trainer scans every `shared/run_*` directory, `multi_run_lora` launches also mark inactive run directories evicted before starting `rl-multi-run`; only the current static wave or Optuna wave should be discoverable by the trainer.
Optuna pruning reports only finite intermediate metrics with non-negative integer `step` values; malformed, boolean, missing, or negative steps are ignored instead of being sent to `optuna_trial.report`. Final objectives read from `metrics.jsonl` use the same step rule and fall back to legacy `final_summary.json` only when no valid-step sidecar row supplies the metric. `FileMonitor` treats the explicit `monitor.log(..., step=...)` argument as the canonical sweep step even if the metrics payload also contains a `step` key, rejects non-integer, boolean, and negative explicit steps before mutating history or writing the sidecar, and replaces non-finite floats with `null` inside dict/list/tuple metric payloads so `metrics.jsonl` stays standard JSON. When `FileMonitor` is combined with W&B or Prime in `MultiMonitor`, its errors must propagate because the sweep sidecar is required for objective attribution; optional remote-monitor errors can remain warnings. A pruning decision should only be applied while the trial is still running; re-check subprocess liveness after reporting an intermediate metric and immediately before `should_prune()`. In `multi_run_lora`, `rl-multi-run` writes each `control/exit_code` as soon as that orchestrator exits, and the Optuna poller must re-read that file immediately before pruning so a completed run is not reclassified as pruned while sibling runs keep the wave alive. If the launcher has already exited, final objective reconciliation wins.
If the target launcher itself cannot be spawned (`OSError`, such as a missing command), local, pruner-enabled Optuna, SLURM, and `multi_run_lora` static/Optuna schedulers retry according to `retry_budget`; only after the final failed launch attempt should the runtime mark the affected trial(s) `failed` with `returncode = -1`, `failure_stage = "launch"`, and the error message in `status.json` before continuing or exiting according to `continue_on_failure`.

### Discriminated unions

Some config fields use discriminated unions (e.g. loss type, data type). Set the `type` field to select the variant:

```toml
[trainer.loss]
type = "sft"

[data]
type = "fake"
batch_size = 2
```

On the CLI:

```bash
uv run sft --data.type fake --data.batch-size 4
```

If you wish to configure values of the default variant, you don't need to set the `type` field.

### SFT hard distill override

For hosted multi-tenant runs where the trainer image's `trainer.loss.type` is fixed, the orchestrator exposes a per-run override that forces SFT loss on every micro-batch without rebuilding the trainer. Set `orchestrator.use_sft_loss = true` alongside `orchestrator.teacher_rollout_model`; both must be configured together (the orchestrator validator enforces this). The orchestrator stamps each `TrainingSample.sft_loss = True`, which the trainer's `compute_loss` honors by dispatching to `sft_loss_fn` per batch — independent of the trainer's configured default loss.

### Model fields

For `BaseModel | None` fields (like `[ckpt]`, `[wandb]`, `[compile]`), a bare flag enables them with defaults:

```bash
uv run rl @ config.toml --model.compile              # enables compilation with defaults (fullgraph = false)
uv run rl @ config.toml --model.compile.fullgraph    # enables compilation and sets nested field (fullgraph = true)
```

In TOML, an empty section header does the same:

```toml
[ckpt]  # enables checkpointing with defaults
```

## Key files

- `packages/prime-rl-configs/src/prime_rl/utils/config.py` — re-exports `BaseConfig` and `cli` from pydantic_config
- `packages/prime-rl-configs/src/prime_rl/configs/` — all domain-specific config classes
- `configs/debug/` — minimal debug configs for testing
- `examples/` — full example configs for various tasks
