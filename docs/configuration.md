# Configuration

Every `prime-rl` entrypoint — `rl`, `sft`, `trainer`, `orchestrator`, `inference`, `eval` — is configured by the same system: TOML files for reproducible base configs, CLI flags for one-off overrides, and a small set of environment variables for production deployments. Under the hood it is [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) wrapping our Pydantic config models.

## Table of Contents

- [Sources and precedence](#sources-and-precedence)
- [TOML files and composition](#toml-files-and-composition)
- [CLI overrides](#cli-overrides)
- [Environment variables](#environment-variables)
- [Inspecting and validating](#inspecting-and-validating)
- [Special syntax](#special-syntax)
  - [Booleans, `None`, and lists](#booleans-none-and-lists)
  - [Optional sub-configs](#optional-sub-configs)
  - [Discriminated unions](#discriminated-unions)
  - [Environments (`[[orchestrator.train.env]]`)](#environments-orchestratortrainenv)
- [Worked example](#worked-example)
- [Conventions](#conventions)

## Sources and precedence

For a single field, sources are applied in this order — later sources win:

1. **Defaults** — declared on the Pydantic model.
2. **TOML files** — passed with `@`, left to right (later files override earlier ones).
3. **CLI flags** — dotted, kebab-case (`--model.name`).

There is no generic `PRIME_*` env-var override for arbitrary fields. A small number of specific fields (log levels, API keys, monitor URLs) read named env vars as their **default** — see [Environment variables](#environment-variables) below — but a field that has been set in a TOML or on the CLI is not overridden by an env var.

Recommendation: pin reproducible experiments in TOML and override one-off knobs (W&B name, output dir, max steps) on the CLI.

## TOML files and composition

`@` introduces a TOML file. Multiple `@` arguments compose left-to-right, deep-merged — unset fields in an overlay keep the base value:

```bash
uv run rl @ configs/gsm8k/rl.toml                              # one file
uv run rl @ base.toml @ overlay.toml                           # left to right
uv run rl --trainer @ trainer.toml --orchestrator @ orch.toml  # per-section
uv run rl @ base.toml --trainer @ trainer.toml                 # mixed
```

**Mind the space**: `@ path/to/x.toml`, not `@path/to/x.toml`.

The composed `rl` entrypoint splits its config across three processes — `[trainer]`, `[orchestrator]`, and `[inference]` tables become the sub-configs for each. Shared knobs (`model.name`, `output_dir`, `wandb.*`, …) live at the top level and are fanned out automatically. Stand-alone entrypoints (`uv run trainer`, `uv run orchestrator`, …) skip this lifting — their TOMLs have no `[trainer]` table because the whole file _is_ the trainer.

## CLI overrides

CLI flags mirror the TOML tree using dots, with kebab-case for field names (the leading `--` is a kebab-case marker; TOML stays snake_case):

```bash
--max-steps 50                              # top-level
--model.name Qwen/Qwen3-4B                  # nested
--trainer.optim.lr 1e-5                     # double-nested
--inference.parallel.tp 4
```

Field names in TOML use snake_case (`max_model_len`); the same field on the CLI is kebab-case (`--max-model-len`).

Three convenience flags every entrypoint accepts:

- `--help` — prints the full schema (all fields, defaults, types, descriptions).
- `--dry-run` — resolves the full config, writes it to `<output_dir>/configs/`, and exits without launching anything. Use to debug composition.
- `--output-dir <path>` — top-level override for the run's working directory (logs, checkpoints, weight snapshots).

## Environment variables

Only a fixed set of env vars are wired into individual config fields as their default. They're a convenience for things that legitimately vary per deployment (credentials, log levels) — they don't generalize to "set any field via env var."

- `PRIME_LOG_LEVEL` / `PRIME_VF_LOG_LEVEL` — defaults for `[log] level` and `[log] vf_level` (the prime-rl and verifiers loggers).
- `WANDB_API_KEY` / `HF_TOKEN` — read directly by W&B and `huggingface_hub`, never by prime-rl itself.
- `PRIME_API_KEY` — read by `[orchestrator.prime_monitor]` for [platform monitoring](training.md#platform-monitoring). The env var name is itself configurable via `prime_monitor.api_key_var`.
- `INFERENCE_SERVER_API_KEY` (or whatever you set in `client.api_key_var`) — used by the orchestrator to authenticate to the inference server.

Any other field needs to be set in TOML or on the CLI.

## Inspecting and validating

```bash
uv run rl --help                                       # full schema
uv run rl @ rl.toml --dry-run --output-dir /tmp/check  # write resolved configs
```

`--dry-run` is the single most useful debugging tool: it runs every Pydantic validator (catching incompatibilities like CP requiring flash-attention, or NCCL weight broadcast requiring `max_async_level=1`) and dumps the fully merged config to disk. If a run misbehaves in mysterious ways, dry-run it first and inspect `<output_dir>/configs/`.

When a validator fails, the error names the conflicting fields — fix one and re-run dry-run until clean.

## Special syntax

### Booleans, `None`, and lists

**Booleans** —  CLI uses paired flags: `--ckpt` enables, `--no-ckpt` disables. TOML must be explicit:

```toml
ckpt = true
```

**None** — TOML has no `null`. Use the string `"None"`, which the loader coerces:

```toml
[inference.model]
max_model_len = "None"
```

On the CLI: `--inference.model.max-model-len None`.

**Lists** — TOML uses arrays of tables (see the env example below). Overlays **replace** lists wholesale, so an overlay that only wants to add an env still has to include the full list. On the CLI, index by position:

```bash
--orchestrator.train.env.0.id math-env --orchestrator.train.env.1.id reverse-text
```

### Optional sub-configs

Many sub-configs are typed `SomeConfig | None`. Two patterns enable them:

- **Bare flag with defaults**: `--model.compile` or, in TOML, an empty section `[model.compile]`. The sub-config materializes with all-default values.
- **Enable and set fields together**: `--model.compile.fullgraph` (CLI) or any populated `[model.compile]` table (TOML).

This is how `[ckpt]`, `[model.lora]`, `[model.compile]`, `[trainer.wandb]`, etc. are turned on.

### Discriminated unions

Loss, advantage, optimizer, scheduler, weight broadcast transport, and several others are discriminated unions. Set the `type` field to pick a variant:

```toml
[trainer.optim]
type = "muon"
lr = 1e-5
mu = 0.95
```

Omit `type` to keep the default variant. See [Reference](reference.md) for every variant's fields.

### Environments (`[[orchestrator.train.env]]`)

Training environments are an array of tables — set one per env, optionally with sampling weights:

```toml
[[orchestrator.train.env]]
id = "math-env"
name = "gsm8k"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }

[[orchestrator.train.env]]
id = "reverse-text"
ratio = 0.25  # 25% of batches; remaining 75% goes to math-env

[[orchestrator.eval.env]]
id = "math-env"
name = "gsm8k-eval"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }
```

`args` is forwarded verbatim to the environment's `load_environment(**args)`. See each environment's README on the [Hub](https://app.primeintellect.ai/dashboard/environments) for accepted args.

## Worked example

Start from a shipped base config, override two fields on the CLI, and dry-run:

```bash
uv run rl @ configs/gsm8k/rl.toml \
  --wandb.name my-experiment \
  --trainer.optim.lr 5e-6 \
  --dry-run \
  --output-dir /tmp/gsm8k-dry
```

Then inspect the resolved config:

```bash
ls /tmp/gsm8k-dry/configs/
# rl.toml  trainer.toml  orchestrator.toml  inference.toml
```

Each per-process TOML reflects the final, validated configuration that the actual run would consume — exactly what each process sees when started standalone (`uv run trainer @ /tmp/gsm8k-dry/configs/trainer.toml`, etc.). This is the easiest way to bisect a misbehaving config: dry-run a known-good base, dry-run your overlay, diff the two.

## Conventions

- **Reproducible base, mutable overlays.** Commit base TOMLs alongside example dirs (`configs/<task>/rl.toml`). Override on the CLI for one-shot experiments; promote overrides to a new TOML when they stabilize.
- **One W&B name per run.** Pass `--wandb.name <unique>` on every launch. The orchestrator and trainer share the W&B run, so the same name surfaces all metrics together.
- **Always pin `output_dir`.** Per-run output directories prevent rollout files from one run leaking into another's training step. Use `--output-dir outputs/<run-name>` or pin in TOML.
- **Prefer `--ckpt` for any run you might resume.** Without `ckpt`, only HF weight snapshots are written — you can serve them but cannot resume optimizer/scheduler state. See [Training § Checkpointing](training.md#checkpointing).
- **Dry-run before scaling.** A multi-node SLURM job that crashes on a config validator wastes a queue slot. Always `--dry-run` first.

For the full set of fields, defaults, types, and constraints accepted by each entrypoint, jump to [Reference](reference.md).
