# Configuration

Every `prime-rl` entrypoint uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config): TOML files for reproducible base configs, CLI flags for one-off overrides.

> **AI agents working in this repo:** the equivalent runbook is at [`skills/configs/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/configs/SKILL.md), with extra runtime hints (where config classes live, validator conventions, the trainer-side `token_export` flag) that aren't surfaced here.

## Table of Contents

- [Sources and precedence](#sources-and-precedence)
- [TOML composition](#toml-composition)
- [CLI overrides](#cli-overrides)
- [Inspecting and validating](#inspecting-and-validating)
- [Syntax](#syntax)
  - [Booleans](#booleans)
  - [Lists](#lists)
  - [Dicts](#dicts)
  - [Optional sub-configs](#optional-sub-configs)
  - [None](#none)
  - [Discriminated unions](#discriminated-unions)
  - [Environments (`[[orchestrator.train.env]]`)](#environments-orchestratortrainenv)
- [Worked example](#worked-example)

## Sources and precedence

Field values come from three sources — Pydantic defaults, TOML files (passed with `@`), and CLI flags. They're layered in this order, with later sources winning:

1. **Defaults** declared on the Pydantic model.
2. **TOML files** passed with `@`, left to right — later files override earlier ones.
3. **CLI flags** in dotted, kebab-case form (`--model.name`).

## TOML composition

The `@` token introduces a TOML file. Multiple `@` arguments compose left-to-right, deep-merged — unset fields in an overlay keep the base value:

```bash
uv run rl @ configs/gsm8k/rl.toml                              # one file
uv run rl @ base.toml @ overlay.toml                           # left to right
uv run rl --trainer @ trainer.toml --orchestrator @ orch.toml  # per-section
uv run rl @ base.toml --trainer @ trainer.toml                 # mixed
```

> Mind the space: `@ path/to/x.toml`, not `@path/to/x.toml`.

## CLI overrides

CLI flags mirror the TOML tree using dots:

```bash
--max-steps 50                              # top-level
--model.name Qwen/Qwen3-4B                  # nested
--trainer.optim.lr 1e-5                     # double-nested
--inference.parallel.tp 4
```

> Field names are snake_case in TOML (`max_model_len`) and kebab-case on the CLI (`--max-model-len`).

> Renamed fields keep their old name as a validation alias — e.g. `rollouts_per_example` is still accepted in TOML and CLI after being renamed to `group_size`. Mixing the two names across sources is safe.

## Inspecting and validating

```bash
uv run rl --help                                       # full schema
uv run rl @ rl.toml --dry-run --output-dir /tmp/check  # write resolved configs
```

## Syntax

### Booleans

CLI uses paired flags: bare `--flag` sets `True`, `--no-flag` sets `False`. TOML must be explicit:

```bash
uv run rl @ rl.toml --clean-output-dir       # True
uv run rl @ rl.toml --no-clean-output-dir    # False
```

```toml
clean_output_dir = true
```

### Lists

CLI accepts space-separated values or a JSON literal. TOML uses an array literal. Both forms target the same field:

```bash
uv run rl @ rl.toml --trainer.model.lora.target-modules q_proj k_proj v_proj
uv run rl @ rl.toml --trainer.model.lora.target-modules '["q_proj", "k_proj", "v_proj"]'
```

```toml
[trainer.model.lora]
target_modules = ["q_proj", "k_proj", "v_proj"]
```

Overlay TOMLs **replace** lists wholesale — an overlay that wants to add one item must still spell out the full list. For arrays of tables (e.g. environments), see [Environments](#environments-orchestratortrainenv).

### Dicts

CLI takes a JSON literal. TOML uses a table or inline-table. CLI dicts deep-merge with TOML dicts — CLI keys win on conflict but don't wipe the file's keys:

```bash
uv run rl @ rl.toml --orchestrator.train.env.0.args \
  '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}'
```

```toml
[[orchestrator.train.env]]
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }
```

### Optional sub-configs

Many sub-configs are typed `SomeConfig | None`. Two patterns enable them:

- **Bare flag with defaults**: `--model.compile` or, in TOML, an empty section `[model.compile]`. The sub-config materializes with all-default values.
- **Enable and set fields together**: `--model.compile.fullgraph` (CLI) or any populated `[model.compile]` table (TOML).

To **disable** a sub-config that's on by default, use `--no-<name>` on the CLI or assign the string `"None"` in TOML (see [None](#none)). This is how `[ckpt]`, `[model.lora]`, `[model.compile]`, `[trainer.wandb]`, etc. are turned on and off.

### None

TOML has no `null`. Use the string `"None"`, which the loader coerces:

```toml
[inference.model]
max_model_len = "None"
```

On the CLI: `--inference.model.max-model-len None`.

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
  --output-dir /tmp/gsm8k-dry \
  --dry-run
```

Then inspect the resolved config:

```bash
ls /tmp/gsm8k-dry/configs/
# rl.toml  trainer.toml  orchestrator.toml  inference.toml
```

Each per-process TOML reflects the final, validated configuration that the actual run would consume — exactly what each process sees when started standalone (`uv run trainer @ /tmp/gsm8k-dry/configs/trainer.toml`, etc.). This is the easiest way to bisect a misbehaving config: dry-run a known-good base, dry-run your overlay, diff the two.

For the full set of fields, defaults, types, and constraints accepted by each entrypoint, jump to [Reference](reference.md).
