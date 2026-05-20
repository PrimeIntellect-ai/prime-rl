---
name: configs
description: How the prime-rl config system works — TOML files, CLI overrides, composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Configs

prime-rl uses `pydantic_config` (combining `tyro` and `pydantic`). Every entrypoint accepts TOML files via `@` and CLI overrides.

## Loading and composition

```bash
uv run rl @ examples/reverse_text/rl.toml                                  # single TOML
uv run rl @ examples/reverse_text/rl.toml --max-steps 50                   # CLI override
uv run rl @ base.toml @ overlay.toml                                       # left-to-right merge
uv run rl --model @ model.toml --data @ data.toml                          # nested section files
uv run rl @ base.toml --trainer @ trainer.toml --trainer.lr 1e-3           # mixed
```

Resolution order: CLI > config files (left-to-right) > class defaults. Merging is deep — unset fields in an overlay are preserved from the base.

Naming: CLI uses kebab-case (`--model.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect & validate

```bash
uv run rl --help                                  # defaults from class
uv run rl @ rl.toml --help                        # defaults merged with TOML
uv run rl @ rl.toml --dry-run --output-dir /tmp/x # write resolved TOML to /tmp/x/configs
```

## Validators

Incompatible combinations (e.g. CP requires flash attention) must raise in a `model_validator` at resolve time, not at runtime. When renaming a field, emit a deprecation warning with a migration hint — never silently drop.

## Special syntax

**Booleans** — CLI `--flag` / `--no-flag`; TOML must be explicit (`enforce_eager = true`).

**None** — TOML has no null, use the string `"None"` (`max_model_len = "None"`); CLI: `--model.max-model-len None`.

**Lists** — TOML uses array of tables; later config files replace lists wholesale, so overlays must include the full desired list:

```toml
[[orchestrator.env]]
id = "reverse-text"
```

CLI: `--env.0.id reverse-text --env.1.id math-env`.

**Dicts** — TOML uses a section; CLI takes a JSON string: `--vllm-extra '{"key1": "value1"}'`.

**Discriminated unions** — set the `type` field to pick the variant (`[trainer.loss] type = "sft"`). Omit `type` to keep the default variant.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## Domain notes

- **SFT hard distill** — set `orchestrator.training_mode = "sft"` (or top-level `training_mode = "sft"`, which auto-propagates) and configure `orchestrator.teacher`. The trainer's `loss.type` is derived from `training_mode`. `[inference]` is still required for the student server unless `orchestrator.student.client.base_url` is set explicitly.
- **RL rollout client** — text-only RL defaults to renderer-backed TITO (`use_renderer = true`). VLM and external-teacher rollouts must set `use_renderer = false`.

## Key files

- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `packages/prime-rl-configs/src/prime_rl/utils/config.py` — re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `configs/private/` — private configs submodule (internal)
- `examples/` — full example configs
