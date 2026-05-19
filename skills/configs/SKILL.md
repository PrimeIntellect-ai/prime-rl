---
name: configs
description: How the prime-rl config system works — TOML files, CLI overrides, composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Configs

prime-rl uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — a Pydantic-based TOML + CLI config system (no tyro). Every entrypoint accepts TOML files via `@` and CLI overrides.

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
uv run rl --help                                  # all fields and defaults
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

## RL trainer token exports

For rollout visualization/debugging, enable trainer-side token export under `trainer.experimental.token_export` (or `experimental.token_export` when running the trainer entrypoint directly). It writes JSONL records with per-token ids, loss mask, advantage, reward, entropy, mismatch KL, inference logprob/prob, trainer logprob/prob, and masking diagnostics. It does not decode token text in the trainer.

```toml
[trainer.experimental.token_export]
# Optional. Relative paths resolve under trainer.output_dir.
path = "token_exports/sample.jsonl"
```

Leave it unset for normal training. When enabled, it exports every sequence from each exporting rank.

```bash
uv run scripts/token_export_visualizer.py outputs/token_exports/rank_0.jsonl --tokenizer Qwen/Qwen3-0.6B -o /tmp/token_export.html
uv run scripts/token_export_visualizer.py outputs/token_exports --tokenizer Qwen/Qwen3-0.6B -o outputs/token_exports/index.html
```

Directory inputs render every exported sequence as one navigable HTML page. For a single JSONL file, pass `--all-records` to embed every matching record instead of just one `--record-index`.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `configs/private/` — private configs submodule (internal)
- `examples/` — full example configs
