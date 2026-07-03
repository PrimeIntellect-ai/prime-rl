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

**Dicts** — TOML uses a section; CLI takes a JSON string: `--vllm-extra '{"key1": "value1"}'`. This works for plain `dict` fields only — nested pydantic-model fields (e.g. `algo`) reject JSON strings; use dotted keys (`--orchestrator.algo.type max_rl`) or a TOML overlay file.

**Discriminated unions** — set the `type` field to pick the variant (`[orchestrator.algo] type = "max_rl"`). Omit `type` to keep the default variant.

**Algorithms** — `[orchestrator.algo] type = "grpo" | "max_rl" | "opd" | "opsd" | "sft" | "echo"` — the type names the algorithm (credit assignment + loss routing, fused), and each type's class defaults are its vetted setting; any other key you set is your own assembly (e.g. `[orchestrator.algo.roles.user] alpha = 0.1` for echo — setting any echo role replaces the whole role table). There is no preset layer, and no config hook that points at user code — a new algorithm is a named class in the repo (subclass `Algorithm`, register it). Per-env override: `[orchestrator.train.env.algo] type = "opd"` (the env assembles its own algorithm). prime-rl only hosts the trainable policy; frozen models are inline external endpoints on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for opd (the frozen model scored against), `[orchestrator.algo.sampling.source]` for sft (the model it samples from), each with `name` + `base_url`. There is no shared `teacher` slot. opsd declares no model — it self-distills against the live policy. See `docs/algorithms.md`.

**Replay envs** — taskset ids `replay-continue-v1` (`anchor = "compaction" | "tool-call"`) and `replay-recheck-v1` (`instruction`) replay saved rollouts (`<run>/rollouts/step_*/train_rollouts.jsonl`) as training tasks; thin packages over `verifiers.v1.tasksets.replay` (like swebench over harbor), configured as normal train env entries with a `ratio` (ratios are all-or-none, and replay envs need one — they advertise a single virtual task). Flat taskset config, base fields: `buffer_dir` (a prior run's rollouts dir, or `"self"` for this run's own growing buffer — implies online + whole-group dispatch), required `inner` (the source run's taskset config, which scores the derived rollouts), `source_envs` (allowlist over the `info.prime_rl.env_name` stamp; unset = every env except replay envs, listing a replay env opts into chained derivations), `allow_container`. Run replay envs under the source env's harness (must support message prompts) and give the env entry explicit `max_input_tokens` / `max_total_tokens` headroom (replay seeds are long). Debug configs: `configs/debug/replay/`. See `docs/replay.md`.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## RL trainer token exports

For rollout debugging, enable trainer-side token export with `trainer.enable_token_export = true` (or `--enable-token-export` when running the trainer entrypoint directly). It writes one JSONL record per exported sequence. Single-run/fallback exports go under `output_dir/token_exports/step_<step>/rank_<rank>.jsonl`; multi-run trainer exports with packer metadata go under the owning run directory, `output_dir/<run_id>/token_exports/step_<run_step>/rank_<rank>.jsonl`. Each record stores aligned per-token arrays for token ids, loss mask, component weight streams (rl/ce/ref_kl), advantages, entropy, mismatch KL, inference/trainer logprobs, importance ratios, probability deltas, and masking diagnostics. It does not decode token text in the trainer.

```toml
enable_token_export = true
```

Leave it unset for normal training. When enabled, it exports every sequence from each exporting rank.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs
