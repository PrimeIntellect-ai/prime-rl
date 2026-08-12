---
name: configs
description: How the prime-rl config system works — TOML files, CLI overrides, composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Configs

prime-rl uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — a Pydantic-based TOML + CLI config system (no tyro). Every entrypoint accepts TOML files via `@` and CLI overrides.

## Loading and composition

```bash
uv run rl @ examples/basic/reverse-text/rl.toml                                  # single TOML
uv run rl @ examples/basic/reverse-text/rl.toml --max-steps 50                   # CLI override
uv run rl @ base.toml @ overlay.toml                                       # left-to-right merge
uv run rl --model @ model.toml --data @ data.toml                          # nested section files
uv run rl @ base.toml --trainer @ trainer.toml --trainer.lr 1e-3           # mixed
```

Resolution order: CLI > config files (left-to-right) > class defaults. Merging is deep — unset fields in an overlay are preserved from the base.

Naming: CLI uses kebab-case (`--vllm.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect & validate

```bash
uv run rl --help                                  # all fields and defaults
uv run rl @ rl.toml --dry-run --output-dir /tmp/x # write resolved TOML to /tmp/x/configs
```

## Validators

Incompatible combinations (e.g. CP requires flash attention) must raise in a `model_validator` at resolve time, not at runtime. When renaming a field, remove the old spelling: no `validation_alias`, no auto-translating `mode="before"` validator. The old key then fails as an unknown key, which is the signal. An alias that stays forever is worse than a break — it never gets retired, and a key whose *meaning* changed silently misconfigures the run.

## Special syntax

**No inline tables** — checked-in configs use `[section]` headers, never `key = { ... }`. Expand `env.taskset = { id = "..." }` to a full-path header (`[orchestrator.train.source.env.taskset]` — subtable headers after a `[[...]]` entry attach to that entry).

**Booleans** — CLI `--flag` / `--no-flag`; TOML must be explicit (`enforce_eager = true`).

**None** — TOML has no null, use the string `"None"` (`max_model_len = "None"`); CLI: `--vllm.max-model-len None`.

**Lists** — TOML uses array of tables; later config files replace lists wholesale, so overlays must include the full desired list:

```toml
[[orchestrator.train.source]]
name = "reverse-text"

[orchestrator.train.source.env.taskset]
id = "reverse-text"

[orchestrator.train.source.env.agent.harness]
id = "null"

[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[[orchestrator.eval.source]]
name = "reverse-text-eval"

[orchestrator.eval.source.env.taskset]
id = "reverse-text"
split = "test"

[orchestrator.eval.source.env.agent.harness]
id = "null"

[orchestrator.eval.source.env.agent.runtime]
type = "subprocess"
```

CLI: `--orchestrator.train.source.0.env.taskset.id reverse-text` or `--orchestrator.eval.source.0.env.taskset.id reverse-text`.

The `sft` entrypoint takes the same eval shape at the top level for online evals: `[eval]` + `[[eval.source]]` (with `[inference]` for the server), e.g. `--eval.source.0.env.taskset.id reverse-text`.

**Dicts** — TOML uses a section; CLI takes a JSON string: `--trainer.env-vars '{"key1": "value1"}'`. This works for plain `dict` fields only — nested pydantic-model fields (e.g. `algo`) reject JSON strings; use dotted keys (`--orchestrator.algo.type max_rl`) or a TOML overlay file.

**vLLM pass-through** — `[inference.vllm]` uses vLLM's own argument names (`model`, `tensor_parallel_size`, `data_parallel_size`, `max_model_len`, ...) and forwards *any* key to the vLLM server, typed by prime-rl or not: `[inference.vllm] max_num_seqs = 256`, or `--inference.vllm.max-num-seqs 256` on the CLI. CLI values are JSON-coerced, so dict-valued vLLM args work as `--inference.vllm.compilation-config '{"cudagraph_mode": "NONE"}'`. Non-vLLM knobs (router, deployment, weight broadcast, kv-cache offload, env vars) stay on `[inference]` itself.

**Discriminated unions** — set the `type` field to pick the variant (`[orchestrator.algo] type = "max_rl"`). Omit `type` to keep the default variant.

**Algorithms** — `[orchestrator.algo] type = "grpo" | "max_rl" | "rae" | "hierarchical_grpo" | "opd" | "opsd" | "sft" | "echo"` — the type names the algorithm (credit assignment + loss routing, fused), and each type's class defaults are its vetted setting; any other key you set is your own assembly (e.g. `[orchestrator.algo.roles.user] alpha = 0.1` for echo — setting any echo role replaces the whole role table). `hierarchical_grpo` is only valid with a proposer-solver env: it compares solvers with attempts on the same proposed problem and proposers with other proposals in the group. There is no preset layer, and no config hook that points at user code — a new algorithm is a named class in the repo (subclass `Algorithm`, register it). Per-source override: `[orchestrator.train.source.algo] type = "opd"` (the source assembles its own algorithm). prime-rl only hosts the trainable policy; frozen models are inline external endpoints on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for opd (the frozen model scored against), `[orchestrator.algo.sampling.source]` for sft (the model it samples from), each with `name` + `base_url`. There is no shared `teacher` slot. opsd declares no model — it self-distills against the live policy. See `docs/algorithms.md`.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## RL trainer token exports

For rollout debugging, enable trainer-side token export with `trainer.enable_token_export = true` (or `--enable-token-export` when running the trainer entrypoint directly). It writes one JSONL record per exported sequence under `output_dir/token_exports/step_<step>/rank_<rank>.jsonl`. Each record stores aligned per-token arrays for token ids, loss mask, component weight streams (rl/ce/ref_kl), advantages, entropy, mismatch KL, inference/trainer logprobs, importance ratios, probability deltas, and masking diagnostics. It does not decode token text in the trainer.

```toml
enable_token_export = true
```

Leave it unset for normal training. When enabled, it exports every sequence from each exporting rank.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs
