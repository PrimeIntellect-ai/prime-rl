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

Resolution order: CLI > config files (left-to-right) > class defaults. Merging is deep — unset fields in an overlay are preserved from the base. `output_dir` has one extra fallback: CLI > config files > `$PRL_OUTPUT_DIR` > `"outputs"`.

Naming: CLI uses kebab-case (`--vllm.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect & validate

```bash
uv run rl --help                                  # all fields and defaults
uv run rl @ rl.toml --dry-run --output-dir /tmp/x --run.name check # write resolved JSON to /tmp/x/check/configs/latest
```

Each attempt also writes `configs/attempt_<n>/command.txt`. It records the
shell-safe launch command, including CLI overrides. `configs/latest` points to
the current attempt.

## Validators

Incompatible combinations (e.g. CP requires flash attention) must raise in a `model_validator` at resolve time, not at runtime. When renaming a field, remove the old spelling: no `validation_alias`, no auto-translating `mode="before"` validator. The old key then fails as an unknown key, which is the signal. An alias that stays forever is worse than a break — it never gets retired, and a key whose *meaning* changed silently misconfigures the run.

## Special syntax

**No inline tables** — checked-in configs use `[section]` headers or dotted keys, never `key = { ... }`.

**Sources are one block** — inside a `[[...source]]` entry, write nested sub-configs as dotted keys in the same block (`env.taskset.id = "..."`, `env.agent.harness.id = "..."`), not one subsection header per nested config. Nested arrays of tables (e.g. `[[orchestrator.train.source.env.taskset.task.judges]]`) keep full-path headers — they attach to the preceding `[[...source]]` entry.

**Booleans** — CLI `--flag` / `--no-flag`; TOML must be explicit (`enforce_eager = true`).

**None** — TOML has no null, use the string `"None"` (`max_model_len = "None"`); CLI: `--vllm.max-model-len None`.

**Lists** — TOML uses array of tables; later config files replace lists wholesale, so overlays must include the full desired list:

```toml
[[orchestrator.train.source]]
name = "reverse-text"
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.eval.source]]
name = "reverse-text-eval"
env.taskset.id = "reverse-text"
env.taskset.split = "test"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

Each source group (`[orchestrator.train]`, `[orchestrator.eval]`, the `sft` `[eval]` block, the top level of `eval`) holds defaults that its sources inherit: every field that the group and its sources both have (`env`, `sampling`, `select`, `group_size`, plus `algo` on train and `interval` on online eval). Blocks merge key by key, a source's own values win, and a block whose `type` differs is the source's alone.

Source lists are set in TOML. The CLI has no list-index paths (`--orchestrator.eval.source.0.env.taskset.id` does not parse); a whole list can be passed as JSON (`--orchestrator.eval.source '[{"env": {"taskset": {"id": "reverse-text"}}}]'`), which replaces the list wholesale.

The `sft` entrypoint takes the same eval shape at the top level for online evals: `[eval]` + `[[eval.source]]` (with `[inference]` for the server). The `eval` entrypoint flattens it further: `[[source]]`, `[client]`, `[concurrency]`, `[select]`, `group_size` at the top level, plus the single-source shorthands `uv run eval <taskset-id> --env.<field> <value> -n N -s -r N -c N -m MODEL` (see the `eval` skill).

**Dicts** — TOML uses a section; CLI takes a JSON string: `--trainer.env-vars '{"key1": "value1"}'`. This works for plain `dict` fields only — nested pydantic-model fields (e.g. `algo`) reject JSON strings; use dotted keys (`--orchestrator.train.algo.type max_rl`) or a TOML overlay file.

**vLLM pass-through** — `[inference.vllm]` uses vLLM's own argument names (`model`, `tensor_parallel_size`, `data_parallel_size`, `max_model_len`, ...) and forwards *any* key to the vLLM server, typed by prime-rl or not: `[inference.vllm] max_num_seqs = 256`, or `--inference.vllm.max-num-seqs 256` on the CLI. CLI values are JSON-coerced, so dict-valued vLLM args work as `--inference.vllm.compilation-config '{"cudagraph_mode": "NONE"}'`. Non-vLLM knobs (router, deployment, weight broadcast, kv-cache offload, env vars) stay on `[inference]` itself.

**Discriminated unions** — set the `type` field to pick the variant (`[orchestrator.train.algo] type = "max_rl"`). Omit `type` to keep the default variant.

**RL loss** — `[trainer.loss]` defaults to IPO with `eps = 0.1`, `adv_tau = 1.0`, and `kl_tau = 1e-3`. Omit the section to use these defaults. Set `type = "icepop"` for ratio masking with `ratio_low = 0.2`, `ratio_high = 5.0`, and `adv_tau = 1.0`; IcePop has no separate KL term. Set `type = "custom"` with `import_path` and optional `kwargs` to load a custom RL loss. The `ce` and `ref_kl` components are fixed.

**Algorithms** — `[orchestrator.train.algo] type = "grpo" | "max_rl" | "rae" | "hierarchical_grpo" | "opd" | "opsd" | "sft" | "echo"` — the type names the algorithm (credit assignment + loss routing, fused), and each type's class defaults are its vetted setting; any other key you set is your own assembly (e.g. `[orchestrator.train.algo.roles.user] alpha = 0.1` for echo — setting any echo role replaces the whole role table). `hierarchical_grpo` is only valid with a proposer-solver env: it compares solvers with attempts on the same proposed problem and proposers with other proposals in the group. There is no preset layer, and no config hook that points at user code — a new algorithm is a named class in the repo (subclass `Algorithm`, register it). Each train source inherits the group algorithm: a source `algo` with a different `type` is its own algorithm (`[orchestrator.train.source.algo] type = "opd"`), and params without a `type` keep the group's algorithm with those params changed. prime-rl only hosts the trainable policy; frozen models are inline external endpoints on the algorithm, named where the model is used — `[orchestrator.train.algo.teacher]` for opd (the frozen model scored against), `[orchestrator.train.algo.sampling.source]` for sft (the model it samples from), each with `name` + `base_url`. There is no shared `teacher` slot. opsd declares no model — it self-distills against the live policy. See `docs/algorithms.md`.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## Sparse delta weight synchronization

Sparse delta is a filesystem weight-broadcast payload mode. For remote
inference, pair it with transactional streaming upload:

```toml
[weight_broadcast]
type = "filesystem"
mode = "delta"
update_protocol = "stage_commit"
stage_transport = "streaming_upload"
background_stage = true
stage_num_streams = 4
stage_chunk_size_mb = 16
stage_chunk_retries = 3
stage_retries = 1
retain_all_deltas = true
delta_stream_group_size = 4
```

`streaming_upload` writes `delta.stream` and uploads complete records while
later layers are scanned. `chunked_upload` waits for extraction and sends
offset-based chunks; `http_upload` uses multipart; `shared_fs` sends only a
path. Parallel streams and chunk size apply to chunked and streaming uploads;
request retries are exhausted before a complete-stage retry. Direct
trainer/orchestrator entrypoints must set the corresponding
filesystem fields separately because only the shared RL config propagates
them.

Sparse artifacts use logical Hugging Face names and global shapes. The vLLM
worker maps them to fused and TP-local weights at apply time. Changed biases
are included, and values widen when required to reproduce the target bits
exactly; this does not change optimizer or reduction dtypes.

The supported path is unquantized dense Qwen3/Llama-style models with PP=1,
no expert parallelism, and no LoRA. Resume requires a separately synchronized
full base and is rejected by the shared RL config. Transactional endpoints
validate `base_version`, keep failed partial applications paused and dirty,
and require a base reload before replay. Auto-launched vLLM routers use the
weight-aware `/weight_health` endpoint so dirty workers leave rollout routing.
Lease recovery additionally requires
`retain_all_deltas = true` and an HTTP stage transport. See
`docs/sparse_delta.md` for relay and multi-endpoint examples.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs
