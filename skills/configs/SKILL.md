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

CLI: `--orchestrator.train.source.0.env.taskset.id reverse-text` or `--orchestrator.eval.source.0.env.taskset.id reverse-text`.

The `sft` entrypoint takes the same eval shape at the top level for online evals: `[eval]` + `[[eval.source]]` (with `[inference]` for the server), e.g. `--eval.source.0.env.taskset.id reverse-text`.

**Dicts** — TOML uses a section; CLI takes a JSON string: `--trainer.env-vars '{"key1": "value1"}'`. This works for plain `dict` fields only — nested pydantic-model fields (e.g. `algo`) reject JSON strings; use dotted keys (`--orchestrator.algo.type max_rl`) or a TOML overlay file.

**vLLM pass-through** — `[inference.vllm]` uses vLLM's own argument names (`model`, `tensor_parallel_size`, `data_parallel_size`, `max_model_len`, ...) and forwards *any* key to the vLLM server, typed by prime-rl or not: `[inference.vllm] max_num_seqs = 256`, or `--inference.vllm.max-num-seqs 256` on the CLI. CLI values are JSON-coerced, so dict-valued vLLM args work as `--inference.vllm.compilation-config '{"cudagraph_mode": "NONE"}'`. Non-vLLM knobs (router, deployment, weight broadcast, kv-cache offload, env vars) stay on `[inference]` itself.

**Discriminated unions** — set the `type` field to pick the variant (`[orchestrator.algo] type = "max_rl"`). Omit `type` to keep the default variant.

**RL loss** — `[trainer.loss]` defaults to IPO with `eps = 0.1`, `adv_tau = 1.0`, and `kl_tau = 1e-3`. Use `type = "dppo"` for advantage-directed probability-difference masking. Use `type = "icepop"` for ratio-band masking without a KL term. Set `type = "custom"` with `import_path` and optional `kwargs` to load a custom RL loss. The `ce` and `ref_kl` components are fixed.

**Policy cache salts** — `[orchestrator] enable_cache_salt = true` prevents live-policy prefix-cache reuse across weight versions. Set it to `false` only for a controlled cache-reuse ablation.

**Algorithms** — `[orchestrator.algo] type = "grpo" | "max_rl" | "rae" | "hierarchical_grpo" | "opd" | "opsd" | "sft" | "echo"` — the type names the algorithm (credit assignment + loss routing, fused), and each type's class defaults are its vetted setting; any other key you set is your own assembly (e.g. `[orchestrator.algo.roles.user] alpha = 0.1` for echo — setting any echo role replaces the whole role table). `hierarchical_grpo` is only valid with a proposer-solver env: it compares solvers with attempts on the same proposed problem and proposers with other proposals in the group. There is no preset layer, and no config hook that points at user code — a new algorithm is a named class in the repo (subclass `Algorithm`, register it). Per-source override: `[orchestrator.train.source.algo] type = "opd"` (the source assembles its own algorithm). prime-rl only hosts the trainable policy; frozen models are inline external endpoints on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for opd (the frozen model scored against), `[orchestrator.algo.sampling.source]` for sft (the model it samples from), each with `name` + `base_url`. There is no shared `teacher` slot. opsd declares no model — it self-distills against the live policy. See `docs/algorithms.md`.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## GLM Air online blockwise FP8

With unpatched vLLM 0.28.0, GLM-4.5-Air TP8 + EP cannot apply `fp8_per_block` to every
linear layer: the dense MLP down projection consumes 1,368 values per TP rank,
which violates the activation quantizer's 128-element group requirement.
Shared-expert down projections also have a ragged 176-wide TP8 input.
An inference-only smoke-tested workaround is `quantization = "fp8_per_block"`
with `quantization_config.ignore` listing `model.layers.0.mlp.down_proj` and
`model.layers.{1..45}.mlp.shared_experts.down_proj` (expand the numeric range into
individual names). These layers remain BF16; other quantizable linear layers and
routed experts use blockwise FP8. Alternatively, `quantization = "online"` with
`quantization_config.moe = "fp8_per_block"` leaves all linear layers unquantized.
Treat either as mixed precision. Validate actual generated outputs and finite
log-probabilities; engine startup alone does not validate numerical correctness.
The isolated padding implementation in `prime_rl.inference.fp8_padding`, enabled
by `PRIME_RL_FP8_PAD_RAGGED=1`, instead zero-pads the weight/activation input tail
and keeps these projections in FP8. Kernel selection sees the padded shape;
checkpoint loading and logical dimensions retain the original shape. Do not
remove the divisibility assertion or silently change the quantization recipe.
The GLM Air kernel-transfer converter uses the GLM-5 quantization helper, fuses
MHA Q/K/V, and preserves the padded FP8 shape. Its tested kernel-transfer path is
TP1 + DeepGEMM on H200; do not assume TP-sharded kernel-format transfers or other
backend layouts are interchangeable. For reload diagnostics, disable prefix
caching or invalidate it before checking generation with updated weights.
The TP1 H200 check includes CUDA graphs and two FP8 NCCL reloads with prefix
caching disabled; it uses checkpoint weights converted by the trainer's helper,
not a real optimizer update.
Small inference checks do not establish long-context task quality or a full RL
optimizer-step integration.
Actual trainer FP8 compute is separate from FP8 weight transfer: enable
`[trainer.model.quantization] type = "fp8"` for linear layers and
`[trainer.model.moe.compute] type = "deepgemm_fp8"` for routed experts.
`quantize_in_weight_transfer = true` alone leaves trainer compute unchanged.
GLM Air requires preserving attention projection biases and zero-padding ragged
linear dimensions in the trainer as well as inference. The isolated trainer
implementation does both; its GPU forward/backward numerical checks remain
pending until spare capacity is available.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs
