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

When composing multiple TOML files, list fields are replaced wholesale by the later file. To change one
`orchestrator.filters` entry in an overlay, include the full desired filter list in that overlay.

For quick KL smoke runs with very small rollout batches, enforced `zero_advantage` filtering can remove
every rollout and stop the orchestrator. If the goal is only trainer/inference mismatch KL, keep the
filter present but set `enforce = false` in a temporary overlay and call out that the run is not a reward
learning validation.

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

### RL rollout client defaults

For text-only RL rollouts, the orchestrator defaults to renderer-backed TITO (`use_renderer = true`). VLM configs must explicitly fall back to MITO (`use_renderer = false`) so image preprocessing and chat templating stay server-side. External teacher rollouts must also set `use_renderer = false`.

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

### Long-thinking RL on one trainer GPU

For long reasoning runs, especially Qwen/Qwen3.5 with 8k completions, avoid the trainer's vanilla full-logit path. Set the chunked LM head on the trainer model:

```toml
[trainer.model]
fused_lm_head_token_chunk_size = 8192
```

Without this, the trainer may materialize `[batch, seq, vocab]` logits and OOM when scaling logits by per-token temperatures.

When serving long-thinking Qwen/Qwen3.5 on a single inference GPU, also cap vLLM pressure explicitly:

```toml
[orchestrator]
max_inflight_rollouts = 64

[inference]
gpu_memory_utilization = 0.8

[inference.model]
enforce_eager = true

[inference.vllm_extra]
language_model_only = true
max_num_seqs = 64
```

The exact values are experiment-dependent, but leaving these unconstrained can fail during vLLM startup or during the first eval/rollout fanout.

For non-thinking sanity configs, keep the active config close to the proven sanity shape unless the run actually hits a resource limit: do not set `orchestrator.max_inflight_rollouts`, avoid lowering completion length, and disable eval blocks entirely when the goal is to prove training-loop viability.

## Key files

- `packages/prime-rl-configs/src/prime_rl/utils/config.py` — re-exports `BaseConfig` and `cli` from pydantic_config
- `packages/prime-rl-configs/src/prime_rl/configs/` — all domain-specific config classes
- `configs/debug/` — minimal debug configs for testing
- `configs/private/` — private configs via git submodule (internal only)
- `examples/` — full example configs for various tasks
