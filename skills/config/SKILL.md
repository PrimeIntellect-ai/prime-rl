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

For RL runs with shared W&B mode, do not set `[wandb].offline = true`.
Shared W&B requires server connectivity and config validation will fail before
launch. Use online W&B, or explicitly disable shared mode if offline logging is
required.

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

### Baseline eval smoke overrides

For GPQA-style open-ended baselines, do not shrink `max_tokens` so far that
models never reach the required final-answer tag. A smoke run with every rollout
truncated only tests inference plumbing; it does not test parser, grader,
posterior judge decision, or cache artifacts. Prefer reducing
`num_examples`/`rollouts_per_example` first, and keep the production token budget
when the smoke needs to exercise grading.

When testing against a local modified `verifiers` checkout, install both the
checkout and any local env package editably into the PRIME venv, then run with
`uv run --no-sync --env-file .env ...`. Plain `uv run --env-file .env ...` may
re-sync the venv back to the pinned Git dependency before launch, which can hide
new modules such as `judge_evidence_cache` or local envs such as
`hf_singleturn`.

Do not wrap GPU-bearing `uv run` baseline or inference commands in `timeout`.
The approval/sandbox layer may then treat the command differently and hide CUDA
devices. Use the harness' `--wait-timeout-s` or `[launch].wait_timeout_s`
instead; it applies to both local/srun launches and external endpoint waits.
For vLLM `dp > 1`, a startup failure reading
`DP Coordinator process failed to report ZMQ addresses during startup` means
the coordinator child missed vLLM's hard-coded 30s ZMQ-address handshake. Keep
the DP topology intact; the baseline provisioner retries that exact failure via
`[launch].server_start_retries`. On Slurm nodes, set
`VLLM_RPC_BASE_PATH=/tmp/vllm-rpc-$USER` so vLLM IPC sockets live on node-local
storage instead of shared project tmp.

For GPQA open-ended baselines, keep grader model/sampling/cache identity/reward
mode/calibration defaults in the prompt pack when available. The TOML should
usually provide only runtime wiring such as `judge_base_url`,
`judge_api_key_var`, and `judge_persistent_cache_path`. Duplicating calibrated
grader hparams in TOML makes prompt-pack and training runs drift silently.

For local `baseline-eval` runs backed by PRIME/vLLM inference, preserve vLLM
sampling knobs such as `top_k`, `min_p`, and `repetition_penalty` by using the
vLLM-permissive API profile. Local/srun baseline launches do this by default;
external endpoints must opt in explicitly with `api_profile =
"vllm_permissive"` only when the endpoint is actually vLLM-compatible. These
knobs belong in OpenAI `extra_body`; the baseline harness routes known vLLM-only
sampling keys there before dispatch.

For huge HF Parquet-backed baselines, set `dataset_streaming = true` and
`dataset_columns = [...]` in `[env_args]`. Use
`dataset_streaming_shuffle_buffer_size` when a deterministic bounded shuffle is
better than taking the first adjacent rows. The baseline harness threads
`--num-examples` into `num_eval_examples` before environment construction, so
the HF env reads only the bounded sample before normalization. This also avoids
loading unused nested columns whose Parquet metadata may not be understood by
the installed `datasets` version.

For no-thinking probes, put request-side vLLM controls such as
`chat_template_kwargs`, `include_reasoning`, and `thinking_token_budget` in
sampling JSON/TOML; the baseline harness routes those keys into OpenAI
`extra_body`. Qwen may still emit literal generated `</think>` text on long
math traces, so add a decoder-side `bad_words` ban for `"<think>"` and
`"</think>"`; do not put Qwen controls in the task/user prompt, and do not
post-process the response before scoring. For raw base-model probes with no
usable chat template, set
`client_type = "openai_completions"` or pass
`--client-type openai_completions`.

For Omni-MATH-2 matrix runs, launched vLLM servers should use
`generation_config = "vllm"` and explicit request sampling. Treat Qwen3.5
prefix caching as intentionally disabled: vLLM's Qwen3.5 hybrid path does not
support it. Do not retry Trinity Large BF16 or W4A16 from the default matrix:
BF16 needs a real sharded launch instead of DP=8/EP replicas, and W4A16 fails
in `gptq_marlin_moe_repack` with `cudaErrorUnsupportedPtxVersion` on the
current GH200/CUDA/vLLM stack. The runner requires `--include-blocked` to force
known-fatal variants.

For Omni-MATH-2 HybridMath, keep the parser contract explicit: `math-verify`
and judge fallback should see the same parsed boxed answer, empty parses should
fail closed before judge fallback, and task-local aliases should stay narrow
(`(A)`/`(B)` labels for inline choice text, no-solution/empty-set variants, and
`finite` for finite-set claims). If parser or alias policy changes, bump the
judge persistent cache path and `judge_variant_id` together.

Some modern base checkpoints are published as multimodal wrappers even for
text-only use. For text GPQA baselines with vLLM, start mini probes with:

```toml
[launch]
vllm_extra = {
  language_model_only = true,
  skip_mm_profiling = true,
  mm_processor_cache_gb = 0.0,
}
```

For multi-node vLLM on Isambard-AI GH200, do not let vLLM infer addresses from
the first `hostname -I` entry: that is the management network (`nmn0`) on these
nodes. Use the HSN address (`hsn0`) for `VLLM_HOST_IP`, `master_addr`, and
`GLOO_SOCKET_IFNAME`, and load `brics/nccl` plus `brics/aws-ofi-nccl` so NCCL
uses `NCCL_NET="AWS Libfabric"` and `NCCL_SOCKET_IFNAME="hsn"`. For models too
large for one node, use vLLM's documented shape: tensor parallel size = GPUs per
node, pipeline parallel size = node count, `nnodes`, `node_rank`, and `headless`
on non-head nodes. On Isambard, also pass `--network=disable_rdzv_get` to the
`srun` job steps for Slingshot NCCL jobs; validate the allocation with
`brics/nccl-tests` before blaming vLLM.

For Gemma 4 base checkpoints without a tokenizer chat template, provide an
explicit base-model template via `chat_template`. Do not treat "tokenizer loads"
as sufficient readiness for the chat-completions harness.

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

### VLM/text-layer routing

For multimodal Hugging Face wrappers, trainer utilities need the text model, not
the outer wrapper. If model init fails with an error like
`<OuterModel> object has no attribute 'layers'`, check
`src/prime_rl/utils/vlm.py` and add the model family to `VLM_REGISTRY` with the
right `language_model_attr`. Example: Gemma 4 uses
`model.language_model.layers`, so its registry entry must point at
`language_model_attr = "model.language_model"` rather than falling back to the
text-only `model.model` path. Add a small unit test for `get_language_model()`
and `get_layer_prefix()` when adding a new family.

## Key files

- `src/prime_rl/utils/config.py` — re-exports `BaseConfig` and `cli` from pydantic_config
- `src/prime_rl/configs/` — all domain-specific config classes
- `configs/debug/` — minimal debug configs for testing
- `examples/` — full example configs for various tasks
