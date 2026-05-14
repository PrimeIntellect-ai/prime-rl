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

Prime-RL defaults W&B data/cache/config/artifact storage under the run
`output_dir` so long Isambard runs do not fill the home quota. If you override
`WANDB_DATA_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR`, or
`WANDB_ARTIFACT_DIR`, point them at project scratch/Lustre, not `$HOME`.

For canaries launched from inside an already-created Isambard allocation, do
not add `[slurm]` sections just to encode partition/account/time. The allocation
wrapper owns job submission and, from the next mnode allocation onward, exports
`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NPROC_PER_NODE`, `GPUSTAT_DIR`, and the
CUDA 13.1 forward-compat environment from `.env`. Keep the TOML focused on
experiment topology and validate the wrapper environment before launch.

For sub-node RL packing inside an allocation, use
`[deployment].type = "gpu_layout"` instead of pretending the run is homogeneous
`single_node` or `multi_node`. The layout is ordered by selected host: each
`[[deployment.nodes]]` lists one-GPU inference server GPUs in `inference` and
trainer GPUs in `trainer`. If `[deployment].hosts` is unset, the launcher uses
the first `len(nodes)` hosts from `SLURM_JOB_NODELIST`; override hosts on the
CLI as a JSON-style list or in a temporary overlay when running multiple
layouts in parallel. Launch long gpu-layout canaries from an attached tmux pane
so the allocation keeps a live control surface for logs and interruption.
For Omni-MATH-2 canaries, source `.env` and put the HF-task verifiers checkout
before the older shared verifiers checkout on `PYTHONPATH`, then add the
repo-local task environment. The order matters because
`verifiers.utils.hf_tasks` exists in the HF-task checkout but not in the older
shared checkout:
`/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs`,
`/lus/lfs1aip2/projects/a6r/joanv.a6r/work/verifiers`, then
`/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/environments/omni_math2_singleturn`.

For long-rollout Omni-MATH RLVR canaries using token-budget batching, size
`max_off_policy_steps` to the in-flight queue geometry. If
`max_inflight_rollouts` holds several token-batches, setting
`max_off_policy_steps` too low cancels already-generated rollouts and wastes
inference. Keep the cap explicit because higher values improve throughput by
accepting more stale samples.

When deliberately moving toward a more PipelineRL-style high-throughput regime,
do not raise staleness caps under a loss that ignores behavior-policy
probabilities. Prefer `trainer.loss.type = "default"` or the custom
`is_reinforce_loss_fn`, set `importance_ratio_clip` (PipelineRL used 5 in its
experiments), and monitor `importance_ratio_raw`, `importance_ratio`, and
`importance_ratio_clipped` alongside reward/MFU.

Do not confuse `max_async_level` with `max_off_policy_steps`. NCCL weight
broadcast currently requires `max_async_level = 1`, but in-flight rollout groups
can still age across multiple policy updates before being accepted or dropped.
Use W&B/log metrics `scheduler/async_level`, `off_policy_level/*`, and
`scheduler/cancelled_rollouts` to verify the actual staleness and cancellation
rate.

Before increasing `max_off_policy_steps` further, prefer larger batches only
when memory and policy-lag evidence justify it. Token-budget batching can create
more packed micro-batches per optimizer step without raising the per-sequence
activation footprint, but oversized in-flight queues can make the orchestrator
run far ahead of the trainer and turn throughput into stale-policy backlog. For
OLMo3 Omni-MATH canaries, the stable checked-in token-budget shape is
`token_batch_size = 524288`, `max_inflight_rollouts = 768`,
`max_off_policy_steps = 8`, and
`[orchestrator.eval].cancel_inflight_rollouts_on_eval = false`. The
`*_pipelinerl_speed.toml` `3072`-in-flight experiments were speed probes, not
the default recommendation. For fixed-batch long canaries, start with
`batch_size = 256` and `max_inflight_rollouts = 256`; only oversample above one
batch if W&B `time/wait_for_batch` or local rollout queues show real trainer
starvation.

Blocking online eval should not allow the background policy-update poller to
swap in a newly saved trainer checkpoint mid-eval. If eval logs
`Pausing inference engines for weight update` after `Running evals at
ckpt_step=...`, treat that eval as potentially weight-contaminated and make
sure the orchestrator pauses policy updates before launching MaxRL/comparison
runs.

For Omni-MATH eval quality, do not interpret `100×8` online eval as a
checkpoint-selection oracle. It is a heartbeat with high prompt-level variance.
Use it every 50 steps, add a cheap full-set `600×1` sentinel every ~250 steps
for broad p@1 coverage, and run heavier `600×4/8` evals offline/posthoc when
choosing checkpoints. Keep `cancel_inflight_rollouts_on_eval = false` unless
there is concrete inference congestion. `Timeout during comparison` comes from
symbolic `math_verify`; the LLM judge fallback runs after that timeout, so the
timeout still costs scorer latency and can still become a zero if parsing or
judge fallback fails. Eval-only `math_verify_timeout_seconds = 10` is a
reasonable first bump; raising train timeout can slow the rollout pipeline.

For NCCL weight broadcast, pausing orchestrator policy updates during eval is
not enough by itself. Non-master trainer ranks must also wait out-of-band before
entering DTensor/FSDP collectives for the next broadcast. If rank 1 times out in
`NCCLWeightBroadcastSender._resolve_dtensors` / `DTensor.full_tensor()` while
rank 0 is waiting for inference `NCCL_READY`, the bug is broadcast-rank
coordination, not vLLM eval throughput. Keep the trainer-side
`TRAINER_NCCL_READY` marker handshake in place for blocking online evals.

For 1-GPU OLMo3 vLLM servers on GH200, the current 8-node long-context serving
shape is `gpu_memory_utilization = 0.95`, `max_num_seqs = 192`, and
`max_num_batched_tokens = 65536`. The failed `0.95/256/262144` boot was caused
by the larger sequence/token caps, not by `gpu_memory_utilization = 0.95` alone.

For gpu-layout runs with many one-GPU vLLM servers, keep vLLM/Torch/Triton
compile caches server-local on `/tmp`. Concurrent EngineCore warmup can otherwise
race through shared `~/.cache/vllm/torch_compile_cache` and fail with
`torch._inductor.exc.InductorError: ... [Errno 116] Stale file handle` during
autotuning. The launcher should set per-server `VLLM_CACHE_ROOT`,
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `XDG_CACHE_HOME`, and should
preserve those compile cache roots across relaunches inside the same allocation.
It is fine to clear the vLLM RPC socket dir; do not wipe compile caches unless
you are intentionally invalidating them.

Keep `USE_HUB_KERNELS=NO` for OLMo3 unless a model-specific config explicitly
requires hub kernels. If the expected Transformers warning
`kernels hub usage is disabled through the environment USE_HUB_KERNELS=...`
gets noisy, suppress only that message via the PrimeRL logging filter rather
than enabling hub kernels or lowering all Transformers logging.

For current OLMo3 Omni-MATH-2 RLVR full fine-tuning, use solved-only online
filtering: `lr = 1e-6`, `[orchestrator.buffer].easy_threshold = 1.0`,
`online_difficulty_filtering = true`, and no `hard_threshold`. Older
`0.875/0.0625` and `hard_threshold = 0.0` notes are superseded for this run
family: `hard_threshold = 0.0` permanently evicts all-zero prompts that may
become solvable later.

For the 8-node GH200 allocation, the current best-known OLMo3 Omni-MATH-2
topology is 28 inference GPUs / 4 trainer GPUs: `deployment.type =
"multi_node"`, `num_train_nodes = 1`, `num_infer_replicas = 7`, and
`nodes_per_fsdp_group = 1`. Use filesystem weight broadcast with
`max_async_level = 4`, and `enable_prefix_caching = false`. Do not add
`vllm_extra.num_scheduler_steps` unless the installed vLLM exposes that flag;
the 2026-05-12 local environment did not. `batch_size = 256` is the default
throughput/quality point; `batch_size = 512` is viable but inference/KV
pressure makes it slower and higher variance. Add `[trainer.model.compile]`
for the next controlled run, because the 2026-05-12 bs512 probe showed
`compile=None`.

For OLMo3 Omni-MATH RLVR, use higher exploration during training than eval:
`[orchestrator.train.sampling].temperature = 1.0` and
`[orchestrator.eval.sampling].temperature = 0.6`. Keep eval temperature fixed
when comparing against earlier model-card or offline benchmark numbers.

`generation_config = "vllm"` uses vLLM's neutral sampling defaults. In vLLM
0.20.x this does not by itself prove that model EOS IDs are ignored, so treat
`generation_config = "auto"` as a smoke-test axis rather than a confirmed fix
until a short generation run shows reduced truncation.

OLMo3 YaRN configs from HF may encode `beta_fast`/`beta_slow` as JSON integers.
Transformers v5 expects floats there. Prime-RL normalizes those RoPE fields at
the HF config ingestion boundary; do not work around the warning by changing the
model's RoPE values in TOML.

Online eval has separate environment-worker concurrency from training rollouts.
For 100×8 long-context Omni-MATH evals, the `"auto"` worker rule resolves to
only 4 eval workers, which can bottleneck scoring and judge fallback. This is
separate from vLLM request concurrency: long generation volume can still
dominate wall-clock time even when inference GPUs are full. Set
`[orchestrator.eval].num_workers` explicitly for those configs before judging
eval wall-clock time; this does not change model sampling or reward semantics.

For many-server online evals, avoid launching all eval rollouts at once unless
you explicitly want eager round-robin pre-assignment. Set
`[orchestrator.eval].max_concurrent_rollouts_per_client` to enable bounded
dynamic refill across inference clients. This keeps eval size and
`rollouts_per_example` unchanged while reducing late bubbles where some vLLM
servers drain to zero requests and others are stuck on long completions.

For pipelined RLVR runs, choose
`[orchestrator.eval].cancel_inflight_rollouts_on_eval` from the failure mode you
are measuring. `true` gives cleaner checkpoint-boundary evals by clearing stale
train work first, but it can create a severe post-eval refill bubble. `false`
preserves overlap after eval and is useful for long throughput canaries, but it
can leave eval requests behind already-queued train rollouts and make eval
wall-clock harder to interpret. Record which mode was used before comparing
runtime or eval timing.

For small Omni-MATH eval subsets, set `[orchestrator.eval].seed` explicitly.
Without a seed, `get_eval_dataset(n=...)` preserves the dataset's default order,
so `num_examples = 100` can mean the first 100 records rather than a
representative shuffled subset. The seed is inherited by `[[orchestrator.eval.env]]`
entries unless an env overrides it.

CUDA/NCCL package versions must be locked, not only manually installed into the
live venv. If upgrading NCCL, update `uv.lock` so `uv run` does not sync back to
the older wheel. The exact `ctypes.CDLL("libnccl.so.2")` check also requires the
venv NCCL wheel lib directory on `LD_LIBRARY_PATH`.

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

### SFT hard distill override

For hosted multi-tenant runs where the trainer image's `trainer.loss.type` is fixed, the orchestrator exposes a per-run override that forces SFT loss on every micro-batch without rebuilding the trainer. Set `orchestrator.use_sft_loss = true` alongside `orchestrator.teacher_rollout_model`; both must be configured together (the orchestrator validator enforces this). The orchestrator stamps each `TrainingSample.sft_loss = True`, which the trainer's `compute_loss` honors by dispatching to `sft_loss_fn` per batch — independent of the trainer's configured default loss.

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
