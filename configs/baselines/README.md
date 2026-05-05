# Baseline Harness

Run verifier-backed model baselines with PRIME inference:

```bash
uv run --no-sync --env-file .env baseline-eval --config configs/baselines/gpqa_mcq_gemma4_local.toml
```

If an older shell still exports `UV_CACHE_DIR=$PROJECTDIR/joanv.a6r/.uv-cache`,
source `/projects/a6r/joanv.a6r/scripts/primerl_env.sh` once; that script now
points uv and HF caches at writable `$PROJECTDIR/joanv.a6r/tmp/...` locations.

Each run writes:

- `raw_rollouts.jsonl`: verifier outputs with full trajectories and raw judge responses.
- `records.jsonl`: one flat row per rollout for analysis.
- `summary.json`: aggregate single-shot accuracy, mean sample accuracy, pass@k, prefix successes, token, latency, truncation, and error metrics.
- `data/<protocol>.json` and `data/tokens.json`: nanodebate-style extracted records for plotting.

The local configs launch PRIME inference with 4 data-parallel Gemma workers:

```bash
uv run --no-sync --env-file .env baseline-eval \
  --config configs/baselines/gpqa_openended_gemma4_local.toml \
  --output-dir tmp/baseline_gpqa_openended_gemma4_smoke \
  --num-examples 6 \
  --rollouts-per-example 3 \
  --max-concurrency 8 \
  --ks 1,3 \
  --port 8012
```

For SLURM-allocated nodes, either run the same local command inside an allocation
or set `[launch].mode = "srun"` and use the config's `launch_prefix` to match the
allocation. For multi-config sweeps, start PRIME inference once and run configs in
`external` mode against the shared `base_url` to avoid reloading the model.
Use `--wait-timeout-s` or `[launch].wait_timeout_s` for both local/srun and
external endpoint readiness waits. Do not wrap GPU-bearing `uv run` commands in
`timeout`; use the harness timeout instead so CUDA visibility stays tied to the
approved command.
For vLLM `dp > 1`, the harness retries once by default on the exact transient
`DP Coordinator process failed to report ZMQ addresses during startup` failure.
Set `[launch].server_start_retries` to tune this. On Slurm nodes, prefer a
node-local RPC socket directory, e.g. `VLLM_RPC_BASE_PATH=/tmp/vllm-rpc-$USER`,
rather than a shared project tmp path.

Huge HF Parquet datasets should use `dataset_streaming = true` with
`dataset_columns = [...]` in `[env_args]`. Add
`dataset_streaming_shuffle_buffer_size` when taking adjacent first rows would
cluster examples. `--num-examples` is applied before normalization for streaming
envs, so smoke runs and bounded baselines do not materialize full splits. The
Zebra v2 configs use explicit shard paths because the installed `datasets`
version cannot parse the unused `solution_rows` Json feature metadata from the
dataset card/full Parquet schema.

Notes:

- GPQA open-ended uses final-answer extraction before LLM grading. Missing final
  answer extraction fails closed instead of grading the full reasoning transcript.
- Omni-MATH-2 HybridMath uses the same boxed-answer parser for `math-verify` and
  LLM fallback. Empty parsed answers fail closed before judge fallback. The env
  also handles narrow task-local aliases for inline `(A)`/`(B)` choice labels,
  empty-set/no-solution text, and `finite` answers to finite-set claims. Bump
  the judge cache variant/path when changing these parser or alias policies.
- Qwen no-thinking evals need request-side `chat_template_kwargs =
  {enable_thinking = false}` and a decoder-side `bad_words` ban for Qwen's
  native `<think>` / `</think>` strings. Qwen3.5 runs with prefix caching
  disabled because vLLM's hybrid path does not support it. The Omni matrix
  runner keeps those controls out of the task/user prompt.
- The Omni matrix runner sets `generation_config = "vllm"` on launched servers;
  request sampling is the source of truth. Known-fatal Trinity Large variants
  are blocked by default and require `--include-blocked` to force: BF16 needs a
  real sharded launch instead of DP=8/EP replicas, and W4A16 hits
  `cudaErrorUnsupportedPtxVersion` on the current GH200/CUDA/vLLM stack.
- Verifiers' current OpenAI chat client expects one returned choice per request,
  so `rollouts_per_example` is implemented as parallel single-choice requests;
  vLLM still batches them through server-side continuous batching.
- Raw base-model probes can use `client_type = "openai_completions"` or
  `--client-type openai_completions` when the checkpoint has no usable chat
  template.
- `single_shot_accuracy` reports first-draw accuracy per question.
- `mean_sample_accuracy` reports per-rollout accuracy across all sampled draws.
- `pass@k` uses the standard order-invariant estimator from all sampled draws for a question.
- `prefix_pass_at_k` reports whether any of the first `k` ordered draws passed; this is a run-order diagnostic, not the benchmark pass@k.
- `successes_at_k_*` reports successful draws among the first `k` ordered draws.
