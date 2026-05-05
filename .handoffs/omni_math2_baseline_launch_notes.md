# Omni-MATH-2 Baseline Launch Notes

Updated: 2026-05-03 00:41 UTC

## Objective

Keep future Omni-MATH-2 baseline launches on the working path discovered during the GH200/vLLM resmokes, without repeating stale Trinity/Qwen/OLMo mistakes.

## Current Working Rules

- Run scripts with `uv run --env-file .env ...`; do not use conda routes.
- Treat request-level sampling as the source of truth. Launched vLLM servers should use `generation_config = "vllm"` so model-baked generation defaults do not silently drift behavior.
- Qwen3.5 no-thinking runs use:
  - `chat_template_kwargs = {"enable_thinking": false}`
  - `include_reasoning = false`
  - `bad_words = ["<think>", "</think>"]`
  - `presence_penalty = 2.0`
  - `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`
- Qwen3.5 prefix caching is intentionally disabled. vLLM treats Qwen3.5 as a hybrid model path and does not support prefix caching there.
- Trinity Mini direct/no-think-tag runs use the Trinity prompt pack plus:
  - `temperature = 0.15`
  - `top_p = 0.75`
  - `top_k = 50`
  - `min_p = 0.06`
  - `bad_words = ["<think>", "</think>"]`
  - `max_completion_tokens = 16384`
  This is not a fully enforced model-native non-thinking mode; report it as direct/no-think-tag prompting unless a verified model control is added.
- Trinity Large FP8-Block is the viable current large Trinity path:
  - model `arcee-ai/Trinity-Large-Preview-FP8-Block`
  - multinode TP/PP across 2 nodes when available
  - `pipeline_parallel_size = 2`
  - `moe_backend = "triton"`
  - no DeepGEMM
  - Isambard `srun` should use `--network=disable_rdzv_get` and HSN addresses.
- Trinity Large BF16 is blocked by default. The failed BF16 config was DP=8/EP/eager replicas, not a real sharded BF16 launch, and OOMed during load on GH200.
- Trinity Large W4A16 is blocked by default. It fails in `gptq_marlin_moe_repack` with `cudaErrorUnsupportedPtxVersion` on the current GH200/CUDA/vLLM stack.
- Marin is a 4096-context eval unless we deliberately choose a long-context Marin variant or knowingly bypass vLLM's max-length guard. Do not interpret the cap as an accidental truncation bug.
- RNJ's `sliding_attention` / `full_attention` warning is likely Transformers/vLLM config-schema drift, not evidence of wrong attention layout. Judge RNJ by truncation/output behavior instead.
- OLMo3 Think results without an OLMo3 reasoning parser are raw visible-output Think-model results, not clean non-thinking results. The local vLLM install did not expose an `olmo3` reasoning parser.

## Current Code Anchors

- Shared benchmark helpers live in `src/prime_rl/baselines/benchmark.py`.
- Generic baseline runner/provisioner live in `src/prime_rl/baselines/`.
- Omni-specific model matrix lives in `scripts/evals/run_omni_math2_model_bench.py`.
- GPQA comparison runner lives in `scripts/evals/run_gpqa_model_bench.py`.
- Durable runbook notes live in `configs/baselines/README.md` and `skills/config/SKILL.md`.

## Validation Commands

```bash
uv run --env-file .env --no-sync ruff check src/prime_rl/baselines scripts/evals/run_omni_math2_model_bench.py scripts/evals/run_gpqa_model_bench.py tests/unit/baselines
uv run --env-file .env --no-sync pytest tests/unit/baselines
uv run --env-file .env --no-sync python scripts/evals/run_omni_math2_model_bench.py --dry-run --num-examples 1 --rollouts-per-example 1 --model-set trinity --output-root /tmp/omni-math2-dryrun
```

## Known Pitfalls

- Do not retry blocked Trinity BF16/W4A16 by default; use `--include-blocked` only for an explicit toolchain or launch-design experiment.
- Do not pool Trinity Mini default-prompt artifacts with direct/no-think-tag artifacts.
- Do not rely on old smoke dirs as current truth; stale DeepGEMM failures came from old configs that forced DeepGEMM.
- Do not use `--allow-errored-complete`; that path was removed. Complete artifacts must include `summary.json`, `records.jsonl`, expected rollout count, and `error_rate == 0`.
