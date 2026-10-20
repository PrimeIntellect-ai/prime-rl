---
name: add-model
description: Validate a new model in prime-rl against vLLM — a module-by-module dtype table for inference vs training, a short math RL run, then a long SWE RL run, with partial configs to fill in. Use when adding a model family or checkpoint, turning on FP8/MXFP8 for one, or chasing trainer↔inference mismatch KL on a new model.
---

# Add a model

Modeling code, the mini preset and the smoke test are in `docs/development.md` ("Adding a New Model"): code under `src/prime_rl/trainer/models/<arch>/`, registration in `src/prime_rl/trainer/models/__init__.py`, a row in the `docs/advanced.md` "Custom Modeling" table. This skill is what comes after: show that the trainer computes the same policy vLLM samples from. Do the three steps in order and don't start a step until the previous one passes. Validate every config with `--dry-run` first; the user launches runs.

Partial configs in this directory (fill every `<...>` and every `# FILL` line):

| File | What |
|---|---|
| `rl-math.toml` | step 2: 60 steps of `math-env`, lr 1e-6, deterministic scoring |
| `rl-swe.toml` | step 3: `scaleswe` train, `swebench-verified` eval, 131k context, independent replicas |
| `mxfp8.toml` | optional overlay: online MXFP8 vLLM + an MXFP8 trainer, layered as `@ rl-*.toml @ mxfp8.toml` |

```bash
uv run rl @ skills/add-model/rl-math.toml @ skills/add-model/mxfp8.toml --model.name <org>/<model> \
  --dry-run --output-dir <scratch> --run.name check
```

Read the result in `<scratch>/check/configs/latest/resolved/` (`inference.json`, `trainer.json`, `orchestrator.json`), not the TOML.

## 1. Dtype table: inference vs training

Before any run, write this table for the proposed setup and show it to the user. One row per module type the model has:

| Module | vLLM weight / operand / output | Trainer weight / operand / output | Configured by (both sides) | Match? |
|---|---|---|---|---|

Rows: attention projections, special attention parts (compressors, indexers, sparse-attention scorers), dense MLP, shared experts, routed experts, router/gate, lm_head, embeddings, norms, rotary cache, KV cache. Every row that differs gets fixed or a one-line justification.

Where to read it:

- Trainer: `src/prime_rl/trainer/models/<arch>/`, `[trainer.model.quantization]` (`ignore_patterns` are regexes, `re.search` on trainer module names), `[trainer.model.moe.compute]`, `docs/advanced.md` "Low-precision training" and "Router precision".
- vLLM: the model file under `.venv/lib/python3.12/site-packages/vllm/model_executor/models/`, `[inference.vllm] quantization`, `[inference.vllm.quantization_config] ignore` (fnmatch globs on vLLM module names), plus modules the vLLM quant method or model code keeps unquantized on its own. Print vLLM's module tree for real names — they differ from the trainer's (DeepSeek-V4: trainer `attn.compressor.indexer.q_b_proj`, vLLM `attn.indexer.wq_b` beside the compressor).

Rules:

- **Router**: the trainer gate uses BF16 weights and operands, FP32 logits, FP32 routing weights and FP32 expert accumulation (local and torch EP dispatch). vLLM's gate must do the same. `inference.enable_fp32_router_logits` (default true) only injects `hf_overrides.moe_router_dtype = "float32"`; just the models that read it honor it (vLLM `deepseek_v2.py` family, transformers MoE backend). Others hard-code it — vLLM `glm4_moe.py` builds `GateLinear(out_dtype=float32, params_dtype=float32)`, i.e. FP32 gate weights. Read the gate construction and record what it does.
- **lm_head**: `inference.enable_fp32_lm_head` (default true) makes vLLM's head a bf16×bf16→fp32 GEMM. The trainer head (`src/prime_rl/trainer/models/layers/lm_head.py`, fused and vanilla) upcasts a bf16 GEMM output. Record the row.
- **Quantization**: if vLLM quantizes online (`inference.vllm.quantization`), the trainer quantizes the same modules the same way (`mxfp8.toml`), and every module vLLM keeps bf16 — via `quantization_config.ignore` or hard-coded — is in the trainer's `ignore_patterns` too, and vice versa.
- **KV cache**: an FP8 KV cache in vLLM has no trainer counterpart; justify it (e.g. required by the model's attention backend).
- Never change `optimization_dtype` / `reduce_dtype`.

Why it matters: when checkpoint weights lie on the quantization grid (e.g. shipped as FP8, re-exported to bf16), lr-1e-6 updates mostly round back to the base value when vLLM re-quantizes. A bf16 trainer then drifts away from inference step by step. DeepSeek-V4 math: mismatch KL 0.0012 → 0.0082 over 60 steps with a bf16 trainer, flat ~0.002 with an MXFP8 trainer.

Quick checks before the long runs:

- **vLLM self-consistency**: on a `uv run inference` server, generate at temperature 1 with sampled-token logprobs, then re-score the same token ids as a prompt with `prompt_logprobs` and `max_tokens = 1`. The mean decode-minus-prefill logprob estimates the KL between vLLM's own decode and prefill paths (DeepSeek-V4 MXFP8: ~0.0026). Expect step-1 mismatch KL near it; far above means a trainer-side difference.
- **Step-1 mismatch KL** of the math run: before any update, this is pure numerics.

## 2. RL on math

`rl-math.toml`, 30–60 steps, lr 1e-6. Never lr 0: it has to be normal RL, since drift only shows once weights move. When quantized, also run the bf16 baseline (same config without `mxfp8.toml`).

Pass:

- `mismatch_kl/all/mean` low — ~0.001–0.003 for MXFP8, lower for bf16 — and **flat**: no growth over steps.
- Reward (`train/agg/effective/<agent>/reward/mean`) and `entropy/all/mean` sane, no collapse.
- `docs/development.md` merge bar: mismatch KL < 0.015 on every step of 20 `math` steps at `batch_size = 64`.

Report mean `mismatch_kl/all/mean`, reward and entropy per 10-step block, per arm, from `<run_dir>/monitors/file/metrics.jsonl`.

## 3. RL on SWE

`rl-swe.toml`, hundreds of steps.

Pass:

- `mismatch_kl/all/mean` flat over hundreds of steps.
- Eval (`eval/swebench-verified/...`) completes with no errors or cancellations (`has_error`, `dispatcher/errored/eval`).
- `is_masked/mean` small and flat — the default IPO loss masks tokens with |p_train − p_infer| > 0.3, so growth means drift.

See the `training` skill (`monitor-run`) for logs and check-ins.

## Gotchas

- Router replay (`trainer.enable_router_replay`, which turns on `inference.vllm.enable_return_routed_experts`) rules out disaggregated prefill/decode. Use independent replicas (`deployment.num_infer_replicas`).
- MXFP8 trainer: every sample must be padded to a multiple of 32 tokens per CP rank, or the backward crashes. Check that the resolved `orchestrator.pad_to_multiple_of` is `32 * cp`; if it is just `cp`, stop and tell the user.
- vLLM MXFP8 needs each TP shard's input dim divisible by 32. GLM-4.5-Air's dense layer-0 `down_proj` (10944) only works at TP ≤ 2, so it runs TP 2 × DP 4 with expert parallel.
- Eval uses chat completions, so vLLM's `tool_call_parser` and `reasoning_parser` must resolve. `auto` matches the model name (`packages/prime-rl-configs/src/prime_rl/utils/parsers.py`); re-exports under another org need them set. Same for `orchestrator.renderer.name`. Verify by curling `/v1/chat/completions` with `tools`.
- Cap `orchestrator.concurrency.max_inflight`. Uncapped, in-flight episodes grow into the thousands, per-token latency inflates and eval rollouts hit the 1 h timeout.
- Big checkpoints: `orchestrator.model.client.wait_for_ready_timeout` must cover loading and online quantization. Cache FlashInfer autotuning with `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` in top-level `[env_vars]`. Use `[ckpt] keep_last = 1` when the trainer state is TBs.
