---
name: add-model
description: Recipe for bringing a new model into prime-rl and proving it trains correctly — bf16 checkpoint on the PrimeIntellect HF org, modeling code plus a one-node MFU check, a vLLM vs trainer dtype and kernel survey, a short math RL run for mismatch KL, a SWE ablation, and a PR report. Use when adding a model family or checkpoint, turning on FP8/MXFP8 for one, or chasing trainer↔inference mismatch KL on a new model.
---

# Add a model

Five steps, in order. Don't start a step until the previous one passes; if a gate fails, stop and report to the user. Every result goes into the PR description, using [`pr-template.md`](pr-template.md).

| Step | What | Hardware | Gate |
|---|---|---|---|
| 0 | bf16 checkpoint on `PrimeIntellect/` (private) | CPU | only if upstream is not bf16 |
| 1 | modeling code, trainer smoke test, MFU | 1 node, few layers, random weights | runs, parity tests pass, MFU close to a reference family |
| 2 | vLLM vs trainer survey: dtypes, quantization, KV cache, kernels | none | every row matches or is justified |
| 3 | RL on easy math, 8k context, 100 steps | ≤ 2 trainer nodes, 1–2 inference nodes | mismatch KL low and flat |
| 4 | SWE ablation on `scaleswe`, 65k context, 30 turns | minimal | KL flat, reward sane, eval runs clean |

Files in this directory (fill every `<...>` and every `# FILL` line):

| File | Step |
|---|---|
| `trainer-mfu.toml` | 1: `uv run sft`, fake data, random init, truncated layers, forced balanced routing |
| `rl-math.toml` | 3: `math-env`, sign-SGD, full CPU offload, 8k, 100 steps |
| `rl-swe.toml` | 4: `scaleswe` train, `swebench-verified` eval, 65k, `max_turns = 30` |
| `mxfp8.toml` | overlay for 1, 3 and 4 when inference runs MXFP8: `@ rl-*.toml @ mxfp8.toml` |
| `report.py` | per-block metric tables for the PR: `uv run python skills/add-model/report.py <run_dir> ...` |
| `pr-template.md` | PR description to fill |

Validate every config with `--dry-run` and read the resolved JSON in `<output_dir>/<run_name>/configs/latest/resolved/`, not the TOML:

```bash
uv run rl @ skills/add-model/rl-math.toml --model.name <org>/<model> --dry-run --output-dir <scratch> --run.name check
```

Confirm with the user before submitting multi-node jobs. Monitor runs with the `training` skill (`monitor-run`).

## 0. bf16 checkpoint

Skip if every weight upstream is already bf16. Otherwise (FP8, MXFP4, NVFP4, int4, …) the trainer needs a bf16 copy, and RL starts from it on both sides.

1. Check the license allows redistribution. If not, stop and ask the user.
2. Dequantize every layer to bf16. Blockwise FP8 (DeepSeek/GLM format): `uv run python tools/convert_fp8_to_bf16.py <src_dir> <out_dir>`. Other formats: extend that tool with the format's dequantization rather than writing a one-off script.
3. Check the output: no `*scale*` tensors, no `quantization_config` in `config.json`, every floating tensor bf16 (list any tensor upstream ships in fp32, e.g. a router bias, and ask before casting it), the same tensor names and shapes as upstream minus scales, and tokenizer, chat template, `generation_config.json` and any remote-code `.py` files copied.
4. Serve it with `uv run inference --vllm.model <out_dir>` and compare against the original checkpoint on a few prompts (logprobs of the same tokens, or a small `uv run eval`). They should agree up to quantization noise.
5. Upload as private:

   ```bash
   hf repos create PrimeIntellect/<Model>-BF16 --private
   hf upload PrimeIntellect/<Model>-BF16 <out_dir> .
   ```

   Add a model card that links the source checkpoint and says how it was converted.
6. Re-exports don't match the `auto` patterns for the renderer (`MODEL_RENDERER_MAP` in `deps/renderers`) or the vLLM parsers (`packages/prime-rl-configs/src/prime_rl/utils/parsers.py`). Add `PrimeIntellect/<Model>-BF16` there, or set `orchestrator.renderer.name`, `inference.vllm.tool_call_parser` and `reasoning_parser` explicitly in every config.
7. Before the PR is merged: `hf repos settings PrimeIntellect/<Model>-BF16 --no-private`. This is a checkbox in the PR template; don't mark the PR ready without it.

## 1. Modeling code, trainer test, MFU

Modeling code follows `docs/development.md` ("Adding a New Model"):

- Code under `src/prime_rl/trainer/models/<arch>/` (configuration, modeling, converting). Start from the closest family (`glm4_moe/`, `qwen3_moe/`, `laguna/`).
- Register in `src/prime_rl/trainer/models/__init__.py` (`AutoConfig.register` and `_CUSTOM_CAUSAL_LM_MODELS`).
- Row in the `docs/advanced.md` "Custom Modeling" table.
- Mini preset in `scripts/mini_moe.py`; build it and check the HF ↔ prime-rl roundtrip is lossless.
- Parity test `tests/unit/train/models/test_<arch>.py` mirroring `test_glm4_moe.py`: attention only, MLP only, full model, against the HF reference.
- Honor `seq_lens` packing boundaries in `forward()` (varlen `cu_seqlens`, per-document state resets for linear attention).
- MFU needs correct FLOPs: every sequence-mixing module implements `attention_flops(seq_len)` (`AttentionFlops` in `models/base.py`), and `num_active_params()` must count only routed top-k plus shared experts. Add the architecture to `tests/unit/train/test_perf.py` only if its attention type is new.

Then run `trainer-mfu.toml` on one node (`uv run sft`, 8 GPUs, ~20 steps):

- `model.debug.num_layers`: truncation keeps the first N layers, so cover the leading dense layers and at least two MoE layers; for hybrid models use a multiple of the layer-type period.
- `model.debug.random_init = true`, fake data, `model.debug.force_balanced_routing = true` (MoE).
- Run the same file on a reference family of similar shape at the same layer count. Also run with the MXFP8 block uncommented if step 2 will pick MXFP8.

Gate: no errors or NaNs, stable step time after warmup, MFU close to the reference. If it is much lower, profile before moving on. Report MFU, tokens/s per GPU and peak memory for the new model and the reference.

## 2. vLLM vs trainer survey

Before any RL run, write this table for each precision setup you will run (bf16, and the FP8/MXFP8 setup vLLM will serve). One row per module type the model has. Show it to the user.

| Module | vLLM: weight / operand / output dtype | vLLM kernel | Trainer: weight / operand / output dtype | Trainer kernel | Configured by (both sides) | Match? |
|---|---|---|---|---|---|---|

Rows: embeddings, attention projections, attention core (backend, softmax scale, sinks, sliding window), special attention parts (compressors, indexers, sparse scorers), rotary (style, cache dtype, scaling), norms (dtype of the reduction, fused residual add), dense MLP and activation, shared experts, routed experts, router/gate (weights, logits, score function, top-k, routing-weight dtype, expert accumulation), lm_head, KV cache. Every row that differs gets fixed or a one-line justification.

Where to read it:

- Trainer: `src/prime_rl/trainer/models/<arch>/`, `models/layers/`, `[trainer.model.quantization]` (`ignore_patterns` are regexes, `re.search` on trainer module names), `[trainer.model.moe.compute]`, `[trainer.model] attn`, `docs/advanced.md` "Low-precision training" and "Router precision".
- vLLM: the model file under `.venv/lib/python3.12/site-packages/vllm/model_executor/models/`, `[inference.vllm] quantization`, `kv_cache_dtype`, `[inference.vllm.quantization_config] ignore` (fnmatch globs on vLLM module names), and the modules the vLLM quant method or model code keeps unquantized on its own. Print vLLM's module tree for real names; they differ from the trainer's. The attention backend and MoE kernel vLLM picked are in `inference.log` at startup.

Rules:

- **Router**: the trainer gate uses BF16 weights and operands, FP32 logits, FP32 routing weights and FP32 expert accumulation (local and torch EP dispatch). vLLM's gate must do the same. `inference.enable_fp32_router_logits` (default true) only injects `hf_overrides.moe_router_dtype = "float32"`, which only some vLLM models read; others hard-code their gate. Read the gate construction and record what it does.
- **lm_head**: `inference.enable_fp32_lm_head` (default true) makes vLLM's head a bf16×bf16→fp32 GEMM. The trainer head (`models/layers/lm_head.py`) upcasts a bf16 GEMM output. Record the row.
- **Quantization**: if vLLM quantizes (online or from the checkpoint), the trainer quantizes the same modules the same way (`mxfp8.toml`, or `type = "fp8"` / `deepgemm_fp8` for blockwise FP8), and every module vLLM keeps bf16 is in the trainer's `ignore_patterns`, and the other way round.
- **KV cache**: an FP8 KV cache in vLLM has no trainer counterpart. Prefer bf16; if FP8 is required (e.g. by the attention backend), justify it and expect it in the mismatch KL.
- **Kernels**: they don't need to be the same kernel, but they must compute the same thing: same attention variant and masking, same rope convention, same norm epsilon and dtype, same activation (including clamping or gating quirks), same expert combine order and dtype. List each pair and any known numerical difference.
- Never change `optimization_dtype` / `reduce_dtype`.

Why it matters: when checkpoint weights lie on the quantization grid (shipped as FP8, re-exported to bf16), small updates round back to the base value when vLLM re-quantizes, so a bf16 trainer drifts away from inference step by step. DeepSeek-V4 math: mismatch KL 0.0012 → 0.0082 over 60 steps with a bf16 trainer, flat ~0.002 with an MXFP8 trainer.

Quick check before step 3, **vLLM self-consistency**: on a `uv run inference` server, generate at temperature 1 with sampled-token logprobs, then re-score the same token ids as a prompt with `prompt_logprobs` and `max_tokens = 1`. The mean decode-minus-prefill logprob estimates the KL between vLLM's own decode and prefill paths (DeepSeek-V4 MXFP8: ~0.0026). Step-1 mismatch KL in step 3 should be close to it; far above means a trainer-side difference.

## 3. RL on easy math

`rl-math.toml`: `math-env` (Hendrycks MATH) with deterministic scoring, `seq_len = 8192`, batch 64 × group 8, 100 steps, sign-SGD with `model.offload = "full"` so the trainer fits on the fewest nodes. Use 1 trainer node if it fits, else 2; 1–2 inference nodes. Never lr 0: drift only shows once weights move. If inference is quantized, run two arms: bf16 trainer and quantized trainer (`@ mxfp8.toml`).

Gate:

- Step-1 `mismatch_kl/all/mean` close to the vLLM self-consistency estimate.
- `mismatch_kl/all/mean` < 0.015 on every step (merge bar in `docs/development.md`), roughly ~0.001–0.003 for MXFP8 and lower for bf16, and **flat**: the last 10-step block is not clearly above the first.
- `entropy/all/mean` doesn't collapse or explode, `is_masked/mean` small and flat, reward (`train/agg/effective/<agent>/reward/mean`) stable or rising, `optim/grad_norm` without spikes.

Report: `uv run python skills/add-model/report.py <run_dir> [<run_dir_arm2>]` (10-step blocks) plus the W&B link.

## 4. SWE ablation

`rl-swe.toml`: `scaleswe` train with the `bash` harness, `swebench-verified` eval every 50 steps, `seq_len = 65536`, `env.agent.max_turns = 30`. If the model's renderer config has `reasoning_effort`, set the lowest level it supports in `[orchestrator.renderer]`. Same trainer recipe as step 3, CP to fit 65k; the arm (bf16 or quantized) that passed step 3.

Gate:

- `mismatch_kl/all/mean` flat over the whole run, `is_masked/mean` small and flat.
- Reward, `num_turns`, `is_truncated` sane; truncation not dominating.
- Eval completes with no errors or cancellations (`has_error`, `dispatcher/errored/eval`).

Report the same tables (`--block 20` for long runs) and the eval scores per eval step.

## Gotchas

- Router replay (`trainer.enable_router_replay`) needs the vLLM model to return routed experts, and rules out disaggregated prefill/decode; use independent replicas (`deployment.num_infer_replicas`). If vLLM's model can't return them, turn replay off and say so in the report.
- MXFP8 trainer: samples must be padded to a multiple of 32 tokens per CP rank or the backward crashes. Check that the resolved `orchestrator.pad_to_multiple_of` is `32 * cp`.
- vLLM MXFP8 needs each TP shard's input dim divisible by 32 (GLM-4.5-Air's dense layer-0 `down_proj` only works at TP ≤ 2). Lower TP and add DP with expert parallel.
- Eval uses chat completions, so vLLM's `tool_call_parser` and `reasoning_parser` must resolve. Verify by curling `/v1/chat/completions` with `tools` on a `uv run inference` server.
- Sign-SGD with `offload = "full"` disables gradient clipping (`optim.max_norm` resolves to None); expected.
- Big checkpoints: `orchestrator.model.client.wait_for_ready_timeout` must cover loading and online quantization. Use `[ckpt] keep_last = 1` when the trainer state is TBs.
