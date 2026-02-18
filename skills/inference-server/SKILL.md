---
name: inference-server
description: Start and test the prime-rl inference server. Use when asked to run inference, start vLLM, test a model, or launch the inference server.
---

# Inference Server

## Starting the server

Always use the `inference` entry point — never `vllm serve` or `python -m vllm.entrypoints.openai.api_server` directly. The entry point runs `setup_vllm_env()` which configures environment variables (LoRA, multiprocessing) before vLLM is imported.

```bash
# With a TOML config
uv run inference @ path/to/config.toml

# With CLI overrides
uv run inference --model.name Qwen/Qwen3-0.6B --model.max_model_len 2048 --model.enforce_eager

# Combined
uv run inference @ path/to/config.toml --server.port 8001 --gpu-memory-utilization 0.5
```

## FP8 knobs

Use `InferenceConfig` fields for the core FP8 setup:

```toml
[model]
quantization = "fp8"
kv_cache_dtype = "fp8_e4m3"

calculate_kv_scales = true
```

Common constraints:
- FP8 quantization is incompatible with `model.dtype = "float32"`.
- `calculate_kv_scales` requires an FP8 KV cache dtype.
- During `/update_weights`, workers use a non-delegated layerwise reload path: `model.load_weights(...)` wrapped by vLLM `initialize_layerwise_reload(...)` / `finalize_layerwise_reload(...)`.
- `/update_weights` always runs the manual FP8-conversion pass before load; this converts eligible 2D linear weights to FP8 + `*_scale_inv` (with packed-module safeguards) and is a no-op for non-FP8 models.
- The manual FP8 conversion path is Triton-only for FP8 refit and requires Hopper GPUs (SM90+); no non-Triton fallback path is available.
- This wrapper is required for FP8 refit; calling bare `model.load_weights(...)` can fail after FP8 post-processing changes parameter layouts/metadata.

For advanced vLLM tuning not surfaced in `InferenceConfig`, pass flags directly on CLI (unknown args are forwarded to vLLM). Example:

```bash
uv run inference @ path/to/config.toml \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --attention-backend FLASHINFER
```

## Custom endpoints

The server extends vLLM with:

- `/v1/chat/completions/tokens` — accepts token IDs as prompt input (used by multi-turn RL rollouts)
- `/update_weights` — hot-reload model weights from the trainer
- `/load_lora_adapter` — load LoRA adapters at runtime
- `/init_broadcaster` — initialize weight broadcast for distributed training

## Testing the server

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 50
  }'
```

## Key files

- `src/prime_rl/inference/server.py` — entry point, env var setup
- `src/prime_rl/inference/config.py` — `InferenceConfig` and all sub-configs
- `src/prime_rl/inference/vllm/server.py` — FastAPI routes and vLLM monkey-patches
- `configs/debug/infer.toml` — minimal debug config
