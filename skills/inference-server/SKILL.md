---
name: inference-server
description: Start and test the prime-rl inference server. Use when asked to run inference, start vLLM, test a model, or launch the inference server.
---

# Inference Server

## Starting the server

Always use the `inference` entry point — never `vllm serve` or `python -m vllm.entrypoints.openai.api_server` directly. The entry point runs `setup_vllm_env()` which configures environment variables (LoRA, logits processors, multiprocessing) before vLLM is imported.

```bash
# With a TOML config
uv run inference @ path/to/config.toml

# With CLI overrides
uv run inference --model.name Qwen/Qwen3-0.6B --model.max_model_len 2048 --model.enforce_eager

# Combined
uv run inference @ path/to/config.toml --server.port 8001 --gpu-memory-utilization 0.5
```

## TOML config structure

Top-level fields must come before any `[section]` header (TOML rule).

```toml
gpu_memory_utilization = 0.5
seed = 42

[model]
name = "Qwen/Qwen3-0.6B"
max_model_len = 4096
enforce_eager = true
dtype = "bfloat16"

[server]
port = 8000

[parallel]
tp = 2

[logits_processors.gibberish_detection]
# presence of section enables it; all fields have defaults

[logits_processors.repetition_detection]
window = 3000
prob_threshold = 0.99
```

See `src/prime_rl/inference/config.py` for all available fields and defaults.

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
