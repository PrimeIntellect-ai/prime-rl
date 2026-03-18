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

## SLURM scheduling

The inference entrypoint supports optional SLURM scheduling, following the same patterns as SFT and RL.

### Single-node SLURM

```toml
# inference_slurm.toml
output_dir = "/shared/outputs/my-inference"

[model]
name = "Qwen/Qwen3-8B"

[parallel]
tp = 8

[slurm]
job_name = "my-inference"
partition = "cluster"
```

```bash
uv run inference @ inference_slurm.toml
```

### Multi-node SLURM (independent vLLM replicas)

Each node runs an independent vLLM instance. No cross-node parallelism — TP and DP must fit within a single node's GPUs.

```toml
# inference_multinode.toml
output_dir = "/shared/outputs/my-inference"

[model]
name = "PrimeIntellect/INTELLECT-3-RL-600"

[parallel]
tp = 8
dp = 1

[deployment]
type = "multi_node"
num_nodes = 4
gpus_per_node = 8

[slurm]
job_name = "my-inference"
partition = "cluster"
```

### Dry run

Add `dry_run = true` to generate the sbatch script without submitting:

```bash
uv run inference @ config.toml --dry-run true
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

- `src/prime_rl/entrypoints/inference.py` — entrypoint with local/SLURM routing
- `src/prime_rl/inference/server.py` — vLLM env setup
- `src/prime_rl/configs/inference.py` — `InferenceConfig` and all sub-configs
- `src/prime_rl/inference/vllm/server.py` — FastAPI routes and vLLM monkey-patches
- `src/prime_rl/templates/inference.sbatch.j2` — SLURM template (handles both single and multi-node)
- `configs/debug/infer.toml` — minimal debug config
