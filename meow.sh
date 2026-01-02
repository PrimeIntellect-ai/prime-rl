export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

uv run inference @ examples/reverse_text/rl/infer.toml --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 2048 --enable-lora --max-lora-rank 8 --max-loras 2 --parallel.tp 2 --gpu-memory-utilization 0.7
