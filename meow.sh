#export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

#uv run inference @ examples/reverse_text/rl/infer.toml --max-model-len 2048 --enable-lora --max-lora-rank 4 --max-loras 40 --gpu-memory-utilization 0.7 --weight-broadcast.type nixl
uv run inference @ examples/reverse_text/rl/infer.toml --max-model-len 2048

