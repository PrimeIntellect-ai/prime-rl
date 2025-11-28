export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

uv run inference @ configs/reverse_text/rl/infer.toml --max-model-len 2048 --enable-lora
