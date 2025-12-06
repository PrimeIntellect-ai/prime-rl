export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run inference @ examples/reverse_text/rl/infer.toml --max-model-len 2048 --enable-lora -tp 4

