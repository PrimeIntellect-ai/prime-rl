export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora
