export CUDA_VISIBLE_DEVICES=6,7
export VLLM_LOGGING_LEVEL=DEBUG
uv run inference --model.name google/gemma-4-26B-A4B-it \
  --enable-lora --max-lora-rank 16 --max-loras 4 \
  --parallel.tp 2 --gpu-memory-utilization 0.8 --vllm-extra '{"lora_dtype": "bfloat16"}' | tee infer.log
  #--model.max-model-len 65536 \
