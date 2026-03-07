#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/test_fp8_load.out

cd /home/matej/dev/prime-rl
export VLLM_USE_DEEP_GEMM=1

# Try loading FP8 model directly
uv run python -c "
import os
os.environ['VLLM_USE_DEEP_GEMM'] = '1'
from vllm import LLM, SamplingParams
FP8_DIR = '/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-fp8'
llm = LLM(model=FP8_DIR, dtype='bfloat16', enforce_eager=True, gpu_memory_utilization=0.5, max_model_len=256, trust_remote_code=True)
print('FP8_LOADED_OK')
out = llm.generate([{'prompt_token_ids': [1,2,3,4,5]}], SamplingParams(max_tokens=1))
print(f'Generated: {out[0].outputs[0].text!r}')
" 2>&1
