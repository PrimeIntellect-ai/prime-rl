"""Test loading the FP8 model in vLLM with deep_gemm."""
import os
os.environ["VLLM_USE_DEEP_GEMM"] = "1"

from vllm import LLM, SamplingParams

FP8_DIR = "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-fp8"

print("Loading FP8 model in vLLM...")
llm = LLM(
    model=FP8_DIR,
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=0.5,
    max_model_len=256,
    trust_remote_code=True,
)
print("VLLM_FP8_MODEL_LOADED_OK")
