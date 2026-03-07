"""Inspect vLLM model parameter names by loading inside the LLM engine."""

import os
import torch
import json
from vllm import LLM
from vllm.sampling_params import SamplingParams

BF16_DIR = "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-bf16"

# Use LLM with a callback to inspect the model
llm = LLM(
    model=BF16_DIR,
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=0.5,
    max_model_len=256,
    trust_remote_code=True,
)

# Run a simple generation to confirm the engine works
output = llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
print(f"Generation test: {output[0].outputs[0].text!r}")

# Now inspect parameters via collective_rpc
import asyncio

async def inspect():
    engine = llm.llm_engine.engine_core.engine_core
    result = await engine.model_executor.collective_rpc("get_param_info")
    return result

# Actually, let's just use the collective_rpc properly
# The model is in a subprocess. Let's write a custom function.
# Instead, let's use the simplest approach: write a worker extension that dumps params.

# Alternative: parse what load_weights loaded
print("\n=== Checkpoint keys (HF format) ===")
from safetensors import safe_open
with safe_open(os.path.join(BF16_DIR, "model.safetensors"), framework="pt") as f:
    for key in sorted(f.keys()):
        t = f.get_tensor(key)
        print(f"  {key:80s} {str(list(t.shape)):30s} {t.dtype}")
