"""Load vLLM model directly (bypass multiprocess engine) to inspect parameter names."""

import os
os.environ["VLLM_USE_DEEP_GEMM"] = "1"

import torch
torch.cuda.set_device(0)

from vllm.config import VllmConfig, ModelConfig, LoadConfig, CacheConfig, SchedulerConfig, ParallelConfig
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized

init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29501")
ensure_model_parallel_initialized(1, 1)

import sys
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/checkpoints/glm5-tiny-fp8"
FP8_DIR = MODEL_DIR
print(f"Loading from: {MODEL_DIR}")

model_config = ModelConfig(
    model=FP8_DIR,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=256,
)
print(f"Model dtype: {model_config.dtype}")

parallel_config = ParallelConfig()
vllm_config = VllmConfig(
    model_config=model_config,
    load_config=LoadConfig(),
    cache_config=CacheConfig(block_size=64),
    scheduler_config=SchedulerConfig(max_num_seqs=1, max_model_len=256, is_encoder_decoder=False),
    parallel_config=parallel_config,
)

print("Initializing model...")
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = initialize_model(vllm_config=vllm_config)
torch.set_default_dtype(torch.float32)

print("Loading FP8 weights...")
loader = DefaultModelLoader(vllm_config.load_config)
weights_iter = loader._get_weights_iterator(
    DefaultModelLoader.Source(FP8_DIR, revision=None, prefix="", fall_back_to_pt=True)
)
model.load_weights(weights_iter)

device = next(model.parameters()).device
process_weights_after_loading(model, vllm_config.model_config, device)

print("FP8 MODEL LOADED OK")
print()
print("=== vLLM FP8 Model Parameters (kernel naming) ===")
for name, param in model.named_parameters():
    print(f"  PARAM {name:80s} {str(list(param.shape)):30s} {param.dtype}")
print()
print("=== vLLM FP8 Model Buffers ===")
for name, buf in model.named_buffers():
    print(f"  BUF   {name:80s} {str(list(buf.shape)):30s} {buf.dtype}")
