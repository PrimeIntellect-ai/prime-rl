import logging
import os

from prime_rl.inference.patches import (
    monkey_patch_LRUCacheWorkerLoRAManager,
    monkey_patch_minimax_m2_for_lora,
    monkey_patch_no_moe_lora,
    monkey_patch_skip_lora_module_warnings,
)

logger = logging.getLogger(__name__)

# Monkeypatch LRUCacheWorkerLoRAManager to allow loading adapter inplace without doing it every request
monkey_patch_LRUCacheWorkerLoRAManager()
# Skip the per-module regex warning loop in WorkerLoRAManager._load_adapter
# (minutes-long stall on wide MoE models like Qwen3.5-35B-A3B)
monkey_patch_skip_lora_module_warnings()
# Monkeypatch MiniMaxM2 MoE gate dtype and adapter key mapping for LoRA compatibility
monkey_patch_minimax_m2_for_lora()
# Disable LoRA on MoE layers so vLLM picks better kernels (e.g. TRTLLMFlashInfer on Blackwell)
if os.environ.get("PRIME_NO_MOE_LORA") == "1":
    logger.info("PRIME_NO_MOE_LORA=1: disabling LoRA on MoE layers")
    monkey_patch_no_moe_lora()
else:
    logger.info("PRIME_NO_MOE_LORA=0: no patch applied")
