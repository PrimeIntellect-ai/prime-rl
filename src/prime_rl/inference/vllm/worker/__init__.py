import logging
import os

from prime_rl.inference.patches import (
    monkey_patch_fp32_lm_head,
    monkey_patch_fp32_router_logits,
    monkey_patch_minimax_m2_for_lora,
    monkey_patch_no_moe_lora,
)
from prime_rl.inference.vllm.kept_tokens import monkey_patch_kept_tokens_sampler

logger = logging.getLogger(__name__)

# Monkeypatch MiniMaxM2 MoE gate dtype and adapter key mapping for LoRA compatibility
monkey_patch_minimax_m2_for_lora()
# Disable LoRA on MoE layers so vLLM picks better kernels (e.g. TRTLLMFlashInfer on Blackwell)
if os.environ.get("PRIME_NO_MOE_LORA") == "1":
    logger.info("PRIME_NO_MOE_LORA=1: disabling LoRA on MoE layers")
    monkey_patch_no_moe_lora()
else:
    logger.info("PRIME_NO_MOE_LORA=0: no patch applied")

# Install fp32 lm_head patch; self-gates on additional_config["fp32_lm_head"] at call time
monkey_patch_fp32_lm_head()

# Install fp32 router logits patch; self-gates on additional_config["fp32_router_logits"]
monkey_patch_fp32_router_logits()

# Install kept-tokens sampler patch (sampling replay); no-op unless
# PRIME_RETURN_KEPT_TOKENS=1
monkey_patch_kept_tokens_sampler()
