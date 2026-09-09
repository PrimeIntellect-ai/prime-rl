import logging
import os

from prime_rl.inference.patches import (
    monkey_patch_fp32_lm_head,
    monkey_patch_minimax_m2_for_lora,
    monkey_patch_no_moe_lora,
)

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

if os.environ.get("PRIME_DENSE_ALIGNMENT") == "1":
    from prime_rl.trainer.models.layers.dense_alignment import enable_serving_alignment

    enable_serving_alignment(fp32_head=os.environ.get("PRIME_DENSE_FP32_HEAD") == "1")
    logger.info("PRIME_DENSE_ALIGNMENT=1: enabled shared dense Qwen3 forward arithmetic")

if os.environ.get("PRIME_MOE_ALIGNMENT") == "1":
    if os.environ.get("PRIME_DENSE_ALIGNMENT") != "1" or os.environ.get("PRIME_DENSE_FP32_HEAD") != "1":
        raise ValueError("PRIME_MOE_ALIGNMENT requires shared dense arithmetic and the FP32 head")
    from prime_rl.trainer.models.layers.moe_alignment import enable_serving_moe_alignment

    enable_serving_moe_alignment()
    logger.info("PRIME_MOE_ALIGNMENT=1: enabled shared Qwen3 MoE arithmetic")

if os.environ.get("PRIME_GLM_ALIGNMENT") == "1":
    if os.environ.get("PRIME_DENSE_ALIGNMENT") != "1" or os.environ.get("PRIME_DENSE_FP32_HEAD") != "1":
        raise ValueError("PRIME_GLM_ALIGNMENT requires shared dense arithmetic and the FP32 head")
    from prime_rl.trainer.models.layers.moe_alignment import enable_serving_glm_alignment

    enable_serving_glm_alignment()
    logger.info("PRIME_GLM_ALIGNMENT=1: enabled shared GLM MoE arithmetic")
