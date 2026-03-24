from prime_rl.inference.patches import (
    monkey_patch_fused_moe_lora_dp,
    monkey_patch_LRUCacheWorkerLoRAManager,
    monkey_patch_minimax_m2_for_lora,
)

# Monkeypatch LRUCacheWorkerLoRAManager to allow loading adapter inplace without doing it every request
monkey_patch_LRUCacheWorkerLoRAManager()
# Monkeypatch MiniMaxM2 MoE gate dtype and adapter key mapping for LoRA compatibility
monkey_patch_minimax_m2_for_lora()
# NOTE: Fix LoRA + MoE + DP>1 corrupted output (vLLM 0.17.0 bug)
# Applied here so it runs inside every worker subprocess, not just the main API server process.
monkey_patch_fused_moe_lora_dp()
