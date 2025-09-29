# TorchTitan MoE Weight Formats

## TL;DR
Upstream examples focus on dense models. When training TorchTitan MoEs (e.g. `Jackmin108/debug-moe-0.5B`), the SFT warmup finishes fine but the weight checkpoint manager rewrites the state dict into a pure HuggingFace layout for upload. Those checkpoints **drop** the aggregated MoE tensors (`model.layers.N.mlp.experts.w1/w2/w3`, `router.gate.weight`). Prime RL's vLLM inference path still expects the TorchTitan layout, so RL launches crash (`KeyError: 'layers.0.mlp.experts.w1'`).

## What Happens Today
### SFT Weight Saver
Under the hood, `WeightCheckpointManager._convert_tt_moe_to_hf_` rewrites TT MoE checkpoints into HF format. Aggregated tensors get split, TT buffers are removed. This is great for pushing to HF, but vLLM can't load it.

### Inference Defender
vLLM's `Qwen3MoeForCausalLM` loader looks for aggregated tensors. When it only sees per-expert tensors, it raises `KeyError` and exits. Trainer + orchestrator follow suit.

## How to Fix
### Option 1 (short term)
Post-process the SFT checkpoint: starting from the TT `.bin`, rebuild `layers.*.mlp.experts.w*` and `router.gate.weight`, save to a new directory, and point RL trainer/orchestrator/inference there.

### Option 2 (long term)
Patch `WeightCheckpointManager` or vLLM to support both layouts.

## Next Steps
- Decide whether to patch the saver or a post-training script.
- Once aggregated tensors live in the checkpoint, RL goes through.

