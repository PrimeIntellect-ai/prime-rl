# Issue: Nemotron-H adapter export names may not match Hugging Face

## Status

Deferred model-specific issue. The TTT ScaleSWE experiment uses GLM-4.5-Air, not Nemotron-H.

## Summary

Prime-RL's custom Nemotron-H model uses module paths such as `model.layers`, `self_attn`, `mlp`,
and `mamba`. Hugging Face Nemotron-H represents layers under `backbone.layers` and unifies
several layer implementations under `mixer`; shared experts also use a different pluralized
path.

An incomplete `convert_adapter_to_hf` mapping can export LoRA keys that PEFT or serving code
does not attach to the intended Hugging Face modules.

## Change considered on the TTT branch

The mapping added conversions for:

- `model.layers` to `backbone.layers`;
- `mlp.shared_expert` to `mixer.shared_experts`;
- attention, MLP, and Mamba paths to the unified `mixer` path.

TTT replay also added inverse resolution against the custom trainer model.

## Why it is deferred

`convert_adapter_to_hf` is shared by ordinary policy-LoRA export. A model-specific mapping
should be validated with a real Nemotron-H checkpoint and the supported Hugging Face model,
rather than being inferred while implementing a GLM experiment.

## Suggested reproduction

1. Create small LoRA tensors for one attention projection, one Mamba projection, and one shared
   expert projection in the custom model namespace.
2. Export through `convert_adapter_to_hf`.
3. Load into the matching Hugging Face Nemotron-H model with PEFT.
4. Assert every key resolves exactly once and compare a forward pass against the custom model.

## Suggested tests

- Root and per-layer name conversion.
- Attention, MLP, Mamba, and shared-expert targets.
- No ambiguous or silently dropped keys.
- Round-trip numerical parity for a tiny adapter.

## Relevant code

- `src/prime_rl/trainer/models/nemotron_h/modeling_nemotron_h.py`
- `src/prime_rl/trainer/models/base.py`
- `src/prime_rl/trainer/lora.py`
