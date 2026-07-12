# Issue: invalid LoRA CPU capacity fails late

## Status

Deferred from the TTT review. No new global validation is applied by this note.

## Summary

When vLLM LoRA serving is enabled, `max_cpu_loras` is expected to be at least `max_loras`.
Prime-RL configuration can currently express a smaller CPU capacity, allowing an invalid or
unusable combination to reach inference startup.

## Change considered on the TTT branch

An `InferenceConfig` model validator rejected configurations where:

```text
enable_lora = true and max_cpu_loras < max_loras
```

The TTT ScaleSWE config explicitly sets both values to 272, so it does not need a global
validator to launch correctly.

## Why it is deferred

This is a validation change for every LoRA inference user. Before landing it, the exact vLLM
contract should be confirmed for the pinned version, including whether CPU capacity has special
semantics under different deployment modes or zero/offload settings.

## Suggested resolution

If the vLLM invariant is unconditional, add the validator with a direct reference to the vLLM
field contract and place tests in the inference-config suite. If the invariant is conditional,
encode the full condition rather than the simplified TTT assumption.

## Suggested tests

- LoRA disabled preserves existing defaults.
- Equal capacities are accepted.
- Smaller CPU capacity is rejected only under the configurations vLLM itself rejects.
- Disaggregated/prefill/decode overrides are covered if they can set separate capacities.

## Relevant code

- `packages/prime-rl-configs/src/prime_rl/configs/inference.py`
- inference argument construction and vLLM overrides
