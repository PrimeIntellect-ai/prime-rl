# Considerata

## Temperature scheduling
- The orchestrator does not cancel in-flight rollouts when the temperature changes, so a single training batch may mix rollouts generated at different temperatures while `TrainingBatch.temperature` and `sampling/temperature` report the latest value.
- The fused LM head assumes a single temperature per microbatch, so rollouts with different temperatures must be split into separate microbatches (potentially reducing packing efficiency).
- This split is required for correctness because logprob/entropy computation depends on temperature-scaled logits; mixing temperatures in one microbatch would mis-scale some samples.
