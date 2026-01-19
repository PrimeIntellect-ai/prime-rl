# Considerata

## Temperature scheduling
- The orchestrator does not cancel in-flight rollouts when the temperature changes, so a single training batch may mix rollouts generated at different temperatures while `TrainingBatch.temperature` and `sampling/temperature` report the latest value.
