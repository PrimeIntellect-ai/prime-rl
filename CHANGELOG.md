# Changelog

Documenting changes which affect configuration usage patterns (added/moved/removed/renamed fields, notable logic changes).

- **`model.lora`**: Moved from `model.experimental.lora` to `model.lora` (no longer experimental) (#1440, 2025-12-16)
- Auto-set `api_server_count=1` on inference when LoRA is enabled, because vLLM doesn't support hotloading for multiple API servers (#1422, 2025-12-17)
- **`inference.model.rope_scaling`**: Added RoPE scaling configuration passthrough to vLLM (#1447 2025-12-17)
- **`orchestrator.env_mix`**: Deprecated in favor of `orchestrator.buffer.env_ratios` (#1450, 2025-12-18)
- **`orchestrator.buffer.hash_keys`**: Added hash keys configuration for buffer checkpointing (#1450, 2025-12-18)
- **`orchestrator.buffer.env_ratios`**: Added environment ratio configuration for buffer sampling (#1450, 2025-12-18)
- **`orchestrator.buffer.score_rollouts`**: Added configuration to control whether rollouts are scored using the environment's rubric. If False, rewards are always set to 0, online_difficulty_filtering is disabled, and easy/hard thresholds are not used (default: True)
- **`orchestrator.ckpt.buffer_path`**: Deprecated (#1450, 2025-12-18)
- **`orchestrator.buffer.easy_fraction`** and **`orchestrator.buffer.hard_fraction`**: Easy and hard fraction now defines the fraction of easy and hard problems to convert to normal when resuming, whereas previously it was the ratio of easy/ hard samples to sample per step (#1450, 2025-12-18)
- **`orchestrator.teacher_model`**: Added teacher model configuration for computing teacher logprobs (e.g. for distillation). Supports `TeacherModelConfig` (custom model) or `None` (disabled). Renamed from `reference_model` (2025-12-20)
- **`seq_len`**: Added root-level `seq_len` config that sets both `trainer.model.seq_len` and `orchestrator.seq_len`. Added validation that `trainer.model.seq_len >= orchestrator.seq_len` (2025-12-18)
- **`trainer.loss.sequence_mask_ratio_low`** and **`trainer.loss.sequence_mask_ratio_high`**: Renamed to `trainer.loss.sequence_mask_low` and `trainer.loss.sequence_mask_high` (2025-12-19)
- **`trainer.loss.token_mask_high`** and **`trainer.loss.token_mask_low`**: Added token-level importance ratio masking thresholds (2025-12-19)
- **`trainer.loss.sequence_clip_high`**: Added sequence-level importance ratio clipping threshold (2025-12-19)
- **`trainer.loss.geo_mask_high`** and **`trainer.loss.geo_mask_low`**: Added geometric importance ratio masking thresholds (2025-12-19)
- **`trainer.loss.adv_tau`**: Added tau parameter for advantages (default: 1.0)
- **`trainer.loss.teacher_tau`**: Added tau parameter for teacher logprobs (default: 0.0). Renamed from `ref_tau`
- **`{orchestrator,trainer}.transport.zmq`**: Added ZMQ transport for training batches and micro batches (#1446, 2025-12-22)
