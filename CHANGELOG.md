# Changelog

Documenting changes which affect configuration usage patterns (added/moved/removed/renamed fields, notable logic changes).

- **`model.lora`**: Moved from `model.experimental.lora` to `model.lora` (no longer experimental) (#1440, 2025-12-16)
- Auto-set `api_server_count=1` on inference when LoRA is enabled, because vLLM doesn't support hotloading for multiple API servers (#1422, 2025-12-17)
- **`inference.model.rope_scaling`**: Added RoPE scaling configuration passthrough to vLLM (#1447 2025-12-17)
- **`orchestrator.env_mix`**: Deprecated in favor of `orchestrator.buffer.env_ratios` (#1450, 2025-12-18)
- **`orchestrator.buffer.hash_keys`**: Added hash keys configuration for buffer checkpointing (#1450, 2025-12-18)
- **`orchestrator.buffer.env_ratios`**: Added environment ratio configuration for buffer sampling (#1450, 2025-12-18)
- **`orchestrator.ckpt.buffer_path`**: Deprecated (#1450, 2025-12-18)
- **`orchestrator.buffer.easy_fraction`** and **`orchestrator.buffer.hard_fraction`**: Easy and hard fraction now defines the fraction of easy and hard problems to convert to normal when resuming, whereas previously it was the ratio of easy/ hard samples to sample per step (#1450, 2025-12-18)
- **`seq_len`**: Added root-level `seq_len` config that sets both `trainer.model.seq_len` and `orchestrator.seq_len`. Added validation that `trainer.model.seq_len >= orchestrator.seq_len` (2025-12-18)
- **`trainer.loss.sequence_mask_ratio_low`** and **`trainer.loss.sequence_mask_ratio_high`**: Renamed to `trainer.loss.sequence_mask_low` and `trainer.loss.sequence_mask_high` (2025-12-19)
- **`trainer.loss.token_mask_high`** and **`trainer.loss.token_mask_low`**: Added token-level importance ratio masking thresholds (2025-12-19)
- **`trainer.loss.sequence_clip_high`**: Added sequence-level importance ratio clipping threshold (2025-12-19)
- **`trainer.loss.geo_mask_high`** and **`trainer.loss.geo_mask_low`**: Added geometric importance ratio masking thresholds (2025-12-19)
