# Changelog

Documenting changes which affect configuration usage patterns (added/moved/removed/renamed fields, notable logic changes).

- **`inference.model.rope_scaling`**: Added RoPE scaling configuration passthrough to vLLM via `--rope-scaling` (2025-12-17)
- **`model.lora`**: Moved from `model.experimental.lora` to `model.lora` (no longer experimental) (#1440, 2025-12-16)
