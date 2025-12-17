# Changelog

Documenting changes which affect configuration usage patterns (added/moved/removed/renamed fields, notable logic changes).

- **`model.lora`**: Moved from `model.experimental.lora` to `model.lora` (no longer experimental) (#1440, 2025-12-16)
- Auto-set `api_server_count=1` on inference when LoRA is enabled, because vLLM doesn't support hotloading for multiple API servers (#1422, 2025-12-17)
