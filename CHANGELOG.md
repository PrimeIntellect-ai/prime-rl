# Changelog

Documenting changes which affect configuration usage patterns (added/moved/removed/renamed fields, notable logic changes).

- **Per-model defaults**: Added `MODEL_DEFAULTS` registry in `rl.py` that automatically applies model-specific configuration (trajectory strategy, trainer impl, inference settings) based on model name. Supported models include Qwen3 4B/30B/235B Instruct and Thinking variants. The `[inference]` section is no longer required in rl.toml when model defaults provide inference settings. (#1442, 2025-12-16)
- **`model.lora`**: Moved from `model.experimental.lora` to `model.lora` (no longer experimental) (#1440, 2025-12-16)
