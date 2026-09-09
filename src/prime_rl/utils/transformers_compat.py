"""Shims for architectures newer than the pinned transformers version.

Kept free of heavy imports: this is loaded from a vLLM general plugin, in every vLLM
process, before the engine parses a model config.
"""

DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


def allow_deepseek_v4_layer_types() -> None:
    """Add V4's compressed attention names to the vocabulary transformers validates against.

    `PretrainedConfig.validate_layer_type` checks `layer_types` entries against a module-level
    tuple, and transformers only added V4's two compressed names in 5.15. Below that, every
    DeepSeek V4 config raises on construction: prime-rl's own (whose `validate_layer_type`
    override cannot help, since the base one runs first, from `super().__init__()`) and vLLM's
    (`vllm/transformers_utils/configs/deepseek_v4.py`, which has no override at all). Extending
    the tuple is what lets the repo hold a transformers pin below 5.15 and still serve V4.
    """
    from transformers import configuration_utils

    unknown = tuple(
        layer_type
        for layer_type in DEEPSEEK_V4_LAYER_TYPES
        if layer_type not in configuration_utils.ALLOWED_LAYER_TYPES
    )
    if unknown:
        configuration_utils.ALLOWED_LAYER_TYPES += unknown
