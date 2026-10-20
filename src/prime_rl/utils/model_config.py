"""Model-free model-config, tokenizer and processor policy shared by trainer and export."""

from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.logger import get_logger

logger = get_logger()

# Per-layer schedule fields of the supported architectures; anything else is left untouched.
# One entry per decoder layer, truncated to the surviving layers.
_LAYER_SCHEDULE_LISTS = (
    "layer_types",
    "layers_block_type",
    "mlp_layer_types",
    "indexer_types",
    "compress_ratios",
    "num_attention_heads_per_layer",
)
# Scalar prefix counts: layers below the count use a different block; clamped to depth.
# Periodic strides like `decoder_sparse_step` are not prefix counts.
_LAYER_PREFIX_COUNTS = ("num_hash_layers", "num_dense_layers", "first_k_dense_replace")
# Decoder layer indices, filtered to the surviving layers.
_LAYER_INDEX_SCHEDULES = ("mlp_only_layers",)
# Compact per-layer pattern (one character per decoder layer), sliced to the surviving layers.
_LAYER_PATTERN_SCHEDULES = ("hybrid_override_pattern",)


def _get(config, name):
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _set(config, name, value):
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def truncate_model_depth(config, num_layers: int) -> tuple[int, int]:
    """Shorten a decoder config to ``num_layers`` layers, trimming its per-layer schedules.

    Keeps the first-N-layer layout so the shortened config still passes strict HF
    validation on rebuild. Returns ``(depth, original_depth)``.
    """
    if isinstance(config, dict):
        original_depth = config["num_hidden_layers"]
    else:
        original_depth = config.num_hidden_layers
    depth = min(num_layers, original_depth)
    if depth < original_depth:
        for name in _LAYER_SCHEDULE_LISTS:
            schedule = _get(config, name)
            if isinstance(schedule, list) and len(schedule) > depth:
                _set(config, name, schedule[:depth])
        for name in _LAYER_INDEX_SCHEDULES:
            layers = _get(config, name)
            if isinstance(layers, list) and any(layer >= depth for layer in layers):
                _set(config, name, [layer for layer in layers if layer < depth])
        for name in _LAYER_PATTERN_SCHEDULES:
            pattern = _get(config, name)
            if isinstance(pattern, str) and len(pattern) > depth:
                _set(config, name, pattern[:depth])
        for name in _LAYER_PREFIX_COUNTS:
            count = _get(config, name)
            if isinstance(count, int) and count > depth:
                _set(config, name, depth)
    if isinstance(config, dict):
        config["num_hidden_layers"] = depth
    else:
        config.num_hidden_layers = depth
    return depth, original_depth


def load_tokenizer(
    name: str, *, trust_remote_code: bool = False, chat_template: str | None = None
) -> PreTrainedTokenizer:
    """Load a tokenizer with the trainer's chat-template and pad-ID policy."""
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if chat_template is not None:
        path = Path(chat_template)
        if path.is_file():
            logger.info(f"Loading custom chat template from file: {path}")
            tokenizer.chat_template = path.read_text()
            logger.debug(f"Chat template content:\n{tokenizer.chat_template}")
        else:
            logger.info("Using inline custom chat template")
            tokenizer.chat_template = chat_template
    # Some models have no pad token; pad with EOS so batched sequences terminate correctly.
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def try_load_processor(name: str, *, trust_remote_code: bool = False):
    """Load an ``AutoProcessor`` for ``name``; return ``None`` when none is available."""
    try:
        return AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)
    except (ValueError, OSError, KeyError) as exc:
        logger.debug(f"No AutoProcessor available for {name} ({type(exc).__name__})")
        return None


def is_multimodal_processor(processor) -> bool:
    """Whether a processor has image or video components, i.e. the model is not text-only."""
    return bool(getattr(processor, "image_processor", None) or getattr(processor, "video_processor", None))
