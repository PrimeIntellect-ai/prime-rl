# Ported from vLLM main at fb159244e6fdd0b2ecd9d62855cae9a2d282ae87 (Apache-2.0, Copyright contributors to the vLLM
# project); v0.34.0 has no such class:
# - vllm/model_executor/models/config.py: the modeling_overrides check of
#   Qwen3_5ForSlotDecisionConfig.verify_and_update_model_config (lines 1765-1781).
# - vllm/transformers_utils/model_arch_config_convertor.py: what selects the class (lines 656-668), as the class
#   attributes.
# Left out, as they set up the server: the hook's head_dtype and max_model_len defaults, its verify_and_update_config,
# and Qwen3_5SlotDecisionModelState (vllm/model_executor/models/qwen3_5_slot.py), which turns a request's prefix length
# into the attention ranges of the server's kernel.
"""Qwen3.5 slot decision model: a stock Qwen3.5 checkpoint whose config.json has ``modeling_overrides``."""

from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

MODELING_OVERRIDES = "modeling_overrides"
SERVED_OVERRIDES = {
    "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
    "readout": "slot_lm",
}
MAX_SLOTS = 256


def get_n_slots(config) -> int:
    """The number of slot labels that ``config.modeling_overrides`` declares, 256 by default.

    Raises unless vLLM serves the overrides: without ``head_dim`` and ``n_slots`` they equal ``SERVED_OVERRIDES``, and
    ``n_slots`` is an int from 1 to 256."""
    overrides = getattr(config, MODELING_OVERRIDES, None)
    found = dict(overrides) if isinstance(overrides, dict) else {}
    # head_dim sizes a learned readout, which vLLM does not serve
    found.pop("head_dim", None)
    n_slots = found.pop("n_slots", MAX_SLOTS)
    if found != SERVED_OVERRIDES or type(n_slots) is not int or not 1 <= n_slots <= MAX_SLOTS:
        raise ValueError(
            f"Unsupported modeling_overrides {overrides!r}: vLLM serves {SERVED_OVERRIDES} with 1 to {MAX_SLOTS} "
            "n_slots"
        )
    return n_slots


class Qwen3_5ForSlotDecision(Qwen3_5ForCausalLM):
    """Qwen3.5 under the name vLLM serves a checkpoint with ``modeling_overrides`` by, which reads answers from the LM
    head at slot labels. It adds no weights, and refuses overrides that vLLM refuses to serve."""

    extra_config_key = MODELING_OVERRIDES
    stock_model_type = "qwen3_5"
    stock_architecture = "Qwen3_5ForConditionalGeneration"

    def __init__(self, config) -> None:
        get_n_slots(config)
        super().__init__(config)


__all__ = ["Qwen3_5ForSlotDecision", "get_n_slots"]
