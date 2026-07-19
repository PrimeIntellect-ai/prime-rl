"""State dict conversion for Nemotron-VL between HF (checkpoint) and PrimeRL (training) formats.

HF checkpoint layout (produced by models/assemble_nemotron_vl.py):
  - LM keys: either unprefixed text-checkpoint names (`backbone.*`, `lm_head.weight`;
    assembly `--mode hardlink`) or Omni-style `language_model.backbone.*`
    (`--mode rewrite`). Both are accepted; `convert_to_hf` emits the unprefixed form.
  - Vision tower: `vision_model.radio_model.*` (verbatim from Nano Omni / C-RADIOv4-H)
  - Projector: `mlp1.{0,1,3}.weight`

PrimeRL layout (matches NemotronVLForCausalLM.state_dict()):
  - `model.language_model.*` in NemotronH prime format (split mixers, embed_tokens, norm)
  - `model.visual.radio_model.*`
  - `model.mlp1.{0,1,3}.weight`
  - `lm_head.weight`

LM tensor-level conversion (mixer splitting, mtp dropping) is delegated to the
nemotron_h converters; this module only adds the composite prefix handling and
drops Omni-only extras (video embedder, sound tower) if present.
"""

from torch import Tensor

from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import (
    _rename_keys,
    convert_hf_to_prime,
    convert_prime_to_hf,
)
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import (
    _infer_layers_block_type,
    _infer_layers_block_type_from_hf,
)

_HF_TO_PRIME_PREFIXES = {
    "vision_model.": "model.visual.",
    "mlp1.": "model.mlp1.",
}
# Omni composite keys that have no counterpart in Nemotron-VL (video/audio towers).
_HF_DROP_PREFIXES = ("sound_encoder.", "sound_projection.")
_HF_DROP_KEYS = ("vision_model.radio_model.model.patch_generator.video_embedder.weight",)


def _pop_by_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    popped = {key: state_dict.pop(key) for key in [k for k in state_dict if k.startswith(prefix)]}
    return popped


def convert_nemotron_vl_hf_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # Unify the two supported LM layouts to unprefixed text-checkpoint names.
    _rename_keys(state_dict, "language_model.", "")

    for key in list(state_dict):
        if key in _HF_DROP_KEYS or key.startswith(_HF_DROP_PREFIXES):
            del state_dict[key]

    # Set vision/projector keys aside so the LM prefix renames cannot touch them.
    side = {}
    for hf_prefix, prime_prefix in _HF_TO_PRIME_PREFIXES.items():
        for key, tensor in _pop_by_prefix(state_dict, hf_prefix).items():
            side[prime_prefix + key[len(hf_prefix) :]] = tensor

    # LM: backbone.* -> model.*, mixer splits, mtp dropped; then nest under language_model.
    layers_block_type = _infer_layers_block_type_from_hf(state_dict)
    convert_hf_to_prime(state_dict, layers_block_type)
    _rename_keys(state_dict, "model.", "model.language_model.")

    state_dict.update(side)
    return state_dict


def convert_nemotron_vl_prime_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    side = {}
    for hf_prefix, prime_prefix in _HF_TO_PRIME_PREFIXES.items():
        for key, tensor in _pop_by_prefix(state_dict, prime_prefix).items():
            side[hf_prefix + key[len(prime_prefix) :]] = tensor

    _rename_keys(state_dict, "model.language_model.", "model.")
    layers_block_type = _infer_layers_block_type(state_dict)
    convert_prime_to_hf(state_dict, layers_block_type)

    state_dict.update(side)
    return state_dict


__all__ = ["convert_nemotron_vl_hf_to_prime", "convert_nemotron_vl_prime_to_hf"]
