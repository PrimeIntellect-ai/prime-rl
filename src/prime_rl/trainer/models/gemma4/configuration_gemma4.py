"""Prime-RL Gemma4 text config.

Subclasses HF's Gemma4TextConfig to inherit all field definitions, then adds
prime-rl-specific knobs (e.g. grouped_mm toggle for MoE).

Keeping the HF field set intact means:
  - config JSONs from the HF hub load unchanged
  - layer_types / rope_parameters / hidden_size_per_layer_input / etc. all
    flow through without having to re-derive them here.
"""

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig as HFGemma4TextConfig


class Gemma4Config(HFGemma4TextConfig):
    """Gemma4 text config, extended with prime-rl knobs.

    The HF model_type is ``gemma4_text`` for the text-only stack; we keep that
    so ``AutoConfig`` round-trips and existing checkpoints load unchanged.
    """

    model_type = "gemma4_text"

    def __init__(self, *args, use_grouped_mm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # Used by prime-rl's GroupedExperts path; Gemma4TextExperts uses a
        # packed (gate+up) layout so we default off.
        self.use_grouped_mm = use_grouped_mm
