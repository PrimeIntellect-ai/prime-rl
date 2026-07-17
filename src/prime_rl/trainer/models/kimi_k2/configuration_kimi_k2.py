import warnings

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class KimiK2Config(DeepseekV3Config):
    r"""
    Configuration class for Kimi K2 (and K2.7's text backbone).

    Kimi K2 reuses DeepSeek-V3's architecture verbatim (same MLA/MoE hyperparameters, same
    HF weight-key naming — Moonshot ships `transformers`' own `DeepseekV3ForCausalLM` as
    `auto_map` target). This subclasses `DeepseekV3Config` directly rather than redefining
    its ~20 MLA/MoE fields; only the three fields below are prime-rl/Kimi-specific.

    K2.7 (and K2.5, K2.6) are released as a multimodal wrapper (`model_type="kimi_k25"`)
    with a nested `text_config` of this same shape plus a vision tower; this config
    describes only the text backbone — see `converting_kimi_k2.py` for how a K2.7
    checkpoint's `language_model.`-prefixed weights get loaded (vision ignored entirely).

    Args:
        use_grouped_mm (`bool`, defaults to `True`):
            Whether to use grouped matrix multiplication for MoE.
        topk_method (`str`, defaults to `"noaux_tc"`):
            MoE routing top-k method (bias-based load balancing, no auxiliary loss).
        scoring_func (`str`, defaults to `"sigmoid"`):
            Scoring function for the MoE router.
    """

    model_type = "kimi_k2"

    def __init__(self, use_grouped_mm=True, topk_method="noaux_tc", scoring_func="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.use_grouped_mm = use_grouped_mm
        self.topk_method = topk_method
        self.scoring_func = scoring_func

        if not self.use_grouped_mm:
            warnings.warn("not using grouped mm for moe is very slow, should only be used for debugging")
