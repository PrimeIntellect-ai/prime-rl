import warnings

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class KimiK25Config(DeepseekV3Config):
    model_type = "kimi_k25"

    def __init__(self, use_grouped_mm=True, topk_method="noaux_tc", scoring_func="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.use_grouped_mm = use_grouped_mm
        self.topk_method = topk_method
        self.scoring_func = scoring_func

        if not self.use_grouped_mm:
            warnings.warn("not using grouped mm for moe is very slow, should only be used for debugging")
