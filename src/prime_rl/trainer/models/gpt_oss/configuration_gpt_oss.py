# Re-export HF's GptOssConfig so the parameter naming and rope/sliding-window defaults
# stay in sync with upstream. We don't need to subclass it: the only PrimeRL-specific
# behavior lives in the model and checkpoint conversion modules.
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

__all__ = ["GptOssConfig"]
