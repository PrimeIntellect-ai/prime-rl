# Re-export HF's GptOssConfig so the parameter naming and rope/sliding-window defaults
# stay in sync with upstream.
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

__all__ = ["GptOssConfig"]
