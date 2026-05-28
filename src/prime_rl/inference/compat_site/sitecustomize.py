"""Compatibility hooks that must run before vLLM imports model modules.

This directory is added to PYTHONPATH only for vLLM inference processes.
"""

from transformers.models import qwen2_vl
from transformers.models.qwen2_vl import Qwen2VLImageProcessor

qwen2_vl.__dict__.setdefault("Qwen2VLImageProcessorFast", Qwen2VLImageProcessor)
