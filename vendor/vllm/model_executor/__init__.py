
from importlib.metadata import distribution as _vllm_distribution
__path__.append(str(_vllm_distribution("vllm").locate_file("vllm/model_executor")))
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.parameter import BasevLLMParameter, PackedvLLMParameter

__all__ = [
    "BasevLLMParameter",
    "PackedvLLMParameter",
]
