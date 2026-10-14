
from importlib.metadata import distribution as _vllm_distribution
__path__.append(str(_vllm_distribution("vllm").locate_file("vllm/model_executor/determinism")))
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
