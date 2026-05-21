# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block Masking module for Memento-style inference.

This overlay re-exports from the vendored prime_rl.inference.block_masking
package so that vLLM internal imports (e.g. from vllm.v1.core.block_masking)
resolve correctly.
"""

from prime_rl.inference.block_masking.config import BlockMaskingConfig
from prime_rl.inference.block_masking.processor import BlockMaskingProcessor
from prime_rl.inference.block_masking.tracker import BlockInfo, BlockMaskingState

__all__ = [
    "BlockMaskingConfig",
    "BlockMaskingState",
    "BlockInfo",
    "BlockMaskingProcessor",
]
