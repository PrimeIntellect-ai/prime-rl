# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block Masking module for Memento-style inference.

This module provides automatic block compaction during generation,
using vLLM's token span removal infrastructure.
"""

from .config import BlockMaskingConfig
from .position_mapping import CompactedSpanTracker, merge_spans
from .processor import BlockMaskingProcessor
from .tracker import BlockInfo, BlockMaskingState

__all__ = [
    "BlockMaskingConfig",
    "BlockMaskingState",
    "BlockInfo",
    "BlockMaskingProcessor",
    "CompactedSpanTracker",
    "merge_spans",
]
