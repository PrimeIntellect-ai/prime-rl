# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Re-export BlockMaskingConfig from vendored prime_rl package.
"""

from prime_rl.inference.block_masking.config import BlockMaskingConfig

__all__ = ["BlockMaskingConfig"]
