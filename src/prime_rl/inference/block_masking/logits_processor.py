# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block Length Cap Logits Processor for Memento-style block masking.

Forces generation of <|block_end|> when the current block exceeds a
configurable token limit (max_block_tokens). This prevents blocks from
growing unboundedly and triggers the model's summarization behavior.

When a block's token count reaches the threshold, all logits are masked
to -inf except for the block_end token, forcing the model to close the
block and begin generating a summary.

IMPORTANT: Requires --async-scheduling false to work correctly.
In async scheduling mode, output_token_ids contains -1 placeholders
which makes block boundary detection impossible.

Configuration:
    Set max_block_tokens > 0 in BlockMaskingConfig to enable.
    E.g.: --block-masking-config '{"enable": true, "max_block_tokens": 2048, ...}'
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class _RequestBlockState:
    """Per-request block tracking state.

    Tracks block boundaries by scanning output_token_ids (a live
    reference to the request's generated tokens). Each call to apply()
    scans any new tokens appended since the last check.

    Requires sync scheduling (--async-scheduling false) so that
    output_token_ids contains real token IDs, not -1 placeholders.
    """
    output_token_ids: list  # Live reference to request's output tokens
    last_scanned_len: int = 0  # How far we've scanned so far
    in_block: bool = False
    block_token_count: int = 0
    total_forced: int = 0


class BlockLengthCapLogitsProcessor(LogitsProcessor):
    """Force <|block_end|> when block content exceeds max_block_tokens.

    Tracks block boundaries by scanning output_token_ids (which must
    contain real token IDs — requires --async-scheduling false).

    Each step:
    1. Scan any new tokens in output_token_ids since last check.
    2. Update block state (in_block, block_token_count).
    3. If inside a block and over the limit, force block_end.

    The processor is a no-op when:
    - Block masking is not enabled
    - max_block_tokens is 0 or negative
    - No requests are currently inside a long block
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        self.device = device
        self.pin_memory = is_pin_memory

        # Read config
        bm_config = getattr(vllm_config, "block_masking_config", None)
        if bm_config is not None and bm_config.enable:
            self.enabled = True
            self.max_block_tokens = getattr(bm_config, "max_block_tokens", 0)
            self.block_start_id = bm_config.block_start_id
            self.block_end_id = bm_config.block_end_id
            self.summary_start_id = bm_config.summary_start_id
            self.summary_end_id = bm_config.summary_end_id
            self.debug = bm_config.debug
        else:
            self.enabled = False
            self.max_block_tokens = 0
            self.block_start_id = -1
            self.block_end_id = -1
            self.summary_start_id = -1
            self.summary_end_id = -1
            self.debug = False

        if self.enabled and self.max_block_tokens > 0:
            logger.info(
                "[BlockCap] Initialized: max_block_tokens=%d, "
                "block_start_id=%d, block_end_id=%d",
                self.max_block_tokens, self.block_start_id, self.block_end_id,
            )

        # Per-request state: batch_index -> _RequestBlockState
        self._states: dict[int, _RequestBlockState] = {}

        # Pre-allocated tensor for forcing block_end
        self._neg_inf = torch.tensor(
            -float("inf"), dtype=torch.float32, device=device
        )

        # Special token set (these don't count as block content)
        self._special_ids = frozenset()
        if self.enabled:
            self._special_ids = frozenset([
                self.block_start_id, self.block_end_id,
                self.summary_start_id, self.summary_end_id,
            ])

    def is_argmax_invariant(self) -> bool:
        # When forcing block_end, we change the argmax outcome
        return False

    def _scan_new_tokens(self, state: _RequestBlockState) -> None:
        """Scan any new tokens in output_token_ids and update block state."""
        current_len = len(state.output_token_ids)
        if current_len <= state.last_scanned_len:
            return

        for i in range(state.last_scanned_len, current_len):
            tok = state.output_token_ids[i]
            if tok == -1:
                # Async placeholder — skip (shouldn't happen with
                # --async-scheduling false, but be defensive)
                continue
            if tok == self.block_start_id:
                state.in_block = True
                state.block_token_count = 0
                logger.info(
                    "[BlockCap] BLOCK_START at output pos %d", i,
                )
            elif tok == self.block_end_id:
                logger.info(
                    "[BlockCap] BLOCK_END at output pos %d "
                    "(block had %d tokens)", i, state.block_token_count,
                )
                state.in_block = False
                state.block_token_count = 0
            elif state.in_block and tok not in self._special_ids:
                state.block_token_count += 1

        state.last_scanned_len = current_len

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if not self.enabled or self.max_block_tokens <= 0:
            return

        if batch_update is None:
            return

        # Process added requests
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            state = _RequestBlockState(output_token_ids=output_tok_ids)
            # Scan prompt tokens for existing block structure
            if prompt_tok_ids:
                for tok in prompt_tok_ids:
                    if tok == self.block_start_id:
                        state.in_block = True
                        state.block_token_count = 0
                    elif tok == self.block_end_id:
                        state.in_block = False
                        state.block_token_count = 0
                    elif state.in_block and tok not in self._special_ids:
                        state.block_token_count += 1
            self._states[index] = state
            logger.info(
                "[BlockCap] Added idx=%d, prompt_len=%d, "
                "output_len=%d, in_block=%s, "
                "max_block_tokens=%d",
                index,
                len(prompt_tok_ids) if prompt_tok_ids else 0,
                len(output_tok_ids) if output_tok_ids else 0,
                state.in_block,
                self.max_block_tokens,
            )

        # Process removed requests
        for index in batch_update.removed:
            old = self._states.pop(index, None)
            if old is not None:
                logger.info(
                    "[BlockCap] Removed idx=%d, "
                    "output_len=%d, total_forced=%d",
                    index, len(old.output_token_ids), old.total_forced,
                )

        # Process moved requests
        for a_idx, b_idx, direct in batch_update.moved:
            a_state = self._states.pop(a_idx, None)
            b_state = self._states.pop(b_idx, None)
            if a_state is not None:
                self._states[b_idx] = a_state
            if b_state is not None and direct == MoveDirectionality.SWAP:
                self._states[a_idx] = b_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.max_block_tokens <= 0:
            return logits
        if not self._states:
            return logits

        for index, state in self._states.items():
            # Scan any new tokens added since last apply()
            self._scan_new_tokens(state)

            # Check if we need to force block_end
            if (state.in_block
                    and state.block_token_count >= self.max_block_tokens):
                logits[index, :] = self._neg_inf
                logits[index, self.block_end_id] = 0.0
                state.total_forced += 1
                logger.info(
                    "[BlockCap] FORCING block_end idx=%d, "
                    "block_count=%d/%d",
                    index, state.block_token_count, self.max_block_tokens,
                )

        return logits
