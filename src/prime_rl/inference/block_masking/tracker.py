# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block state tracking for Memento-style block masking.

This module defines the per-request state for tracking block boundaries
during generation. The state is attached to each Request object when
block masking is enabled.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlockInfo:
    """Information about a single block."""

    block_id: int
    start_position: int
    end_position: Optional[int] = None
    summary_start: Optional[int] = None
    summary_end: Optional[int] = None

    @property
    def is_complete(self) -> bool:
        """Block is complete when it has both end and summary_end."""
        return self.end_position is not None and self.summary_end is not None

    def get_content_range(self, mask_delimiters: bool) -> Optional[tuple[int, int]]:
        """
        Returns (start, end) range of block content to compact, or None if not ready.

        The range is [start, end) - start inclusive, end exclusive.

        Args:
            mask_delimiters: Whether to include delimiter tokens in the masked range.
                - True: Mask [block_start, block_end] inclusive (Phi3/Phi4 style)
                - False: Mask (block_start, block_end) exclusive (Qwen3/OLMo3 style)

        Returns:
            (start, end) tuple where start is inclusive and end is exclusive,
            or None if the block is not complete.
        """
        if not self.is_complete:
            return None

        if mask_delimiters:
            # Phi3/Phi4 style: mask [block_start, block_end] inclusive
            # Return [start, end+1) to include block_end token
            return (self.start_position, self.end_position + 1)
        else:
            # Qwen3/OLMo3 style: keep delimiters visible, mask content only
            # Return (start+1, end) to exclude both block_start and block_end
            return (self.start_position + 1, self.end_position)

    def get_restart_range(self, mask_delimiters: bool) -> Optional[tuple[int, int]]:
        """
        Returns (start, end) range for restart mode: block tokens only.

        In restart mode, we evict block content + delimiters from the KV cache,
        then rewind num_computed_tokens to summary_start so the engine
        re-prefills summary tokens. The summary KV stays in the cache and is
        overwritten in-place during re-prefill.

        Block delimiters (block_start, block_end) are ALWAYS included in the
        eviction range regardless of mask_delimiters, matching HF restart
        which evicts all block tokens including delimiters.

        Summary tokens are NOT evicted (unlike the old implementation).
        This avoids negative physical positions that occurred when evicting
        block+summary as a contiguous range then rewinding into the gap.

        Args:
            mask_delimiters: Ignored for restart mode — delimiters are always
                evicted to match HF restart behavior.

        Returns:
            (start, end) tuple covering [block_start, block_end] inclusive,
            or None if the block is not complete.
        """
        if not self.is_complete:
            return None

        # Always evict [block_start, block_end] inclusive.
        # Summary tokens stay in cache — they'll be overwritten during
        # re-prefill after the rewind to summary_start.
        return (self.start_position, self.end_position + 1)

    # Keep content_range as property for backwards compatibility, defaults to Qwen3/OLMo3 style
    @property
    def content_range(self) -> Optional[tuple[int, int]]:
        """Backwards compatible property. Use get_content_range() for explicit control."""
        return self.get_content_range(mask_delimiters=False)


@dataclass
class BlockMaskingState:
    """
    Per-request state for block masking.

    Tracks:
    - Open blocks (via stack for nesting support)
    - Completed blocks ready for compaction
    - Position adjustments from previous compactions
    - Assistant section detection
    """

    # Stack of currently open blocks: [(block_id, start_pos), ...]
    # Supports nested blocks
    open_blocks: list[tuple[int, int]] = field(default_factory=list)

    # All blocks (including completed ones)
    # Maps block_id -> BlockInfo
    blocks: dict[int, BlockInfo] = field(default_factory=dict)

    # Counter for assigning block IDs
    next_block_id: int = 0

    # Completed blocks ready for compaction (in order of completion)
    # These are block_ids that have both block_end and summary_end
    pending_compactions: list[int] = field(default_factory=list)

    # Blocks that have been compacted
    compacted_block_ids: list[int] = field(default_factory=list)

    # Total tokens compacted so far (for position adjustment)
    # When compacting, logical positions > compacted region need adjustment
    total_compacted_tokens: int = 0

    # Track assistant section (for filtering false positives in prompts)
    assistant_start_pos: Optional[int] = None
    last_im_start_pos: Optional[int] = None
    in_assistant_section: bool = False

    # Currently active summary region
    in_summary: bool = False
    current_summary_block_id: Optional[int] = None

    # Restart mode: when set, the scheduler should rewind num_computed_tokens
    # to this position to trigger summary re-prefill with clean KV.
    pending_restart_rewind: Optional[int] = None

    # Flag to track that we've scheduled a deferred compaction
    # (to avoid skipping sampling multiple times)
    deferred_compaction_scheduled: bool = False

    def start_block(self, position: int) -> int:
        """
        Start a new block at the given position.

        Returns:
            The block_id of the new block
        """
        block_id = self.next_block_id
        self.next_block_id += 1

        self.open_blocks.append((block_id, position))
        self.blocks[block_id] = BlockInfo(
            block_id=block_id,
            start_position=position,
        )

        return block_id

    def end_block(self, position: int) -> Optional[int]:
        """
        End the current (innermost) block at the given position.

        Returns:
            The block_id that was ended, or None if no open block
        """
        if not self.open_blocks:
            return None

        block_id, _ = self.open_blocks.pop()
        self.blocks[block_id].end_position = position

        return block_id

    def start_summary(self, position: int) -> Optional[int]:
        """
        Start a summary region at the given position.

        Associates with the most recently closed block that doesn't have a summary.

        Returns:
            The block_id associated with this summary, or None
        """
        # Find the most recently closed block without a summary
        for block_id in reversed(list(self.blocks.keys())):
            block = self.blocks[block_id]
            if block.end_position is not None and block.summary_start is None:
                block.summary_start = position
                self.in_summary = True
                self.current_summary_block_id = block_id
                return block_id

        return None

    def end_summary(self, position: int) -> Optional[int]:
        """
        End the current summary region at the given position.

        Returns:
            The block_id of the completed block, or None
        """
        if not self.in_summary or self.current_summary_block_id is None:
            return None

        block_id = self.current_summary_block_id
        block = self.blocks[block_id]
        block.summary_end = position

        self.in_summary = False
        self.current_summary_block_id = None

        # Block is now complete - add to pending compactions
        if block.is_complete:
            self.pending_compactions.append(block_id)

        return block_id

    def get_compaction_range(
        self,
        keep_last_n: int,
        mask_delimiters: bool = False,
        restart_mode: bool = False,
    ) -> Optional[tuple[int, int]]:
        """
        Get the next compaction range if one is ready.

        Respects keep_last_n setting:
        - -1: Never compact
        - 0: Compact immediately when ready
        - N: Keep last N blocks, compact older ones

        Args:
            keep_last_n: Number of blocks to keep visible
            mask_delimiters: Whether to include delimiter tokens in compaction range
                - True: Mask [block_start, block_end] inclusive (Phi3/Phi4 style)
                - False: Mask content only, keep delimiters visible (Qwen3/OLMo3 style)
            restart_mode: If True, use get_restart_range() to include summary in
                the eviction range. This is needed because we will rewind and
                recompute the summary KV.

        Returns:
            (start, end) logical positions to compact, or None
        """
        if keep_last_n < 0:
            # Never compact
            return None

        num_pending = len(self.pending_compactions)

        # Only compact if we have more pending than keep_last_n
        if num_pending <= keep_last_n:
            return None

        # Get the oldest pending block
        block_id = self.pending_compactions.pop(0)
        block = self.blocks[block_id]

        if restart_mode:
            # Restart mode: evict block content + summary
            content_range = block.get_restart_range(mask_delimiters=mask_delimiters)
        else:
            # Normal mode: evict only block content
            content_range = block.get_content_range(mask_delimiters=mask_delimiters)
        if content_range is None:
            return None

        start, end = content_range

        # NOTE: Return LOGICAL positions, not adjusted positions!
        # The scheduler's mask_token_span() handles logical-to-physical
        # conversion internally using request.compacted_spans.
        # We only need to track total_compacted_tokens for internal state.

        # Update tracking
        tokens_to_compact = end - start
        self.total_compacted_tokens += tokens_to_compact
        self.compacted_block_ids.append(block_id)

        return (start, end)

    def set_in_assistant_section(self, position: int) -> None:
        """Mark that we've entered the assistant section."""
        self.assistant_start_pos = position
        self.in_assistant_section = True

    def record_im_start(self, position: int) -> None:
        """Record position of <|im_start|> token."""
        self.last_im_start_pos = position

    def check_assistant_start(self, position: int) -> bool:
        """
        Check if current position marks start of assistant section.

        Returns True if this is the "assistant" token right after <|im_start|>.
        """
        if self.in_assistant_section:
            return False  # Already in assistant section

        if self.last_im_start_pos is None:
            return False

        # Check if we're within 2 positions of <|im_start|>
        # (allows for possible whitespace token)
        return position - self.last_im_start_pos <= 2

    def should_process_token(self, position: int, require_assistant: bool) -> bool:
        """
        Check if we should process this token for block tracking.

        Args:
            position: Current token position
            require_assistant: Whether assistant section is required

        Returns:
            True if token should be processed for block tracking
        """
        if not require_assistant:
            return True

        if not self.in_assistant_section:
            return False

        # Don't process the assistant token itself
        if position <= self.assistant_start_pos:
            return False

        return True

    def get_stats(self) -> dict:
        """Get statistics about block tracking state."""
        return {
            "total_blocks": len(self.blocks),
            "open_blocks": len(self.open_blocks),
            "pending_compactions": len(self.pending_compactions),
            "compacted_blocks": len(self.compacted_block_ids),
            "total_compacted_tokens": self.total_compacted_tokens,
            "in_assistant_section": self.in_assistant_section,
        }
