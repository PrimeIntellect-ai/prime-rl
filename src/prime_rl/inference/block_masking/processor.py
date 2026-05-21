# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block masking processor for handling token-by-token block tracking.

This module provides the main interface for processing tokens and
determining when to trigger compaction.
"""

from typing import TYPE_CHECKING, Optional

from .tracker import BlockMaskingState

if TYPE_CHECKING:
    from .config import BlockMaskingConfig


class BlockMaskingProcessor:
    """
    Processor for block masking during generation.

    This class handles the token-by-token processing of generated output,
    detecting block boundaries and determining when to trigger compaction.

    Usage in scheduler:
        processor = BlockMaskingProcessor(config)

        # For each generated token:
        compaction = processor.process_token(state, token_id, position)
        if compaction:
            start, end = compaction
            self.mask_token_span(request_id, start, end)
    """

    def __init__(self, config: "BlockMaskingConfig"):
        """
        Initialize the processor with configuration.

        Args:
            config: Block masking configuration (must be initialized)
        """
        self.config = config
        assert config._initialized, "Config must be initialized with token IDs"

    def create_state(self) -> BlockMaskingState:
        """Create a new state for a request."""
        return BlockMaskingState()

    def process_token(
        self,
        state: BlockMaskingState,
        token_id: int,
        position: int,
    ) -> Optional[tuple[int, int]]:
        """
        Process a newly generated token and return compaction range if needed.

        This is the main entry point called by the scheduler for each
        generated token.

        Args:
            state: The request's block masking state
            token_id: The generated token ID
            position: The absolute position (prompt + generated)

        Returns:
            Optional (start, end) tuple for compaction, or None
        """
        config = self.config

        # Track <|im_start|> for assistant section detection
        if config.im_start_id is not None and token_id == config.im_start_id:
            state.record_im_start(position)

        # Check for assistant section start
        if (config.require_assistant_section and
            config.assistant_id is not None and
            token_id == config.assistant_id and
            state.check_assistant_start(position)):
            state.set_in_assistant_section(position)
            if config.debug:
                print(f"[BlockMasking] Assistant section detected at position {position}")

        # Check if we should process this token for block tracking
        if not state.should_process_token(position, config.require_assistant_section):
            return None

        # Process block boundary tokens
        compaction = None

        if token_id == config.block_start_id:
            # If keep_last_block_for_answer: a new block is starting, so the
            # previous block (sitting in pending_compactions) is NOT the last
            # one. Flush it now — this gives pure keep0 KV usage mid-generation.
            if (config.keep_last_block_for_answer
                    and config.keep_last_n_blocks == 0
                    and state.pending_compactions):
                compaction = state.get_compaction_range(
                    keep_last_n=0,
                    mask_delimiters=config.mask_delimiters,
                    restart_mode=config.restart_mode,
                )
                if config.debug and compaction:
                    print(f"[BlockMasking] Deferred compaction flushed on "
                          f"block_start: {compaction}")
                # Restart mode: schedule rewind for the deferred block's summary
                if config.restart_mode and compaction is not None:
                    flushed_block_id = state.compacted_block_ids[-1]
                    flushed_block = state.blocks[flushed_block_id]
                    state.pending_restart_rewind = flushed_block.summary_start
                    if config.debug:
                        print(f"[BlockMasking] Restart rewind scheduled "
                              f"to position {flushed_block.summary_start}")

            block_id = state.start_block(position)
            if config.debug:
                print(f"[BlockMasking] Block {block_id} started at position {position}")

        elif token_id == config.block_end_id:
            block_id = state.end_block(position)
            if config.debug and block_id is not None:
                print(f"[BlockMasking] Block {block_id} ended at position {position}")

        elif token_id == config.summary_start_id:
            block_id = state.start_summary(position)
            if config.debug and block_id is not None:
                print(f"[BlockMasking] Summary started for block {block_id} at {position}")

        elif token_id == config.summary_end_id:
            block_id = state.end_summary(position)
            if config.debug and block_id is not None:
                print(f"[BlockMasking] Summary ended for block {block_id} at {position}")

            # Check if we should compact
            if config.compact_on_summary_end and block_id is not None:
                if (config.keep_last_block_for_answer
                        and config.keep_last_n_blocks == 0):
                    # Defer: don't compact now. If <|block_start|> comes next,
                    # we'll flush it then. If </think> comes instead, the block
                    # stays in KV cache for the final answer.
                    if config.debug:
                        print(f"[BlockMasking] Deferring compaction of block "
                              f"{block_id} (keep_last_block_for_answer)")
                else:
                    compaction = state.get_compaction_range(
                        config.keep_last_n_blocks,
                        mask_delimiters=config.mask_delimiters,
                        restart_mode=config.restart_mode,
                    )
                    if config.debug and compaction:
                        print(f"[BlockMasking] Compaction triggered: {compaction} "
                              f"(mask_delimiters={config.mask_delimiters}, "
                              f"restart_mode={config.restart_mode})")
                    # Restart mode: schedule rewind to recompute summary KV
                    if config.restart_mode and compaction is not None:
                        block = state.blocks[block_id]
                        state.pending_restart_rewind = block.summary_start
                        if config.debug:
                            print(f"[BlockMasking] Restart rewind scheduled "
                                  f"to position {block.summary_start}")

        return compaction

    def process_prompt_tokens(
        self,
        state: BlockMaskingState,
        prompt_token_ids: list[int],
    ) -> None:
        """
        Process prompt tokens to initialize block state.

        This scans the prompt for existing block structures (e.g., in
        multi-turn conversations) to initialize the state correctly.

        Note: We don't trigger compactions during prompt processing.

        Args:
            state: The request's block masking state
            prompt_token_ids: The prompt token IDs
        """
        config = self.config

        for position, token_id in enumerate(prompt_token_ids):
            # Track <|im_start|> for assistant section detection
            if config.im_start_id is not None and token_id == config.im_start_id:
                state.record_im_start(position)

            # Check for assistant section start
            if (config.require_assistant_section and
                config.assistant_id is not None and
                token_id == config.assistant_id and
                state.check_assistant_start(position)):
                state.set_in_assistant_section(position)

            # Skip block tracking if not in assistant section
            if not state.should_process_token(position, config.require_assistant_section):
                continue

            # Track block boundaries (but don't compact)
            if token_id == config.block_start_id:
                state.start_block(position)
            elif token_id == config.block_end_id:
                state.end_block(position)
            elif token_id == config.summary_start_id:
                state.start_summary(position)
            elif token_id == config.summary_end_id:
                state.end_summary(position)

        # Restart mode: schedule rewind for prompt blocks that need compaction
        if config.restart_mode and state.pending_compactions:
            earliest_summary_start = None
            for block_id in state.pending_compactions:
                block = state.blocks[block_id]
                if block.summary_start is not None:
                    if (earliest_summary_start is None or
                            block.summary_start < earliest_summary_start):
                        earliest_summary_start = block.summary_start
            if earliest_summary_start is not None:
                state.pending_restart_rewind = earliest_summary_start
                if config.debug:
                    print(f"[BlockMasking] Restart rewind scheduled "
                          f"to position {earliest_summary_start} "
                          f"(prompt, {len(state.pending_compactions)} blocks)")

        if config.debug:
            stats = state.get_stats()
            print(f"[BlockMasking] Prompt processed: {stats}")

    def force_compact_pending(
        self,
        state: BlockMaskingState,
    ) -> list[tuple[int, int]]:
        """
        Force compaction of all pending blocks.

        This can be called at the end of generation to ensure all
        completed blocks are compacted.

        Returns:
            List of (start, end) compaction ranges
        """
        compactions = []

        while state.pending_compactions:
            # Temporarily set keep_last_n to 0 to force compaction
            compaction = state.get_compaction_range(
                keep_last_n=0,
                mask_delimiters=self.config.mask_delimiters,
            )
            if compaction:
                compactions.append(compaction)
                if self.config.debug:
                    print(f"[BlockMasking] Forced compaction: {compaction}")

        return compactions
