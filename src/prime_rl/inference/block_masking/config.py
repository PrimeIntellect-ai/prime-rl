# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block Masking Configuration for Memento-style inference.

Block masking enables automatic compaction of "thinking" blocks during
generation, keeping only their summaries visible for subsequent attention.

This implements the inference-time behavior of Memento-trained models,
using vLLM's token compaction infrastructure for memory efficiency.

Example usage:
    from vllm import LLM
    from vllm.config import BlockMaskingConfig

    llm = LLM(
        model="path/to/memento-model",
        block_masking_config=BlockMaskingConfig(
            enable=True,
            keep_last_n_blocks=0,
        ),
    )
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class BlockMaskingConfig:
    """
    Configuration for Memento-style block masking during inference.

    When enabled, the engine monitors generated tokens for block boundary
    markers and automatically compacts completed blocks using mask_token_span().

    Attributes:
        enable: Whether to enable block masking. Default False.

        keep_last_n_blocks: How many completed blocks to keep visible.
            -1 = Keep all blocks (no compaction, standard generation)
            0 = Compact all completed blocks immediately (default when enabled)
            N = Keep last N blocks visible, compact older ones

        block_start_token: Token string for block start. Default "<|block_start|>"
        block_end_token: Token string for block end. Default "<|block_end|>"
        summary_start_token: Token string for summary start. Default "<|summary_start|>"
        summary_end_token: Token string for summary end. Default "<|summary_end|>"

        require_assistant_section: Only process blocks after <|im_start|>assistant.
            This prevents false positives from block tokens in prompts.
            Default True.

        im_start_token: Token for detecting assistant section. Default "<|im_start|>"
        assistant_token: Token after im_start for assistant. Default "assistant"

        compact_on_summary_end: Trigger compaction when summary_end is seen.
            If False, compaction must be triggered manually. Default True.

        mask_delimiters: Whether to mask the block delimiter tokens (block_start, block_end)
            in addition to the block content. REQUIRED - no default value.
            - True: Mask delimiters (matches Phi3/Phi4 training)
            - False: Keep delimiters visible (matches Qwen3/OLMo3 training)
            This must be set explicitly to avoid train/inference mismatch.

        keep_last_block_for_answer: When True with keep_last_n_blocks=0, defers
            compaction of the most recently completed block so it remains visible
            in the KV cache when the model generates the final answer (after
            </think>). Older blocks are still evicted immediately.

            NOTE: This is NOT the same as keep_last_n_blocks=1. With keep1,
            during block N generation block N-1 content is still visible (it's
            only compacted when block N's summary completes). With this option,
            block N-1 is compacted as soon as block N's block_start is seen,
            so during reasoning the model sees no old block content (matching
            keep0 training distribution). Only the very last block is preserved
            for the final answer. Default False.

        debug: Enable debug logging. Default False.

    Example:
        # Default configuration (compact all blocks)
        config = BlockMaskingConfig(enable=True)

        # Keep last block visible (useful for reasoning chains)
        config = BlockMaskingConfig(enable=True, keep_last_n_blocks=1)

        # Custom tokens
        config = BlockMaskingConfig(
            enable=True,
            block_start_token="<think>",
            block_end_token="</think>",
        )
    """

    enable: bool = False
    keep_last_n_blocks: int = 0

    # Block boundary tokens
    block_start_token: str = "<|block_start|>"
    block_end_token: str = "<|block_end|>"
    summary_start_token: str = "<|summary_start|>"
    summary_end_token: str = "<|summary_end|>"

    # Assistant section detection
    require_assistant_section: bool = True
    im_start_token: str = "<|im_start|>"
    assistant_token: str = "assistant"

    # Behavior
    compact_on_summary_end: bool = True
    mask_delimiters: Optional[bool] = None  # REQUIRED when enable=True, no default
    keep_last_block_for_answer: bool = False
    restart_mode: bool = False
    """When True, after compacting block content, rewind num_computed_tokens
    to the summary_start position so the engine re-prefills summary tokens
    with clean KV (block content no longer in cache). This matches HF
    restart behavior: evict block+summary KV, recompute summary."""
    async_mode: Literal["safe_sync", "async_barrier", "async_native"] = "safe_sync"
    """How block masking interacts with vLLM async scheduling.

    safe_sync disables async scheduling when block masking is enabled.
    async_barrier keeps async scheduling enabled but serializes compaction steps.
    async_native is reserved for a future placeholder-aware implementation.
    """
    max_block_tokens: int = 0
    """Maximum tokens allowed inside a single block before forcing block_end.
    0 = disabled (no cap). When > 0, a logits processor will force the model
    to generate <|block_end|> once a block reaches this many content tokens."""
    debug: bool = False

    # Resolved token IDs (set during initialization)
    _block_start_id: Optional[int] = field(default=None, repr=False)
    _block_end_id: Optional[int] = field(default=None, repr=False)
    _summary_start_id: Optional[int] = field(default=None, repr=False)
    _summary_end_id: Optional[int] = field(default=None, repr=False)
    _im_start_id: Optional[int] = field(default=None, repr=False)
    _assistant_id: Optional[int] = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def initialize_token_ids(self, tokenizer) -> "BlockMaskingConfig":
        """
        Resolve token strings to IDs using the provided tokenizer.

        This must be called before the config is used. It's automatically
        called by the engine during initialization.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            self (for chaining)

        Raises:
            ValueError: If required tokens are not in vocabulary
        """
        if self._initialized:
            return self

        def get_token_id(token: str, required: bool = True) -> Optional[int]:
            """Get token ID, optionally raising if not found.
            Accepts integer strings (e.g. "151669") as direct token IDs."""
            # Support direct integer token IDs (e.g. for models without
            # named special tokens like <|block_start|>)
            try:
                return int(token)
            except (ValueError, TypeError):
                pass
            try:
                # Try encode first
                ids = tokenizer.encode(token, add_special_tokens=False)
                if len(ids) == 1:
                    return ids[0]
                # Try convert_tokens_to_ids for special tokens
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    return token_id
            except Exception:
                pass

            if required:
                raise ValueError(
                    f"Token '{token}' not found in vocabulary. "
                    f"Ensure the model was trained with this token or add it "
                    f"using tokenizer.add_special_tokens()."
                )
            return None

        # Resolve required block tokens
        self._block_start_id = get_token_id(self.block_start_token, required=True)
        self._block_end_id = get_token_id(self.block_end_token, required=True)
        self._summary_start_id = get_token_id(self.summary_start_token, required=True)
        self._summary_end_id = get_token_id(self.summary_end_token, required=True)

        # Resolve optional assistant detection tokens
        if self.require_assistant_section:
            self._im_start_id = get_token_id(self.im_start_token, required=False)
            self._assistant_id = get_token_id(self.assistant_token, required=False)
            if self._im_start_id is None or self._assistant_id is None:
                # Disable assistant section requirement if tokens not found
                self.require_assistant_section = False

        self._initialized = True
        return self

    @property
    def block_start_id(self) -> int:
        assert self._initialized, "Call initialize_token_ids() first"
        return self._block_start_id

    @property
    def block_end_id(self) -> int:
        assert self._initialized, "Call initialize_token_ids() first"
        return self._block_end_id

    @property
    def summary_start_id(self) -> int:
        assert self._initialized, "Call initialize_token_ids() first"
        return self._summary_start_id

    @property
    def summary_end_id(self) -> int:
        assert self._initialized, "Call initialize_token_ids() first"
        return self._summary_end_id

    @property
    def im_start_id(self) -> Optional[int]:
        return self._im_start_id

    @property
    def assistant_id(self) -> Optional[int]:
        return self._assistant_id

    def __post_init__(self):
        if self.async_mode not in ("safe_sync", "async_barrier", "async_native"):
            raise ValueError(
                "async_mode must be one of: safe_sync, async_barrier, async_native"
            )
        if self.enable and self.keep_last_n_blocks < -1:
            raise ValueError("keep_last_n_blocks must be >= -1")
        if self.enable and self.mask_delimiters is None:
            raise ValueError(
                "mask_delimiters must be explicitly set when block masking is enabled.\n"
                "  - mask_delimiters=True: Mask block_start and block_end tokens (Phi3/Phi4 style)\n"
                "  - mask_delimiters=False: Keep delimiters visible (Qwen3/OLMo3 style)\n"
                "This ensures inference matches the model's training configuration."
            )
