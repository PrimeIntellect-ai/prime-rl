"""
Pure logic tests for Memento block masking.

Tests the vendored modules (config, tracker, processor, position_mapping)
on CPU with no vLLM or GPU dependency.
"""

import pytest

from prime_rl.inference.block_masking.config import BlockMaskingConfig
from prime_rl.inference.block_masking.tracker import BlockInfo, BlockMaskingState
from prime_rl.inference.block_masking.processor import BlockMaskingProcessor
from prime_rl.inference.block_masking.position_mapping import (
    CompactedSpanTracker,
    merge_spans,
)

# Token IDs used throughout tests
BLOCK_START = 100
BLOCK_END = 101
SUMMARY_START = 102
SUMMARY_END = 103
IM_START = 104
ASSISTANT = 105
UNK = 0
CONTENT = 200  # generic content token


class FakeTokenizer:
    """Minimal tokenizer for testing token ID resolution."""

    def __init__(self, vocab: dict[str, int], unk_token_id: int = UNK):
        self._vocab = vocab
        self._reverse = {v: k for k, v in vocab.items()}
        self.unk_token_id = unk_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self._vocab:
            return [self._vocab[text]]
        return [self.unk_token_id, self.unk_token_id]  # multi-token = not found

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)


def make_tokenizer() -> FakeTokenizer:
    return FakeTokenizer({
        "<|block_start|>": BLOCK_START,
        "<|block_end|>": BLOCK_END,
        "<|summary_start|>": SUMMARY_START,
        "<|summary_end|>": SUMMARY_END,
        "<|im_start|>": IM_START,
        "assistant": ASSISTANT,
    })


def make_config(**overrides) -> BlockMaskingConfig:
    defaults = dict(
        enable=True,
        mask_delimiters=False,
        keep_last_n_blocks=0,
        require_assistant_section=False,
    )
    defaults.update(overrides)
    config = BlockMaskingConfig(**defaults)
    config.initialize_token_ids(make_tokenizer())
    return config


def run_generation(
    processor: BlockMaskingProcessor,
    state: BlockMaskingState,
    token_ids: list[int],
    start_pos: int = 0,
) -> list[tuple[int, int]]:
    """Feed tokens one by one, collect all compaction ranges."""
    compactions = []
    for i, token_id in enumerate(token_ids):
        result = processor.process_token(state, token_id, start_pos + i)
        if result is not None:
            compactions.append(result)
    return compactions


# ---------------------------------------------------------------------------
# Test 1: Config initialization and validation
# ---------------------------------------------------------------------------

def test_config_initialization_and_validation():
    tokenizer = make_tokenizer()

    # --- Basic initialization ---
    config = BlockMaskingConfig(enable=True, mask_delimiters=False)
    config.initialize_token_ids(tokenizer)
    assert config._initialized
    assert config.block_start_id == BLOCK_START
    assert config.block_end_id == BLOCK_END
    assert config.summary_start_id == SUMMARY_START
    assert config.summary_end_id == SUMMARY_END
    assert config.im_start_id == IM_START
    assert config.assistant_id == ASSISTANT

    # --- Integer string token resolution ---
    config_int = BlockMaskingConfig(
        enable=True,
        mask_delimiters=False,
        block_start_token="42",
        block_end_token="43",
        summary_start_token="44",
        summary_end_token="45",
    )
    config_int.initialize_token_ids(tokenizer)
    assert config_int.block_start_id == 42
    assert config_int.block_end_id == 43
    assert config_int.summary_start_id == 44
    assert config_int.summary_end_id == 45

    # --- Missing required token raises ---
    config_bad = BlockMaskingConfig(
        enable=True,
        mask_delimiters=False,
        block_start_token="<|nonexistent|>",
    )
    with pytest.raises(ValueError, match="not found in vocabulary"):
        config_bad.initialize_token_ids(tokenizer)

    # --- mask_delimiters=None with enable=True raises ---
    with pytest.raises(ValueError, match="mask_delimiters must be explicitly set"):
        BlockMaskingConfig(enable=True, mask_delimiters=None)

    # --- keep_last_n_blocks < -1 raises ---
    with pytest.raises(ValueError, match="keep_last_n_blocks must be >= -1"):
        BlockMaskingConfig(enable=True, mask_delimiters=False, keep_last_n_blocks=-2)

    # --- async mode validation ---
    config_barrier = BlockMaskingConfig(
        enable=True,
        mask_delimiters=False,
        async_mode="async_barrier",
    )
    assert config_barrier.async_mode == "async_barrier"
    with pytest.raises(ValueError, match="async_mode must be one of"):
        BlockMaskingConfig(
            enable=True,
            mask_delimiters=False,
            async_mode="bad_mode",
        )

    # --- require_assistant_section auto-disables if tokens missing ---
    no_assistant_tokenizer = FakeTokenizer({
        "<|block_start|>": BLOCK_START,
        "<|block_end|>": BLOCK_END,
        "<|summary_start|>": SUMMARY_START,
        "<|summary_end|>": SUMMARY_END,
    })
    config_no_asst = BlockMaskingConfig(
        enable=True,
        mask_delimiters=False,
        require_assistant_section=True,
    )
    config_no_asst.initialize_token_ids(no_assistant_tokenizer)
    assert not config_no_asst.require_assistant_section
    assert config_no_asst.im_start_id is None

    # --- Idempotent re-initialization ---
    config2 = BlockMaskingConfig(enable=True, mask_delimiters=False)
    result1 = config2.initialize_token_ids(tokenizer)
    result2 = config2.initialize_token_ids(tokenizer)
    assert result1 is result2 is config2

    # --- Disabled config skips validation ---
    config_disabled = BlockMaskingConfig(enable=False, mask_delimiters=None)
    assert not config_disabled.enable  # should not raise

    # --- encode() path: single-token encode ---
    encode_tokenizer = FakeTokenizer({
        "<|block_start|>": BLOCK_START,
        "<|block_end|>": BLOCK_END,
        "<|summary_start|>": SUMMARY_START,
        "<|summary_end|>": SUMMARY_END,
    })
    config_enc = BlockMaskingConfig(enable=True, mask_delimiters=False,
                                    require_assistant_section=False)
    config_enc.initialize_token_ids(encode_tokenizer)
    assert config_enc.block_start_id == BLOCK_START


# ---------------------------------------------------------------------------
# Test 2: Full block lifecycle — single block
# ---------------------------------------------------------------------------

def test_full_block_lifecycle_single_block():
    # --- mask_delimiters=False (Qwen3/OLMo3 style) ---
    config = make_config(mask_delimiters=False, keep_last_n_blocks=0)
    processor = BlockMaskingProcessor(config)
    state = processor.create_state()

    # Prompt: 6 tokens (positions 0-5)
    prompt = [CONTENT] * 6
    processor.process_prompt_tokens(state, prompt)
    assert len(state.blocks) == 0
    assert len(state.open_blocks) == 0

    # Generation starts at position 6
    # pos 6: block_start
    result = processor.process_token(state, BLOCK_START, 6)
    assert result is None
    assert len(state.open_blocks) == 1
    assert len(state.blocks) == 1
    assert state.blocks[0].start_position == 6
    assert state.blocks[0].end_position is None

    # pos 7-10: content tokens
    for pos in range(7, 11):
        result = processor.process_token(state, CONTENT, pos)
        assert result is None

    # pos 11: block_end
    result = processor.process_token(state, BLOCK_END, 11)
    assert result is None
    assert len(state.open_blocks) == 0
    assert state.blocks[0].end_position == 11

    # pos 12: summary_start
    result = processor.process_token(state, SUMMARY_START, 12)
    assert result is None
    assert state.in_summary
    assert state.current_summary_block_id == 0
    assert state.blocks[0].summary_start == 12

    # pos 13-14: summary content
    for pos in range(13, 15):
        result = processor.process_token(state, CONTENT, pos)
        assert result is None

    # pos 15: summary_end → triggers compaction
    result = processor.process_token(state, SUMMARY_END, 15)
    assert result is not None
    assert not state.in_summary
    assert state.blocks[0].summary_end == 15
    assert state.blocks[0].is_complete

    # mask_delimiters=False: range is (start+1, end) = (7, 11)
    start, end = result
    assert start == 7
    assert end == 11
    assert end - start == 4  # 4 content tokens

    assert len(state.compacted_block_ids) == 1
    assert state.total_compacted_tokens == 4

    # --- mask_delimiters=True (Phi3/Phi4 style) ---
    config_md = make_config(mask_delimiters=True, keep_last_n_blocks=0)
    processor_md = BlockMaskingProcessor(config_md)
    state_md = processor_md.create_state()

    processor_md.process_prompt_tokens(state_md, prompt)

    tokens = [BLOCK_START, CONTENT, CONTENT, CONTENT, CONTENT,
              BLOCK_END, SUMMARY_START, CONTENT, CONTENT, SUMMARY_END]
    compactions = run_generation(processor_md, state_md, tokens, start_pos=6)

    assert len(compactions) == 1
    start, end = compactions[0]
    assert start == 6   # block_start position (inclusive)
    assert end == 12    # block_end position + 1
    assert end - start == 6  # block_start + 4 content + block_end

    # --- require_assistant=True: blocks before assistant are ignored ---
    config_asst = make_config(require_assistant_section=True, mask_delimiters=False)
    processor_asst = BlockMaskingProcessor(config_asst)
    state_asst = processor_asst.create_state()

    # Pre-assistant tokens with block markers → should be ignored
    pre_tokens = [CONTENT, BLOCK_START, CONTENT, BLOCK_END,
                  SUMMARY_START, CONTENT, SUMMARY_END]
    compactions_pre = run_generation(processor_asst, state_asst, pre_tokens, start_pos=0)
    assert len(compactions_pre) == 0
    assert len(state_asst.blocks) == 0
    assert not state_asst.in_assistant_section

    # Now enter assistant section: im_start at pos 7, assistant at pos 8
    result = processor_asst.process_token(state_asst, IM_START, 7)
    assert result is None
    result = processor_asst.process_token(state_asst, ASSISTANT, 8)
    assert result is None
    assert state_asst.in_assistant_section

    # Now block markers should be tracked
    result = processor_asst.process_token(state_asst, BLOCK_START, 9)
    assert result is None
    assert len(state_asst.blocks) == 1

    # --- require_assistant=False: blocks tracked everywhere ---
    config_no_asst = make_config(require_assistant_section=False, mask_delimiters=False)
    processor_no_asst = BlockMaskingProcessor(config_no_asst)
    state_no_asst = processor_no_asst.create_state()

    tokens_all = [BLOCK_START, CONTENT, CONTENT, BLOCK_END,
                  SUMMARY_START, CONTENT, SUMMARY_END]
    compactions_all = run_generation(processor_no_asst, state_no_asst, tokens_all, start_pos=0)
    assert len(compactions_all) == 1
    assert compactions_all[0] == (1, 3)  # (start+1, end) = (1, 3)


# ---------------------------------------------------------------------------
# Test 3: Multi-block keep_last_n and deferred compaction
# ---------------------------------------------------------------------------

def _make_three_blocks(config: BlockMaskingConfig, prompt_len: int = 6):
    """Create processor+state and generate 3 complete blocks.

    Block layout (starting at position prompt_len):
      Block 0: BS content content BE SS content SE
      Block 1: BS content content BE SS content SE
      Block 2: BS content content BE SS content SE

    Returns (processor, state, all_compactions, final_pos)
    """
    processor = BlockMaskingProcessor(config)
    state = processor.create_state()
    processor.process_prompt_tokens(state, [CONTENT] * prompt_len)

    all_compactions = []
    pos = prompt_len
    for _ in range(3):
        tokens = [BLOCK_START, CONTENT, CONTENT, BLOCK_END,
                  SUMMARY_START, CONTENT, SUMMARY_END]
        for token_id in tokens:
            result = processor.process_token(state, token_id, pos)
            if result is not None:
                all_compactions.append(result)
            pos += 1

    return processor, state, all_compactions, pos


def test_multi_block_keep_last_n_and_deferred_compaction():
    # --- keep_last_n=0: compact all immediately ---
    config_k0 = make_config(keep_last_n_blocks=0, mask_delimiters=False)
    _, state_k0, compactions_k0, _ = _make_three_blocks(config_k0)
    assert len(compactions_k0) == 3
    assert len(state_k0.compacted_block_ids) == 3
    assert len(state_k0.pending_compactions) == 0
    # Each block: (start+1, end) = 2 content tokens
    for c in compactions_k0:
        assert c[1] - c[0] == 2

    # --- keep_last_n=1: keep 1 block, compact older ---
    config_k1 = make_config(keep_last_n_blocks=1, mask_delimiters=False)
    _, state_k1, compactions_k1, _ = _make_three_blocks(config_k1)
    assert len(compactions_k1) == 2
    assert len(state_k1.compacted_block_ids) == 2
    assert len(state_k1.pending_compactions) == 1  # block 2 still pending

    # --- keep_last_n=-1: never compact ---
    config_km1 = make_config(keep_last_n_blocks=-1, mask_delimiters=False)
    _, state_km1, compactions_km1, _ = _make_three_blocks(config_km1)
    assert len(compactions_km1) == 0
    assert len(state_km1.compacted_block_ids) == 0
    assert len(state_km1.pending_compactions) == 3

    # --- keep_last_block_for_answer with keep_last_n=0 ---
    config_defer = make_config(
        keep_last_n_blocks=0,
        keep_last_block_for_answer=True,
        mask_delimiters=False,
    )
    processor_defer, state_defer, compactions_defer, final_pos = \
        _make_three_blocks(config_defer)

    # Block 0 completes → deferred (no compaction)
    # Block 1 starts → flushes block 0 (1 compaction)
    # Block 1 completes → deferred
    # Block 2 starts → flushes block 1 (1 compaction)
    # Block 2 completes → deferred
    # Total: 2 compactions (blocks 0, 1). Block 2 is deferred.
    assert len(compactions_defer) == 2
    assert len(state_defer.compacted_block_ids) == 2
    assert len(state_defer.pending_compactions) == 1  # block 2 deferred
    assert state_defer.pending_compactions[0] == 2

    # --- force_compact_pending drains all ---
    forced = processor_defer.force_compact_pending(state_defer)
    assert len(forced) == 1
    assert len(state_defer.pending_compactions) == 0
    assert len(state_defer.compacted_block_ids) == 3

    # --- force_compact_pending on keep_last_n=-1 ---
    config_force = make_config(keep_last_n_blocks=-1, mask_delimiters=False)
    proc_force, state_force, _, _ = _make_three_blocks(config_force)
    assert len(state_force.pending_compactions) == 3
    forced_all = proc_force.force_compact_pending(state_force)
    assert len(forced_all) == 3
    assert len(state_force.pending_compactions) == 0

    # --- Verify block content tracking ---
    config_verify = make_config(keep_last_n_blocks=0, mask_delimiters=False)
    _, state_v, _, _ = _make_three_blocks(config_verify)
    assert state_v.total_compacted_tokens == 6  # 3 blocks × 2 content tokens


# ---------------------------------------------------------------------------
# Test 4: Position mapping and span merging
# ---------------------------------------------------------------------------

def test_position_mapping_and_span_merging():
    # --- Single span (5, 10): removes positions 5-9 ---
    tracker = CompactedSpanTracker()
    tracker.spans = [(5, 10)]

    # Active tokens: 0-4, 10+
    assert tracker.is_token_active(0)
    assert tracker.is_token_active(4)
    assert not tracker.is_token_active(5)
    assert not tracker.is_token_active(9)
    assert tracker.is_token_active(10)

    # logical→physical
    assert tracker.logical_to_physical(0) == 0
    assert tracker.logical_to_physical(4) == 4
    assert tracker.logical_to_physical(10) == 5  # 10 - 5 removed = 5
    assert tracker.logical_to_physical(15) == 10

    # physical→logical
    assert tracker.physical_to_logical(0) == 0
    assert tracker.physical_to_logical(4) == 4
    assert tracker.physical_to_logical(5) == 10  # gap: phys 5 → logical 10
    assert tracker.physical_to_logical(10) == 15

    # get_active_positions
    active = tracker.get_active_positions(15)
    assert active == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]

    # get_compacted_token_count
    assert tracker.get_compacted_token_count() == 5

    # --- Two spans (5,10) + (15,20): cumulative offset ---
    tracker2 = CompactedSpanTracker()
    tracker2.spans = [(5, 10), (15, 20)]

    assert tracker2.logical_to_physical(0) == 0
    assert tracker2.logical_to_physical(4) == 4
    assert tracker2.logical_to_physical(10) == 5
    assert tracker2.logical_to_physical(14) == 9
    assert tracker2.logical_to_physical(20) == 10  # 20 - 5 - 5 = 10
    assert tracker2.logical_to_physical(25) == 15

    assert tracker2.physical_to_logical(0) == 0
    assert tracker2.physical_to_logical(5) == 10
    assert tracker2.physical_to_logical(9) == 14
    assert tracker2.physical_to_logical(10) == 20
    assert tracker2.physical_to_logical(15) == 25

    assert tracker2.get_compacted_token_count() == 10

    active2 = tracker2.get_active_positions(25)
    assert active2 == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]

    # --- merge_spans: 7 edge cases ---
    # Overlapping
    assert merge_spans([(0, 5), (3, 8)]) == [(0, 8)]
    # Adjacent
    assert merge_spans([(0, 5), (5, 10)]) == [(0, 10)]
    # Gap
    assert merge_spans([(0, 5), (7, 10)]) == [(0, 5), (7, 10)]
    # Out of order
    assert merge_spans([(7, 10), (0, 5)]) == [(0, 5), (7, 10)]
    # Empty
    assert merge_spans([]) == []
    # Cascading
    assert merge_spans([(0, 3), (2, 6), (5, 9)]) == [(0, 9)]
    # Containment
    assert merge_spans([(0, 10), (3, 7)]) == [(0, 10)]

    # --- Sequential add_span like the scheduler does ---
    tracker3 = CompactedSpanTracker()
    tracker3.add_span(5, 10)
    assert tracker3.spans == [(5, 10)]
    assert tracker3.get_compacted_token_count() == 5

    tracker3.add_span(15, 20)
    assert tracker3.spans == [(5, 10), (15, 20)]
    assert tracker3.get_compacted_token_count() == 10

    tracker3.add_span(10, 15)  # fills the gap
    assert tracker3.spans == [(5, 20)]
    assert tracker3.get_compacted_token_count() == 15

    # --- Round-trip property ---
    tracker_rt = CompactedSpanTracker()
    tracker_rt.spans = [(3, 7), (12, 18), (25, 30)]

    # For every active logical position, round-trip should work
    active_positions = tracker_rt.get_active_positions(40)
    for logical in active_positions:
        physical = tracker_rt.logical_to_physical(logical)
        back = tracker_rt.physical_to_logical(physical)
        assert back == logical, (
            f"Round-trip failed: logical={logical} → physical={physical} → {back}"
        )

    # --- Empty tracker ---
    empty = CompactedSpanTracker()
    assert empty.logical_to_physical(5) == 5
    assert empty.physical_to_logical(5) == 5
    assert empty.is_token_active(5)
    assert empty.get_active_positions(5) == [0, 1, 2, 3, 4]
    assert empty.get_compacted_token_count() == 0


# ---------------------------------------------------------------------------
# Test 5: Restart mode rewind
# ---------------------------------------------------------------------------

def test_restart_mode_rewind():
    # --- restart_range always includes delimiters ---
    block = BlockInfo(block_id=0, start_position=6, end_position=11,
                      summary_start=12, summary_end=15)
    assert block.is_complete

    # mask_delimiters=False: content range = (7, 11), restart range = (6, 12)
    assert block.get_content_range(mask_delimiters=False) == (7, 11)
    assert block.get_restart_range(mask_delimiters=False) == (6, 12)

    # mask_delimiters=True: content range = (6, 12), restart range = (6, 12)
    assert block.get_content_range(mask_delimiters=True) == (6, 12)
    assert block.get_restart_range(mask_delimiters=True) == (6, 12)

    # --- restart_mode sets pending_restart_rewind ---
    config = make_config(
        restart_mode=True,
        mask_delimiters=False,
        keep_last_n_blocks=0,
    )
    processor = BlockMaskingProcessor(config)
    state = processor.create_state()

    prompt = [CONTENT] * 6
    processor.process_prompt_tokens(state, prompt)
    assert state.pending_restart_rewind is None

    # Generate a complete block
    tokens = [BLOCK_START, CONTENT, CONTENT, CONTENT, CONTENT,
              BLOCK_END, SUMMARY_START, CONTENT, CONTENT, SUMMARY_END]
    compactions = run_generation(processor, state, tokens, start_pos=6)

    assert len(compactions) == 1
    # restart_mode=True → uses restart range (includes delimiters)
    start, end = compactions[0]
    assert start == 6   # block_start
    assert end == 12    # block_end + 1
    assert state.pending_restart_rewind == 12  # summary_start position

    # --- deferred + restart combo ---
    config_defer_restart = make_config(
        restart_mode=True,
        keep_last_block_for_answer=True,
        keep_last_n_blocks=0,
        mask_delimiters=False,
    )
    proc_dr = BlockMaskingProcessor(config_defer_restart)
    state_dr = proc_dr.create_state()
    proc_dr.process_prompt_tokens(state_dr, [CONTENT] * 6)

    # Block 0: deferred
    tokens_b0 = [BLOCK_START, CONTENT, CONTENT, BLOCK_END,
                 SUMMARY_START, CONTENT, SUMMARY_END]
    compactions_b0 = run_generation(proc_dr, state_dr, tokens_b0, start_pos=6)
    assert len(compactions_b0) == 0
    assert len(state_dr.pending_compactions) == 1

    # Block 1 starts → flushes block 0 with restart range
    result = proc_dr.process_token(state_dr, BLOCK_START, 13)
    assert result is not None
    start, end = result
    assert start == 6   # block_start of block 0
    assert end == 10    # block_end + 1 of block 0 (pos 9 + 1)
    assert state_dr.pending_restart_rewind == 10  # summary_start of block 0

    # --- prompt processing with restart mode ---
    config_prompt_restart = make_config(
        restart_mode=True,
        mask_delimiters=False,
        keep_last_n_blocks=0,
    )
    proc_pr = BlockMaskingProcessor(config_prompt_restart)
    state_pr = proc_pr.create_state()

    # Prompt contains a complete block
    prompt_with_block = [
        CONTENT, CONTENT,
        BLOCK_START, CONTENT, CONTENT, BLOCK_END,
        SUMMARY_START, CONTENT, SUMMARY_END,
        CONTENT,
    ]
    proc_pr.process_prompt_tokens(state_pr, prompt_with_block)

    assert len(state_pr.pending_compactions) == 1
    assert state_pr.pending_restart_rewind == 6  # summary_start in prompt

    # --- incomplete block: no restart range ---
    incomplete = BlockInfo(block_id=0, start_position=6, end_position=11)
    assert incomplete.get_restart_range(mask_delimiters=False) is None
    assert incomplete.get_content_range(mask_delimiters=False) is None
