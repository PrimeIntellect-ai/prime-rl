# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Position mapping for compacted token spans.

Extracted from Memento's vLLM v1/request.py. Provides O(log n)
logical<->physical position conversion after KV cache compaction.

This is used by the overlay Request class (compacted_spans property)
and can be tested standalone.
"""

import bisect


class CompactedSpanTracker:
    """Tracks compacted (removed) token spans and provides position mapping.

    Maintains sorted, non-overlapping spans and precomputed auxiliary arrays
    for O(log n) binary-search position lookups.
    """

    def __init__(self) -> None:
        self._spans: list[tuple[int, int]] = []
        self._cs_starts: list[int] = []
        self._cs_cumulative: list[int] = []
        self._cs_gap_physical: list[int] = [0]
        self._cs_gap_logical: list[int] = [0]

    @property
    def spans(self) -> list[tuple[int, int]]:
        return self._spans

    @spans.setter
    def spans(self, spans: list[tuple[int, int]]) -> None:
        self._spans = spans
        self._recompute()

    def _recompute(self) -> None:
        """Recompute auxiliary arrays from current spans."""
        starts: list[int] = []
        cumulative: list[int] = []
        gap_physical: list[int] = [0]
        gap_logical: list[int] = [0]
        total = 0
        for s, e in self._spans:
            starts.append(s)
            total += e - s
            cumulative.append(total)
            gap_physical.append(e - total)
            gap_logical.append(e)
        self._cs_starts = starts
        self._cs_cumulative = cumulative
        self._cs_gap_physical = gap_physical
        self._cs_gap_logical = gap_logical

    def add_span(self, start: int, end: int) -> None:
        """Add a span and merge with existing overlapping/adjacent spans."""
        if start >= end:
            return
        new_spans = merge_spans(self._spans + [(start, end)])
        self.spans = new_spans

    def is_token_active(self, position: int) -> bool:
        """Check if a token is active (not compacted/removed). O(log n)."""
        if not self._cs_starts:
            return True
        idx = bisect.bisect_right(self._cs_starts, position) - 1
        if idx < 0:
            return True
        return position >= self._spans[idx][1]

    def get_active_positions(self, num_tokens: int) -> list[int]:
        """Get list of active (non-compacted) logical positions."""
        if not self._spans:
            return list(range(num_tokens))
        active: list[int] = []
        prev_end = 0
        for start, end in self._spans:
            if start >= num_tokens:
                break
            active.extend(range(prev_end, min(start, num_tokens)))
            prev_end = end
        if prev_end < num_tokens:
            active.extend(range(prev_end, num_tokens))
        return active

    def logical_to_physical(self, logical_pos: int) -> int:
        """Convert a logical position to physical position after compaction. O(log n)."""
        if not self._cs_starts:
            return logical_pos
        idx = bisect.bisect_right(self._cs_starts, logical_pos) - 1
        if idx < 0:
            return logical_pos
        _, end = self._spans[idx]
        if logical_pos >= end:
            return logical_pos - self._cs_cumulative[idx]
        prev_cum = self._cs_cumulative[idx - 1] if idx > 0 else 0
        return logical_pos - prev_cum - (logical_pos - self._cs_starts[idx])

    def physical_to_logical(self, physical_pos: int) -> int:
        """Convert a physical position back to logical position. O(log n)."""
        if not self._cs_gap_physical:
            return physical_pos
        idx = bisect.bisect_right(self._cs_gap_physical, physical_pos) - 1
        return (self._cs_gap_logical[idx]
                + (physical_pos - self._cs_gap_physical[idx]))

    def get_compacted_token_count(self) -> int:
        """Get total number of tokens that have been compacted."""
        return self._cs_cumulative[-1] if self._cs_cumulative else 0


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent spans into sorted, non-overlapping list."""
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
