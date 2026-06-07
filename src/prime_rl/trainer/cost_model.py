"""Per-sequence forward-cost proxy used to balance packed micro-batches.

In a packed micro-batch with sequence-masked attention, each sequence
attends only within itself, so attention is O(n^2) per sequence while
linear ops (QKV proj, FFN, attn-out) are O(n). FFD packs bins to
~max_seq_len, so the linear term is approximately constant across bins
and inter-bin work variance is dominated by attention.

`bin_cost` returns just the n^2 term. It ranks bins correctly for
standard MHA / GQA setups. Add a model-aware estimator here only if a
specific model shows measured wallclock skew that this proxy misses.
"""

from collections.abc import Iterable


def bin_cost(seqlens: Iterable[int]) -> int:
    return sum(n * n for n in seqlens)
