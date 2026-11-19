"""Bin packing."""

import bisect

import numpy as np


def fill_rows(lengths: list[int] | np.ndarray, num_rows: int, capacity: int) -> np.ndarray:
    """Items of ``num_rows * capacity`` tokens in all, none longer than ``capacity``, in ``num_rows`` full rows of
    ``capacity``: longest first, each into the fullest row with room for it (best fit decreasing; Ding et al., 2024),
    then the items that fit nowhere cut to fill the gaps, the largest gap first. Returns the pieces row by row, in the
    order placed: the row, the item, the offset in it and the length. Rows are found by their free space, which has at
    most ``capacity`` values."""
    lengths = np.asarray(lengths, dtype=np.int64)
    by_space = {capacity: list(range(num_rows))[::-1]}  # free space -> the rows with that much free space
    spaces = [capacity]  # the keys of by_space, sorted

    def take(k: int) -> tuple[int, int]:
        space = spaces[k]
        row = by_space[space].pop()
        if not by_space[space]:
            del by_space[space], spaces[k]
        return row, space

    def give(row: int, space: int) -> None:
        if space:
            if space not in by_space:
                by_space[space] = []
                bisect.insort(spaces, space)
            by_space[space].append(row)

    order = np.argsort(-lengths, kind="stable")
    rows, left = [], []  # the row of every item in order, -1 if it fits nowhere
    for item, size in zip(order.tolist(), lengths[order].tolist()):
        k = bisect.bisect_left(spaces, size)
        if k == len(spaces):
            rows.append(-1)
            left.append(item)
            continue
        row, space = take(k)
        rows.append(row)
        give(row, space - size)
    cuts = []
    for item in left:
        offset, size = 0, int(lengths[item])
        while offset < size:
            row, space = take(len(spaces) - 1)
            length = min(space, size - offset)
            cuts.append((row, item, offset, length))
            give(row, space - length)
            offset += length
    rows = np.array(rows, dtype=np.int64)
    whole = order[rows >= 0]
    pieces = np.concatenate(
        [
            np.stack([rows[rows >= 0], whole, np.zeros_like(whole), lengths[whole]], axis=1),
            np.array(cuts, dtype=np.int64).reshape(-1, 4),
        ]
    )
    return pieces[np.argsort(pieces[:, 0], kind="stable")]
