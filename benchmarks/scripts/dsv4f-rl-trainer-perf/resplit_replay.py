"""Cut a dp-8 replay set to a fixed number of micro batches per rank and re-split the same micro batches for dp 16."""

import argparse
from pathlib import Path

from prime_rl.transports.batch.replay import (
    ReplayMeta,
    read_replay_meta,
    read_replay_step,
    write_replay_meta,
    write_replay_step,
)


def num_tokens(micro_batch) -> int:
    return sum(micro_batch.sequence_lengths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dp8", type=Path)
    parser.add_argument("out_dp16", type=Path)
    parser.add_argument("--micro-batches-per-dp8-rank", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    meta = read_replay_meta(args.source)
    assert meta.num_train_workers == 8, meta
    assert args.micro_batches_per_dp8_rank % 2 == 0
    half = args.micro_batches_per_dp8_rank // 2

    for directory, workers in ((args.out_dp8, 8), (args.out_dp16, 16)):
        directory.mkdir(parents=True, exist_ok=False)
        write_replay_meta(
            directory,
            ReplayMeta(
                num_steps=args.num_steps,
                seq_len=meta.seq_len,
                num_train_workers=workers,
                pad_to_multiple_of=meta.pad_to_multiple_of,
            ),
        )

    for step in range(args.num_steps):
        dp8_grid = []
        for rank in range(8):
            micro_batches = read_replay_step(args.source, step, rank)
            assert len(micro_batches) >= args.micro_batches_per_dp8_rank, (step, rank, len(micro_batches))
            dp8_grid.append(micro_batches[: args.micro_batches_per_dp8_rank])
        dp16_grid = [part for micro_batches in dp8_grid for part in (micro_batches[:half], micro_batches[half:])]
        write_replay_step(args.out_dp8, step, dp8_grid)
        write_replay_step(args.out_dp16, step, dp16_grid)
        tokens = sum(num_tokens(mb) for micro_batches in dp8_grid for mb in micro_batches)
        print(f"step {step}: {8 * args.micro_batches_per_dp8_rank} micro batches, {tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
