"""Convert a run's recorded episodes into packed trainer batches for `trainer.data.replay`.

Reads trace stream chunks (`<output_dir>/monitors/file/traces/stream/*.jsonl.zst`) in the order
given, keeps the trainable traces of successful train episodes, and packs every
`--rollouts-per-step` of them into one step exactly as the orchestrator does (see
`prime_rl.orchestrator.replay` for which fields are recorded and which are synthesized). Each step
is written as one file per data rank, beside a `meta.json` the trainer checks its topology
against.

Usage (from the prime-rl repo):
    uv run python tools/convert_traces_to_replay_batches.py <model> <output_dir> <stream chunk>...
        --seq-len 262144 --rollouts-per-step 1024 [--num-train-workers 1]
        [--pad-to-multiple-of 1] [--num-steps N]

`<model>` names the trained model (HF id or path): micro batches are balanced across data ranks by
its training FLOPs, like the orchestrator does. Match `--num-train-workers` to the trainer's data
parallel size and `--pad-to-multiple-of` to the orchestrator's (the rl entrypoint sets it to
`model.cp`, or `32 * model.cp` with MXFP8 quantization). A run that set
`orchestrator.pad_to_seq_len` is reproduced with `--pad-to-multiple-of <seq_len>`.
"""

import argparse
import statistics
from itertools import islice
from pathlib import Path

import torch

from prime_rl.configs.trainer import ModelConfig
from prime_rl.orchestrator.replay import read_records, replay_batches
from prime_rl.trainer.model import get_model
from prime_rl.trainer.perf import FlopCounter
from prime_rl.transports.batch.replay import ReplayMeta, write_replay_meta, write_replay_step
from prime_rl.transports.batch.types import MicroBatch


def flop_counter(model_name: str, trust_remote_code: bool) -> FlopCounter:
    """The orchestrator's bin cost: training FLOPs of the model, built on the meta device."""
    # The default sparse-attention backend refuses to build without a GPU. Backends select kernels, not modules.
    model_config = ModelConfig(
        name=model_name, attn="flash_attention_3", dsa_backend="tilelang", trust_remote_code=trust_remote_code
    )
    return FlopCounter(get_model(model_config, device=torch.device("meta")))


def step_summary(step: int, micro_batch_grid: list[list[MicroBatch]]) -> str:
    micro_batches = [micro_batch for rank in micro_batch_grid for micro_batch in rank]
    row_lengths = [len(micro_batch.input_ids) for micro_batch in micro_batches]
    padding = sum(micro_batch.padding for micro_batch in micro_batches)
    docs = [len(micro_batch.sequence_lengths) for micro_batch in micro_batches if micro_batch.trace_ids is not None]
    return (
        f"step {step}: {sum(row_lengths)} tokens ({padding} padding), "
        f"{len(micro_batch_grid[0])} micro batches per rank, "
        f"row length min/median/max {min(row_lengths)}/{int(statistics.median(row_lengths))}/{max(row_lengths)}, "
        f"{statistics.mean(docs):.1f} docs per row"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="trained model (HF id or path), for the FLOP-balanced rank assignment")
    parser.add_argument("output_dir", type=Path, help="replay directory to write")
    parser.add_argument("streams", type=Path, nargs="+", help="trace stream chunks, in order")
    parser.add_argument("--seq-len", type=int, required=True, help="tokens per packed micro batch")
    parser.add_argument("--rollouts-per-step", type=int, required=True, help="trainable traces per step")
    parser.add_argument("--num-train-workers", type=int, default=1, help="trainer data parallel size")
    parser.add_argument("--pad-to-multiple-of", type=int, default=1, help="pad each micro batch to a multiple of this")
    parser.add_argument("--num-steps", type=int, help="stop after this many steps (default: all full steps)")
    parser.add_argument("--trust-remote-code", action="store_true", help="trust remote code when building the model")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"replay: {args.output_dir} is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches = replay_batches(
        read_records(args.streams),
        rollouts_per_step=args.rollouts_per_step,
        seq_len=args.seq_len,
        num_train_workers=args.num_train_workers,
        bin_cost=flop_counter(args.model, args.trust_remote_code),
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
    num_steps = 0
    for step, micro_batch_grid in enumerate(islice(batches, args.num_steps)):
        write_replay_step(args.output_dir, step, micro_batch_grid)
        print(f"replay: {step_summary(step, micro_batch_grid)}", flush=True)
        num_steps += 1
    if num_steps == 0:
        raise SystemExit("replay: the streams hold fewer trainable traces than one step")
    write_replay_meta(
        args.output_dir,
        ReplayMeta(
            num_steps=num_steps,
            seq_len=args.seq_len,
            num_train_workers=args.num_train_workers,
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
    )
    print(f"replay: wrote {num_steps} step(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
