# /// script
# requires-python = ">=3.10"
# ///
"""Tabulate torch.compile recompiles from a `TORCH_LOGS=recompiles` log, by step, function and guard-failure reason.

Each "Recompiling function F" record is followed by its guard failures ("- <frame>/<entry>: <reason>"). Reasons are
normalized so that the same kind of failure groups together: standalone numbers are replaced by N, so "size mismatch
at index 1. expected 84, actual 187" becomes "size mismatch at index N. expected N, actual N"; `--keep-numbers` turns
that off. A record's step is the last `--step-regex` match seen before it in the log (its first capture group), so
recompiles can be bucketed by training step when the program prints a step line. A program that prints the line when
a step ends, as prime-rl does (`Step 3 | ...`), shows each recompile under the previous step (`-` for the first).

`--rank R` keeps only lines of rank R: those carrying torch's `[rankR]:` log prefix (added once a process group
exists), else a torchrun `--tee` prefix such as `[default0]:` or prime-rl's `[trainer0]:` (all lines when omitted).
The step regex is matched after those prefixes are stripped, on lines of rank R or with no rank prefix. torchrun
`--tee` copies stdout and stderr separately, so print the step line to stderr, where torch logs go, or the two
interleave out of order.

Usage:
  uv run --script recompile_reasons.py LOG --rank 0 --step-regex '^step (\\d+):'
  uv run --script recompile_reasons.py LOG --rank 0 --by step,reason --top 30
  uv run --script recompile_reasons.py <run dir>/logs/attempt_1/trainer.log --rank 0 --step-regex 'Step (\\d+) \\|'
"""

import argparse
import re
from collections import Counter

RECOMPILE = re.compile(r"\[(\d+)/(\d+)\] \[__recompiles\] Recompiling function (\S+)")
REASON = re.compile(r"\[__recompiles\]\s+- (\d+)/(\d+): (.*)")
PREFIXES = re.compile(r"^(?:\[[A-Za-z_]*\d+\]:)+")
TORCH_RANK = re.compile(r"\[rank(\d+)\]:")
TEE_RANK = re.compile(r"\[[A-Za-z_]*(\d+)\]:")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument(
        "--step-regex",
        default=None,
        help="regex whose first group is the step, e.g. '^step (\\d+):'",
    )
    parser.add_argument(
        "--by",
        default="function,reason",
        help="comma list of: step, function, frame, reason",
    )
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--keep-numbers", action="store_true")
    parser.add_argument("--reason-width", type=int, default=110)
    return parser.parse_args()


def line_rank(prefixes: str) -> int | None:
    match = TORCH_RANK.search(prefixes) or TEE_RANK.search(prefixes)
    return int(match.group(1)) if match else None


def normalize(reason: str, keep_numbers: bool, width: int) -> str:
    reason = reason.split("  #")[0].strip()
    if not keep_numbers:
        reason = re.sub(r"\b\d+\b", "N", reason)
    return reason[:width]


def main():
    args = parse_args()
    step_re = re.compile(args.step_regex) if args.step_regex else None
    keys = args.by.split(",")
    counts: Counter = Counter()
    records = 0
    step, function, frame = "-", None, None
    with open(args.log, errors="replace") as f:
        for line in f:
            prefix_match = PREFIXES.match(line)
            prefixes = prefix_match.group(0) if prefix_match else ""
            body = line[len(prefixes) :].strip()
            rank = line_rank(prefixes)
            if step_re and (args.rank is None or rank in (None, args.rank)) and (m := step_re.search(body)):
                step = m.group(1)
            if args.rank is not None and rank != args.rank:
                continue
            if m := RECOMPILE.search(line):
                frame, function = m.group(1), m.group(3)
                records += 1
                continue
            if function and (m := REASON.search(line)):
                values = {
                    "step": step,
                    "function": function,
                    "frame": frame,
                    "reason": normalize(m.group(3), args.keep_numbers, args.reason_width),
                }
                counts[tuple(values[k] for k in keys)] += 1
    print(f"{records} recompile records" + (f" on rank {args.rank}" if args.rank is not None else ""))
    print("| " + " | ".join(keys) + " | guard failures |")
    print("|" + "---|" * (len(keys) + 1))
    for key, n in counts.most_common(args.top):
        print("| " + " | ".join(str(k).replace("|", "\\|") for k in key) + f" | {n} |")


if __name__ == "__main__":
    main()
