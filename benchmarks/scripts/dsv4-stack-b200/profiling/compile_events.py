# /// script
# requires-python = ">=3.10"
# ///
"""Place every rank's torch.compile recompiles and every node's JIT kernel compiles in the training step they stalled.

Paths are relative to RUN_DIR unless absolute. The defaults match a prime-rl run directory, and
`projects/<repo>.md` lists the flags a repo's runs need.
- Steps: `--metrics` (default `monitors/file/metrics.jsonl`) holds JSON lines. Each record carrying `--step-key`
  (default `step`), `--duration-key` (default `time/step`) and `--end-key` (default `time`, the step's end in epoch
  seconds) defines one step, so step k spans (end of step k-1, end of step k]. `--row-filter KEY=VALUE` (repeatable)
  keeps only matching records, for logs where several processes write the same keys under one step index.
- Dynamo: every `Recompiling function` record in the `*.log` files under `--dynamo-logs` (default
  `logs/attempt_<attempt>/trainer`), written with `TORCH_LOGS=recompiles`. The rank comes from torch's `[rankR]:`
  prefix and the time from the glog timestamp (`V1019 19:49:02.48`, local time; the year is taken from the first step).
- JIT caches: `<kind>_cache_<host>.txt` files anywhere under `--listings` (default `logs/attempt_<attempt>`), one line
  per cache entry,
  `<mtime epoch seconds> <entry name>`, written after training. Each entry's mtime is when that node first compiled
  the kernel, and `<kind>` names the column. Point each cache at a fresh per-run directory, or entries from earlier
  runs land in the listing. Listings for DeepGEMM (`kernel.*` entries only) and Triton:
    find $DG_JIT_CACHE_DIR/cache -mindepth 1 -maxdepth 1 -printf "%T@ %f\n" > deep_gemm_cache_$(hostname).txt
    find $TRITON_CACHE_DIR -mindepth 2 -maxdepth 2 -name "*.cubin" -printf "%T@ %f\n" > triton_cache_$(hostname).txt
  Triton writes one `<hash>/<kernel>.cubin` per specialization, so the Triton listing catches hand-written kernels
  that Dynamo never logs, e.g. a second variant compiled when an int argument first becomes divisible by 16. It
  also holds Inductor's generated kernels (`triton_poi_*`, `triton_red_*`, ...), which first compile in step 1.
  `--deepgemm-listing FILE[:HOST]` (repeatable) replaces the DeepGEMM listings found under `--listings`, e.g. for
  runs launched outside the `rl` launcher. A listing may also use
  `find <cache> -type f -printf "%TY-%Tm-%Td %TT %p\n"` lines; each `kernel.*` entry counts once, at its first file's
  mtime.

Prints a markdown table per step with any event (step time, Dynamo recompiles with their ranks, compiles per JIT cache
with their nodes), then `last compile step: K` over all events. Pass K + 1 as the start of `run_metrics.py --steady` to
report steady-state step time after every compile. `--events-csv` writes one row per event: kind, step, time, where
(rank or host), what (function and first guard failure, or cache entry).

Usage:
  uv run --script compile_events.py <run dir>
  uv run --script compile_events.py <run dir> --attempt 1 --events-csv events.csv
  uv run --script compile_events.py <run dir> --metrics metrics.jsonl --duration-key step_seconds \\
    --end-key wall_time --dynamo-logs logs --listings caches --row-filter role=trainer
  uv run --script compile_events.py <run dir> --dynamo-logs torchrun --deepgemm-listing listing.txt:node0
"""

import argparse
import csv
import json
import re
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

RECOMPILE = re.compile(r"\[rank(\d+)\]:[VIWE](\d{4}) (\d\d:\d\d:\d\d\.\d+) .*\[__recompiles\] Recompiling function (\S+)")
REASON = re.compile(r"\[__recompiles\]\s+- \d+/\d+: (.*)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attempt", type=int, default=1, help="fills <attempt> in the default log paths")
    parser.add_argument("--metrics", type=Path, default=Path("monitors/file/metrics.jsonl"))
    parser.add_argument("--step-key", default="step")
    parser.add_argument("--duration-key", default="time/step")
    parser.add_argument("--end-key", default="time", help="the step's end time in epoch seconds")
    parser.add_argument("--row-filter", action="append", default=[], metavar="KEY=VALUE",
                        help="keep only metrics records whose KEY equals VALUE (repeatable)")
    parser.add_argument("--dynamo-logs", type=Path, default=None, help="default logs/attempt_<attempt>/trainer")
    parser.add_argument("--listings", type=Path, default=None, help="default logs/attempt_<attempt>")
    parser.add_argument("--deepgemm-listing", action="append", default=[], metavar="FILE[:HOST]",
                        help="extra DeepGEMM cache listing; replaces the deep_gemm_cache_<host>.txt search when given")
    parser.add_argument("--events-csv", type=Path, default=None)
    args = parser.parse_args()
    args.row_filter = [tuple(spec.split("=", 1)) for spec in args.row_filter]
    return args


def load_steps(path: Path, step_key: str, end_key: str, duration_key: str,
               row_filters: list[tuple[str, str]]) -> list[tuple[int, float, float]]:
    """(step, end epoch seconds, step seconds), sorted by step, from the records passing every row filter."""
    steps = {}
    for line in path.open():
        record = json.loads(line)
        if duration_key in record and all(str(record.get(key)) == value for key, value in row_filters):
            steps[record[step_key]] = (record[end_key], record[duration_key])
    return [(step, end, seconds) for step, (end, seconds) in sorted(steps.items())]


def glog_epoch(mmdd: str, clock: str, year: int) -> float:
    whole, _, fraction = clock.partition(".")
    struct = time.strptime(f"{year}{mmdd} {whole}", "%Y%m%d %H:%M:%S")
    return time.mktime(struct) + float(f"0.{fraction}")


def dynamo_events(log_dir: Path, year: int) -> list[tuple[float, str, str]]:
    """(epoch seconds, rank, function: first guard failure) per recompile, deduplicated across tee'd copies."""
    events = set()
    for log in sorted(log_dir.rglob("*.log")):
        pending = None
        for line in log.open(errors="replace"):
            if match := RECOMPILE.search(line):
                pending = (glog_epoch(match.group(2), match.group(3), year), f"rank{match.group(1)}", match.group(4))
            elif pending and (match := REASON.search(line)):
                events.add((pending[0], pending[1], f"{pending[2]}: {match.group(1).split('  #')[0].strip()}"))
                pending = None
    return sorted(events)


def listing_entries(listing: Path, kind: str) -> list[tuple[float, str]]:
    """(mtime epoch seconds, entry) from `<epoch> <entry>` or `<YYYY-MM-DD> <HH:MM:SS.frac> <path>` lines."""
    entries = []
    for line in listing.open():
        fields = line.split()
        if len(fields) >= 3 and re.fullmatch(r"\d{4}-\d\d-\d\d", fields[0]):
            whole, _, fraction = fields[1].partition(".")
            epoch = time.mktime(time.strptime(f"{fields[0]} {whole}", "%Y-%m-%d %H:%M:%S")) + float(f"0.{fraction or 0}")
            path = fields[2]
        elif len(fields) == 2:
            epoch, path = float(fields[0]), fields[1]
        else:
            continue
        entries.append((epoch, path))
    if kind != "deep_gemm":
        return entries
    first = {}
    for epoch, path in entries:
        entry = next((part for part in Path(path).parts if part.startswith("kernel.")), None)
        if entry is not None:
            first[entry] = min(epoch, first.get(entry, epoch))
    return [(epoch, entry) for entry, epoch in first.items()]


def cache_events(listings: list[tuple[Path, str, str]]) -> list[tuple[str, float, str, str]]:
    """(kind, mtime epoch seconds, host, entry) per entry of every (listing, kind, host)."""
    events = []
    for listing, kind, host in listings:
        events += [(kind, epoch, host, entry) for epoch, entry in listing_entries(listing, kind)]
    return sorted(events)


def main():
    args = parse_args()
    steps = load_steps(args.run_dir / args.metrics, args.step_key, args.end_key, args.duration_key, args.row_filter)
    ends = [end for _, end, _ in steps]
    year = time.localtime(ends[0]).tm_year
    attempt_dir = Path("logs") / f"attempt_{args.attempt}"
    dynamo_dir = args.run_dir / (args.dynamo_logs or attempt_dir / "trainer")
    listings_dir = args.run_dir / (args.listings or attempt_dir)

    def step_of(epoch: float) -> int:
        return steps[min(bisect_left(ends, epoch), len(steps) - 1)][0]

    listings = []
    for path in sorted(listings_dir.rglob("*_cache_*.txt")):
        kind, _, host = path.stem.rpartition("_cache_")
        if not (args.deepgemm_listing and kind == "deep_gemm"):
            listings.append((path, kind, host))
    for spec in args.deepgemm_listing:
        path, _, host = spec.partition(":")
        listings.append((args.run_dir / path, "deep_gemm", host or "node"))
    events = [("dynamo", *e) for e in dynamo_events(dynamo_dir, year)]
    if not events:
        print(f"note: no `Recompiling function` records under {dynamo_dir} (logs missing, or not run with "
              "TORCH_LOGS=recompiles); Dynamo counts below are not evidence of zero recompiles.\n")
    if not listings:
        print("note: no JIT cache listing found; the missing compile columns are not evidence of zero compiles.\n")
    events += cache_events(listings)
    cache_kinds = sorted({kind for kind, *_ in events} - {"dynamo"})
    by_step: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    rows = []
    for kind, epoch, where, what in events:
        step = step_of(epoch)
        by_step[step][kind].append(where)
        rows.append((kind, step, f"{epoch:.3f}", where, what))

    seconds = {step: s for step, _, s in steps}
    cache_headers = "".join(f" {kind} compiles | nodes |" for kind in cache_kinds)
    print("| step | step time (s) | Dynamo recompiles | ranks |" + cache_headers)
    print("|---:|---:|---:|---|" + "---:|---|" * len(cache_kinds))
    for step in sorted(by_step):
        dynamo = by_step[step]["dynamo"]
        ranks = ",".join(sorted(set(dynamo), key=lambda r: int(r[4:])))
        cells = "".join(f" {len(by_step[step][k])} | {','.join(sorted(set(by_step[step][k])))} |" for k in cache_kinds)
        print(f"| {step} | {seconds[step]:.2f} | {len(dynamo)} | {ranks} |{cells}")
    totals = [f"{sum(k == 'dynamo' for k, *_ in rows)} Dynamo recompiles"]
    totals += [f"{sum(k == kind for k, *_ in rows)} {kind} compiles" for kind in cache_kinds]
    print(f"\n{len(steps)} steps, " + ", ".join(totals))
    print(f"last compile step: {max(by_step) if by_step else 0}")

    if args.events_csv:
        with args.events_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["kind", "step", "epoch", "where", "what"])
            writer.writerows(sorted(rows, key=lambda r: (r[1], r[0], r[3])))


if __name__ == "__main__":
    main()
