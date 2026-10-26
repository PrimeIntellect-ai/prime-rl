"""Count Dynamo graph breaks, recompiles and CP guard failures per rank from the TORCH_LOGS output of A/B runs."""

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

GLOG = re.compile(
    r"\[rank(?P<rank>\d+)\]:(?P<level>[VIWEC])(?P<mmdd>\d{4}) (?P<clock>\d\d:\d\d:\d\d\.\d+) (?P<pid>\d+) "
    r"(?P<src>\S+?:\d+)\](?P<trace> \[[^\]]*\])?(?P<artifact> \[__\w+\])? ?(?P<body>.*)$"
)
FRAME = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<function>\S+)')
BREAK_START = re.compile(
    r"^Graph break (?:\(user stack suppressed due to duplicate graph break\) )?in user code at (\S+)"
)
GUARD = re.compile(r"^\s*- \d+/\d+: (.*)")
RECOMPILE_START = re.compile(r"^Recompiling function (\S+) in (\S+)")
GB_ID = re.compile(r"/gb/(gb\d+)\.html")
ADDRESS = re.compile(r" at 0x[0-9a-f]+")
DEFAULT_FRAGMENTS = [
    "qwen3_8_flash_next/position_learning.py",
    "qwen3_8_flash_next/ngram_embedding.py",
    "distributed/embedding_parallel.py",
    "layers/ngram.py",
    "distributed/ngram_parallel.py",
    "torch/distributed/fsdp",
]
DEFAULT_CP_PATTERN = r"cu_seqlens|seqlen|seq_len|cp_context|cp_group|cp_rank|cp_world"
ROW_WIDTH = 120


@dataclass
class Record:
    rank: int
    epoch: float
    artifact: str | None
    lines: list[str] = field(default_factory=list)


@dataclass
class Run:
    label: str
    out_dir: Path
    run_dir: Path
    meta: dict


def resolve_run(path: Path) -> Run:
    path = path.resolve()
    meta_path = path / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run_dir = Path(meta["run_dir"]) if meta else path
    label = f"{meta['arm']}/{meta['cell_dir']}/r{meta['repeat']}" if meta else path.name
    return Run(label=label, out_dir=path, run_dir=run_dir, meta=meta)


def rank_stderr_logs(run_dir: Path) -> dict[int, Path]:
    logs = {int(p.parent.name): p for p in run_dir.glob("logs/attempt_*/trainer/torchrun/*/attempt_*/*/stderr.log")}
    return dict(sorted(logs.items()))


def run_year(run: Run) -> int:
    start = run.meta.get("start_time") or run.run_dir.stat().st_mtime
    return time.localtime(start).tm_year


def glog_epoch(mmdd: str, clock: str, year: int) -> float:
    whole, _, fraction = clock.partition(".")
    return time.mktime(time.strptime(f"{year}{mmdd} {whole}", "%Y%m%d %H:%M:%S")) + float(f"0.{fraction}")


def read_records(path: Path, year: int) -> list[Record]:
    records: list[Record] = []
    current_key = None
    for raw in path.open(errors="replace"):
        match = GLOG.search(raw.rstrip("\n"))
        if not match:
            current_key = None
            continue
        body = match["body"]
        key = match.group(0)[: match.start("body") - match.start(0)]
        starts_record = BREAK_START.match(body) or RECOMPILE_START.match(body)
        if key != current_key or starts_record:
            artifact = match["artifact"].strip()[3:-1] if match["artifact"] else None
            records.append(
                Record(
                    rank=int(match["rank"]),
                    epoch=glog_epoch(match["mmdd"], match["clock"], year),
                    artifact=artifact,
                )
            )
            current_key = key
        records[-1].lines.append(body)
    return records


def parse_graph_break(record: Record) -> dict:
    location = BREAK_START.match(record.lines[0]).group(1)
    reason_lines = []
    files = [location.rsplit(":", 1)[0]]
    in_traceback = False
    for line in record.lines[1:]:
        if line.startswith("User code traceback:"):
            in_traceback = True
        elif in_traceback:
            if frame := FRAME.match(line):
                files.append(frame["file"])
        elif line.startswith("Graph Break Reason: "):
            reason_lines.append(line.removeprefix("Graph Break Reason: ").strip())
        elif reason_lines and line.strip():
            reason_lines.append(line.strip())
    return {
        "rank": record.rank,
        "duplicate": "user stack suppressed" in record.lines[0],
        "location": location,
        "reason": summarize_reason(reason_lines),
        "files": files,
    }


def summarize_reason(lines: list[str]) -> str:
    fields = {}
    gb_type = None
    for line in lines[1:]:
        key, _, value = line.partition(": ")
        if key in ("Higher Order Operator", "Explanation", "Hint", "Developer debug context"):
            fields[key] = value
        elif (match := GB_ID.search(line)) and "gb_id" not in fields:
            fields["gb_id"] = match.group(1)
        elif gb_type is None:
            gb_type = line
    gb_type = gb_type if lines and lines[0].startswith("Encountered graph break") else (lines[0] if lines else "?")
    reason = " ".join(part for part in (fields.get("gb_id"), gb_type) if part)
    if "Higher Order Operator" in fields:
        reason += f" in {fields['Higher Order Operator']}"
    if "Developer debug context" in fields:
        reason += f" ({ADDRESS.sub('', fields['Developer debug context'])})"
    return reason


def parse_recompile(record: Record) -> dict:
    start = RECOMPILE_START.match(record.lines[0])
    guards = [m.group(1).split("  #")[0].strip() for line in record.lines[1:] if (m := GUARD.match(line))]
    return {
        "rank": record.rank,
        "epoch": record.epoch,
        "function": start.group(1),
        "where": start.group(2),
        "guards": guards,
    }


def run_records(run: Run) -> dict[int, list[Record]]:
    year = run_year(run)
    return {rank: read_records(path, year) for rank, path in rank_stderr_logs(run.run_dir).items()}


def recompile_events(run: Run) -> list[dict]:
    events = []
    for records in run_records(run).values():
        events += [
            parse_recompile(r) for r in records if r.artifact == "recompiles" and RECOMPILE_START.match(r.lines[0])
        ]
    return sorted(events, key=lambda event: (event["epoch"], event["rank"]))


def recompile_limit_hits(records: list[Record]) -> int:
    return sum(any("recompile_limit" in line or "cache_size_limit" in line for line in r.lines) for r in records)


def normalize(text: str) -> str:
    return re.sub(r"\b\d+\b", "N", text).replace("|", "/")


def short_path(location: str) -> str:
    return location.split("/src/")[-1].split("site-packages/")[-1]


def table(headers: list[str], rows: list[list], right: set[int] | None = None) -> str:
    right = right if right is not None else set(range(1, len(headers)))
    cells = [[str(c) for c in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def fmt(row):
        return (
            "| "
            + " | ".join(c.rjust(w) if i in right else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths)))
            + " |"
        )

    rule = "| " + " | ".join(("-" * (w - 1) + ":") if i in right else "-" * w for i, w in enumerate(widths)) + " |"
    return "\n".join([fmt(cells[0]), rule, *(fmt(row) for row in cells[1:])])


def summarize(run: Run, fragments: list[str], cp_pattern: re.Pattern) -> dict:
    per_rank = {}
    breaks_all, recompiles_all = [], []
    for rank, records in run_records(run).items():
        breaks = [
            parse_graph_break(r) for r in records if r.artifact == "graph_breaks" and BREAK_START.match(r.lines[0])
        ]
        recompiles = [
            parse_recompile(r) for r in records if r.artifact == "recompiles" and RECOMPILE_START.match(r.lines[0])
        ]
        per_rank[rank] = {
            "graph_breaks": len(breaks),
            "graph_breaks_duplicate": sum(b["duplicate"] for b in breaks),
            "graph_breaks_touching": sum(any(f in path for f in fragments for path in b["files"]) for b in breaks),
            "recompiles": len(recompiles),
            "cp_guard_failures": sum(bool(cp_pattern.search(g)) for r in recompiles for g in r["guards"]),
            "recompile_limit_hits": recompile_limit_hits(records),
        }
        breaks_all += breaks
        recompiles_all += recompiles
    return {
        "label": run.label,
        "out_dir": str(run.out_dir),
        "head": run.meta.get("head"),
        "torch_logs": run.meta.get("env_overrides", {}).get("TORCH_LOGS"),
        "per_rank": per_rank,
        "fragments": {f: sum(any(f in path for path in b["files"]) for b in breaks_all) for f in fragments},
        "fragments_any": sum(any(f in path for f in fragments for path in b["files"]) for b in breaks_all),
        "reasons": Counter(normalize(b["reason"]) for b in breaks_all),
        "locations": Counter(short_path(b["location"]) for b in breaks_all),
        "guards": Counter(normalize(g) for r in recompiles_all for g in r["guards"]),
        "cp_guards": Counter(normalize(g) for r in recompiles_all for g in r["guards"] if cp_pattern.search(g)),
    }


def counter_table(title: str, column: str, summaries: list[dict], key: str, short: list[str]) -> str:
    keys = sorted(set().union(*(s[key] for s in summaries)), key=lambda k: -sum(s[key][k] for s in summaries))
    if not keys:
        return f"{title}: none\n"
    width = max(40, ROW_WIDTH - 4 - sum(max(len(name), 5) + 3 for name in short))
    rows = [[k[:width], *(s[key][k] for s in summaries)] for k in keys]
    return (
        f"{title}, records over all ranks:\n\n"
        + table([column, *short], rows, right=set(range(1, len(short) + 1)))
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run output dirs (with run.json) or prime-rl run dirs")
    parser.add_argument("--fragments", nargs="+", default=DEFAULT_FRAGMENTS, help="path fragments to match user stacks")
    parser.add_argument("--cp-pattern", default=DEFAULT_CP_PATTERN, help="regex marking a guard failure CP-related")
    parser.add_argument("--json", type=Path, default=None, help="also write the summaries as JSON")
    args = parser.parse_args()

    cp_pattern = re.compile(args.cp_pattern)
    summaries = [summarize(resolve_run(path), args.fragments, cp_pattern) for path in args.runs]
    short = [f"run {i}" for i in range(len(summaries))]

    print("Runs:\n")
    print(
        table(
            ["run", "label", "head", "TORCH_LOGS"],
            [[short[i], s["label"], (s["head"] or "?")[:10], s["torch_logs"]] for i, s in enumerate(summaries)],
            right=set(),
        )
    )
    print(
        "\nPer rank. Breaks: graph break records. Dup: those logged with the user stack suppressed as a repeat. Touch "
        "frag.: breaks whose location or user stack touches a fragment below. Recompiles: 'Recompiling function' "
        f"records. Seqlen guards: guard failures matching /{args.cp_pattern}/, the CP parameters but also any varlen "
        "recompile at cp=1. Limit hits: recompile_limit warnings. Fewer is better in every column.\n"
    )
    rows = []
    for i, s in enumerate(summaries):
        for rank, c in s["per_rank"].items():
            rows.append(
                [
                    short[i],
                    rank,
                    c["graph_breaks"],
                    c["graph_breaks_duplicate"],
                    c["graph_breaks_touching"],
                    c["recompiles"],
                    c["cp_guard_failures"],
                    c["recompile_limit_hits"],
                ]
            )
    print(table(["run", "rank", "breaks", "dup", "touch frag.", "recompiles", "seqlen guards", "limit hits"], rows))
    print("\nBreak records whose location or user stack touches each fragment (all ranks):\n")
    frag_rows = [[f, *(s["fragments"][f] for s in summaries)] for f in args.fragments]
    frag_rows.append(["any fragment", *(s["fragments_any"] for s in summaries)])
    print(table(["fragment", *short], frag_rows))
    print()
    print(counter_table("Graph break reasons (numbers as N)", "reason", summaries, "reasons", short))
    print(counter_table("Graph break locations", "location", summaries, "locations", short))
    print(counter_table("Recompile guard failures (numbers as N)", "guard failure", summaries, "guards", short))
    print(
        counter_table(
            "Sequence-length and CP guard failures (numbers as N)", "guard failure", summaries, "cp_guards", short
        )
    )

    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2, default=dict) + "\n")


if __name__ == "__main__":
    main()
