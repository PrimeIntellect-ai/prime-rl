"""Parse per-push timing and RDMA BW from a prime-rl-mx-on-nixl run log.

Extracts the per-push `[nixl rank=...]` lines emitted by PI's TransportPlan
and the `[mx-rendezvous]` lines added by the MX overlay. Aggregates:

- Per-step `update_weights` wall time
- Wire BW + net BW per rank
- Rendezvous discovery latency (MX scenarios only)
- Pipeline-replication source counts (scenario C only)

Usage:
    python parse_mx_metrics.py --log /path/to/trainer.log [--json out.json]
    kubectl -n kavin logs prime-rl-mx-on-nixl-trainer-0 | python parse_mx_metrics.py --stdin
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Iterable, TextIO


# Regex patterns for PI's nixl push log line. Example:
# [nixl rank=0] push bytes=3553.26MB handles=339 convert=120.10ms post+wait=560.22ms
#   barrier=82.33ms total=762.65ms wire_bw=7.12GB/s net_bw=6.34GB/s
_PUSH_RE = re.compile(
    r"\[nixl rank=(?P<rank>\d+)\]\s+push\s+"
    r"bytes=(?P<bytes_mb>[\d.]+)MB\s+"
    r"handles=(?P<handles>\d+)\s+"
    r"convert=(?P<convert_ms>[\d.]+)ms\s+"
    r"post\+wait=(?P<postwait_ms>[\d.]+)ms\s+"
    r"barrier=(?P<barrier_ms>[\d.]+)ms\s+"
    r"total=(?P<total_ms>[\d.]+)ms\s+"
    r"wire_bw=(?P<wire_gbps>[\d.]+)GB/s\s+"
    r"net_bw=(?P<net_gbps>[\d.]+)GB/s"
)

_MX_DISCOVER_RE = re.compile(
    r"\[mx-rendezvous\].*?(?:role=|)(?P<role>trainer|inference).*?"
    r"rank=(?P<rank>\d+).*?discovered coordinator=(?P<endpoint>[^\s]+).*?after (?P<polls>\d+)\s+polls"
)

_MX_COORDINATOR_RE = re.compile(
    r"\[mx-rendezvous\] coordinator published source_id=(?P<source_id>\S+)\s+endpoint=(?P<endpoint>\S+)"
)

_MX_PIPELINE_RE = re.compile(
    r"\[mx-rendezvous\] published rollout-as-source rank=(?P<rank>\d+)"
)


@dataclass
class PushSample:
    rank: int
    bytes_mb: float
    handles: int
    convert_ms: float
    postwait_ms: float
    barrier_ms: float
    total_ms: float
    wire_gbps: float
    net_gbps: float


@dataclass
class MetricsReport:
    pushes: list[PushSample] = field(default_factory=list)
    mx_discovered: list[dict] = field(default_factory=list)
    mx_coordinator: dict | None = None
    mx_pipeline_publishes: int = 0

    def summary(self) -> dict:
        if not self.pushes:
            return {"pushes": 0, "note": "no nixl push lines found"}

        per_rank: dict[int, list[PushSample]] = {}
        for p in self.pushes:
            per_rank.setdefault(p.rank, []).append(p)

        def _stats(values: list[float]) -> dict:
            if not values:
                return {}
            return {
                "count": len(values),
                "mean": round(statistics.mean(values), 3),
                "median": round(statistics.median(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            }

        report = {"num_pushes": len(self.pushes), "ranks": {}}
        for rank, samples in sorted(per_rank.items()):
            report["ranks"][rank] = {
                "total_ms": _stats([s.total_ms for s in samples]),
                "wire_gbps": _stats([s.wire_gbps for s in samples]),
                "net_gbps": _stats([s.net_gbps for s in samples]),
                "convert_ms": _stats([s.convert_ms for s in samples]),
                "barrier_ms": _stats([s.barrier_ms for s in samples]),
                "bytes_mb": samples[0].bytes_mb,
                "handles": samples[0].handles,
            }

        # Cluster-level aggregate: mean total_ms across all ranks (the
        # bottleneck rank dominates, but the mean is a useful summary).
        report["aggregate"] = {
            "mean_total_ms": round(
                statistics.mean([s.total_ms for s in self.pushes]), 3
            ),
            "mean_wire_gbps": round(
                statistics.mean([s.wire_gbps for s in self.pushes]), 3
            ),
        }

        if self.mx_coordinator:
            report["mx_coordinator"] = self.mx_coordinator
        if self.mx_discovered:
            report["mx_discovered"] = {
                "count": len(self.mx_discovered),
                "avg_polls": round(
                    statistics.mean([d["polls"] for d in self.mx_discovered]), 1
                ),
            }
        if self.mx_pipeline_publishes:
            report["mx_pipeline_publishes"] = self.mx_pipeline_publishes

        return report


def parse_lines(lines: Iterable[str]) -> MetricsReport:
    report = MetricsReport()
    for line in lines:
        m = _PUSH_RE.search(line)
        if m:
            report.pushes.append(
                PushSample(
                    rank=int(m.group("rank")),
                    bytes_mb=float(m.group("bytes_mb")),
                    handles=int(m.group("handles")),
                    convert_ms=float(m.group("convert_ms")),
                    postwait_ms=float(m.group("postwait_ms")),
                    barrier_ms=float(m.group("barrier_ms")),
                    total_ms=float(m.group("total_ms")),
                    wire_gbps=float(m.group("wire_gbps")),
                    net_gbps=float(m.group("net_gbps")),
                )
            )
            continue

        m = _MX_DISCOVER_RE.search(line)
        if m:
            report.mx_discovered.append(
                {
                    "role": m.group("role"),
                    "rank": int(m.group("rank")),
                    "endpoint": m.group("endpoint"),
                    "polls": int(m.group("polls")),
                }
            )
            continue

        m = _MX_COORDINATOR_RE.search(line)
        if m:
            report.mx_coordinator = {
                "source_id": m.group("source_id"),
                "endpoint": m.group("endpoint"),
            }
            continue

        m = _MX_PIPELINE_RE.search(line)
        if m:
            report.mx_pipeline_publishes += 1

    return report


def _read_source(path: str | None, use_stdin: bool) -> TextIO:
    if use_stdin:
        return sys.stdin
    if not path:
        raise SystemExit("must pass --log PATH or --stdin")
    return open(path, "r")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Path to log file")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--json", help="Optional JSON output path")
    args = parser.parse_args()

    with _read_source(args.log, args.stdin) as src:
        report = parse_lines(src)

    summary = report.summary()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
