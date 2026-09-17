"""Summarize the runtime benchmark from finished `uv run eval` run directories.

    uv run python benchmarks/runtimes/analyze.py outputs/runtime-bench-prime outputs/runtime-bench-modal

Per run and eval source: episode outcomes, p50/p90 of boot, setup and rollout wall clock,
the resources each sandbox asked for, the summed sandbox lifetime, and the cost that
lifetime implies at the provider's published per-resource rates.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from prime_rl.eval.resume import archives, read_records
from prime_rl.monitors.file.traces import get_trace_stream

# USD per hour of one core / one GB of memory / one GB of disk, from the providers' pricing
# pages. Modal and E2B size the disk from the image, so their disk rate is zero.
RATES: dict[str, tuple[float, float, float]] = {
    # https://docs.primeintellect.ai/sandboxes/overview
    "prime": (0.05, 0.01, 0.001),
    # https://modal.com/pricing ($0.00003942/core/s, $0.00000667/GiB/s)
    "modal": (0.00003942 * 3600, 0.00000667 * 3600, 0.0),
    # https://e2b.dev/pricing ($0.000014/vCPU/s, $0.0000045/GiB/s)
    "e2b": (0.000014 * 3600, 0.0000045 * 3600, 0.0),
}


@dataclass
class Rollout:
    runtime: str
    ok: bool
    errors: list[str]
    stop: str | None
    boot: float
    setup: float
    wallclock: float
    lifetime: float
    cpu: float | None
    memory: float | None
    disk: float | None
    image_cached: bool | None


@dataclass
class Bucket:
    rollouts: list[Rollout] = field(default_factory=list)


def span(timing: dict, phase: str) -> tuple[float | None, float | None]:
    part = timing.get(phase) or {}
    return part.get("start"), part.get("end")


def duration(timing: dict, phase: str) -> float:
    start, end = span(timing, phase)
    return max(0.0, end - start) if start and end else 0.0


def rollout(rec: dict, trace: dict) -> Rollout:
    timing = trace.get("timing") or {}
    info = trace.get("info") or {}
    runtime = (trace.get("agent") or {}).get("runtime") or {}
    dispatched = (info.get("dispatch") or {}).get("time")
    arrived = (info.get("arrival") or {}).get("time")
    boot_start, boot_end = span(timing, "boot")
    ends = [end for phase in ("setup", "agent", "finalize", "scoring") for _, end in [span(timing, phase)] if end]
    if dispatched and arrived:
        wallclock = arrived - dispatched
    else:
        wallclock = (max(ends) - boot_start) if ends and boot_start else 0.0
    # Billing starts once the box is up; the wait for it (image builds included) is `boot`.
    lifetime = (max(ends) - boot_end) if ends and boot_end else 0.0
    return Rollout(
        runtime=runtime.get("type") or "?",
        ok=bool(rec.get("ok")),
        errors=[error.get("type") or "?" for error in (rec.get("errors") or []) + (trace.get("errors") or [])],
        stop=trace.get("stop_condition"),
        boot=duration(timing, "boot"),
        setup=duration(timing, "setup"),
        wallclock=max(0.0, wallclock),
        lifetime=max(0.0, lifetime),
        cpu=runtime.get("cpu"),
        memory=runtime.get("memory"),
        disk=runtime.get("disk"),
        image_cached=runtime.get("image_cached"),
    )


def streams(run_dir: Path):
    current = get_trace_stream(run_dir)
    for archive in archives(run_dir):
        yield archive / current.relative_to(current.parents[1])
    yield current


def load(run_dir: Path) -> dict[str, Bucket]:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for stream in streams(run_dir):
        if not stream.exists():
            continue
        for rec in read_records(stream):
            env = (rec.get("env") or {}).get("name") or (rec.get("env") or {}).get("id") or "?"
            for trace in rec.get("traces") or []:
                buckets[env].rollouts.append(rollout(rec, trace))
    return buckets


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def hms(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds:.0f} s"


def resources(rollouts: list[Rollout]) -> str:
    shapes = Counter((r.cpu, r.memory, r.disk) for r in rollouts)
    (cpu, memory, disk), _ = shapes.most_common(1)[0]
    if cpu is None and memory is None:
        return "n/a"
    text = f"{cpu:g} cpu / {memory:g} GB" + (f" / {disk:g} GB disk" if disk else "")
    return text if len(shapes) == 1 else f"{text} (mode, {len(shapes)} shapes)"


def cost(rollouts: list[Rollout]) -> float | None:
    rates = RATES.get(rollouts[0].runtime)
    if rates is None:
        return None
    cpu_rate, memory_rate, disk_rate = rates
    return sum(
        r.lifetime / 3600 * ((r.cpu or 0) * cpu_rate + (r.memory or 0) * memory_rate + (r.disk or 0) * disk_rate)
        for r in rollouts
    )


def report(run_dir: Path, buckets: dict[str, Bucket]) -> None:
    print(f"\n## {run_dir.name}\n")
    print(
        "| source | runtime | rollouts | ok | sandbox errors | boot p50/p90 | setup p50/p90 | wallclock p50/p90 | resources per sandbox | Σ lifetime | est. cost |"
    )
    print("|---|---|---:|---:|---:|---|---|---|---|---:|---:|")
    for env, bucket in sorted(buckets.items()):
        rows = bucket.rollouts
        boots = [r.boot for r in rows]
        setups = [r.setup for r in rows]
        walls = [r.wallclock for r in rows]
        sandbox_errors = sum(1 for r in rows if "SandboxError" in r.errors)
        total = cost(rows)
        print(
            f"| {env} | {rows[0].runtime} | {len(rows)} | {sum(r.ok for r in rows) / len(rows):.0%} | {sandbox_errors} "
            f"| {hms(percentile(boots, 0.5))} / {hms(percentile(boots, 0.9))} "
            f"| {hms(percentile(setups, 0.5))} / {hms(percentile(setups, 0.9))} "
            f"| {hms(percentile(walls, 0.5))} / {hms(percentile(walls, 0.9))} "
            f"| {resources(rows)} | {sum(r.lifetime for r in rows) / 3600:.1f} h "
            f"| {'n/a' if total is None else f'${total:,.2f}'} |"
        )
        cached = [r.boot for r in rows if r.image_cached is True]
        built = [r.boot for r in rows if r.image_cached is False]
        if built:
            print(
                f"|  ↳ boot split | | | | | cached {hms(percentile(cached, 0.5))} / {hms(percentile(cached, 0.9))} "
                f"({len(cached)}), built {hms(percentile(built, 0.5))} / {hms(percentile(built, 0.9))} ({len(built)}) | | | | | |"
            )
    rows = [r for bucket in buckets.values() for r in bucket.rollouts]
    total = cost(rows) if rows else None
    print(
        f"\nΣ lifetime {sum(r.lifetime for r in rows) / 3600:.1f} h, est. cost {'n/a' if total is None else f'${total:,.2f}'}"
    )
    stops = Counter(r.stop for r in rows if not r.ok)
    if stops:
        print("failed rollouts by stop condition: " + ", ".join(f"{stop}={n}" for stop, n in stops.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="eval run directories (output_dir/run.name)")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        report(run_dir, load(run_dir))


if __name__ == "__main__":
    main()
