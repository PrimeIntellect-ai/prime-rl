"""Per-phase event-loop-lag attribution for the orch step boundary.

Why this exists: phase wallclock timings (process_rollout_gather=8s, etc) tell
us where time goes but NOT what blocks the loop. A 5s gather in to_thread does
not necessarily block the loop for 5s — but in prod we see 5-8s of loop lag.
Wallclock != loop lag.

This harness loads a real raw_rollouts/step_N.pkl and runs the orch's
post-rollouts phases incrementally, each time with a 1ms-granularity event-loop
lag monitor running on the same loop:

    [P0]                            (baseline: empty loop)
    [P1]  compute_advantages
    [P2]  compute_advantages + apply_filters
    [P3]  + save_rollouts
    [P4]  + pretokenize
    [P5]  + interleave_rollout_gather  (the suspected culprit)
    [P6]  + training_batch_send
    [P7]  + pandas + metrics_aggregation
    [Pall] all phases run together

Each level reports event-loop lag stats during the phase work — that's the
metric we actually care about. Layers that materially worsen lag identify the
loop blockers; layers that only add wallclock without raising lag are fine.

Usage:
    uv run python scripts/perf_r3/replay_orch_attribution.py \\
        --dump <path/to/step_N.pkl> [--gather-chunk-size 128] [--synthetic-traffic 200]
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.filters import apply_filters, setup_filters
from prime_rl.orchestrator.trajectories import (
    interleave_rollout,
    pretokenize_rollout_trajectory,
)
from prime_rl.orchestrator.vf_utils import get_seq_len, save_rollouts
from prime_rl.transport.filesystem import FileSystemTrainingBatchSender
from prime_rl.transport.types import TrainingBatch

# lag_monitor.py is a sibling — add this script's dir to sys.path so it imports cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lag_monitor import LagMonitor, fmt, fmt_ms  # noqa: E402


def _load_rollouts(path: Path) -> list[dict]:
    print(f"[load] reading {path}  ({path.stat().st_size/1e9:.2f} GB)")
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        rollouts = pickle.load(f)
    print(f"[load] unpickled {len(rollouts)} rollouts in {fmt_ms(time.perf_counter()-t0)}")
    return rollouts


async def _synthetic_traffic_task(stop_event: asyncio.Event):
    import random
    while not stop_event.is_set():
        d = {i: i * 1.5 for i in range(800)}
        sum(d.values())
        await asyncio.sleep(random.uniform(0.0005, 0.002))


async def _measured(name: str, body, *, warmup: float = 0.05):
    """Run `body` (a no-arg awaitable factory) with a fine-grained lag monitor.

    Returns (wall_seconds, snapshot).
    """
    monitor = LagMonitor(interval=0.001)  # 1ms granularity
    monitor.start()
    await asyncio.sleep(warmup)  # let monitor warm up before measuring
    t0 = time.perf_counter()
    result = await body()
    wall = time.perf_counter() - t0
    await asyncio.sleep(warmup)
    snap = monitor.snapshot()
    await monitor.stop()
    print(f"  {name:36s} wall={wall:6.2f}s   lag {fmt(snap)}")
    return wall, snap, result


def _interleave_gather_factory(rollouts, chunk_size):
    """Returns an async factory that runs interleave_rollout × N as the orch does.

    Mirrors the chunked gather added in orchestrator.py: if chunk_size is None
    or >= len(rollouts), one big gather; otherwise batches with sleep(0) yield
    between.
    """
    async def go():
        if chunk_size is None or chunk_size >= len(rollouts):
            return await asyncio.gather(
                *(
                    asyncio.to_thread(interleave_rollout, r, None, idx, None)
                    for idx, r in enumerate(rollouts)
                )
            )
        results: list = []
        for start in range(0, len(rollouts), chunk_size):
            batch = rollouts[start : start + chunk_size]
            batch_results = await asyncio.gather(
                *(
                    asyncio.to_thread(interleave_rollout, r, None, start + i, None)
                    for i, r in enumerate(batch)
                )
            )
            results.extend(batch_results)
            await asyncio.sleep(0)
        return results
    return go


async def main_async(args):
    rollouts = _load_rollouts(Path(args.dump))
    print(f"[shape] rollouts={len(rollouts)} turns: median={sorted(len(r['trajectory']) for r in rollouts)[len(rollouts)//2]} max={max(len(r['trajectory']) for r in rollouts)}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── traffic
    traffic_stop = asyncio.Event()
    traffic_tasks: list = []
    if args.synthetic_traffic > 0:
        print(f"[traffic] spawning {args.synthetic_traffic} background tasks")
        traffic_tasks = [
            asyncio.create_task(_synthetic_traffic_task(traffic_stop))
            for _ in range(args.synthetic_traffic)
        ]

    # Pre-compute filters (cheap setup, run once)
    filters = setup_filters([], vocab_size=200000)

    # Phase factories — each returns an async function that performs the phase
    # against a deepcopy-free, side-effecting view of `rollouts`. We re-load
    # rollouts between levels to keep them pristine for the next test.

    def _build_factories(rollouts):
        advantage_cfg = DefaultAdvantageConfig()

        async def compute_adv():
            await asyncio.to_thread(compute_advantages, rollouts, advantage_cfg)

        async def filter_phase():
            await asyncio.to_thread(apply_filters, filters, rollouts)

        async def save_phase():
            jsonl_path = outdir / "replay_attr.jsonl"
            await asyncio.to_thread(save_rollouts, rollouts, jsonl_path, {"trajectory"})

        async def pretok_phase():
            # Call-site skip mirrors the prod fix: avoid the to_thread fanout
            # entirely when every step already has tokens populated.
            if not any(step["tokens"] is None for r in rollouts for step in r["trajectory"]):
                return
            await asyncio.gather(
                *(asyncio.to_thread(pretokenize_rollout_trajectory, r, None) for r in rollouts)
            )

        gather_factory = _interleave_gather_factory(rollouts, args.gather_chunk_size)
        async def gather_phase():
            results = await gather_factory()
            return results

        async def send_phase(results):
            sender_root = outdir / "send_root"
            sender_root.mkdir(parents=True, exist_ok=True)
            sender = FileSystemTrainingBatchSender(sender_root)
            train_examples = []
            for rollout, samples in zip(rollouts, results):
                samples = samples or []
                for sample in samples:
                    if not rollout.get("is_filtered", False):
                        train_examples.append(sample)
            batch = TrainingBatch(examples=train_examples, step=0)
            await sender.send(batch)

        async def df_phase():
            # quick pandas + metrics aggregation pass — happens on the loop synchronously
            results_df = pd.DataFrame(
                {
                    "example_id": [r.get("example_id", 0) for r in rollouts],
                    "env_name": [r.get("env_name", "") for r in rollouts],
                    "reward": [r.get("reward", 0.0) for r in rollouts],
                    "seq_len": [get_seq_len(r) for r in rollouts],
                }
            )
            by_example = results_df.groupby(["env_name", "example_id"])
            _ = by_example.seq_len.mean().mean()
            _ = by_example.reward.mean().mean()

        return {
            "compute_adv": compute_adv,
            "filter": filter_phase,
            "save": save_phase,
            "pretok": pretok_phase,
            "gather": gather_phase,
            "send": send_phase,
            "df": df_phase,
        }

    # ── Baseline — empty loop with traffic
    print("\n=== P0 baseline (empty loop) ===")
    async def baseline_body():
        await asyncio.sleep(1.0)  # idle 1s
    await _measured("baseline_idle", baseline_body)

    # ── Each phase ALONE (uses fresh rollouts to avoid filter/advantage mutation)
    print("\n=== Each phase alone ===")
    fac = _build_factories(rollouts)
    await _measured("compute_advantages", fac["compute_adv"])
    await _measured("apply_filters", fac["filter"])
    await _measured("save_rollouts (orjson)", fac["save"])
    await _measured("pretokenize_all", fac["pretok"])
    _, _, gather_results = await _measured("interleave_rollout_gather", fac["gather"])
    await _measured("training_batch_send", lambda: fac["send"](gather_results))
    await _measured("pandas + metrics", fac["df"])

    if not args.quick:
        # ── Incremental cumulation (re-load between tests so apply_filters etc don't double-apply)
        print("\n=== Cumulative (each layer adds one) ===")
        levels = [
            ("L1 advantages",                    ["compute_adv"]),
            ("L2 + filters",                     ["compute_adv", "filter"]),
            ("L3 + save",                        ["compute_adv", "filter", "save"]),
            ("L4 + pretokenize",                 ["compute_adv", "filter", "save", "pretok"]),
            ("L5 + gather",                      ["compute_adv", "filter", "save", "pretok", "gather"]),
            ("L6 + send",                        ["compute_adv", "filter", "save", "pretok", "gather", "send"]),
            ("L7 + dataframes",                  ["compute_adv", "filter", "save", "pretok", "gather", "send", "df"]),
        ]

        for label, names in levels:
            # Re-load fresh rollouts so state isn't polluted across levels
            rollouts_fresh = _load_rollouts(Path(args.dump))
            fac_fresh = _build_factories(rollouts_fresh)
            gather_res_holder = {"r": None}

            async def run_all():
                for n in names:
                    if n == "gather":
                        gather_res_holder["r"] = await fac_fresh["gather"]()
                    elif n == "send":
                        if gather_res_holder["r"] is None:
                            gather_res_holder["r"] = await fac_fresh["gather"]()
                        await fac_fresh["send"](gather_res_holder["r"])
                    else:
                        await fac_fresh[n]()

            await _measured(label, run_all)

    # ── stop traffic
    if traffic_tasks:
        traffic_stop.set()
        for t in traffic_tasks:
            t.cancel()
        await asyncio.gather(*traffic_tasks, return_exceptions=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--output-dir", default="/beegfs/outputs/replay_attr_scratch")
    p.add_argument("--gather-chunk-size", type=int, default=None)
    p.add_argument("--synthetic-traffic", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Skip the cumulative levels (which reload the pkl 7x). Just baseline + each phase alone.")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
