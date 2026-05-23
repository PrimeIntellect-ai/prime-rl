"""Replay an orchestrator step boundary from a saved raw-rollout dump.

Usage:

  uv run python scripts/perf_r3/replay_orch_step.py \\
      --dump <path/to/raw_rollouts/step_N.pkl> \\
      --output-dir /tmp/replay_out

Loads the pickled list[vf.RolloutOutput] that the orch wrote when
`config.dump_raw_rollouts=True`, then runs the same post-process pipeline as
the real orch step boundary, timing each phase:

  compute_advantages → apply_filters → save_rollouts(JSONL) → pretokenize →
  interleave_rollout × N → result-collection loop → TrainingBatch.send (sidecar
  stream encode) → pandas DataFrames → metrics aggregation → wandb log (stub)

Output is a single line per phase plus a summary table — same kind of thing
the real orch now logs (Step N phase breakdown). 30 s iteration loop vs a
multi-minute SLURM run, lets us actually attribute residual orch event-loop
lag to specific code paths without guessing.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from prime_rl.configs.orchestrator import (
    DefaultAdvantageConfig,
    OrchestratorConfig,
)
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.filters import apply_filters, setup_filters
from prime_rl.orchestrator.trajectories import (
    interleave_rollout,
    pretokenize_rollout_trajectory,
)
from prime_rl.orchestrator.vf_utils import get_seq_len, save_rollouts
from prime_rl.transport.filesystem import FileSystemTrainingBatchSender
from prime_rl.transport.types import TrainingBatch


SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from lag_monitor import LagMonitor, fmt, fmt_ms  # noqa: E402


def _load_rollouts(path: Path) -> list[dict]:
    print(f"[load] reading {path}  ({path.stat().st_size/1e9:.2f} GB)")
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        rollouts = pickle.load(f)
    print(f"[load] unpickled {len(rollouts)} rollouts in {fmt_ms(time.perf_counter()-t0)}")
    return rollouts


def _summarize_rollouts(rollouts: list[dict]) -> None:
    import numpy as np
    n = len(rollouts)
    turns = [len(r.get("trajectory", [])) for r in rollouts]
    rewards = [r.get("reward", 0.0) for r in rollouts]
    re_bytes_per_rollout = []
    for r in rollouts:
        total = 0
        for step in r.get("trajectory", []):
            tok = step.get("tokens") or {}
            re = tok.get("routed_experts")
            if re and isinstance(re, dict) and isinstance(re.get("data"), (bytes, bytearray, str)):
                d = re["data"]
                total += len(d) if isinstance(d, (bytes, bytearray)) else int(len(d) * 0.75)  # base64 ≈ 3/4
        re_bytes_per_rollout.append(total)
    print(
        f"[summary] rollouts={n} turns: median={int(np.median(turns))} max={max(turns)} | "
        f"reward mean={np.mean(rewards):.3f} | routed_experts MB/rollout: median={np.median(re_bytes_per_rollout)/1e6:.1f} "
        f"max={max(re_bytes_per_rollout)/1e6:.1f} total={sum(re_bytes_per_rollout)/1e9:.2f} GB"
    )


async def _synthetic_traffic_task(stop_event: asyncio.Event):
    """Burn small CPU on the loop at high frequency to simulate prod oversubscription.

    Prod loop is concurrently servicing ~1024 active rollouts (recv/parse/dispatch)
    while gather runs. We mimic that with periodic short bursts of dict/loop work
    that hold the GIL on the loop thread for a few hundred microseconds.
    """
    import random
    while not stop_event.is_set():
        # tiny burst of work: ~300us
        d = {i: i * 1.5 for i in range(800)}
        sum(d.values())
        await asyncio.sleep(random.uniform(0.0005, 0.002))


async def main_async(args):
    rollouts = _load_rollouts(Path(args.dump))
    _summarize_rollouts(rollouts)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Spin up synthetic traffic to simulate prod loop oversubscription.
    traffic_stop = asyncio.Event()
    traffic_tasks: list = []
    if args.synthetic_traffic > 0:
        print(f"[traffic] spawning {args.synthetic_traffic} background tasks")
        traffic_tasks = [
            asyncio.create_task(_synthetic_traffic_task(traffic_stop))
            for _ in range(args.synthetic_traffic)
        ]

    if args.profile_interleave:
        # Run interleave_rollout once on the longest-trajectory rollout under
        # cProfile so we see which Python-level lines hold the GIL. Pick the
        # max-turns rollout because that's where the per-step list/bool work
        # dominates.
        import cProfile
        import pstats
        worst = max(rollouts, key=lambda r: len(r.get("trajectory", [])))
        print(f"[profile] running interleave_rollout on worst rollout: turns={len(worst['trajectory'])}")
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.profile_repeat):
            interleave_rollout(worst, None, 0, None)
        pr.disable()
        stats = pstats.Stats(pr).sort_stats("cumulative")
        stats.print_stats(30)
        stats.sort_stats("tottime").print_stats(30)
        return

    phase_times: dict[str, float] = {}

    monitor = LagMonitor(interval=0.001)

    async def _timed_phase(name: str, fn, *args, **kwargs):
        monitor.reset()
        monitor.start()
        await asyncio.sleep(0.05)  # let monitor warm up
        t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(0.05)
        snap = monitor.snapshot()
        await monitor.stop()
        phase_times[name] = elapsed
        print(f"[phase] {name:32s} wall={elapsed:6.2f}s   lag {fmt(snap)}")
        return result

    # ── compute_advantages (to_thread today)
    advantage_cfg = DefaultAdvantageConfig()
    await _timed_phase("compute_advantages", asyncio.to_thread, compute_advantages, rollouts, advantage_cfg)

    # ── apply_filters (to_thread today). Use default filter set.
    filters = setup_filters([], vocab_size=200000)  # vocab_size unused for default filters
    await _timed_phase("apply_filters", asyncio.to_thread, apply_filters, filters, rollouts)

    # ── save_rollouts JSONL (to_thread today)
    jsonl_path = outdir / "replay_rollouts.jsonl"
    await _timed_phase(
        "save_rollouts_jsonl",
        asyncio.to_thread,
        save_rollouts,
        rollouts,
        jsonl_path,
        {"trajectory"},
    )

    # ── pretokenize. We don't have a real tokenizer here, but pretokenize is
    # a no-op when the renderer client already populated `tokens` per step.
    # The router-replay path always populates tokens, so we can pass None and
    # the function will detect already-tokenized.
    # If it doesn't no-op cleanly, skip; phase wall-time is then 0.
    try:
        await _timed_phase(
            "pretokenize_all",
            asyncio.gather,
            *(
                asyncio.to_thread(pretokenize_rollout_trajectory, r, None)
                for r in rollouts
            ),
        )
    except Exception as e:
        print(f"[phase] pretokenize_all                skipped ({e})")
        phase_times["pretokenize_all"] = 0.0

    # ── interleave_rollout × N (to_thread today, now O(N) per rollout).
    # Optional chunking: fan out `chunk` to_thread tasks, await the batch, yield
    # the loop with sleep(0), then move on. Idea: give the main loop a
    # guaranteed slice between batches so other tasks (recv, scheduler) make
    # progress. Compared to one big gather of all 256 tasks at once.
    chunk = args.gather_chunk_size
    async def _do_gather():
        if chunk is None or chunk >= len(rollouts):
            return await asyncio.gather(
                *(
                    asyncio.to_thread(interleave_rollout, r, None, idx, None)
                    for idx, r in enumerate(rollouts)
                )
            )
        results: list = []
        for start in range(0, len(rollouts), chunk):
            batch = rollouts[start : start + chunk]
            batch_results = await asyncio.gather(
                *(
                    asyncio.to_thread(interleave_rollout, r, None, start + i, None)
                    for i, r in enumerate(batch)
                )
            )
            results.extend(batch_results)
            await asyncio.sleep(0)
        return results

    results = await _timed_phase("interleave_rollout_gather", _do_gather)

    # ── sync result-collection loop (on the loop today)
    async def _collect():
        train_examples = []
        rollout_prefill_lens = []
        rollout_decode_lens = []
        rollout_samples_per_rollout = []
        for rollout, samples in zip(rollouts, results):
            rpf = 0
            rdc = 0
            samples = samples or []
            rollout_samples_per_rollout.append(len(samples))
            for sample in samples:
                sdt = sum(sample.completion_mask)
                spt = len(sample.prompt_ids) + len(sample.completion_mask) - sdt
                rdc += sdt
                rpf += spt
                if not rollout.get("is_filtered", False):
                    train_examples.append(sample)
            rollout_prefill_lens.append(rpf)
            rollout_decode_lens.append(rdc)
        return train_examples, rollout_prefill_lens, rollout_decode_lens, rollout_samples_per_rollout

    train_examples, rollout_prefill_lens, rollout_decode_lens, rollout_samples_per_rollout = await _timed_phase(
        "result_collection_loop", _collect
    )
    print(f"          collected {len(train_examples)} train_examples")

    # ── TrainingBatch.send (async, sidecar+stream today)
    sender_root = outdir / "send_root"
    sender_root.mkdir(parents=True, exist_ok=True)
    sender = FileSystemTrainingBatchSender(sender_root)
    batch = TrainingBatch(examples=train_examples, step=0)
    async def _send():
        await sender.send(batch)
    await _timed_phase("training_batch_send", _send)

    # ── pandas DataFrames (sync today)
    async def _build_dfs():
        results_df = pd.DataFrame(
            {
                "example_id": [r.get("example_id", 0) for r in rollouts],
                "env_name": [r.get("env_name", "") for r in rollouts],
                "reward": [r.get("reward", 0.0) for r in rollouts],
                "is_truncated": [r.get("is_truncated", False) for r in rollouts],
                "is_filtered": [r.get("is_filtered", False) for r in rollouts],
                "stop_condition": [r.get("stop_condition") for r in rollouts],
                "seq_len": [get_seq_len(r) for r in rollouts],
                "prefill_len": rollout_prefill_lens,
                "decode_len": rollout_decode_lens,
                "samples_per_rollout": rollout_samples_per_rollout,
                "num_turns": [len(r.get("trajectory", [])) for r in rollouts],
            }
        )
        metrics_df = pd.DataFrame([r.get("metrics") or {} for r in rollouts])
        filter_df = pd.DataFrame([r.get("filters") or {} for r in rollouts])
        timing_df = pd.DataFrame(
            [
                {
                    "total": (r.get("timing") or {}).get("total", 0.0),
                }
                for r in rollouts
            ]
        )
        return results_df, metrics_df, filter_df, timing_df

    results_df, metrics_df, filter_df, timing_df = await _timed_phase("pandas_dataframes", _build_dfs)

    # ── metrics aggregation (groupby + per-env loops) — approximate
    async def _aggregate():
        by_example = results_df.groupby(["env_name", "example_id"])
        agg = {
            "seq_len/mean": by_example.seq_len.mean().mean(),
            "reward/mean": by_example.reward.mean().mean(),
        }
        # mimic per-env loop
        for env, env_df in results_df.groupby("env_name"):
            env_by_example = env_df.groupby("example_id")
            agg[f"reward/{env}/mean"] = env_by_example.reward.mean().mean()
        return agg

    await _timed_phase("metrics_aggregation", _aggregate)

    # ── wandb_log stub (just construct the dict; no actual upload here)
    async def _wandb_stub():
        d = {f"k{i}": i * 0.1 for i in range(500)}
        return d

    await _timed_phase("wandb_log_stub", _wandb_stub)

    # ── stop synthetic traffic
    if traffic_tasks:
        traffic_stop.set()
        for t in traffic_tasks:
            t.cancel()
        await asyncio.gather(*traffic_tasks, return_exceptions=True)

    # ── summary
    print("\n── phase breakdown (descending) ──")
    for name, dur in sorted(phase_times.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {dur:6.2f}s")
    total = sum(phase_times.values())
    print(f"  {'TOTAL':32s} {total:6.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="path/to/raw_rollouts/step_N.pkl")
    p.add_argument("--output-dir", default="/tmp/replay_orch_out")
    p.add_argument("--profile-interleave", action="store_true",
                   help="cProfile interleave_rollout on the worst rollout, then exit")
    p.add_argument("--profile-repeat", type=int, default=5,
                   help="Number of interleave_rollout iterations to profile")
    p.add_argument("--gather-chunk-size", type=int, default=None,
                   help="Chunk size for the interleave_rollout gather. None = one gather of all 256")
    p.add_argument("--synthetic-traffic", type=int, default=0,
                   help="N background asyncio tasks that simulate prod loop oversubscription")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
