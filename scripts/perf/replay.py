"""Orchestrator postprocess replay: measure event-loop lag for each intervention.

Usage:
  uv run python scripts/perf_r3/replay.py --batch <path/to/train_rollouts.bin> \
      --modes baseline,stream,sidecar,stream_sidecar,to_thread,uvloop_baseline,uvloop_stream_sidecar

Loads a saved TrainingBatch msgpack blob, then re-runs the orchestrator-side
"build batch bytes + write to disk" path under each intervention mode, while
an asyncio event-loop lag monitor records samples. Writes CSV with
per-mode {p50, p90, p99, max} lag + wallclock to /tmp/perf_r3_out/.

Intervention modes:

  baseline                 — encode whole TrainingBatch with one msgspec.Encoder.encode()
                             call on the event loop (current orchestrator.py:596).
  stream                   — encode per-TrainingSample, write length-prefixed frames,
                             `await asyncio.sleep(0)` between samples (O2).
  sidecar                  — strip routed_experts.data into raw-bytes sidecar file
                             with a small msgpack header carrying shape/dtype/offset;
                             rest of TrainingBatch through msgspec (O1).
  stream_sidecar           — combine O1 + O2 (per-sample stream of the slimmed batch,
                             routed_experts.data written separately).
  to_thread                — current encoder.encode wrapped in asyncio.to_thread.
                             (Expected: small win; GIL is held by C ext.)
  uvloop_baseline          — same as baseline but under uvloop.
  uvloop_stream_sidecar    — uvloop + stream + sidecar.

Each mode runs `--repeats` times. The reported max lag is the worst observed
across all repeats.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Callable

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import msgspec  # noqa: E402

from lag_monitor import LagMonitor, fmt, fmt_ms  # noqa: E402
from prime_rl.transport.types import RoutedExperts, TrainingBatch, TrainingSample  # noqa: E402


# ---------------------------------------------------------------------------
# Mode implementations: each returns total bytes written (not necessarily equal
# across modes, since sidecar splits the payload).
# ---------------------------------------------------------------------------


def _make_encoder() -> msgspec.msgpack.Encoder:
    return msgspec.msgpack.Encoder()


async def mode_baseline(tb: TrainingBatch, outdir: Path) -> int:
    """Current orchestrator path: one big msgspec.encode of TrainingBatch."""
    enc = _make_encoder()
    buf = enc.encode(tb)
    (outdir / "baseline.bin").write_bytes(buf)
    return len(buf)


async def mode_stream(tb: TrainingBatch, outdir: Path) -> int:
    """Per-sample length-prefixed stream encode (O2)."""
    enc = _make_encoder()
    out = io.BytesIO()
    header = {
        "version": 2,
        "step": tb.step,
        "run_idx": tb.run_idx,
        "n_samples": len(tb.examples),
    }
    h = enc.encode(header)
    out.write(len(h).to_bytes(4, "little"))
    out.write(h)
    for sample in tb.examples:
        frame = enc.encode(sample)
        out.write(len(frame).to_bytes(4, "little"))
        out.write(frame)
        await asyncio.sleep(0)
    data = out.getvalue()
    (outdir / "stream.bin").write_bytes(data)
    return len(data)


async def mode_sidecar(tb: TrainingBatch, outdir: Path) -> int:
    """Strip routed_experts.data into raw sidecar bytes, rest via msgspec (O1).

    File layout:
      meta.bin  = msgpack({sidecar_offsets, shapes, dtypes, n_samples, step})
                  + msgpack-encoded TrainingBatch with routed_experts.data
                  replaced by empty bytes (shape/dtype preserved).
      sidecar.bin = raw concatenation of all routed_experts.data, byte-aligned.
    Re-build on receive: read meta, slice sidecar.bin at offsets.
    """
    enc = _make_encoder()
    sidecar_path = outdir / "sidecar_routed_experts.bin"
    meta_path = outdir / "sidecar_meta.bin"

    # Stripped batch: same TrainingSample structs but with empty routed_experts.data.
    # We allocate new RoutedExperts with empty data; original bytes are written separately.
    stripped_examples = []
    offsets: list[int] = []
    shapes: list[list[int] | None] = []
    dtypes: list[str | None] = []
    running = 0
    with open(sidecar_path, "wb") as sf:
        for sample in tb.examples:
            if sample.routed_experts is None:
                offsets.append(-1)
                shapes.append(None)
                dtypes.append(None)
                stripped_examples.append(sample)
                continue
            re = sample.routed_experts
            sf.write(re.data)
            offsets.append(running)
            shapes.append(re.shape)
            dtypes.append(re.dtype)
            running += len(re.data)
            # New TrainingSample with empty routed_experts.data; msgspec sees only the
            # tiny shape/dtype metadata; the big bytes blob has been peeled off.
            empty_re = RoutedExperts(data=b"", shape=re.shape, dtype=re.dtype)
            stripped_examples.append(
                msgspec.structs.replace(sample, routed_experts=empty_re)
            )

    stripped_tb = msgspec.structs.replace(tb, examples=stripped_examples)
    meta_buf = enc.encode(
        {
            "version": 3,
            "format": "sidecar",
            "step": tb.step,
            "run_idx": tb.run_idx,
            "offsets": offsets,
            "shapes": shapes,
            "dtypes": dtypes,
            "sidecar_total_bytes": running,
        }
    )
    batch_buf = enc.encode(stripped_tb)
    with open(meta_path, "wb") as mf:
        mf.write(len(meta_buf).to_bytes(4, "little"))
        mf.write(meta_buf)
        mf.write(len(batch_buf).to_bytes(4, "little"))
        mf.write(batch_buf)
    return meta_path.stat().st_size + sidecar_path.stat().st_size


async def mode_stream_sidecar(tb: TrainingBatch, outdir: Path) -> int:
    """O1 + O2: sidecar bytes + per-sample stream of the slimmed batch."""
    enc = _make_encoder()
    sidecar_path = outdir / "stream_sidecar_routed_experts.bin"
    meta_path = outdir / "stream_sidecar_meta.bin"

    offsets: list[int] = []
    shapes: list[list[int] | None] = []
    dtypes: list[str | None] = []
    running = 0

    # Open both files; stream samples without holding the stripped batch in memory.
    with open(sidecar_path, "wb") as sf, open(meta_path, "wb") as mf:
        # Write a placeholder meta header; will patch offsets after.
        # Simpler: write samples first, then append the meta after sample stream.
        # File layout: <hdr_len:u4><hdr_msgpack><n_samples × <frame_len:u4><frame_msgpack>>
        # but we patch the header at the end with offsets, so write meta to a separate
        # file and read both on the receive side.
        n_examples = len(tb.examples)
        # 4-byte placeholder for n_samples kept first for simplicity.
        mf.write(n_examples.to_bytes(4, "little"))
        for sample in tb.examples:
            if sample.routed_experts is None:
                offsets.append(-1)
                shapes.append(None)
                dtypes.append(None)
                out_sample = sample
            else:
                re = sample.routed_experts
                sf.write(re.data)
                offsets.append(running)
                shapes.append(re.shape)
                dtypes.append(re.dtype)
                running += len(re.data)
                out_sample = msgspec.structs.replace(
                    sample,
                    routed_experts=RoutedExperts(data=b"", shape=re.shape, dtype=re.dtype),
                )
            frame = enc.encode(out_sample)
            mf.write(len(frame).to_bytes(4, "little"))
            mf.write(frame)
            await asyncio.sleep(0)
        # Trailing manifest: shapes/dtypes/offsets — encoded separately at end.
        manifest = enc.encode(
            {
                "version": 4,
                "format": "stream_sidecar",
                "step": tb.step,
                "run_idx": tb.run_idx,
                "offsets": offsets,
                "shapes": shapes,
                "dtypes": dtypes,
                "sidecar_total_bytes": running,
            }
        )
        mf.write(len(manifest).to_bytes(4, "little"))
        mf.write(manifest)
    return meta_path.stat().st_size + sidecar_path.stat().st_size


async def mode_to_thread(tb: TrainingBatch, outdir: Path) -> int:
    """Current encode call wrapped in asyncio.to_thread. Tests GIL hypothesis."""
    enc = _make_encoder()
    buf = await asyncio.to_thread(enc.encode, tb)
    (outdir / "to_thread.bin").write_bytes(buf)
    return len(buf)


async def mode_stream_sidecar_threaded_io(tb: TrainingBatch, outdir: Path) -> int:
    """O1 + O2 + thread the I/O.

    Collect the routed_experts bytes pointers and write them in a single
    threaded pass after the metadata-only stream completes. Decouples the
    25MB-per-sample sync write from the event loop.
    """
    enc = _make_encoder()
    sidecar_path = outdir / "stream_sidecar_threaded_io_routed_experts.bin"
    meta_path = outdir / "stream_sidecar_threaded_io_meta.bin"

    offsets: list[int] = []
    shapes: list[list[int] | None] = []
    dtypes: list[str | None] = []
    re_payloads: list[bytes] = []
    running = 0

    n_examples = len(tb.examples)
    # Build meta in-memory (small) then write to disk once via to_thread.
    meta_buf = io.BytesIO()
    meta_buf.write(n_examples.to_bytes(4, "little"))
    for sample in tb.examples:
        if sample.routed_experts is None:
            offsets.append(-1)
            shapes.append(None)
            dtypes.append(None)
            out_sample = sample
        else:
            re = sample.routed_experts
            offsets.append(running)
            shapes.append(re.shape)
            dtypes.append(re.dtype)
            re_payloads.append(re.data)
            running += len(re.data)
            out_sample = msgspec.structs.replace(
                sample,
                routed_experts=RoutedExperts(data=b"", shape=re.shape, dtype=re.dtype),
            )
        frame = enc.encode(out_sample)
        meta_buf.write(len(frame).to_bytes(4, "little"))
        meta_buf.write(frame)
        await asyncio.sleep(0)

    manifest = enc.encode(
        {
            "version": 5,
            "format": "stream_sidecar_threaded_io",
            "step": tb.step,
            "run_idx": tb.run_idx,
            "offsets": offsets,
            "shapes": shapes,
            "dtypes": dtypes,
            "sidecar_total_bytes": running,
        }
    )
    meta_buf.write(len(manifest).to_bytes(4, "little"))
    meta_buf.write(manifest)

    def _write_all(meta_bytes: bytes, payloads: list[bytes]) -> int:
        # All sync I/O happens off the event loop.
        with open(meta_path, "wb") as mf:
            mf.write(meta_bytes)
        with open(sidecar_path, "wb") as sf:
            for p in payloads:
                sf.write(p)
        return meta_path.stat().st_size + sidecar_path.stat().st_size

    return await asyncio.to_thread(_write_all, meta_buf.getvalue(), re_payloads)


async def mode_pure_idle(tb: TrainingBatch, outdir: Path) -> int:
    """Control: no work at all, just yield. Establishes monitor noise floor."""
    for _ in range(128):
        await asyncio.sleep(0)
    return 0


async def mode_production(tb: TrainingBatch, outdir: Path) -> int:
    """Call the actual FileSystemTrainingBatchSender from prime-rl/perf/r3.

    Confirms that the production code path matches our inline microbench
    implementation. Output goes to a sandboxed dir inside outdir.
    """
    from prime_rl.transport.filesystem import FileSystemTrainingBatchSender, BATCH_FILE_NAME, SIDECAR_FILE_NAME

    prod_dir = outdir / "production_root"
    prod_dir.mkdir(parents=True, exist_ok=True)
    sender = FileSystemTrainingBatchSender(prod_dir)
    await sender.send(tb)

    from prime_rl.utils.pathing import get_rollout_dir, get_step_path

    step_dir = get_step_path(get_rollout_dir(prod_dir), tb.step)
    return (step_dir / BATCH_FILE_NAME).stat().st_size + (step_dir / SIDECAR_FILE_NAME).stat().st_size


# ---------------------------------------------------------------------------


MODE_DISPATCH: dict[str, Callable[[TrainingBatch, Path], Awaitable[int]]] = {
    "baseline": mode_baseline,
    "stream": mode_stream,
    "sidecar": mode_sidecar,
    "stream_sidecar": mode_stream_sidecar,
    "stream_sidecar_threaded_io": mode_stream_sidecar_threaded_io,
    "to_thread": mode_to_thread,
    "pure_idle": mode_pure_idle,
    "production": mode_production,
}


async def run_mode(name: str, tb: TrainingBatch, outdir: Path, repeats: int, monitor_interval: float) -> dict:
    fn = MODE_DISPATCH[name]
    mode_outdir = outdir / name
    mode_outdir.mkdir(parents=True, exist_ok=True)
    monitor = LagMonitor(interval=monitor_interval)
    monitor.reset()
    monitor.start()
    # quiet baseline sample
    await asyncio.sleep(0.25)
    quiet = monitor.snapshot()
    monitor.reset()
    t0 = perf_counter()
    sizes = []
    for _ in range(repeats):
        size = await fn(tb, mode_outdir)
        sizes.append(size)
    elapsed = perf_counter() - t0
    await asyncio.sleep(0.2)
    measured = monitor.snapshot()
    await monitor.stop()
    return {
        "mode": name,
        "repeats": repeats,
        "elapsed_total_s": elapsed,
        "per_call_s": elapsed / max(repeats, 1),
        "bytes_per_call": sizes[0] if sizes else 0,
        "quiet": quiet,
        "lag": measured,
    }


async def amain(args):
    print(f"[load] reading {args.batch}")
    t0 = perf_counter()
    raw = Path(args.batch).read_bytes()
    print(f"[load] {len(raw)/1e6:.1f}MB read in {fmt_ms(perf_counter()-t0)}")

    t0 = perf_counter()
    tb = msgspec.msgpack.decode(raw, type=TrainingBatch)
    print(f"[decode] TrainingBatch decoded in {fmt_ms(perf_counter()-t0)}")

    re_bytes = sum(len(s.routed_experts.data) for s in tb.examples if s.routed_experts is not None)
    re_n = sum(1 for s in tb.examples if s.routed_experts is not None)
    print(
        f"[summary] examples={len(tb.examples)} re_samples={re_n} re_bytes={re_bytes/1e6:.1f}MB"
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    requested = args.modes.split(",")
    results = []
    for name in requested:
        if name not in MODE_DISPATCH:
            print(f"[skip] unknown mode '{name}'", file=sys.stderr)
            continue
        print(f"\n[mode] {name}")
        r = await run_mode(name, tb, outdir, args.repeats, args.monitor_interval)
        results.append(r)
        print(f"  elapsed={fmt_ms(r['elapsed_total_s'])} per_call={fmt_ms(r['per_call_s'])}")
        print(f"  bytes/call={r['bytes_per_call']/1e6:.1f}MB")
        print(f"  lag:    {fmt(r['lag'])}")
        print(f"  quiet:  {fmt(r['quiet'])}")

    # Write CSV.
    csv_path = outdir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "repeats",
                "per_call_s",
                "bytes_per_call_mb",
                "lag_max_s",
                "lag_p99_s",
                "lag_p90_s",
                "lag_mean_s",
                "lag_n",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["mode"],
                    r["repeats"],
                    f"{r['per_call_s']:.4f}",
                    f"{r['bytes_per_call']/1e6:.1f}",
                    f"{r['lag'].get('max',0):.4f}",
                    f"{r['lag'].get('p99',0):.4f}",
                    f"{r['lag'].get('p90',0):.4f}",
                    f"{r['lag'].get('mean',0):.4f}",
                    r["lag"].get("n", 0),
                ]
            )
    print(f"\n[csv] wrote {csv_path}")

    # Also write json for richer inspection.
    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[json] wrote {outdir/'results.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch",
        default="/beegfs/outputs/qwen30b-rlm-router-replay-debug/run_default/rollouts/step_3/train_rollouts.bin",
        help="Path to saved TrainingBatch msgpack blob (orch-side train_rollouts.bin).",
    )
    p.add_argument("--modes", default="baseline,stream,sidecar,stream_sidecar,to_thread")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--monitor-interval", type=float, default=0.01,
                   help="Lag-monitor sample interval (seconds). Smaller = finer measurement.")
    p.add_argument("--outdir", default="/tmp/perf_r3_out")
    p.add_argument("--uvloop", action="store_true", help="Install uvloop as the event loop policy.")
    args = p.parse_args()

    if args.uvloop:
        try:
            import uvloop  # type: ignore

            uvloop.install()
            print("[uvloop] installed")
        except ImportError:
            print("[uvloop] NOT INSTALLED — pip install uvloop to enable", file=sys.stderr)
            sys.exit(2)

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
