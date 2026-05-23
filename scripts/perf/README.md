# Orchestrator performance harness

Offline tools for attributing orchestrator event-loop lag at step boundaries.
The orchestrator dumps each step's raw rollouts to a pickle when
`orchestrator.dump_raw_rollouts = true`; these scripts replay the post-rollout
phases against that pickle so changes can be benchmarked without a SLURM run.

## Scripts

- `lag_monitor.py` — 1 ms-granularity event-loop lag sampler used by the other
  scripts. Standalone, no prime-rl imports.
- `replay_orch_step.py` — runs the full orch step boundary (advantages,
  filters, save, pretokenize, interleave-gather, batch-send, dataframes) once
  with per-phase wallclock + lag stats.
- `replay_orch_attribution.py` — same input, but runs each phase **alone** with
  fine-grained lag measurement so blockers are attributed by phase. Supports
  `--synthetic-traffic N` to spawn N background loop tasks that simulate the
  prod orchestrator's competing work (active rollouts, HTTP recv handlers,
  inference replies). The right knob for finding "what blocks the loop."
- `replay.py` — older microbench focused on the encode/sender side. Useful for
  iterating on `FileSystemTrainingBatchSender` without spinning up the trainer.

## Typical workflow

```bash
# 1. Run training with raw-rollout dumping
#    set in your orchestrator config:
#      dump_raw_rollouts = true
#    rollouts are written to <output_dir>/run_default/raw_rollouts/step_N.pkl

# 2. Pick a worst-case step (longer rollouts → more loop pressure)
DUMP=/path/to/run_default/raw_rollouts/step_8.pkl

# 3. Run the attribution
uv run python scripts/perf/replay_orch_attribution.py \
  --dump $DUMP \
  --output-dir /tmp/replay_attr \
  --gather-chunk-size 128 \
  --synthetic-traffic 200 \
  --quick
```

The `--quick` flag skips the cumulative-levels pass (which re-loads the pickle
seven times). Per-phase-alone is enough signal in most cases.

## What to look for

For each phase the harness reports:
- `wall=` — wallclock duration of the phase
- `lag` — event-loop lag stats *measured at 1 ms granularity during the phase*

If `wall` is large but `max` lag is small (< ~50 ms) the phase is parallel /
GIL-releasing and is **not blocking the loop** — wallclock alone is misleading.
If `wall ≈ max` lag, the phase is GIL-bound and starves the loop for that
duration. Those are the blockers worth fixing.
