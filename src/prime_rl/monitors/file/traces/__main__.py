"""``uv run python -m prime_rl.monitors.file.traces <run_dir> [<trace_id>]`` — the run's
live (in-flight) traces: a table of them, or one assembled from its deltas."""

from prime_rl.monitors.file.traces.live import main

main()
