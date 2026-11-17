#!/usr/bin/env bash
# Summarize DSv4 proxy A/B runs with the profiling skill scripts, inside one srun step of <job>; every invocation is appended to commands.sh.
# Usage: analyze.sh <job> <run dir> ...
# Run dirs are named <node label>-<arm><variant>-r<k>[-retry]: the node label is the summary group, the rest the variant.
# Pick a job with no timing run in progress. BUSY_WATTS (default 400; idle B200s draw about 190 W) sets gpu_clocks.py --busy-watts.
set -euo pipefail

ROOT=$INVESTIGATION
SCRIPTS=$PROFILING_SCRIPTS
OUT=$ROOT/derived
BUSY_WATTS=${BUSY_WATTS:-400}

(($# >= 2)) || {
    echo "usage: analyze.sh <job> <run dir> ..." >&2
    exit 1
}
JOB=$1
shift

if [[ -z ${SLURM_STEP_ID:-} ]]; then
    exec srun --jobid="$JOB" --overlap -N1 -n1 "$(readlink -f "$0")" "$JOB" "$@"
fi

record_and_run() {
    local out=$1
    shift
    printf '%s> %q\n' "$(printf '%q ' "$@")" "$out" >>"$ROOT/commands.sh"
    "$@" >"$out" || echo "failed (exit $?): $*" >&2
}

mkdir -p "$OUT/compile_events" "$OUT/gpu_clocks"
runs=()
clock_logs=()
for dir in "$@"; do
    dir=$(readlink -f "$dir")
    name=$(basename "$dir")
    runs+=(--run "${name%%-*}:${name#*-}=$dir")
    record_and_run "$OUT/compile_events/$name.md" uv run --script "$SCRIPTS/compile_events.py" "$dir" \
        --row-filter producer=trainer --events-csv "$OUT/compile_events/$name.csv"
    if [[ -f $dir/clocks_100ms.csv ]]; then
        clock_logs+=("$name=$dir/clocks_100ms.csv")
        record_and_run "$OUT/gpu_clocks/$name.md" uv run --script "$SCRIPTS/gpu_clocks.py" --per-gpu \
            --busy-watts "$BUSY_WATTS" "$name=$dir/clocks_100ms.csv"
    else
        echo "no clocks_100ms.csv in $dir" >&2
    fi
done

summaries=(--prefer producer=trainer --summarize "fwd_bwd_s=time/forward_backward" --summarize "step_s=time/step"
    --summarize "peak_gib=perf/peak_memory")
record_and_run "$OUT/ab_summary_steady4-20.md" uv run --script "$SCRIPTS/run_metrics.py" \
    --out "$OUT/ab_summary_steady4-20.csv" --steady 4 20 "${summaries[@]}" "${runs[@]}"
record_and_run "$OUT/ab_summary_all_steps.md" uv run --script "$SCRIPTS/run_metrics.py" \
    --out "$OUT/ab_summary_all_steps.csv" "${summaries[@]}" "${runs[@]}"
if ((${#clock_logs[@]} > 0)); then
    record_and_run "$OUT/gpu_clocks/all_runs.md" uv run --script "$SCRIPTS/gpu_clocks.py" --busy-watts "$BUSY_WATTS" \
        "${clock_logs[@]}"
fi
