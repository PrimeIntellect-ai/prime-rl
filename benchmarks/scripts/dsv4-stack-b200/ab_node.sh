#!/usr/bin/env bash
# Run DSv4 proxy run specs strictly serially on one node, each with a 100 ms clock log, JIT cache listings and a hang watchdog.
# Usage: ab_node.sh <job> <node label> <arm>:<run name>[:<extra overrides>] ...
# Extra overrides are split on whitespace (quotes are not interpreted); the word <TARGETS> is replaced, when the run
# starts, by the JSON list in inputs/sac_targets.json. DRY_RUN=1 prints every command instead of running it.
# FRESH_CUTE_DSL_CACHE=1 gives each run its own CUTE_DSL_CACHE_DIR inside run_arm.sh's per-run cache dir.
# AB_ROOT, AB_ARM_ROOT and AB_OUT_ROOT redirect the logs and notes, the arm worktrees and the run dirs, for testing the driver.
set -Eeuo pipefail

INVESTIGATION=$INVESTIGATION
ROOT=${AB_ROOT:-$INVESTIGATION}
ARM_ROOT=${AB_ARM_ROOT:-$WORKTREES}
NOTES=$ROOT/NOTES.md
COMMANDS=$ROOT/commands.sh
SAC_TARGETS_FILE=${SAC_TARGETS_FILE:-$INVESTIGATION/inputs/sac_targets.json}
TARGETS_WORD='<TARGETS>'
KNOWN_ARMS=" main 4354 4355 4356 4357 4358 "
DRY_RUN=${DRY_RUN:-0}
FRESH_CUTE_DSL_CACHE=${FRESH_CUTE_DSL_CACHE:-0}
WATCH_POLL_S=${WATCH_POLL_S:-30}
LOG_STALL_S=${LOG_STALL_S:-1200}
FIRST_STEP_DEADLINE_S=${FIRST_STEP_DEADLINE_S:-2700}
STEP_STALL_S=${STEP_STALL_S:-1200}
TARGETS_POLL_S=60
IDLE_POLL_S=30
IDLE_WAIT_MAX_S=600
IDLE_MAX_MIB=1024
CLOCK_QUERY=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active,memory.used
BASE_OVERRIDES_TEXT='--model.debug.force-balanced-routing --model.debug.num-layers 6 --no-monitors.prime --model.name "$DSV4_CKPT" --model.conversion-dir "$DSV4_CONV"'

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [%s] %s\n' "$(utc)" "${NODE:-?}" "$*" >&2; }
die() { log "error: $*"; exit 1; }
dry() { [[ $DRY_RUN == 1 ]]; }
q() { printf '%q' "$1"; }

append() {
    local file=$1 text=$2
    if dry; then
        printf 'DRY: append to %s:\n%s\n' "$file" "$text" >&2
        return 0
    fi
    { flock -x 9; printf '%s\n' "$text" >&9; } 9>>"$file"
}

node_step() {
    if dry; then
        printf 'DRY: srun --jobid=%s --overlap -N1 -n1 bash -c %s\n' "$JOB" "$(q "$1")" >&2
        return 0
    fi
    srun --jobid="$JOB" --overlap -N1 -n1 bash -c "$1"
}

read_sac_targets() {
    local targets
    while true; do
        if [[ -f $SAC_TARGETS_FILE ]] && targets=$(jq -ce \
            'if type == "array" and length > 0 and all(.[]; type == "string") then . else error("expected a non-empty JSON list of strings") end' \
            "$SAC_TARGETS_FILE"); then
            printf '%s' "$targets"
            return 0
        fi
        if dry; then
            log "DRY: $SAC_TARGETS_FILE missing or invalid: would poll every ${TARGETS_POLL_S}s; keeping $TARGETS_WORD"
            printf '%s' "$TARGETS_WORD"
            return 0
        fi
        log "waiting for a valid $SAC_TARGETS_FILE (polling every ${TARGETS_POLL_S}s)"
        sleep "$TARGETS_POLL_S"
    done
}

resolve_extra() {
    local words=() word
    read -r -a words <<<"$1"
    RESOLVED=()
    for word in "${words[@]}"; do
        if [[ $word == "$TARGETS_WORD" ]]; then
            word=$(read_sac_targets)
        fi
        RESOLVED+=("$word")
    done
}

wait_gpus_idle() {
    local waited=0 used
    while true; do
        used=$(node_step "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits") || used="query failed"
        if [[ $used != "query failed" ]] && ! awk -v max="$IDLE_MAX_MIB" '$1 + 0 > max { busy = 1 } END { exit !busy }' <<<"$used"; then
            return 0
        fi
        if ((waited >= IDLE_WAIT_MAX_S)); then
            IDLE_REPORT=${used//$'\n'/,}
            return 1
        fi
        log "GPUs not idle (memory.used MiB: ${used//$'\n'/,}), waiting ${IDLE_POLL_S}s"
        sleep "$IDLE_POLL_S"
        waited=$((waited + IDLE_POLL_S))
    done
}

start_clock_log() {
    local csv=$1 script
    script="exec nvidia-smi --query-gpu=$CLOCK_QUERY --format=csv -lms 100 > $(q "$csv")"
    CLOCK_PID=""
    if dry; then
        printf 'DRY: srun --jobid=%s --overlap -N1 -n1 bash -c %s > %s 2>&1 &\n' "$JOB" "$(q "$script")" "$CLOCK_LOG" >&2
        return 0
    fi
    srun --jobid="$JOB" --overlap -N1 -n1 bash -c "$script" >"$CLOCK_LOG" 2>&1 &
    CLOCK_PID=$!
}

stop_clock_log() {
    if dry; then
        printf 'DRY: kill -TERM <clock srun pid>, wait up to 30 s, then kill -KILL\n' >&2
        return 0
    fi
    [[ -n $CLOCK_PID ]] || return 0
    kill -TERM "$CLOCK_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$CLOCK_PID" 2>/dev/null && ((waited < 30)); do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$CLOCK_PID" 2>/dev/null; then
        log "clock srun $CLOCK_PID ignored SIGTERM for 30s, sending SIGKILL"
        kill -KILL "$CLOCK_PID" 2>/dev/null || true
    fi
    wait "$CLOCK_PID" 2>/dev/null || true
}

check_orphan_smi() {
    local found
    found=$(node_step 'pgrep -a nvidia-smi || true') || found="pgrep step failed"
    [[ -z $found ]] && return 0
    log "nvidia-smi still running on the node after $RUN: ${found//$'\n'/; }; killing it"
    node_step "pkill -u $(q "$USER") -x nvidia-smi; sleep 2; pgrep -a nvidia-smi || true" >&2 || true
    ORPHAN_REPORT="orphan nvidia-smi after run: ${found//$'\n'/; }"
}

write_cache_listings() {
    local cache=/tmp/$USER/dsv4-perf-proxy/$RUN script
    script="set -u; h=\$(hostname); cd $(q "$LOG_DIR")
find $(q "$cache/triton") -mindepth 2 -maxdepth 2 -name '*.cubin' -printf '%T@ %f\n' > triton_cache_\$h.txt
[[ -e deep_gemm_cache_\$h.txt ]] || find $(q "$cache/deep_gemm/cache") -mindepth 1 -maxdepth 1 -printf '%T@ %f\n' > deep_gemm_cache_\$h.txt"
    if ! dry && [[ ! -d $LOG_DIR ]]; then
        return 0
    fi
    node_step "$script" >&2 || log "cache listing for $RUN failed (see the lines above)"
}

rank_logs() {
    find "$LOG_DIR/trainer/torchrun" -name 'std*.log' 2>/dev/null | sort
}

capture_hang() {
    local wd=$RUN_DIR/watchdog f rank
    mkdir -p "$wd"
    printf '%s %s\n' "$(utc)" "$HANG_REASON" >"$wd/reason.txt"
    while read -r f; do
        [[ -n $f ]] || continue
        rank=$(basename "$(dirname "$f")")
        tail -n 50 "$f" >"$wd/rank${rank}_$(basename "$f" .log).tail"
    done < <(rank_logs)
    tail -n 50 "$LOG_DIR/trainer.log" >"$wd/trainer.log.tail" 2>/dev/null || true
    node_step 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv; nvidia-smi' \
        >"$wd/nvidia_smi.txt" 2>&1 || log "nvidia-smi snapshot failed for $RUN"
}

cancel_run_step() {
    local srun_pid step_id
    srun_pid=$(pgrep -P "$RUN_PID" -x srun | head -n 1 || true)
    if [[ -z $srun_pid ]]; then
        log "no srun child of run_arm.sh pid $RUN_PID found; sending SIGTERM to run_arm.sh"
        kill -TERM "$RUN_PID" 2>/dev/null || true
        return 0
    fi
    step_id=$(scontrol -o show step "$JOB" | grep -E "SrunHost:Pid=[^ ]*:${srun_pid}( |$)" | grep -o 'StepId=[^ ]*' | cut -d= -f2 || true)
    if [[ $step_id != "$JOB".* ]]; then
        log "no step of job $JOB has srun pid $srun_pid; sending SIGTERM to srun"
        kill -TERM "$srun_pid" 2>/dev/null || true
        return 0
    fi
    log "scancel $step_id (srun pid $srun_pid)"
    scancel "$step_id"
    CANCELLED_STEP=$step_id
}

wait_run_exit() {
    local waited=0
    while kill -0 "$RUN_PID" 2>/dev/null && ((waited < 300)); do
        sleep 5
        waited=$((waited + 5))
    done
    if kill -0 "$RUN_PID" 2>/dev/null; then
        log "run_arm.sh $RUN_PID still alive 300s after cancel; killing its process group"
        pkill -KILL -P "$RUN_PID" 2>/dev/null || true
        kill -KILL "$RUN_PID" 2>/dev/null || true
    fi
}

max_steps_for() {
    local toml=$1 i
    shift
    for ((i = 1; i < $#; i++)); do
        if [[ ${!i} == --max-steps ]]; then
            i=$((i + 1))
            printf '%s' "${!i}"
            return 0
        fi
    done
    sed -n 's/^max_steps *= *\([0-9]*\).*/\1/p' "$toml"
}

watch_run() {
    local metrics=$RUN_DIR/monitors/file/metrics.jsonl trainer_log=$LOG_DIR/trainer.log
    local size last_size=0 last_change=$LAUNCHED now log_mtime last_step
    HANG_REASON=""
    TEARDOWN_HANG=0
    while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep "$WATCH_POLL_S"
        kill -0 "$RUN_PID" 2>/dev/null || break
        now=$(date +%s)
        size=$(stat -c %s "$metrics" 2>/dev/null || echo 0)
        if ((size != last_size)); then
            last_size=$size
            last_change=$now
        fi
        if ((size == 0)); then
            log_mtime=$(stat -c %Y "$trainer_log" 2>/dev/null || echo "$LAUNCHED")
            if ((now - log_mtime >= LOG_STALL_S && now - LAUNCHED >= FIRST_STEP_DEADLINE_S)); then
                HANG_REASON="no metrics $((now - LAUNCHED))s after launch and trainer.log unchanged for $((now - log_mtime))s"
                return 1
            fi
        elif ((now - last_change >= STEP_STALL_S)); then
            last_step=$(tail -n 1 "$metrics" | jq -r '.step // empty' 2>/dev/null || true)
            HANG_REASON="no new metrics line for $((now - last_change))s (last step ${last_step:-unknown})"
            if [[ $last_step == "$MAX_STEPS" ]]; then
                TEARDOWN_HANG=1
                HANG_REASON="$HANG_REASON: hung after the final step"
            fi
            return 1
        fi
    done
    return 0
}

failure_tail() {
    local f
    printf '%s\n' "  \`\`\`"
    printf '  ==> %s (last 20 lines) <==\n' "$LOG_DIR/trainer.log"
    tail -n 20 "$LOG_DIR/trainer.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g; s/^/  /' || true
    while read -r f; do
        [[ -s $f ]] || continue
        printf '  ==> %s (last 5 lines) <==\n' "$f"
        tail -n 5 "$f" | sed 's/\x1b\[[0-9;]*m//g; s/^/  /'
    done < <(rank_logs)
    printf '%s\n' "  \`\`\`"
}

launch_run() {
    local arm=$1
    RUN=$2
    shift 2
    local worktree=$ARM_ROOT/prime-rl-arm-$arm
    local run_arm=$worktree/benchmarks/scripts/dsv4-perf-proxy/run_arm.sh
    local extra_text="" sha start end rc=0 waited
    RUN_DIR=$OUT_ROOT/$RUN
    LOG_DIR=$RUN_DIR/logs/attempt_1
    CLOCK_LOG=$ROOT/logs/clocks-$RUN.log
    OUTCOME=ok
    IDLE_REPORT=""
    ORPHAN_REPORT=""
    CANCELLED_STEP=""
    if (($# > 0)); then
        extra_text=$(printf '%q ' "$@")
        extra_text=${extra_text% }
    fi
    sha=$(git -C "$worktree" rev-parse HEAD)
    MAX_STEPS=$(max_steps_for "$worktree/benchmarks/scripts/dsv4-perf-proxy/proxy.toml" "$@")
    start=$(utc)
    local run_env=() env_text=""
    if [[ $FRESH_CUTE_DSL_CACHE == 1 ]]; then
        run_env=(env "CUTE_DSL_CACHE_DIR=/tmp/$USER/dsv4-perf-proxy/$RUN/cute_dsl")
        env_text="${run_env[*]} "
    fi
    local cmd=("${run_env[@]}" "$run_arm" "$JOB" "$RUN" --model.debug.force-balanced-routing --model.debug.num-layers 6 --no-monitors.prime
        --model.name "$DSV4_CKPT" --model.conversion-dir "$DSV4_CONV" "$@")

    if [[ -e $RUN_DIR ]]; then
        OUTCOME="skipped: run dir already exists"
        rc=1
    elif ! wait_gpus_idle; then
        OUTCOME="skipped: GPUs still busy after ${IDLE_WAIT_MAX_S}s (memory.used MiB: $IDLE_REPORT)"
        rc=1
    else
        log "launch $RUN: arm $arm @ $sha, extra overrides: ${extra_text:-none}"
        append "$COMMANDS" "source $INVESTIGATION/env.sh && $env_text$(q "$run_arm") $JOB $(q "$RUN") $BASE_OVERRIDES_TEXT${extra_text:+ $extra_text}"
        if dry; then
            printf 'DRY: %s > %s 2>&1 &\n' "$(printf '%q ' "${cmd[@]}")" "$ROOT/logs/run_arm-$RUN.log" >&2
            printf 'DRY:   (run_arm.sh then runs: srun --jobid=%s --overlap -N1 -n1 --gres=gpu:8 bash -c <trainer launch>)\n' "$JOB" >&2
            printf 'DRY: wait for %s, then start the clock log\n' "$RUN_DIR/overrides.txt" >&2
            start_clock_log "$RUN_DIR/clocks_100ms.csv"
            printf 'DRY: watchdog every %ss on %s (before step 1: trainer.log idle %ss and %ss since launch; after: %ss without a new line; last step %s)\n' \
                "$WATCH_POLL_S" "$RUN_DIR/monitors/file/metrics.jsonl" "$LOG_STALL_S" "$FIRST_STEP_DEADLINE_S" "$STEP_STALL_S" "$MAX_STEPS" >&2
            stop_clock_log
        else
            LAUNCHED=$(date +%s)
            "${cmd[@]}" >"$ROOT/logs/run_arm-$RUN.log" 2>&1 &
            RUN_PID=$!
            waited=0
            while [[ ! -e $RUN_DIR/overrides.txt ]] && kill -0 "$RUN_PID" 2>/dev/null && ((waited < 120)); do
                sleep 1
                waited=$((waited + 1))
            done
            CLOCK_PID=""
            if [[ -e $RUN_DIR/overrides.txt ]]; then
                start_clock_log "$RUN_DIR/clocks_100ms.csv"
            else
                log "run dir of $RUN not created after ${waited}s; no clock log"
            fi
            if ! watch_run; then
                log "watchdog: $RUN: $HANG_REASON"
                capture_hang
                cancel_run_step
                wait_run_exit
                OUTCOME=hang
                ((TEARDOWN_HANG == 0)) || OUTCOME=teardown-hang
            fi
            wait "$RUN_PID" || rc=$?
            stop_clock_log
        fi
        check_orphan_smi
        write_cache_listings
    fi

    end=$(utc)
    if [[ $OUTCOME == ok && $rc != 0 ]]; then
        OUTCOME="failed"
    fi
    log "$RUN finished: exit $rc ($OUTCOME)"
    append "$NOTES" "- ab run \`$RUN\`: start $start, end $end, node $NODE, job $JOB, host $HOST, arm $arm, worktree HEAD $sha, ${env_text:-CuTe DSL cache node-shared, }extra overrides \`${extra_text:-none}\`, exit $rc ($OUTCOME)${ORPHAN_REPORT:+, $ORPHAN_REPORT}"
    if [[ $OUTCOME == hang || $OUTCOME == teardown-hang ]]; then
        append "$NOTES" "  - watchdog \`$RUN\`: $HANG_REASON; captured $RUN_DIR/watchdog/; cancelled step ${CANCELLED_STEP:-none (signalled srun or run_arm.sh instead)}"
    elif [[ $OUTCOME == failed ]] && ! dry; then
        append "$NOTES" "  - failed \`$RUN\` with exit $rc; tail of rank logs:
$(failure_tail)"
    fi
}

run_spec() {
    local arm=$1 run=$2 extra=$3
    resolve_extra "$extra"
    launch_run "$arm" "$run" "${RESOLVED[@]}"
    if [[ $OUTCOME != hang ]]; then
        return 0
    fi
    log "relaunching $run once as $run-retry"
    append "$NOTES" "  - watchdog: relaunching \`$run\` once as \`$run-retry\`"
    resolve_extra "$extra"
    launch_run "$arm" "$run-retry" "${RESOLVED[@]}"
    if [[ $OUTCOME == hang ]]; then
        append "$NOTES" "  - watchdog: \`$run-retry\` hung again; moving on to the next spec"
    fi
}

main() {
    (($# >= 3)) || die "usage: ab_node.sh <job> <node label> <arm>:<run name>[:<extra overrides>] ..."
    JOB=$1
    NODE=$2
    shift 2
    source "$INVESTIGATION/env.sh"
    OUT_ROOT=${AB_OUT_ROOT:-$OUT_ROOT}
    : "${OUT_ROOT:?} ${DSV4_CKPT:?} ${DSV4_CONV:?}"
    HOST=$(squeue -h -j "$JOB" -o %N)
    [[ -n $HOST ]] || die "job $JOB is not running"
    mkdir -p "$ROOT/logs"

    local spec arm rest run extra seen=" " arms=() runs=() extras=() i
    for spec in "$@"; do
        arm=${spec%%:*}
        rest=${spec#*:}
        [[ $spec == *:* && -n $rest ]] || die "bad run spec '$spec'"
        run=${rest%%:*}
        extra=""
        [[ $rest == *:* ]] && extra=${rest#*:}
        [[ $KNOWN_ARMS == *" $arm "* ]] || die "unknown arm '$arm' in '$spec'"
        [[ -x $ARM_ROOT/prime-rl-arm-$arm/benchmarks/scripts/dsv4-perf-proxy/run_arm.sh ]] || die "no run_arm.sh in arm $arm"
        [[ $run =~ ^${NODE}-${arm}[a-z]*-r[0-9]+$ ]] || die "run name '$run' does not match <node label>-<arm><variant>-r<k>"
        [[ $seen != *" $run "* ]] || die "duplicate run name $run"
        [[ ! -e $OUT_ROOT/$run && ! -e $OUT_ROOT/$run-retry ]] || die "run dir for $run already exists under $OUT_ROOT"
        if [[ " $extra " == *" $TARGETS_WORD "* ]]; then
            command -v jq >/dev/null || die "jq is required for $TARGETS_WORD"
        fi
        seen+="$run "
        arms+=("$arm")
        runs+=("$run")
        extras+=("$extra")
    done

    trap 'log "driver error at line $LINENO: $BASH_COMMAND"' ERR
    log "job $JOB on $HOST: ${#runs[@]} runs: ${runs[*]}"
    for i in "${!runs[@]}"; do
        run_spec "${arms[$i]}" "${runs[$i]}" "${extras[$i]}"
    done
    log "all runs done"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
