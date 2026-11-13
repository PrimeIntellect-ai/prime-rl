#!/usr/bin/env bash
# Usage: run_one.sh <prime-rl checkout with synced .venv> <layout toml> <out dir>
# Starts prime-rl's inference server, runs a warmup and a measured vllm bench serve pass, then stops the server with SIGINT and times until every process exits and GPU memory is freed.
set -uo pipefail

checkout=$(readlink -f "$1")
layout=$(readlink -f "$2")
mkdir -p "$3"
out=$(readlink -f "$3")
bench_checkout=${BENCH_CHECKOUT:-$checkout}
model=Qwen/Qwen3-30B-A3B-Instruct-2507
port=8000
backend_port=8100
num_prompts=${NUM_PROMPTS:-1024}
warmup_prompts=${WARMUP_PROMPTS:-256}
max_concurrency=${MAX_CONCURRENCY:-128}
ready_timeout_s=${READY_TIMEOUT_S:-1200}
sigterm_after_s=${SIGTERM_AFTER_S:-90}
hang_timeout_s=${HANG_TIMEOUT_S:-120}
gpu_free_mib=${GPU_FREE_MIB:-1024}

log() { echo "$(date -u +%FT%T) $*" | tee -a "$out/driver.log"; }
gpu_mem_max() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; }
live_session_pids() { ps -o pid=,stat= -s "$sid" 2>/dev/null | awk '$2 !~ /^Z/ {print $1}'; }

dump_stuck() {
    for p in $(live_session_pids); do
        echo "== pid $p: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"
        grep -E '^(Name|State|PPid)' /proc/$p/status 2>/dev/null
        echo "wchan: $(cat /proc/$p/wchan 2>/dev/null)"
        echo "stack:"
        cat /proc/$p/stack 2>&1
        for t in /proc/$p/task/*; do
            echo "task $(basename "$t") comm=$(cat "$t/comm" 2>/dev/null) wchan=$(cat "$t/wchan" 2>/dev/null) stat=$(cat "$t/stat" 2>/dev/null)"
        done
    done
}

log "host $(hostname), checkout $checkout ($(git -C "$checkout" rev-parse --short HEAD)), layout $layout"
if [[ $(gpu_mem_max) -ge $gpu_free_mib ]]; then
    log "ABORT: GPU memory already in use"
    nvidia-smi | tee -a "$out/driver.log"
    exit 1
fi

export PATH="$checkout/.venv/bin:$PATH" VIRTUAL_ENV="$checkout/.venv"
cd "$checkout"
setsid env --default-signal=INT,QUIT NCCL_DEBUG=VERSION "$checkout/.venv/bin/inference" @ "$layout" --output-dir "$out/outputs" > "$out/server.log" 2>&1 < /dev/null &
pid=$!
sleep 1
sid=$(ps -o sid= -p "$pid" | tr -d ' ')
log "server pid $pid, session $sid"

start=$(date +%s)
until curl -sf "localhost:$backend_port/health" > /dev/null && curl -sf "localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 "$pid" 2> /dev/null; then
        log "FAIL: server exited during startup"
        break
    fi
    if (( $(date +%s) - start > ready_timeout_s )); then
        log "FAIL: server not ready after $ready_timeout_s s"
        break
    fi
    sleep 5
done
ready=$(( $(date +%s) - start ))

if kill -0 "$pid" 2> /dev/null && (( ready <= ready_timeout_s )); then
    log "server ready after $ready s"
    bench() {
        local name=$1 prompts=$2 seed=$3
        "$bench_checkout/.venv/bin/vllm" bench serve --backend openai --base-url "http://localhost:$port" --endpoint /v1/completions \
            --model "$model" --dataset-name random --random-input-len 1024 --random-output-len 512 --random-range-ratio 0 --ignore-eos \
            --num-prompts "$prompts" --max-concurrency "$max_concurrency" --request-rate inf --seed "$seed" \
            --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
            --save-result --result-dir "$out" --result-filename "$name.json" --disable-tqdm > "$out/$name.log" 2>&1
        log "bench $name exit $?"
    }
    bench warmup "$warmup_prompts" 1
    bench measured "$num_prompts" 0
fi

nvidia-smi > "$out/nvidia-smi-before-stop.txt"
log "sending SIGINT to process group $pid, as Ctrl-C would"
t0=$(date +%s.%N)
kill -INT -- "-$pid"
elapsed() { awk -v now="$(date +%s.%N)" -v t0="$t0" 'BEGIN { printf "%.1f", now - t0 }'; }
at_least() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
result=CLEAN
procs_exit_s=""
sent_term=0
while true; do
    procs=$(live_session_pids)
    mem=$(gpu_mem_max)
    el=$(elapsed)
    if [[ -z $procs && -z $procs_exit_s ]]; then
        procs_exit_s=$el
        log "all server processes exited after $el s"
    fi
    if [[ -z $procs && $mem -lt $gpu_free_mib ]]; then
        log "GPU memory freed after $el s (max used $mem MiB)"
        break
    fi
    if [[ -n $procs && $sent_term -eq 0 ]] && at_least "$el" "$sigterm_after_s"; then
        log "still alive after $el s, sending SIGTERM to: $(echo $procs)"
        kill -TERM $procs 2> /dev/null
        sent_term=1
        result=CLEAN_AFTER_SIGTERM
    fi
    if at_least "$el" "$hang_timeout_s"; then
        result=HANG
        log "HANG: after $el s, live processes: $(echo $procs), max GPU memory $mem MiB"
        ps -o pid,ppid,stat,etime,cmd -s "$sid" | tee -a "$out/driver.log"
        nvidia-smi > "$out/nvidia-smi-hang.txt"
        dump_stuck > "$out/hang_dump.txt" 2>&1
        kill -KILL $(live_session_pids) 2> /dev/null
        sleep 5
        log "after SIGKILL: live $(echo $(live_session_pids)), max GPU memory $(gpu_mem_max) MiB"
        break
    fi
    sleep 0.5
done
wait "$pid"
rc=$?
log "RESULT $result: inference exit code $rc, processes gone at ${procs_exit_s:-never} s, total $(elapsed) s"
log "leftover processes of $USER: $(ps -u "$USER" -o pid=,comm= | grep -vE 'slurm_script|sleep|ps$|bash$|srun|grep|awk|tr$' | tr '\n' ' ')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$out/driver.log"
grep -m4 -E "NCCL version|nccl==" "$out/server.log" | tee -a "$out/driver.log"
