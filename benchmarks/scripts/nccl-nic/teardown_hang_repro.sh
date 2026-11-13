#!/usr/bin/env bash
# Usage: teardown_hang_repro.sh <prime-rl checkout with synced .venv> <repeats> [pattern ...]
# Runs teardown_hang_repro.py on 8 local GPUs per pattern and prints PASS, HANG or FAIL per run.
set -uo pipefail

checkout=$1
repeats=$2
shift 2
if (( $# )); then
    patterns=("$@")
else
    patterns=(lazy-p2p eager-p2p batched-p2p eager-subgroup-p2p)
fi
script=$(dirname "$(readlink -f "$0")")/teardown_hang_repro.py
python=$checkout/.venv/bin/python
timeout_s=${TIMEOUT_S:-120}
log_dir=${LOG_DIR:-$(mktemp -d)}

nccl=$("$python" -c 'import ctypes, torch; v = ctypes.c_int(); ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(v)); print(v.value)')
echo "checkout $(git -C "$checkout" rev-parse --short HEAD), torch $("$python" -c 'import torch; print(torch.__version__)'), NCCL $nccl, logs $log_dir"

for pattern in "${patterns[@]}"; do
    for run in $(seq "$repeats"); do
        log=$log_dir/$pattern-$run.log
        start=$(date +%s)
        timeout -k 10 "$timeout_s" "$python" -m torch.distributed.run --standalone --nproc-per-node 8 "$script" "$pattern" > "$log" 2>&1
        status=$?
        pkill -KILL -f "$script" 2>/dev/null
        entered=$(grep -o "entering destroy_process_group" "$log" | wc -l)
        destroyed=$(grep -o "destroy_process_group took" "$log" | wc -l)
        if [[ $status -eq 0 && $destroyed -eq 8 ]]; then
            result=PASS
        elif [[ $status -eq 124 || $status -eq 137 ]]; then
            result="HANG ($entered/8 ranks entered destroy_process_group, $destroyed/8 finished)"
        else
            result="FAIL (exit $status)"
        fi
        echo "NCCL $nccl $pattern run $run: $result in $(( $(date +%s) - start )) s"
    done
done
