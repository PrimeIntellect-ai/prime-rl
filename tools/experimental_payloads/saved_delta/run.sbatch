#!/bin/bash
#SBATCH --job-name=hubert-deferred-matched
#SBATCH --nodes=1
#SBATCH --nodelist=ltc-idc3-hgx8-h200-25
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hubert/perplexity/deferred-payload-saved-delta-20261008/job-%j.log
set -euo pipefail
run_root=/home/hubert/perplexity/deferred-payload-saved-delta-20261008
archive=/home/hubert/perplexity/delta-episode-contract-20261008
options=/home/hubert/perplexity/delta-decoding-options-20261008
interpreter=/home/hubert/qwen38-vlm/prime-rl/.venv/bin/python
export PYTHONPATH="$archive/native-source:$options/pinned-deps:$options/extras"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
unset LD_PRELOAD
cd "$run_root"
scontrol show job "$SLURM_JOB_ID" > job-metadata.txt
for tier in smoke full; do
    count=1
    cycles=1
    [[ "$tier" == full ]] && count=32 && cycles=2
    for spec in thread:off:1:1 thread:on:1:1 typed-bytes:on:1:1 typed-views:off:1:1 typed-views:on:1:1 deferred:off:4:1 deferred:off:4:0.25; do
        IFS=: read -r mode callbacks queues consumption <<< "$spec"
        flag=()
        [[ "$callbacks" == on ]] && flag=(--callbacks)
        name="$tier-$mode-callback-$callbacks-consumption-$consumption"
        srun --cpu-bind=cores uv run --no-project --python "$interpreter" \
            code/observe_processes.py "$run_root/observations/$SLURM_JOB_ID/$name" \
            "$interpreter" \
            code/replay.py "$archive" "$run_root/results/$SLURM_JOB_ID/$name" \
            --mode "$mode" --queues "$queues" --replicas "$count" --cycles "$cycles" \
            --consumption "$consumption" "${flag[@]}"
    done
done
