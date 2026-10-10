#!/bin/bash
#SBATCH --job-name=hubert-deferred-exact
#SBATCH --nodes=1
#SBATCH --nodelist=ltc-idc3-hgx8-h200-30
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hubert/perplexity/deferred-payload-saved-delta-20261008/exact-%j.log
set -euo pipefail
run_root=/home/hubert/perplexity/deferred-payload-saved-delta-20261008
archive=/home/hubert/perplexity/delta-episode-contract-20261008
options=/home/hubert/perplexity/delta-decoding-options-20261008
interpreter=/home/hubert/qwen38-vlm/prime-rl/.venv/bin/python
export PYTHONPATH="$archive/native-source:$options/pinned-deps:$options/extras"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
unset LD_PRELOAD
cd "$run_root"
scontrol show job "$SLURM_JOB_ID" > exact-job-metadata.txt
for spec in thread:off:1 thread:on:1 typed-bytes:on:1 typed-views:off:1 typed-views:on:1 delta-pickle:on:4 delta-shm:on:4 owner-shm:off:4; do
    IFS=: read -r mode callbacks queues <<< "$spec"
    flag=()
    [[ "$callbacks" == on ]] && flag=(--callbacks)
    name="exact-$SLURM_JOB_ID-$mode-callback-$callbacks"
    srun --cpu-bind=cores uv run --no-project --python "$interpreter" \
        code/observe_processes.py "$run_root/exact-observations/$name" "$interpreter" \
        code/exact_harness.py "$run_root/legacy/contract_bench_v3.py" "$run_root/legacy" \
        --mode "$mode" --queues "$queues" --replicas 32 --cycles 2 --name "$name" "${flag[@]}"
done
