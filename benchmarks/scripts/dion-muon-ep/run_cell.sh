#!/usr/bin/env bash
# Run one A/B cell inside an existing Slurm allocation.
#
#   run_cell.sh <arm worktree> <p1|p2|f8> <run name> <comma-separated hosts> [extra sft args...]
#
# p1 runs single-node torchrun on the one host through srun. p2 and f8 render the launcher's sbatch
# script with --dry-run and run it attached to the allocation on exactly the given hosts.
set -euo pipefail

ARM=$1 CELL=$2 RUN=$3 HOSTS=$4
shift 4
ALLOC=${ALLOC:-3409}
HARNESS=$ARM/benchmarks/scripts/dion-muon-ep
export PRL_OUTPUT_DIR=${PRL_OUTPUT_DIR:-/home/garrett/prl_output_dir}
export TILELANG_CACHE_DIR=$HOME/tmp/profiling/caches/$RUN/tilelang
export TRITON_CACHE_DIR=/tmp/$USER/$RUN/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/$RUN/inductor
RUN_DIR=$PRL_OUTPUT_DIR/$RUN
NUM_NODES=$(tr ',' '\n' <<<"$HOSTS" | wc -l)

cd "$ARM"
echo "arm=$ARM commit=$(git rev-parse HEAD) cell=$CELL run=$RUN hosts=$HOSTS"

if [ "$CELL" = p1 ]; then
    [ "$NUM_NODES" -eq 1 ] || { echo "p1 takes one host" >&2; exit 1; }
    exec srun --jobid="$ALLOC" --overlap -N1 -n1 --nodelist="$HOSTS" \
        uv run --no-sync sft @ "$HARNESS/common.toml" @ "$HARNESS/p1.toml" --run.name "$RUN" "$@"
fi

uv run --no-sync sft @ "$HARNESS/common.toml" @ "$HARNESS/$CELL.toml" --run.name "$RUN" --dry-run "$@"

# srun numbers tasks in the allocation's node order, not the order given, and the script assigns
# node ranks by its host list, so feed it the hosts in SLURM_PROCID order.
ORDERED=$(srun --jobid="$ALLOC" --overlap -N"$NUM_NODES" --ntasks-per-node=1 --nodelist="$HOSTS" \
    bash -c 'echo "$SLURM_PROCID $(hostname -s)"' | sort -n | awk '{print $2}' | paste -sd,)
ATTACH="--jobid=$ALLOC --overlap -N$NUM_NODES --ntasks-per-node=1 --nodelist=$ORDERED"
sed -e "s|^srun bash -s|srun $ATTACH bash -s|" \
    -e "s|^srun --kill-on-bad-exit=1 bash -s|srun $ATTACH --kill-on-bad-exit=1 bash -s|" \
    "$RUN_DIR/launcher/sft.sbatch" >"$RUN_DIR/launcher/sft.attached.sh"
[ "$(grep -c -- "--jobid=$ALLOC" "$RUN_DIR/launcher/sft.attached.sh")" -eq 2 ] || {
    echo "expected to attach exactly two srun steps" >&2
    exit 1
}

export SLURM_JOB_ID=$ALLOC SLURM_JOB_NODELIST=$ORDERED SLURM_JOB_NUM_NODES=$NUM_NODES
echo "ordered hosts: $ORDERED"
exec bash "$RUN_DIR/launcher/sft.attached.sh"
