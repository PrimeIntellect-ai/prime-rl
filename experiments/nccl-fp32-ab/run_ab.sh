#!/usr/bin/env bash
# Launch the NCCL fp32 weight-transfer A/B described in PLAN.md.
#
# Three arms, two replicates each, named <code>-<transport>-<replicate> so the run/job name says
# exactly what ran with no separate legend:
#
#   main-nccl-{a,b}         origin/main (bug present) over the NCCL transport
#   fix-nccl-{a,b}          this branch (fix present) over the NCCL transport
#   main-filesystem-{a,b}   origin/main over the filesystem transport, which already honors
#                           keep_in_fp32_for_weight_transfer and so predicts the post-fix value a
#                           priori. If main-nccl already matches main-filesystem, the experiment
#                           has no power and the config is wrong, not the fix.
#
# The shared config sets trainer.optim.lr = 0, so every arm's weights are frozen at the values
# from the startup broadcast for the whole run: AdamW's update and its decoupled weight-decay
# term are both scaled by lr, so both vanish. That removes per-step policy drift as a confound,
# turning each step into a near-independent repeated measurement of the one fixed transfer effect
# instead of a moving target.
#
# Every arm reads this one config file by absolute path, so the arms differ only in which
# worktree's code runs (--slurm.project-dir) and, for main-filesystem, the transport.
#
# TWO WORKTREES, AND WHICH IS WHICH
#
# This experiment compares two checkouts of the same repo, so "where the script runs" matters more
# than usual:
#
#   AFTER  is the worktree this script file itself lives in, found via `git rev-parse
#          --show-toplevel` on the script's own directory. It is therefore chosen by WHICH COPY OF
#          THIS SCRIPT YOU INVOKE, not by your shell's cwd. Run the copy in the in-progress
#          worktree holding the fix; that is the arm under test.
#
#   BEFORE is the baseline, passed in as BEFORE_DIR. It must be a SEPARATE worktree checked out at
#          this branch's fork point (`git merge-base HEAD origin/main`), i.e. the same code minus
#          the fix. It needs its own venv, synced to the same uv.lock: run
#          `uv sync --all-extras --all-packages --locked` there first. A stale venv there is the
#          single most likely way to get a wrong answer, because a different vLLM version moves
#          the very kernels mismatch_kl compares.
#
# Both worktrees run the same commit for everything except the fix, so any difference in
# mismatch_kl is attributable to the weight-transfer dtype and nothing else.
#
# ENVIRONMENT
#
#   BEFORE_DIR  required. Baseline worktree, as above.
#   PRL_OUTPUT_DIR  required, absolute. prime-rl's standard output-dir variable, so these runs land
#               beside every other run and `uv run dashboard` finds them with no argument. Must be
#               on storage shared with the compute nodes; /tmp is node-local and would strand
#               metrics.jsonl there. Run dirs are named $RUN_PREFIX-<arm> to stay identifiable.
#   CONFIG      optional, defaults to this directory's rl.toml. Point at a sibling toml (e.g.
#               rl-wordle.toml) to run the same fix against a different task/model.
#   RUN_PREFIX  optional, defaults to "nccl-fp32". Set alongside a non-default CONFIG so runs
#               land in their own directories instead of overwriting a prior variant's.
#   ARMS        optional, defaults to all six. Space-separated subset.
#   DRY_RUN     optional. Any non-empty value generates the sbatch without submitting.
#
#   BEFORE_DIR=/path/to/baseline-worktree ./experiments/nccl-fp32-ab/run_ab.sh
#   BEFORE_DIR=... DRY_RUN=1 ./experiments/nccl-fp32-ab/run_ab.sh    # generate sbatch, do not submit
#   BEFORE_DIR=... ARMS="fix-nccl-a main-filesystem-a" ./experiments/nccl-fp32-ab/run_ab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFTER_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
# CONFIG/RUN_PREFIX let a second task variant reuse this script unchanged: point CONFIG at a
# sibling toml (e.g. rl-wordle.toml) and set RUN_PREFIX so its run/job names cannot collide
# with a prior experiment's output directories.
CONFIG="${CONFIG:-$SCRIPT_DIR/rl.toml}"
RUN_PREFIX="${RUN_PREFIX:-nccl-fp32}"

# BEFORE_DIR is named explicitly rather than derived: picking the baseline is the one decision
# that determines what the experiment measures, and it should be a deliberate act.
: "${BEFORE_DIR:?BEFORE_DIR must name the baseline worktree (a separate checkout at this branch fork point, with its own synced venv). See the header of this script.}"
BEFORE_DIR="$(cd "$BEFORE_DIR" && pwd)"
FORK_POINT="$(git -C "$AFTER_DIR" merge-base HEAD origin/main)"
echo "AFTER  (fix):      $AFTER_DIR"
echo "BEFORE (baseline): $BEFORE_DIR"
echo "fork point:        ${FORK_POINT:0:9}"
# prime-rl already resolves output_dir as CLI > config file > $PRL_OUTPUT_DIR > "outputs"
# (packages/prime-rl-configs/src/prime_rl/utils/config.py:12), so this script sets no output path
# of its own and just requires the standard variable. It must be absolute: the "outputs" fallback
# is relative and would resolve inside each project_dir, splitting the arms across two worktrees.
: "${PRL_OUTPUT_DIR:?PRL_OUTPUT_DIR must be set to an absolute path on storage shared with the compute nodes}"
case "$PRL_OUTPUT_DIR" in
  /*) ;;
  *) echo "PRL_OUTPUT_DIR must be absolute, got: $PRL_OUTPUT_DIR" >&2; exit 1 ;;
esac
echo "output dir:        $PRL_OUTPUT_DIR"
ARMS="${ARMS:-main-nccl-a main-nccl-b fix-nccl-a fix-nccl-b main-filesystem-a main-filesystem-b}"
DRY_RUN="${DRY_RUN:-}"

if [ "$AFTER_DIR" = "$BEFORE_DIR" ]; then
  echo "BEFORE_DIR and the script's own worktree are the same path; the arms would run identical code." >&2
  exit 1
fi

# Guard the whole point of the experiment: the arms must differ by exactly the fix.
if ! git -C "$BEFORE_DIR" merge-base --is-ancestor HEAD "$(git -C "$AFTER_DIR" rev-parse HEAD)" 2>/dev/null; then
  echo "warning: $BEFORE_DIR HEAD is not an ancestor of this branch; the arms may differ by more than the fix." >&2
fi

launch() {
  local arm="$1" project_dir="$2" run_name="$3" tags="$4"
  shift 4
  echo "=== $arm  (project_dir=$project_dir, run.name=$run_name)"
  # shellcheck disable=SC2086
  (cd "$project_dir" && uv run rl @ "$CONFIG" \
    --slurm.project-dir "$project_dir" \
    --slurm.job-name "$RUN_PREFIX-$arm" \
    --run.name "$run_name" \
    --monitors.wandb.tags "$tags" \
    "$@" ${DRY_RUN:+--dry-run})
}

for arm in $ARMS; do
  run_name="$RUN_PREFIX-$arm"
  case "$arm" in
    main-nccl-*) launch "$arm" "$BEFORE_DIR" "$run_name" '["code:main","transport:nccl"]' ;;
    fix-nccl-*) launch "$arm" "$AFTER_DIR" "$run_name" '["code:fix","transport:nccl"]' ;;
    main-filesystem-*) launch "$arm" "$BEFORE_DIR" "$run_name" '["code:main","transport:filesystem"]' --weight-broadcast.type filesystem ;;
    *) echo "unknown arm: $arm" >&2; exit 1 ;;
  esac
done
