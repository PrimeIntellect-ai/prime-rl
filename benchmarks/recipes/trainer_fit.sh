#!/bin/bash
#SBATCH --job-name=recipe-trainer-fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

# Sweep trainer-only variants of one model on one node with fake 256k samples.
#
#   sbatch --partition=dev --output=<out>/job.log benchmarks/recipes/trainer_fit.sh \
#       <out> benchmarks/recipes/<model>/trainer.toml benchmarks/recipes/<model>/trainer_variants.txt
#
# Each line of the variants file is `<name> <cli overrides...>`. Blank lines and `#` lines are
# skipped. Every variant runs in its own output dir under <out>; summarize with summarize_trainer.py.

set -u
OUT=$(realpath -m "$1")
BASE=$(realpath "$2")
VARIANTS=$(realpath "$3")
VARIANT_TIMEOUT=${VARIANT_TIMEOUT:-900}
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}

mkdir -p "$OUT"
cd "$PROJECT_DIR"
[ -f .env ] && source .env
source .venv/bin/activate
uv sync --all-extras --all-packages --frozen >/dev/null

echo "node=$(hostname) commit=$(git rev-parse --short HEAD) base=$BASE" | tee "$OUT/job.meta"
cp "$BASE" "$VARIANTS" "$OUT/"

while read -r name overrides; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    dir="$OUT/$name"
    mkdir -p "$dir"
    echo "=== $(date -u +%H:%M:%S) start $name: $overrides"
    start=$(date +%s)
    # shellcheck disable=SC2086
    timeout --signal=TERM --kill-after=60 "$VARIANT_TIMEOUT" \
        torchrun --nproc-per-node=8 --standalone src/prime_rl/trainer/rl/train.py \
        @ "$BASE" --output-dir "$dir" --no-monitors.prime $overrides \
        >"$dir/stdout.log" 2>&1 </dev/null
    rc=$?
    echo "$rc $(( $(date +%s) - start ))" >"$dir/exit"
    echo "=== $(date -u +%H:%M:%S) end $name rc=$rc wall=$(( $(date +%s) - start ))s"
    pkill -9 -f "src/prime_rl/trainer/rl/train.py" 2>/dev/null
    sleep 5
done <"$VARIANTS"

python benchmarks/recipes/summarize_trainer.py "$OUT" | tee "$OUT/summary.md"
