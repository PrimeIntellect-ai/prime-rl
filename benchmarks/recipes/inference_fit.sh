#!/bin/bash
#SBATCH --job-name=recipe-inference-fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

# Sweep inference-server variants of one model on one node under the synthetic agentic workload.
#
#   sbatch --partition=dev --output=<out>/job.log benchmarks/recipes/inference_fit.sh \
#       <out> benchmarks/recipes/<model>/inference.toml benchmarks/recipes/<model>/inference_variants.txt
#
# Each line of the variants file is `<name> <dp> <concurrency,...> <cli overrides...>`. Blank
# lines and `#` lines are skipped. Summarize with summarize_inference.py.

set -u
OUT=$(realpath -m "$1")
BASE=$(realpath "$2")
VARIANTS=$(realpath "$3")
WARMUP=${WARMUP:-90}
DURATION=${DURATION:-240}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1500}
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}

mkdir -p "$OUT"
cd "$PROJECT_DIR"
[ -f .env ] && source .env
source .venv/bin/activate
uv sync --all-extras --all-packages --frozen >/dev/null

MODEL=$(python -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['vllm']['model'])" "$BASE")
echo "node=$(hostname) commit=$(git rev-parse --short HEAD) base=$BASE model=$MODEL" | tee "$OUT/job.meta"
cp "$BASE" "$VARIANTS" "$OUT/"

stop_server() {
    kill -TERM "$1" 2>/dev/null
    sleep 15
    pkill -9 -f "vllm|prime_rl.entrypoints.inference|vllm-router" 2>/dev/null
    sleep 5
}

while read -r name dp concurrencies overrides; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    dir="$OUT/$name"
    mkdir -p "$dir"
    echo "=== $(date -u +%H:%M:%S) start $name: $overrides"
    start=$(date +%s)
    # shellcheck disable=SC2086
    uv run --frozen inference @ "$BASE" --output-dir "$dir" $overrides >"$dir/server.log" 2>&1 </dev/null &
    server=$!
    ready=0
    for _ in $(seq "$((STARTUP_TIMEOUT / 10))"); do
        if curl -sf localhost:8000/health >/dev/null 2>&1 && curl -sf localhost:8100/v1/models >/dev/null 2>&1; then
            ready=1; break
        fi
        kill -0 "$server" 2>/dev/null || break
        sleep 10
    done
    echo "$(( $(date +%s) - start ))" >"$dir/startup_s"
    if [ "$ready" != 1 ]; then
        echo "=== $name: server not ready (see $dir/server.log)"
        echo "not_ready" >"$dir/status"
        stop_server "$server"
        continue
    fi
    echo "=== $(date -u +%H:%M:%S) $name ready after $(cat "$dir/startup_s")s"
    for c in ${concurrencies//,/ }; do
        python benchmarks/recipes/bench_inference.py --model "$MODEL" --concurrency "$c" --dp "$dp" \
            --warmup "$WARMUP" --duration "$DURATION" --out "$dir/c$c.json" >"$dir/c$c.log" 2>&1
        echo "=== $(date -u +%H:%M:%S) $name c=$c rc=$? $(python -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"out/s/gpu={r['output_tok_s_per_gpu']:.0f} prefill/s={r['engine_prompt_tok_s']:.0f} kv={r['kv_usage_mean']} preempt={r['preemptions']} err={r['errors']}\")" "$dir/c$c.json" 2>/dev/null)"
    done
    echo "ok" >"$dir/status"
    stop_server "$server"
done <"$VARIANTS"

python benchmarks/recipes/summarize_inference.py "$OUT" | tee "$OUT/summary.md"
