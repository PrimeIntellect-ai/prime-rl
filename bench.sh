#!/usr/bin/env bash
# A/B benchmark for `model.skip_masked_lm_head_tokens`.
#
# Runs the same SFT config twice — once with the LM head skipping tokens no loss
# component reads, once without — and reports the difference in throughput, MFU
# and peak memory, plus whether the two loss curves agree.
#
#   ./bench.sh                                  # Qwen3-8B on 8 GPUs
#   MODEL=Qwen/Qwen3.5-35B-A3B BATCH_SIZE=32 EXTRA_ARGS="--model.impl custom" ./bench.sh
#   DRY_RUN=1 ./bench.sh                        # validate the configs, run nothing
#
# The saving is (LM head's share of the step) x (fraction of tokens skipped), so
# read `perf/lm_head_token_fraction` first: near 1.0 means this dataset has
# nothing to skip and any throughput delta is noise.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-Qwen/Qwen3-8B}
CONFIG=${CONFIG:-examples/basic/wordle/sft.toml}
GPUS=${GPUS:-8}
SEQ_LEN=${SEQ_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-30}
WARMUP=${WARMUP:-5}          # steps excluded from the averages
TAG=${TAG:-bench}
EXTRA_ARGS=${EXTRA_ARGS:-}
DRY_RUN=${DRY_RUN:-}

ON_RUN="${TAG}-on"
OFF_RUN="${TAG}-off"

echo "=============================================================="
echo " skip_masked_lm_head_tokens A/B"
echo "   model      : ${MODEL}"
echo "   config     : ${CONFIG}"
echo "   gpus       : ${GPUS}"
echo "   seq_len    : ${SEQ_LEN}   batch: ${BATCH_SIZE}   micro: ${MICRO_BATCH_SIZE}"
echo "   steps      : ${MAX_STEPS} (first ${WARMUP} dropped from averages)"
echo "   branch     : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "                $(git status --porcelain 2>/dev/null | grep -q . && echo 'dirty working tree' || echo 'clean')"
echo "=============================================================="

# Fail early rather than benchmarking a checkout without the flag.
if ! uv run python -c "
from prime_rl.configs.trainer import ModelConfig
raise SystemExit(0 if 'skip_masked_lm_head_tokens' in ModelConfig.model_fields else 1)
" 2>/dev/null; then
  echo "ERROR: model.skip_masked_lm_head_tokens is not in this checkout." >&2
  echo "       Check out the branch that adds it before benchmarking." >&2
  exit 1
fi

run_side() {
  local name=$1 run_dir=$2 flag=$3
  echo
  echo ">>> ${name}"
  # shellcheck disable=SC2086
  uv run sft @ "${CONFIG}" \
    --model.name "${MODEL}" \
    --deployment.num-train-gpus "${GPUS}" --deployment.gpus-per-node "${GPUS}" \
    --data.seq-len "${SEQ_LEN}" \
    --data.batch-size "${BATCH_SIZE}" \
    --data.micro-batch-size "${MICRO_BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --no-ckpt --monitors.file --clean \
    --run.name "${run_dir}" \
    ${flag} ${EXTRA_ARGS} ${DRY_RUN:+--dry-run}
}

run_side "skipping ON  (default)" "${ON_RUN}"  ""
run_side "skipping OFF (baseline)" "${OFF_RUN}" "--no-model.skip-masked-lm-head-tokens"

if [[ -n "${DRY_RUN}" ]]; then
  echo
  echo "Dry run complete — both configs parse. Unset DRY_RUN to benchmark."
  exit 0
fi

echo
uv run python - "outputs/${ON_RUN}/metrics.jsonl" "outputs/${OFF_RUN}/metrics.jsonl" "${WARMUP}" <<'PY'
import json, sys
from statistics import mean

on_path, off_path, warmup = sys.argv[1], sys.argv[2], int(sys.argv[3])


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(rs, key):
    return {r["step"]: r[key] for r in rs if key in r and r["step"] > warmup}


on, off = rows(on_path), rows(off_path)

print("=" * 62)
print(" RESULT".ljust(62))
print("=" * 62)

frac = series(on, "perf/lm_head_token_fraction")
if frac:
    f = mean(frac.values())
    print(f"  tokens the LM head scored     {f:>8.1%}   (skipped {1 - f:.1%})")
    if f > 0.95:
        print("  -> almost nothing to skip on this data; expect no speedup.")

print()
for key, label, fmt, agg in [
    ("perf/throughput", "throughput (tok/s)", "{:>10,.0f}", mean),
    ("perf/mfu", "MFU (%)", "{:>10.1f}", mean),
    ("perf/peak_memory", "peak memory (GiB)", "{:>10.1f}", max),
]:
    a, b = series(on, key), series(off, key)
    if not a or not b:
        continue
    va, vb = agg(a.values()), agg(b.values())
    delta = (va - vb) / vb * 100 if vb else 0.0
    print(f"  {label:<24}  on {fmt.format(va)}   off {fmt.format(vb)}   {delta:+6.1f}%")

# The two runs must agree numerically — a diverging loss means a real bug.
la, lb = series(on, "loss/mean"), series(off, "loss/mean")
common = sorted(set(la) & set(lb))
if common:
    worst = max(abs(la[s] - lb[s]) for s in common)
    rel = worst / max(abs(lb[s]) for s in common)
    print()
    print(f"  loss agreement over {len(common)} steps   max |delta| {worst:.2e}  ({rel:.1e} relative)")
    print("  " + ("OK — the two runs are numerically the same."
                  if rel < 1e-3 else
                  "WARNING — losses diverge; this is a correctness bug, not a perf result."))
print("=" * 62)
PY
