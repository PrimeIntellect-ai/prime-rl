# Canonical Launch Commands

Use `prime_rl.entrypoints.launch` as the single repo-local launch surface. The
`prime-launch` console alias exists after syncing the environment, but
`python -m` works with `uv run --no-sync`.

## RLVR

Dry-run:

```bash
uv run --no-sync python -m prime_rl.entrypoints.launch rlvr \
  --config configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml \
  --output-dir /tmp/prime-rl-dryrun \
  --dry-run
```

Launch:

```bash
uv run --no-sync python -m prime_rl.entrypoints.launch rlvr \
  --config configs/omni_math2/rl_olmo3_dpo_default_8node_28i4t_compile_fsasync4_refill.toml \
  --output-dir outputs/omni_math2_rlvr_canary/<run_name>
```

This deliberately delegates to the existing `rl` entrypoint. The canonical
layer exists to stop hand-writing different `uv run rl ...` variants.

## Offline Eval

Run inside an existing 8-node allocation:

```bash
uv run --no-sync python -m prime_rl.entrypoints.launch offline-eval \
  --in-allocation \
  --arm refill_lr1e6_28i4t \
  --run-root outputs/omni_math2_rlvr_canary/<run_name>/run_default \
  --output-dir outputs/omni_math2_rlvr_canary/<run_name>/offline_eval_600x8_8node_router \
  --steps 25,50,75,100 \
  --nodes 8 \
  --router-policy round_robin
```

Submit as a dependency after a training job:

```bash
uv run --no-sync python -m prime_rl.entrypoints.launch offline-eval \
  --after-job-id <train_job_id> \
  --arm refill_lr1e6_28i4t \
  --run-root outputs/omni_math2_rlvr_canary/<run_name>/run_default \
  --output-dir outputs/omni_math2_rlvr_canary/<run_name>/offline_eval_600x8_8node_router \
  --steps 25,50,75,100 \
  --nodes 8 \
  --sbatch-nodes 8 \
  --router-policy round_robin
```

Defaults are routed 8-node eval, 4 GPUs/node, DP local 4, API server count 4,
router port 9800, backend port 9900, `gpu_memory_utilization=0.95`,
`max_model_len=16384`, `max_num_seqs=192`, and
`max_num_batched_tokens=65536`.

The old scripts remain as compatibility shims:

```bash
scripts/evals/run_omni_math2_offline_eval_28i4t.sh
scripts/evals/submit_omni_math2_postrun_offline_eval.sh <job_id> <arm> <run_root> <out_dir>
```

## Data Generation / Filtering

Run a baseline generation config, then build a perfectible subset from the
resulting rollouts:

```bash
uv run --no-sync python -m prime_rl.entrypoints.launch data \
  --baseline-config configs/baselines/omni_math2_perfectible_olmo3_fp8kv.toml \
  --dataset benchmarks/datasets/omni_math2/omni_math2_olmo3_perfectible_seed42.jsonl \
  --filter-output benchmarks/datasets/omni_math2/perfectible_card/olmo3.jsonl
```

Filter from existing rollouts without regenerating:

```bash
uv run --no-sync python -m prime_rl.entrypoints.launch data \
  --baseline-rollouts outputs/baselines/<run>/eval_rollouts.jsonl \
  --dataset benchmarks/datasets/omni_math2/source.jsonl \
  --filter-output benchmarks/datasets/omni_math2/perfectible.jsonl \
  --low 0.2 \
  --high 0.8 \
  --min-rollouts 8
```
