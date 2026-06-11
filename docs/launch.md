# Canonical Launch Commands

Use `prime_rl.entrypoints.launch` as the single repo-local launch surface. The
`prime-launch` console alias exists after syncing the environment, but
`python -m` works with `uv run --no-sync`.

Before launching from an allocation, bind the shell to this checkout. The
allocation shell normally already has the Prime-RL environment loaded; source
the env script only if `uv`, `PRIME_RL_ROOT`, or fabric variables are missing.
Export repo `.env` values before launching so W&B/cache/API variables propagate
through `uv run` and nested `srun` children.

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main
# source scripts/env/activate-prime-rl.sh
set -a
source .env
set +a
```

The helper loads the repo-local `.env`, clears any mismatched active venv, sets
`PRIME_RL_ROOT`, and activates `.venv`. Model/cache paths such as `HF_HOME`,
`HF_HUB_CACHE`, and `UV_CACHE_DIR` should live in the repo `.env`; launch
scripts should not hard-code another checkout path.

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

## In-allocation multi-node (lane) launch

On Isambard-AI you hold a Slurm allocation as a node **pool**, then carve it into
**lanes** — each lane is one full `multi_node` run on a disjoint node-slice. This
runs *inside* the held allocation (no `sbatch`): the launcher fans out with
`srun --jobid=$SLURM_JOB_ID --exact -w <slice>` (`--exact`, **not** `--overlap`),
so disjoint slices coexist. See [Scaling § In-allocation multi-node lanes](scaling.md#in-allocation-multi-node-lanes)
for the pool model.

A config runs in-allocation **when it has no `[slurm]` block**. Presence of a
`[slurm]` block switches back to the legacy path that submits a fresh allocation
via `sbatch` (use that when you do *not* already hold nodes). Example lane config:
[`configs/isambard/rl_2node_inalloc.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/configs/isambard/rl_2node_inalloc.toml)
(1 train + 1 infer = a 2-node lane, `gpus_per_node = 4`).

### Hold the nodes

Hold the pool with the cc-wrapper launcher (e.g. 4 nodes), which drops you into a
shell inside the allocation. That shell normally already has the Prime-RL
environment loaded; source the env script only if `uv` or the fabric variables
are missing. Then bind the shell to this checkout and export the repo `.env`:

```bash
launch-script-mnode 4
# source scripts/env/activate-prime-rl.sh
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main
set -a
source .env
set +a
```

### Launch one lane

```bash
uv run --no-sync rl @ <base>.toml @ configs/isambard/rl_2node_inalloc.toml
```

With no `hosts` set, the launcher treats the whole allocation as a single lane.

### Carve 4 nodes into 2+2 (two concurrent lanes)

Two invocations of the same config, each pinned to a disjoint 2-node slice with a
distinct `port_base` (spaced ≥100 apart) and `lane_tag`:

```bash
# lane 0 — nodes nid001000,nid001001
uv run --no-sync rl @ <base>.toml @ configs/isambard/rl_2node_inalloc.toml \
  --deployment.hosts nid001000,nid001001 \
  --deployment.port-base 29500 \
  --deployment.lane-tag ${SLURM_JOB_ID}-lane0 &

# lane 1 — nodes nid001002,nid001003
uv run --no-sync rl @ <base>.toml @ configs/isambard/rl_2node_inalloc.toml \
  --deployment.hosts nid001002,nid001003 \
  --deployment.port-base 29700 \
  --deployment.lane-tag ${SLURM_JOB_ID}-lane1 &

wait   # both lanes run concurrently on disjoint slices
```

The three lane knobs map to `MultiNodeDeploymentConfig` fields:

| Flag | Field | Purpose |
|---|---|---|
| `--deployment.hosts` | `deployment.hosts` | hostnames of this lane's slice (disjoint per lane) |
| `--deployment.port-base` | `deployment.port_base` | base port; every service port is an offset from it; ≥100 apart per lane |
| `--deployment.lane-tag` | `deployment.lane_tag` | namespaces caches, shm, rendezvous-id, and output subdir |

Key facts:

- `gpus_per_node = 4` is **mandatory** on Isambard AIP2 (4 GH200/node); upstream
  examples default to 8.
- Lanes use `srun --exact` (**not** `--overlap`) so each lane gets only its slice.
- The Slingshot fabric is set automatically — the launch path sources
  `scripts/env/isambard-fabric.sh` (`module load brics/nccl`). It is libfabric,
  not InfiniBand: no `ibv_devinfo` / `NCCL_IB_HCA`.

### Pitfalls

These trip people (and agents) consistently — the [`in-alloc-launch`](../skills/in-alloc-launch/SKILL.md) skill carries the same list for agent retrieval.

- **Run `rl` plainly — never under `srun`/`sbatch`.** Inside an allocation `rl` is itself the head-node orchestrator and fans out via its own internal `srun --exact`. Wrapping it in an outer `srun` nests step creation and deadlocks.
- **In-allocation requires NO `[slurm]` block** in the config. A `[slurm]` block switches to the sbatch-submit path; use it only when you do not already hold nodes.
- **Set `weight_broadcast.type = "nccl"` for real non-LoRA runs.** It moves weights GPU→GPU over Slingshot — roughly an order of magnitude faster than the `filesystem` default at multi-billion-parameter scale, which round-trips a checkpoint through Lustre. LoRA runs must keep `filesystem`; PRIME-RL rejects NCCL broadcast when `[trainer.model.lora]` is enabled.
- **Concurrent lanes must be fully disjoint:** distinct host slices, `port-base` ≥100 apart, distinct `lane-tag`. The `lane-tag` namespaces outputs, rollouts, caches, and the broadcast rendezvous; reuse or omit it and concurrent lanes overwrite each other's rollouts (`FileNotFoundError` on `rollouts/step_N`).
- **`vllm-router` must resolve for aarch64 in `uv.lock`.** If multi_node inference dies with `vllm-router: command not found`, the lock is missing the `manylinux_2_28_aarch64` router wheel. After any `pyproject.toml` dependency change, run `uv lock` and commit the regenerated lock alongside the manifest.
- **Use `scripts/sync-prime-rl-env.sh` for GH200 syncs.** Raw `uv sync` can leave the FA2/FA4 namespace invalid on aarch64 — and once uv's build cache is pruned it insists on a full nvcc rebuild of flash-attn just to reconcile the lock. The wrapper excludes flash-attn from the locked sync, restores the locked version from `PRIME_RL_WHEELHOUSE` (default `./wheels`; on Isambard symlink that to the project wheelhouse carrying the pre-built aarch64 wheel), repairs FA4 clobbering, and verifies FA2/FA3/FA4 imports before returning. If FA2 really must be rebuilt, run the wrapper on a compute node with `PRIME_RL_ALLOW_FLASH_ATTN_BUILD=1`.
- **Generated lane component commands must use `uv run --no-sync`.** Plain inner `uv run inference` / `orchestrator` / `torchrun` can resync a shared `.venv` concurrently across nodes; symptoms include transient missing CUDA libraries such as `libcusparseLt.so.0`.
- **Export `.env` before launching `rl`.** Plain `source .env` does not export bare assignments. Use `set -a; source .env; set +a` so `WANDB_API_KEY`, cache roots, and API keys reach child processes.
- **Stagger large inference fanout if vLLM startup livelocks.** Set `INFERENCE_START_STAGGER_SECONDS=30` before `rl` to delay inference node `N` by `N*30s`; default is 0. This preserves CUDA graph mode and avoids simultaneous multi-node vLLM startup pressure.
- **`/tmp` and `/dev/shm` are a shared, RAM-backed tmpfs that is not wiped between jobs.** A lane can hit `OSError [Errno 28] No space left on device` from other users' leftover caches on a node — an environment condition, not a launcher bug. Pick fresh nodes or clear space rather than debugging it as code.

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
