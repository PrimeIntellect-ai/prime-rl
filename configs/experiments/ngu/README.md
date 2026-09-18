# NGU: SWE training comparison and difficulty profile

Prepared on `feat/ngu`, merged with `origin/main` `776053131`.
The PRL eval CLI is present (merged by `c394c2e1b`, PR #3471). Profiling finished and job 737 was released; runtime artifacts are under `outputs/ngu-profile-20260917`.

## Training configurations

- [`train-static.toml`](train-static.toml): static GRPO, K=16, batch target 256. Validated with the native RL dry-run entrypoint.
- [`train-ngu.toml`](train-ngu.toml): NGU, K=16, continuation probability .875, inclusive payload history age 4, historical binary baseline and positive anchoring.
- [`difficulty-eval.toml`](difficulty-eval.toml): optional overlay adding the four fixed training-difficulty subsets while preserving both held-out sources.

Both arms start from `PrimeIntellect/GLM-4.5-Air-Scaleswe` with fresh optimizer state. They use two H200 trainer nodes and six independent eight-GPU inference replicas (64 GPUs total), TP8 + EP, 131072 context, router replay, CP4/ulysses, Muon LR 3e-6, and IPO with epsilon 0.3, advantage tau 1.0, and KL tau 0. Both log to W&B project `ngu-ablations`. No length penalty or sampling override. Adaptive concurrency is 256–1000. The model's numerical dtype defaults are unchanged. `max_steps=10000` is a guard, not an enforced GPU-hour budget; compare checkpoints at equal allocated H200-hours. Checkpoints save every 50 steps.

Training uses **all 1,000** IDs in `tools/ngu/sample-1000.json`, including the three tasks with no valid profiling attempts. This is a single training source: bucket membership does not change task weights. NGU is intended to allocate extra rounds online from actual training outcomes, not from the profiling labels.

Evaluation runs at step 0 and every 20 updates with one rollout per task:

| Source | Tasks | Purpose |
|---|---:|---|
| `swebench-verified` | full taskset (500) | Benchmark evaluation |
| `swerebench-heldout-500` | 500 | In-distribution held-out evaluation |

The held-out manifest `tools/ngu/eval-500.json` uses the same pinned snapshot as training. Selection is `random.Random(43).sample(sorted(all_instance_ids - train_instance_ids), 500)`: 5,272 eligible tasks, 500 unique selected IDs, zero overlap with the training manifest. The manifest records revision, seed, eligible population and excluded manifest. This verifies instance-ID separation; it does not establish decontamination against the base checkpoint's earlier ScaleSWE training.

Every source uses the bash harness, Prime sandboxes, a one-hour solve budget and a **two-hour scoring timeout**. Scoring failures remain errors. The manifest-backed SWE environment counts solve-budget exhaustion as a valid zero, as in profiling.

Commands below prepare/launch training only when explicitly requested; no SWE training has been submitted. From the repository root, export the taskset path for config resolution (the config also propagates it to the orchestrator and env servers):

```bash
export PYTHONPATH="$PWD/tools/ngu/tasksets${PYTHONPATH:+:$PYTHONPATH}"
uv run rl @ configs/experiments/ngu/train-static.toml --dry-run
# Optional diagnostics, also evaluated every 20 updates:
uv run rl @ configs/experiments/ngu/train-static.toml @ configs/experiments/ngu/difficulty-eval.toml --dry-run
```

NGU's configuration matches the static arm except run labels and algorithm settings. Both enable whole-cohort batching (`preserve_groups=true`) and a 100-batch-equivalent no-output guard. NGU retries failed rounds with fresh episodes, keeps historical reward counts after payload expiry, and rescores anchored advantages after freshness filtering. See [algorithm semantics](../../../docs/algorithms.md#never-give-up-ngu) for memory limits and resume behavior. The small reverse-text integration run is documented in [the smoke report](../../../notes/ngu/smoke.md). The eight-node SWE training runs have not been launched.

Track each held-out source's resolved rate, error rate and rollout length against allocated GPU-hours, alongside throughput, trainer idle time and trained/generated samples. The optional buckets are **training-set diagnostics**, not held-out evaluation. NGU logs retry rounds, first-success attempt totals, history eviction, give-up rates and accepted cohort sizes under `ngu/<source>/`; trace annotations expose the anchored advantages and history counts.

## Completed profile

The profile produced 1,707 solves / 7,827 valid attempts (21.81% pooled); the mean per-task pass rate was 21.62% over 997 scorable tasks. There were 173 errors, including 13 graders stopped after two hours. No attempts were rerun to fill incomplete groups.

Frozen manifests under `tools/ngu/` contain 168 easy, 93 medium, 72 hard and 664 extra-hard tasks. Three tasks without any valid attempts are excluded from the diagnostic subsets only. For partial groups, classification uses observed valid pass rate (≥75%, ≥37.5%, >0%, and 0%). Zero observed solves does not imply a true zero probability of success.

All 8,000 episodes were uploaded to Prime Traces; all trace IDs and example retrieval were verified. Search by SDK context `run_name=ngu-swerebench-1k-glm45air-20260917` (the service did not index the native run ID). Full receipts, verification and retrieval instructions are in `outputs/ngu-profile-20260917/`.

## Measurement

- Model: `PrimeIntellect/GLM-4.5-Air-Scaleswe`.
- Four independent H200 nodes, each eight GPUs: TP=8 + expert parallelism within each node, DP=1 per replica. One native router balances the four replicas. No trainer allocation.
- SWE-rebench-V2 verified dataset, pinned to `03cc767ee33126b7fc7890ad57047e9dd6914cca`.
- `tools/ngu/sample-1000.json` selects exactly 1,000 unique tasks, uniformly without replacement from the 6,272 source tasks’ sorted instance IDs using Python `random.Random(42)`. The manifest order is the evaluation order. It is not the first 1,000 dataset rows.
- Eight independent episodes per task: 8,000 total. Bash harness, fresh Prime runtime, default sampling settings, 131072 model context. No length penalty.
- A 3,600-second **solve/agent** budget per episode, excluding setup and scoring. This is not a one-hour whole-job cap. No job time limit is specified; cluster policy applies.
- Adaptive concurrency from 256 to 1,000 inflight episodes across the four replicas, driven by engine metrics. Track sandbox capacity and errors during launch.

`avg@8` here means total solves / 8 per task, averaged across tasks (an estimate of pass@1), not pass@8's “any attempt solved.”

Confirmed buckets:

| Temporary taskset | Solves / 8 |
|---|---:|
| easy | 6–8 |
| medium | 3–5 |
| hard | 1–2 |
| extra-hard | 0 |

## One allocation, eval on its master node

Use the standard `inference` launcher to allocate the four nodes, then attach eval as an overlapping SLURM step on the allocation's master node. No custom launcher or template is required. The login node only submits these commands; dataset loading, env serving, rollout dispatch and trace processing run in the allocation. Task sandboxes run on Prime.

The local taskset module is added to `PYTHONPATH` in the eval step. It reuses SWE-rebench's task class, setup and verifier, loads the dataset by pinned revision, and selects the exact manifest IDs. Original source indices are retained, so identities remain stable between the profile and its bucket subsets.

Native verifiers treats solve-budget exhaustion as a failed trace and skips grading. `NGUSWEEnv` narrowly converts the exact `HarnessError: agent timeout: rollout exceeded its 3600s budget` into a valid zero. It retains the error for inspection and emits `solve_timeout=1`. All other failures remain errors. This prevents one-hour timeouts from disappearing from the denominator or being retried as infrastructure failures on eval resume.

## Submission

From this branch on the cluster, initialize submodules and install the normal project environment first:

```bash
git submodule update --init --recursive
uv sync --all-extras --all-packages
```

Use a persistent shared output path. The SLURM partition uses the launcher default. Ensure existing Hugging Face model access and Prime sandbox credentials are available in the job environment / `.env`.

```bash
uv run inference @ configs/experiments/ngu/inference.toml \
  --slurm.project-dir "$PWD" \
  --output-dir "$NGU_SHARED_OUTPUT"
```

This submits the standard inference job. Add `--dry-run` to render without submitting. Once it is running and all four backends and the router are ready, use its job ID below. `srun` attaches to that allocation; it does not request a separate node. Wait until native node cleanup and model startup have completed before attaching eval.

```bash
export NGU_JOB_ID="<inference-job-id>"
export NGU_SHARED_OUTPUT="<same-shared-output-path>"
export NGU_PROJECT_DIR="$PWD"
NGU_MASTER=$(squeue --noheader --jobs "$NGU_JOB_ID" --format '%B')
srun --jobid "$NGU_JOB_ID" --overlap --nodes=1 --ntasks=1 \
  --nodelist "$NGU_MASTER" --gres=none bash -s <<'EVAL'
set -euo pipefail
cd "$NGU_PROJECT_DIR"
[ ! -f .env ] || source .env
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/tools/ngu/tasksets${PYTHONPATH:+:$PYTHONPATH}"
uv run eval @ configs/experiments/ngu/eval.toml \
  --client.base-url http://localhost:8000/v1 \
  --output-dir "$NGU_SHARED_OUTPUT/eval"
uv run python tools/ngu_difficulty.py split \
  tools/ngu/sample-1000.json \
  "$NGU_SHARED_OUTPUT/eval/swerebench-1k" \
  "$NGU_SHARED_OUTPUT/difficulty" --allow-partial
EVAL
```

The inference job remains running after the eval step exits. Inspect the results, then release the allocation with `scancel "$NGU_JOB_ID"`. On interruption, rerun the eval step with `--resume` on the eval command before splitting. Do not run eval directly on the login node. The completed profile exercised this model on all four GPU replicas.

Outputs under `$NGU_SHARED_OUTPUT`:

- `launcher/logs/`: standard inference allocation logs; the attached eval step also streams to its invoking terminal.
- `logs/latest/inference/`: native replica/router logs.
- `eval/swerebench-1k/`: normal PRL eval configs, metrics and trace stream.
- `difficulty/{easy,medium,hard,extra-hard}.json`: four frozen task manifests, generated only after complete measurement.
- `difficulty/{easy,medium,hard,extra-hard}.toml`: four taskset configuration fragments.
- `difficulty/results.json`: per-task solve counts, aggregate avg@8 and bucket sizes.
- `difficulty/online-eval.toml`: training-time eval overlay with four separately named sources plus SWE-Bench Verified. Empty buckets retain manifests but are omitted from online eval and reported as size zero.

The splitter deduplicates episode IDs across resumed trace archives and rejects inconsistent copies/task hashes. This run was split with `--allow-partial` after it finished, without rerunning failed attempts. Failed episodes are excluded from each task's denominator; a solve timeout is a valid zero. Buckets use observed pass rate: easy ≥75%, medium ≥37.5%, hard >0%, extra-hard 0%. Tasks with no valid outcomes remain unclassified.

`results.json` records valid counts and rates per task, the equally weighted task-mean pass rate, and the pooled valid-attempt pass rate. `avg_at_8` is null unless every task has eight valid outcomes. With no `--allow-partial`, the splitter requires exactly eight valid outcomes per task. Existing frozen split directories are never overwritten.

## Verification performed

- Both configs resolve against the actual current config classes.
- Actual `uv run eval ... --dry-run` exercised successfully in a temporary CPU environment.
- Native inference entrypoint renders its standard job; generated Bash syntax checked.
- All 1,000 unique task objects loaded from the real pinned Hub snapshot.
- Isolated tests cover every bucket boundary, resume duplicate handling, incomplete/error rejection, disjoint/exhaustive splits, and preservation of the existing eval source.

Training configs and frozen manifests are ready for review. NGU runtime is implemented; use the native dry-run before deployment.
