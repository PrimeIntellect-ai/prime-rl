# NGU: SWE-rebench difficulty profile

Prepared on `feat/ngu`, based on latest fetched `origin/main` `a563a03d6`.
The PRL eval CLI is present (merged by `c394c2e1b`, PR #3471). No GPU job has been submitted.

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
  "$NGU_SHARED_OUTPUT/difficulty"
EVAL
```

The inference job remains running after the eval step exits. Inspect the results, then release the allocation with `scancel "$NGU_JOB_ID"`. On interruption, rerun the eval step with `--resume` on the eval command before splitting. Do not run eval directly on the login node. Model access has not been smoke-tested on GPUs here.

Outputs under `$NGU_SHARED_OUTPUT`:

- `launcher/logs/`: standard inference allocation logs; the attached eval step also streams to its invoking terminal.
- `logs/latest/inference/`: native replica/router logs.
- `eval/swerebench-1k/`: normal PRL eval configs, metrics and trace stream.
- `difficulty/{easy,medium,hard,extra-hard}.json`: four frozen task manifests, generated only after complete measurement.
- `difficulty/{easy,medium,hard,extra-hard}.toml`: four taskset configuration fragments.
- `difficulty/results.json`: per-task solve counts, aggregate avg@8 and bucket sizes.
- `difficulty/online-eval.toml`: training-time eval overlay with four separately named sources plus SWE-Bench Verified. Empty buckets retain manifests but are omitted from online eval and reported as size zero.

The splitter deduplicates episode IDs across resumed trace archives, rejects inconsistent copies/task hashes, and requires exactly eight valid outcomes for every selected task. It does not silently bin missing/error results as failures. On an incomplete run, resume the **same native eval** in a compute allocation with `--resume`; repeat splitting only once eight valid outcomes/task are available. Existing frozen split directories are never overwritten.

## Observe curves during the two training runs

Keep `tools/ngu/tasksets` on `PYTHONPATH` on every relevant process/node, and compose the generated overlay onto both training configs:

```bash
export PYTHONPATH="$PWD/tools/ngu/tasksets${PYTHONPATH:+:$PYTHONPATH}"
uv run rl @ notes/ngu/swe-baseline.toml @ "$NGU_SHARED_OUTPUT/difficulty/online-eval.toml"
uv run rl @ notes/ngu/swe-ngu.toml @ "$NGU_SHARED_OUTPUT/difficulty/online-eval.toml"
```

These are later training commands, not part of this profiling job; the NGU training algorithm still requires stage 1. The generated source array retains SWE-Bench Verified because config array composition replaces lists. Each difficulty source evaluates its full fixed subset every 20 steps with one rollout/task and the same one-hour solve budget. The source names produce separate curves. These sampled tasks are from the training distribution; their curves are learning diagnostics, not additional held-out generalization claims.

## Verification performed

- Both configs resolve against the actual current config classes.
- Actual `uv run eval ... --dry-run` exercised successfully in a temporary CPU environment.
- Native inference entrypoint renders its standard job; generated Bash syntax checked.
- All 1,000 unique task objects loaded from the real pinned Hub snapshot.
- Isolated tests cover every bucket boundary, resume duplicate handling, incomplete/error rejection, disjoint/exhaustive splits, and preservation of the existing eval source.

Preparation artifacts are ready for review; actual bucket membership and metrics do not exist until the GPU eval completes.
