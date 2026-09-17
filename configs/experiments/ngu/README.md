# NGU: SWE-rebench difficulty profile

Prepared on `feat/ngu`, based on latest fetched `origin/main` `a563a03d6`.
The PRL eval CLI is present (merged by `c394c2e1b`, PR #3471). No GPU job has been submitted.

## Measurement

- Model: `PrimeIntellect/GLM-4.5-Air-Scaleswe`.
- Four independent H200 nodes, each eight GPUs: TP=8 + expert parallelism within each node, DP=1 per replica. One native router balances the four replicas. No trainer allocation.
- SWE-rebench-V2 verified dataset, pinned to `03cc767ee33126b7fc7890ad57047e9dd6914cca`.
- `sample-1000.json` selects exactly 1,000 unique tasks, uniformly without replacement from the 6,272 source tasks’ sorted instance IDs using Python `random.Random(42)`. The manifest order is the evaluation order. It is not the first 1,000 dataset rows.
- Eight independent episodes per task: 8,000 total. Bash harness, fresh Prime runtime, temperature/top-p 1/1, 131072 model context. No length penalty.
- A 3,600-second **solve/agent** budget per episode, excluding setup and scoring. This is not a one-hour whole-job cap. The allocation wall-time guard is 48 hours.
- 256 concurrent episodes across the four replicas. Track sandbox capacity and errors during launch; this is a concurrency ceiling, not a batch size.

`avg@8` here means total solves / 8 per task, averaged across tasks (an estimate of pass@1), not pass@8's “any attempt solved.”

Confirmed buckets:

| Temporary taskset | Solves / 8 |
|---|---:|
| easy | 6–8 |
| medium | 3–5 |
| hard | 1–2 |
| extra-hard | 0 |

## One allocation, eval on its master node

`profile.sbatch.j2` wraps the **native PRL inference template**, through symlinks to its current template and includes. It starts native inference in a process group, waits for all four backend health endpoints and the router, and executes `uv run eval` in the batch shell on the first allocated node. The eval driver has `CUDA_VISIBLE_DEVICES` empty; replica zero still uses that node's GPUs. Splitting also runs there. The wrapper terminates the inference process group on success, failure or a termination signal.

The login node only parses/submits the job. Dataset loading, env serving, rollout dispatch and trace processing happen on the allocated master; task sandboxes run on Prime. The native inference template's standard node cleanup runs before the eval driver starts.

The local taskset module is added to `PYTHONPATH` by the wrapper. It reuses SWE-rebench's task class, setup and verifier, loads the dataset by pinned revision, and selects the exact manifest IDs. Original source indices are retained, so identities remain stable between the profile and its bucket subsets.

Native verifiers treats solve-budget exhaustion as a failed trace and skips grading. `NGUSWEEnv` narrowly converts the exact `HarnessError: agent timeout: rollout exceeded its 3600s budget` into a valid zero. It retains the error for inspection and emits `solve_timeout=1`. All other failures remain errors. This prevents one-hour timeouts from disappearing from the denominator or being retried as infrastructure failures on eval resume.

## Submission

From this branch on the cluster, initialize submodules and install the normal project environment first:

```bash
git submodule update --init --recursive
uv sync --all-extras --all-packages
```

Use the cluster's H200 partition and a persistent shared output path. The checked-in `partition="all"` matches the existing GLM Air recipe; override it if that partition is not exclusively suitable H200 nodes. Ensure existing Hugging Face model access and Prime sandbox credentials are available in the job environment / `.env`.

```bash
uv run inference @ configs/experiments/ngu/inference.toml \
  --slurm.partition "$NGU_H200_PARTITION" \
  --slurm.project-dir "$PWD" \
  --output-dir "$NGU_SHARED_OUTPUT"
```

This submits inference **and eval together**. Add `--dry-run` to render without submitting. Do not launch `uv run eval` separately on the login node. The model name is user-specified; model access has not been smoke-tested on GPUs here.

Outputs under `$NGU_SHARED_OUTPUT`:

- `launcher/logs/profile_<jobid>.log`: combined allocation/controller log.
- `logs/latest/inference/`: native replica/router logs.
- `eval/swerebench-1000-avg8/`: normal PRL eval configs, metrics and trace stream.
- `difficulty/{easy,medium,hard,extra-hard}.json`: four frozen task manifests, generated only after complete measurement.
- `difficulty/{easy,medium,hard,extra-hard}.toml`: four taskset configuration fragments.
- `difficulty/results.json`: per-task solve counts, aggregate avg@8 and bucket sizes.
- `difficulty/online-eval.toml`: training-time eval overlay with four separately named sources plus SWE-Bench Verified. Empty buckets retain manifests but are omitted from online eval and reported as size zero.

The splitter deduplicates episode IDs across resumed trace archives, rejects inconsistent copies/task hashes, and requires exactly eight valid outcomes for every selected task. It does not silently bin missing/error results as failures. On an incomplete run, resume the **same native eval** in a compute allocation with `--resume`; repeat splitting only once eight valid outcomes/task are available. Existing frozen split directories are never overwritten.

## Observe curves during the two training runs

Keep `configs/experiments/ngu/tasksets` on `PYTHONPATH` on every relevant process/node, and compose the generated overlay onto both training configs:

```bash
export PYTHONPATH="$PWD/configs/experiments/ngu/tasksets${PYTHONPATH:+:$PYTHONPATH}"
uv run rl @ notes/ngu/swe-baseline.toml @ "$NGU_SHARED_OUTPUT/difficulty/online-eval.toml"
uv run rl @ notes/ngu/swe-ngu.toml @ "$NGU_SHARED_OUTPUT/difficulty/online-eval.toml"
```

These are later training commands, not part of this profiling job; the NGU training algorithm still requires stage 1. The generated source array retains SWE-Bench Verified because config array composition replaces lists. Each difficulty source evaluates its full fixed subset every 20 steps with one rollout/task and the same one-hour solve budget. The source names produce separate curves. These sampled tasks are from the training distribution; their curves are learning diagnostics, not additional held-out generalization claims.

## Verification performed

- Both configs resolve against the actual current config classes.
- Actual `uv run eval ... --dry-run` exercised successfully in a temporary CPU environment.
- Native inference entrypoint renders the combined job; generated Bash syntax checked.
- All 1,000 unique task objects loaded from the real pinned Hub snapshot.
- Isolated tests cover every bucket boundary, resume duplicate handling, incomplete/error rejection, disjoint/exhaustive splits, and preservation of the existing eval source.

Preparation artifacts are ready for review; actual bucket membership and metrics do not exist until the GPU eval completes.
