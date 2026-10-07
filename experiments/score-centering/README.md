# Score-centering experiment

Read `PROTOCOL.md` for the estimand, controls, and decision rules.
Read `STATE.md` for live job IDs and recovery steps.

- `common.toml`: shared model, tasks, trainer, quantization, and resource settings.
- `baseline.toml`: current-main IPO with eps 0.3.
- `centered.toml`: the same IPO with score centering enabled.
- `ipo-centered.toml`: equivalent explicit IPO-plus-centering overlay.
- `standalone-centered.toml`: archived treatment from the earlier standalone-estimator comparison.
- `eval-preflight.toml`: bounded Prime VM checks on Terminal Lego and Terminal Bench 2.
- `smoke.toml`: three updates with a smaller batch and evaluation sample.
- `kv-fp8.toml`: archived FP8-cache diagnostic override; the main config now uses the same cache precision.
- `smoke-fast.toml`: save only at the final smoke step after intermediate saves are verified.
- `verify_math.py`: gradient checks and comparison with the official formula.
- `verify_transport.py`: sampler-head transport and packing checks.
- `benchmark_compact_logprobs.py`: CPU comparison of compact and nested logprob responses.
- `COMPACT_LOGPROBS.md`: transport format, controls, and benchmark results.
- `analyze.py`: metric CSV, summary JSON, and PNG/PDF curves by update and token budget.
- `eval_results.py`: task rewards, errors, task-level confidence intervals, and missing-reward bounds.
- `select_holdout.py`: select the supplemental unseen-Lego audit after both training runs stop.

Validate VM provisioning, tool execution, and scoring first:

```bash
uv run eval @ experiments/score-centering/eval-preflight.toml --run.name runtime-preflight --no-dashboard
```

Prepare launch scripts that resume both arms from the matched step-250 checkpoints:

```bash
uv run rl @ experiments/score-centering/common.toml @ experiments/score-centering/baseline.toml --run.name ipo-eps03-seed42-compact-v1 --resume.step 250 --no-dashboard --dry-run
uv run rl @ experiments/score-centering/common.toml @ experiments/score-centering/centered.toml --run.name ipo-sc-eps03-seed42-compact-v1 --resume.step 250 --no-dashboard --dry-run
```

After the live preflight, schedule both arms in one four-node allocation:

```bash
sbatch experiments/score-centering/launch-pair.sbatch
```

The launcher gives each arm one trainer node and one inference node, with 116 CPUs per node.
Monitoring does not cancel jobs automatically. A failed arm does not stop the other arm.
Preserve the earlier standalone-estimator results separately; their launch script and lineage are under `archive/`.
For a three-update smoke check, add `@ experiments/score-centering/smoke.toml` before the CLI overrides and use separate names.
Keep at most eight experiment nodes allocated across all checks and runs.

The renderer and verifiers changes are archived in `patches/` against their pinned submodules.
Apply these patches once when reconstructing this experiment from a fresh checkout:

```bash
git -C deps/renderers apply ../../experiments/score-centering/patches/renderers.patch
git -C deps/verifiers apply ../../experiments/score-centering/patches/verifiers.patch
```

The dependency changes are archived as patches; their pinned submodule commits remain unchanged.
The model snapshot path in `common.toml` points to the revision recorded in `model.json`.

```bash
uv run python experiments/score-centering/analyze.py outputs/score-centering/ipo-eps03-seed42-compact-v1 outputs/score-centering/ipo-sc-eps03-seed42-compact-v1 --output experiments/score-centering/results/ipo-versus-ipo-sc
```

The exporter also records numerical warnings, including nonfinite values dropped by metric writers.
Review these warnings before classifying numerical stability.
Export trace-level results for task uncertainty and failure classification:

```bash
uv run python experiments/score-centering/eval_results.py outputs/score-centering/ipo-eps03-seed42-compact-v1 outputs/score-centering/ipo-sc-eps03-seed42-compact-v1 --output experiments/score-centering/results/ipo-versus-ipo-sc
```

Valid-task reward estimates exclude failed episodes. Report error rates and missing-reward bounds alongside them.


Token-budget plots count output tokens from training episodes received by each optimizer update.
They include rejected and failed episodes when their tokens were recorded.
They exclude unfinished episodes and tokens absent from failed responses.
The CSV also records server generation totals, including evaluation tokens, from the latest metrics poll.

After both main runs finish, create the score-independent Lego audit manifest:

```bash
uv run python experiments/score-centering/select_holdout.py outputs/score-centering/ipo-eps03-seed42-compact-v1 outputs/score-centering/ipo-sc-eps03-seed42-compact-v1 --output experiments/score-centering/results/ipo-versus-ipo-sc/heldout
```

The selector records the full dataset order and excludes every task in either dispatch log.
Do not use a manifest generated while either run is still dispatching tasks.
Evaluate the initial model and both final checkpoints on the selected tasks before interpreting capability retention.

The selector also writes `eval.json` with the recorded agent budgets, sampling settings, and four attempts per task.
It pins audit concurrency at 128 for all three models.
Start the base model and each exported final model with the recorded inference quantization.
Run `uv run eval @ <heldout>/eval.json --model <served-model-id> --client.base-url <endpoint>/v1 --run.name <audit-name>`.
Use the exact model identifier returned by that endpoint's `/v1/models`.
The default endpoint in the generated config is only a local placeholder.
