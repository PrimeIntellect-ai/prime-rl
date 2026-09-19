# Score-centering experiment

Read `PROTOCOL.md` for the estimand, controls, and decision rules.
Read `STATE.md` for live job IDs and recovery steps.

- `common.toml`: shared model, tasks, trainer, quantization, and resource settings.
- `baseline.toml`: current-main IPO with eps 0.3.
- `centered.toml`: the paper's standalone score-centering loss.
- `ipo-centered.toml`: optional weighted IPO-centering ablation.
- `eval-preflight.toml`: bounded Prime VM checks on Terminal Lego and Terminal Bench 2.
- `smoke.toml`: three updates with a smaller batch and evaluation sample.
- `kv-fp8.toml`: archived FP8-cache diagnostic override; the main config now uses the same cache precision.
- `smoke-fast.toml`: save only at the final smoke step after intermediate saves are verified.
- `verify_math.py`: gradient checks and comparison with the official formula.
- `verify_transport.py`: sampler-head transport and packing checks.
- `analyze.py`: metric CSV, summary JSON, and PNG/PDF curves by update and token budget.
- `eval_results.py`: task rewards, infrastructure errors, and task-level confidence intervals.

Validate VM provisioning, tool execution, and scoring first:

```bash
uv run eval @ experiments/score-centering/eval-preflight.toml --run.name runtime-preflight --no-dashboard
```

Launch both arms together, using distinct fresh run names:

```bash
uv run rl @ experiments/score-centering/common.toml @ experiments/score-centering/baseline.toml --run.name ipo-seed42 --no-dashboard &
uv run rl @ experiments/score-centering/common.toml @ experiments/score-centering/centered.toml --run.name sc-seed42 --no-dashboard &
wait
```

Append `@ experiments/score-centering/smoke.toml` before the CLI overrides for smoke runs.
Each launch allocates four nodes. Stop or finish previous experiment allocations first.
Keep at most eight experiment nodes allocated across all checks and runs.

The renderer and verifiers changes are archived in `patches/` against their pinned submodules.
Apply these patches when reconstructing this experiment from a fresh checkout.
The model snapshot path in `common.toml` points to the revision recorded in `model.json`.

```bash
uv run python experiments/score-centering/analyze.py outputs/score-centering/ipo-seed42 outputs/score-centering/sc-seed42 --output experiments/score-centering/results/main
```

The exporter also records numerical warnings, including nonfinite values dropped by metric writers.
Review these warnings before classifying numerical stability.
Export trace-level results for task uncertainty and failure classification:

```bash
uv run python experiments/score-centering/eval_results.py outputs/score-centering/ipo-seed42 outputs/score-centering/sc-seed42 --output experiments/score-centering/results/main
```

Valid-task reward estimates exclude failed episodes. Report error rates and missing-reward bounds alongside them.


Token-budget plots count output tokens from training episodes received by each optimizer update.
They include rejected and failed episodes when their tokens were recorded.
They exclude unfinished episodes and tokens absent from failed responses.
The CSV also records server generation totals, including evaluation tokens, from the latest metrics poll.
