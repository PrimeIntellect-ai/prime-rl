---
name: transcript-analysis
description: "Set up and run Prime RL transcript analysis with Docent. Use when Joan asks to inspect, cluster, label, or analyze rollout transcripts/failure modes across training or eval runs."
---

# Prime RL Transcript Analysis

Use this workflow for Docent-style transcript analysis of Prime RL rollouts.

## Capture

Full transcripts are opt-in because trajectories can be large.

For RL configs:

```toml
[orchestrator]
save_full_rollouts = true
```

For orchestrator-only configs:

```toml
save_full_rollouts = true
```

Expected files:

```text
{output_dir}/rollouts/step_N/train_rollouts_full.jsonl
{output_dir}/rollouts/step_N/eval_rollouts_<env>.full.jsonl
```

The existing reduced `train_rollouts.jsonl` and `eval_rollouts_<env>.jsonl`
still exclude `trajectory`.

## Ingest

Dry-run:

```bash
uv run --no-project --with docent-python \
  python scripts/docent/ingest_prime_rollouts.py \
  {output_dir}/rollouts \
  --dry-run
```

Create a Docent collection:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/ingest_prime_rollouts.py \
  {output_dir}/rollouts \
  --collection-name "prime-rl <run-name> transcript analysis" \
  --source-label <run-name>
```

Add to an existing collection:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/ingest_prime_rollouts.py \
  {output_dir}/rollouts \
  --collection-id <collection-id> \
  --source-label <run-name>
```

Avoid `--allow-reduced` unless full rollouts are unavailable; it only ingests
top-level prompt/completion.

## Starter Plan

Create the reviewable Docent analysis plan:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/create_prime_analysis_plan.py <collection-id>
```

Useful filters:

```bash
--env-name <env>
--limit 500
--reward-below 0.5
--include-successes
```

Pass `--auto-approve` only after reviewing the generated plan.

The starter plan creates:

- `prime-rl/candidate-runs/v1`
- `prime-rl/failure-classification/v1`
- `prime-rl/failure-mode-aggregate/v1`
- `prime-rl/high-confidence-failures/v1`
- `prime-rl/failure-classification-verifier/v1`

## Analysis Discipline

Follow the Transluce analysis-plan pattern:

1. Query candidate runs with explicit metadata filters.
2. Read transcripts into structured outputs with citation fields.
3. Aggregate labels before drawing conclusions.
4. Verify high-confidence or high-stakes labels with a second reading pass.
5. Treat any label without transcript evidence as unsupported.
