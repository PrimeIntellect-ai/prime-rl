# Transcript Analysis With Docent

Prime RL can produce opt-in full rollout snapshots for transcript analysis with
Docent. The default rollout JSONL stays compact and excludes `trajectory`; full
snapshots are written only when explicitly enabled.

## Capture Full Rollouts

Set the orchestrator flag:

```toml
[orchestrator]
save_full_rollouts = true
```

For orchestrator-only configs, set it at top level:

```toml
save_full_rollouts = true
```

This writes the usual reduced files plus full transcript files:

```text
{output_dir}/rollouts/step_N/
├── train_rollouts.jsonl
├── train_rollouts_full.jsonl
├── eval_rollouts_<env>.jsonl
└── eval_rollouts_<env>.full.jsonl
```

The full snapshot files preserve the rollout `trajectory`, including each
trajectory step's prompt, completion, rewards, extras, and token metadata.

## Ingest Into Docent

Dry-run first:

```bash
uv run --no-project --with docent-python \
  python scripts/docent/ingest_prime_rollouts.py \
  outputs/my-run/rollouts \
  --dry-run
```

Create a collection and upload:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/ingest_prime_rollouts.py \
  outputs/my-run/rollouts \
  --collection-name "prime-rl my-run transcript analysis" \
  --source-label my-run
```

Or add to an existing collection:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/ingest_prime_rollouts.py \
  outputs/my-run/rollouts \
  --collection-id <collection-id> \
  --source-label my-run
```

Reduced rollout JSONL can be ingested with `--allow-reduced`, but that only
captures top-level prompt/completion and is much weaker for transcript analysis.

## Create The Starter Analysis Plan

The starter plan follows the Transluce analysis-plan pattern:

1. Query candidate Prime RL runs from Docent metadata.
2. Run a structured reading pass that labels outcome and failure mode with
   transcript citations.
3. Aggregate failure modes with DQL.
4. Run a verifier reading over high-confidence failures.

Create the plan for review:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/create_prime_analysis_plan.py <collection-id>
```

Restrict to one environment:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/create_prime_analysis_plan.py <collection-id> \
  --env-name reverse-text
```

Submit immediately only when the plan looks right:

```bash
DOCENT_API_KEY=... uv run --no-project --with docent-python \
  python scripts/docent/create_prime_analysis_plan.py <collection-id> \
  --auto-approve
```

Default result names:

```text
prime-rl/candidate-runs/v1
prime-rl/failure-classification/v1
prime-rl/failure-mode-aggregate/v1
prime-rl/high-confidence-failures/v1
prime-rl/failure-classification-verifier/v1
```

Use these as stable handles for follow-up DQL queries and second-pass readings.
