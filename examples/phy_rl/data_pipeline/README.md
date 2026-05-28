# Physics RLVR Data Pipeline

Local curation tools for English, text-only, rule-verifiable physics RLVR data.
The verified PDF path uses Gemini OCR and writes canonical Markdown/JSON for
downstream alignment, filtering, deduplication, and export.

## Setup

```bash
uv sync --project examples/phy_rl/data_pipeline
```

Run from the repository root:

```bash
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data --help
```

## Flow

```bash
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data init

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data manifest-local \
  --pdf-root /path/to/pdfs \
  --out examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --source-id ipho_olimpicos

GEMINI_API_KEY=... \
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data extract \
  --manifest examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --out-dir examples/phy_rl/data_pipeline/data/extracted \
  --extractor gemini \
  --vlm-model gemini-3.1-flash-lite

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data build-candidates \
  --manifest examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --extracted-dir examples/phy_rl/data_pipeline/data/extracted \
  --out examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data extract-answer-key \
  --input examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl \
  --out examples/phy_rl/data_pipeline/data/manual_corrections/auto_answer_key.jsonl \
  --review examples/phy_rl/data_pipeline/data/audits/answer_key_review.jsonl

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data build-rlvr-subproblems \
  --input examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl \
  --verified examples/phy_rl/data_pipeline/data/verified/subproblems_verified.jsonl \
  --review examples/phy_rl/data_pipeline/data/audits/subproblems_review.jsonl \
  --rejected examples/phy_rl/data_pipeline/data/rejected/subproblems_rejected.jsonl \
  --judge-model gemini-3.1-flash-lite

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data apply-answer-key \
  --input examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl \
  --answer-key examples/phy_rl/data_pipeline/data/manual_corrections/answer_key.jsonl \
  --out examples/phy_rl/data_pipeline/data/candidates/candidates_with_answers.jsonl

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data filter-candidates \
  --input examples/phy_rl/data_pipeline/data/candidates/candidates_with_answers.jsonl \
  --verified examples/phy_rl/data_pipeline/data/verified/verified.jsonl \
  --rejected examples/phy_rl/data_pipeline/data/rejected/rejected.jsonl

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data dedup \
  --input examples/phy_rl/data_pipeline/data/verified/verified.jsonl \
  --out examples/phy_rl/data_pipeline/data/verified/deduped.jsonl \
  --report examples/phy_rl/data_pipeline/data/audits/duplicate_report.csv

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data export-final \
  --input examples/phy_rl/data_pipeline/data/verified/deduped.jsonl \
  --out-dir examples/phy_rl/data_pipeline/data/final
```

Use `--extractor embedded` only as a fast diagnostic path for PDFs with clean
embedded text. For quick Gemini smoke tests, pass `--vlm-max-pages 1`. The
Gemini judge receives OCR-derived problem text, solution context, and heuristic
answer candidates, then returns the full question/answer rows to keep. The
pipeline still admits only rows with deterministic verifier metadata; uncertain
or unsupported rows stay in `subproblems_review.jsonl`.

## Acceptance Rule

The pipeline never invents labels. A candidate is rejected until it has explicit
answers and deterministic verifier metadata, either from a structured source or
an answer-key JSONL.

For P1-style RLVR data, each admitted item should have a structured
question/solution/answer schema: answer values are split from units, multi-part
answers are represented as separate entries, and ambiguous auto-extractions are
kept in review files instead of being used as rewards.
