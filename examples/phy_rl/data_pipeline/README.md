# Physics RLVR Data Pipeline

Local-first curation tools for English, text-only, rule-verifiable physics RLVR
data. The verified PDF path uses `zai-org/GLM-OCR` through Hugging Face
Transformers and writes canonical Markdown/JSON for downstream alignment,
filtering, deduplication, and export.

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

HF_HOME=/private/tmp/hf-cache \
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data extract \
  --manifest examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --out-dir examples/phy_rl/data_pipeline/data/extracted \
  --extractor glm-ocr \
  --vlm-model zai-org/GLM-OCR \
  --vlm-prompt "Text Recognition:"

uv run --project examples/phy_rl/data_pipeline physics-rlvr-data build-candidates \
  --manifest examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --extracted-dir examples/phy_rl/data_pipeline/data/extracted \
  --out examples/phy_rl/data_pipeline/data/candidates/candidates.jsonl

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
embedded text. For quick GLM-OCR smoke tests, pass `--vlm-max-pages 1`.

## Acceptance Rule

The pipeline never invents labels. A candidate is rejected until it has explicit
answers and deterministic verifier metadata, either from a structured source or
an answer-key JSONL.
