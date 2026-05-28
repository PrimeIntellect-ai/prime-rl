# Physics RLVR Data Pipeline

Use this skill for `examples/phy_rl/data_pipeline`.

## Commands

Set up the standalone data-pipeline environment:

```bash
uv sync --project examples/phy_rl/data_pipeline
```

Run the CLI from the repo root:

```bash
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data --help
```

Typical PDF flow:

```bash
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
```

Then apply an answer key, filter, deduplicate, and export with the matching CLI
subcommands.

For local GLM-OCR fallback extraction, run persistent workers and pin one worker
to each CUDA device:

```bash
uv run --project examples/phy_rl/data_pipeline physics-rlvr-data extract \
  --manifest examples/phy_rl/data_pipeline/data/manifest.jsonl \
  --out-dir examples/phy_rl/data_pipeline/data/extracted \
  --extractor glm-ocr \
  --vlm-model zai-org/GLM-OCR \
  --vlm-prompt "Text Recognition:" \
  --jobs 2 \
  --vlm-devices 0,1
```

## Rules

- Use `gemini` for real PDF extraction; use `embedded` only as a diagnostic.
- `ipho.olimpicos.net` requires browser-like HTTP headers for crawl/download;
  keep the pipeline HTTP helper in use instead of raw default urllib requests.
- IPhO filenames use `Q1`/`S1` conventions; metadata inference must preserve
  `paper_type` and `problem_number` from those names before building candidates.
- Gemini OCR requires `GEMINI_API_KEY` in the environment. Never write API keys
  into repo files, configs, or audit artifacts.
- Auto answer-key extraction is high precision only. By default, only explicit
  P1-style answer blocks or boxed answers should be admitted. Rows in
  `answer_key_review.jsonl` need manual/consensus audit before they can be used
  for RLVR.
- Use `build-rlvr-subproblems --judge-model gemini-3.1-flash-lite` before final
  filtering for Olympiad PDFs. Full IPhO problems are usually containers, not
  single RLVR items; split labels such as `A.1`, pass OCR context plus candidate
  answers to the judge, and keep uncertain labels in `subproblems_review.jsonl`.
- P1-style RLVR rows need structured question, expert solution, and
  rule-verifiable answers. Split units into the `unit` field and represent
  multi-part answers as separate answer entries.
- Do not admit an item without explicit answer metadata and a deterministic verifier.
- Keep 2024-2025 Olympiad items and HiPhO out of train.
- Keep rejected rows and audit reports.
