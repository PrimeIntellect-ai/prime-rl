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
```

Then apply an answer key, filter, deduplicate, and export with the matching CLI
subcommands.

## Rules

- Use `glm-ocr` for real PDF extraction; use `embedded` only as a diagnostic.
- Do not admit an item without explicit answer metadata and a deterministic verifier.
- Keep 2024-2025 Olympiad items and HiPhO out of train.
- Keep rejected rows and audit reports.
