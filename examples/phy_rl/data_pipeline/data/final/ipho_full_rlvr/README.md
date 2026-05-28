---
license: other
language:
- en
task_categories:
- question-answering
pretty_name: IPhO Physics RLVR
tags:
- physics
- rlvr
- olympiad
- verification
---

# IPhO Physics RLVR

English IPhO physics problems curated into RLVR-ready question/answer rows.

Each admitted row contains:

- `problem_text`: full prompt context plus the focused question
- `shared_context`: reusable context needed to answer the question
- `question`: focused answerable question
- `official_solution`: solution evidence supporting the answer
- `answers`: structured verifier targets with value, unit, answer type, tolerance, verifier, equivalent forms, and subproblem id
- `split`: `train` or `frozen_test`
- `provenance`: source URL/hash and OCR metadata

## Splits

| split | rows |
| --- | ---: |
| train | 510 |
| frozen_test | 23 |
| dev | 0 |

The frozen-test split contains held-out recent IPhO rows according to the local
training policy. Rows requiring manual review or missing solutions were not
included in the exported splits.

## Verification Contract

Rows are text-only, English, and include deterministic verifier metadata. The
supported verifier families include numeric, symbolic/expression, multiple
choice, set/multi-select, tuple, and interval checks.

Generation/audit artifacts in the source workspace produced:

- 533 verified rows before split export
- 46 rows held for review
- 5 rejected rows with missing solutions
- 0 validation errors after deduplication
