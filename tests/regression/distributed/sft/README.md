# Distributed SFT suite

A nightly functionality suite for the trainer. It runs trainer-only SFT (`uv run sft`) on seeded tiny checkpoints of every trainer arch, across a few parallel layouts and memory features, on one 4-GPU node. A case passes if training exits cleanly with a finite loss. Selected cases also check that checkpoint resume continues the step count and that the compiled model has no graph breaks.

## Layout

- `archs.py`: the archs the suite runs. Each `Arch` holds its tiny config, the tokenizer matching its vocab, and how the suite runs it (layouts, feature rows, resume). It also writes the seeded tiny checkpoints.
- `cases.py`: the building blocks (`Layout`, `Feature`, `Arch`, `Case`), the rendering of a case as `uv run sft` arguments, sharding, and the parsers for metrics and logs.
- `test_matrix.py`: runs each case once and checks it with `test_sft` and `test_no_graph_breaks`.
- `../conftest.py`: the session fixtures (tiny checkpoints, tokenizer downloads), the `--shard` option and the GPU-count skip.
- `tests/regression/configs/distributed/sft/`: the shared `base.toml` and the training data.

## Coverage

Every arch runs a base case on each of its layouts. To save launches, only qwen3_moe also runs the feature rows and the resume case. It is the plainest MoE arch, with standard attention and routing, so its features and resume exercise only shared code and stand in for every arch.

## Running

`scripts/ci/run_distributed_nightly.sh` runs the suite and passes extra args to pytest: `-k` selects cases by arch, layout or feature name, and `--shard I/N` runs one of N shards.
