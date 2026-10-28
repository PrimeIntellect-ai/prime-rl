# Distributed SFT suite

A nightly functionality suite for the trainer. It runs trainer-only SFT (`uv run sft`) on seeded tiny checkpoints of every trainer arch, across a few parallel layouts and memory features, on one node. A case passes if training exits cleanly with a finite loss. Selected cases also check that checkpoint resume continues the step count and that the compiled model has no graph breaks.

## Layout

- `../tiny_models.py`: one `TinyModel` per arch, with its tiny config, the tokenizer matching its vocab and, where needed, its renderer. It also writes the seeded tiny checkpoints. It holds nothing SFT-specific, so other suites can reuse it.
- `archs.py`: the matrix. It defines the classes (`Layout`, `Feature`, `Arch`, `Case`), the parallel layouts and features, and `ARCHS`, one entry per arch pairing its `TinyModel` with how the suite runs it. It expands `ARCHS` into `CASES`.
- `harness.py`: running and checking a case: its `uv run sft` arguments, sharding, and the parsers for metrics and logs.
- `test_matrix.py`: runs each case once and checks it with `test_sft` and `test_no_graph_breaks`.
- `../conftest.py`: the session fixtures (tiny checkpoints, tokenizer downloads), the `--shard` option and the GPU-count skip.
- `tests/regression/configs/distributed/sft/`: the shared `base.toml` and the training data.

## Coverage

The suite has a list of features, settings that change how the trainer runs (for example selective activation checkpointing, optimizer offloading or `fullgraph` compile). Running every feature on every arch and parallel layout would take far too many launches, so the full feature list runs on one representative arch. Today that arch is qwen3_moe. It is the plainest MoE arch, with standard attention and routing, so its features exercise only shared code and stand in for every arch. It also runs the checkpoint-resume check.

Every other arch runs one base case per parallel layout, a combination of FSDP, context-parallel and expert-parallel degrees across the node's GPUs. A base case checks that training runs (clean exit, finite loss), and on the CP layout also that the compiled model has no graph breaks.

## Running

`scripts/ci/run_distributed_nightly.sh` runs the suite and passes extra args to pytest. `-k` selects cases by arch, layout or feature name, as in `-k "qwen3_moe and cp4_ep4"`, and `--shard I/N` runs one of N shards.
