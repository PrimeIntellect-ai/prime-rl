# Handoff: verify the async AnnotationWriter on a GPU node

Branch `fix/trainer-async-annotation-writer` in prime-rl (base `a566e5d8e` on `main`) moves rank 0's per-step annotation serialize-and-write in `AnnotationWriter.flush()` (`src/prime_rl/trainer/rl/annotations.py`) onto a single background thread, so the other ranks stop waiting for it at the gradient-norm all_reduce. The gather to rank 0 stays on the critical path. Also: `AnnotationWriter.wait()` runs before every checkpoint save and `close()` runs before `monitors.finalize` (`src/prime_rl/trainer/rl/train.py`). This file covers the testing that could not be done without a GPU node.

## Why it matters

A trainer-bound production run (DeepSeek-V4-Flash architecture, 16 trainer nodes at DP 32 x CP 4) had median `time/wait_for_batch` 21 s, `time/step` ~930 s, `time/forward_backward` ~750 s, 4e8 to 5e8 trained tokens per step. Scaling the one-node proxy's measurement (2.1M tokens per step: ~200 ms serialize, ~72 ms gather) linearly gives ~45 s serialize plus ~15 s gather per step in production. `time/forward_backward` (`train.py:348` to `:669`) contains `flush()`, so this stall sits in the denominator of `perf/mfu` and `perf/throughput`. Expected gain when trainer-bound: ~40 to 50 s per step, ~4 to 5%. None of these production-scale numbers have been measured yet; the benchmark below is what measures them.

## What you need

- One node with 8 GPUs (B300 is what the estimates assume; the production run also used B300), and node-local disk with ~60 GB free per arm (the benchmark writes ~6 GB of JSON per step).
- Two prime-rl checkouts: `<sync>` at `a566e5d8e` (the branch's base on `main`, without the change) and `<async>` at the branch head. Initialize submodules as recorded, and create each venv only with `uv sync --all-extras --all-packages`.
- `bench.py`, next to this file on the branch. Both arms run the branch's copy (`<async>/benchmarks/annotation_writer/bench.py`) from their own checkout, so each arm imports its own `prime_rl`.

## Steps

1. Unit tests, on the node, from the branch checkout: `uv run pytest tests/unit/train/rl/test_annotations.py -v`. They have never been run. Never run pytest on a login node.
2. Smoke run of both arms at small scale to shake out errors (~2 min each). From each arm's checkout (`cd <sync>` or `cd <async>`): `uv run torchrun --nproc-per-node 8 <async>/benchmarks/annotation_writer/bench.py --out-dir /local/annot-bench/smoke-<arm> --tokens-per-step 2e7 --steps 3 --gpu-seconds 5`. The script prints `arm=sync` or `arm=async`; check it matches the checkout.
3. Full runs, one per arm, from that arm's checkout: `uv run torchrun --nproc-per-node 8 <async>/benchmarks/annotation_writer/bench.py --out-dir /local/annot-bench/<arm>` (defaults: 5e8 tokens per step, 6 steps with step 0 as warmup, 90 s of GPU work per step). If the async arm's `background_write_s` exceeds `--gpu-seconds`, rerun both arms with a larger `--gpu-seconds`, since production steps are ~900 s and a write longer than the step makes the next `flush()` wait.
4. Confirm both arms wrote identical annotations: `diff -rq /local/annot-bench/sync/monitors/file/traces/annotations /local/annot-bench/async/monitors/file/traces/annotations` should print nothing.
5. Delete the benchmark output directories afterwards.

## What to report

Per arm, the table `bench.py` prints (also in `<out-dir>/results.json`), and the comparison:

| Quantity | Sync arm (expected) | Async arm (expected) |
|---|---|---|
| `others_stall_s` (ranks 1-7, flush to all_reduce done) | gather + serialize, ~60 s | ~gather, ~15 s |
| `rank0_flush_return_s` | same as stall | ~gather |
| `rank0_gpu_work_s` vs `others_gpu_work_s` | equal | close; a gap is GIL contention |
| `background_write_s` | n/a | well below `--gpu-seconds` |
| `step_s` | GPU work + ~60 s | GPU work + ~15 s |

Call out any deviation from these expectations, especially a rank-0 GPU-work slowdown in the async arm (the background thread's `orjson` serialization holds the GIL while rank 0 launches kernels), since that would eat into the gain. The benchmark's 90 s step is much shorter than production's ~900 s, so any contention is a larger fraction of the step here than it would be in production.

## Production follow-up (not for this handoff)

On a trainer-bound production run with the branch, `time/forward_backward` should drop by tens of seconds at equal `progress/tokens`, with `perf/mfu` rising by a matching fraction.
