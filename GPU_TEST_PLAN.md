# GPU test plan: RL micro batch workers (`data.num_workers`)

Temporary file for an agent on a Linux GPU cluster. It will be deleted before merge.

## Context

Branch `feat/rl-data-workers` adds `WorkerMap` (`src/prime_rl/utils/worker_map.py`), an ordered, lazy map that runs a function in persistent forked worker processes, keeping at most `2 * num_workers` items in flight. The RL trainer (`src/prime_rl/trainer/rl/train.py`, search `mm_materializer`) uses it to decode and preprocess each micro batch's images ahead of the forward pass. Only each micro batch's `mm_refs` (base64 image strings) go to the workers; only the materialized image tensors come back. It is configured by `trainer.data.num_workers` (default 0, which materializes inline exactly as on `main`). PR.md in the repo root (if present) describes the design. It is an alternative to PR #3907, which does the same work on a background thread.

Nothing has run on a GPU yet. Everything below was reasoned about or measured on macOS only.

Run training with the `rl` entrypoint (see `skills/training/start-run/SKILL.md`), for example:

```bash
uv run rl @ configs/ci/nightly/multimodal_color_codeword.toml --run.name <name> --trainer.data.num-workers 4
```

Use the dashboard or the run's logs (`skills/training/monitor-run/SKILL.md`) for the metrics `time/step` and `time/forward_backward`. Do not change the design based on these results; report them so the design decisions can be made with numbers.

## Experiments, in priority order

### 1. Correctness and speedup

Run `configs/ci/nightly/multimodal_color_codeword.toml` (Qwen3.5-4B VLM, 25 steps) three times with distinct run names: `num-workers 0`, `2` and `4`.

- Check that losses and other training metrics match across the three runs (same seed; small numeric noise is fine).
- Compare mean `time/forward_backward` and `time/step`, excluding the first two steps.
- If the speedup is small, check whether image work is a meaningful part of the step at all (for example, time `materialize_mm_refs` with `num-workers 0`).

### 2. Memory pinned by forked workers (copy-on-write)

Workers are forked from the trainer at the first micro batch of step 0 and live for the whole run. After a fork, pages are shared until written; when the trainer later rewrites a page, the worker keeps the old copy alive. Each worker can therefore accumulate a private snapshot of trainer memory that the trainer rewrites, for example CPU-offloaded optimizer state (`model.optim_cpu_offload` defaults to true; its buffers are pinned, and it is unknown whether pinned memory is shared with forked children at all).

- During a `num-workers 4` run, find the worker processes (children of each trainer rank process, `pgrep -P <trainer_pid>`), and record `grep -E '^(Rss|Pss|Private_Clean|Private_Dirty)' /proc/<pid>/smaps_rollup` for each worker at step 2 and at the last step. Also record the trainer rank's RSS.
- Repeat with `--no-trainer.model.optim-cpu-offload` and with full offload (`--no-trainer.model.optim-cpu-offload --trainer.model.full-offload`) if the model supports it.
- Then apply a local, uncommitted one-line patch in `worker_map.py`, `multiprocessing.get_context("fork")` to `multiprocessing.get_context("forkserver")`, and repeat the default run. Record worker private memory, worker startup time, and whether training behaves identically. First check that the worker function pickles: `uv run python -c "import pickle; ..."` on `partial(materialize_micro_batch_mm, processor=<the loaded processor>, mm_adapter=...)` and report its pickled size.

Report: per-worker private memory growth over the run under fork vs forkserver, for each offload setting.

### 3. Shared memory and open files

During a `num-workers 4` run, sample every few steps:

- `df -h /dev/shm` on the node.
- Open file descriptors of each trainer rank and each worker: `ls /proc/<pid>/fd | wc -l`.

Expected: both stay bounded (roughly the image tensors of `2 * num_workers` micro batches per rank) and do not grow with step count. Report peaks and whether they grow.

### 4. Threads (#3907) vs processes

Check out PR #3907 in a separate worktree (`gh pr checkout 3907` inside a fresh worktree) and run the same config. Compare its `time/forward_backward` and `time/step` with experiment 1. The design question is whether image decoding on a thread competes with the training loop for the GIL in practice; if #3907 matches `num-workers 4`, the process design's extra complexity is not buying speed.

### 5. Robustness

- **Parent death:** during a `num-workers 4` run, `kill -9` one trainer rank process. Its workers should exit within about 5 seconds (`pgrep -P` on the dead pid's former children, or `ps -o pid,ppid,cmd` filtered by the worker command line). Report whether any worker survives.
- **Long run:** run at least 100 steps with `num-workers 4` and confirm no hang or slowdown over time. A Python `DeprecationWarning` about forking a multi-threaded process is expected once per worker; report any other warnings or errors from the workers.

### 6. Optional: cost of a generic whole-micro-batch worker function

Only if time allows. A generic worker function that receives whole micro batches is possible if `WorkerMap.__call__` copies each item's tensors into fresh shared memory before submitting (for example `torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: torch.empty_like(t).share_memory_().copy_(t), item)`), which keeps the step's own tensors out of `/dev/shm` and bounds shared memory to the in-flight items. Prototype that locally (uncommitted) with a function that takes and returns whole micro batches, and measure the main-process cost per micro batch and the `/dev/shm` peak, ideally on a MoE config with router replay (`enable_return_routed_experts`), where `routed_experts` is large.

## Reporting

For each experiment, report the commands run, the run names, the numbers, and anything unexpected. A small table per experiment is enough. Do not push commits or change the design; leave local patches uncommitted and describe them.
