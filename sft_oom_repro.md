# SFT host-memory reproducer

`sft_oom_repro.py` measures host memory of the SFT dataloader on CPU only, to find out
why PR #3772 (`PackedDataLoader`) exhausted host memory on B300 nodes before step 1
while the same job on `main` did not
([report](https://github.com/PrimeIntellect-ai/prime-rl/pull/3772#issuecomment-6058095937)).

The script builds a synthetic multi-subset dataset, constructs the real dataloader of
whichever commit is checked out, pulls a few optimizer steps of micro-batches, and
reports memory summed over the whole `torchrun` process tree. It needs no GPU and no
model. Run it on Linux: macOS spawns DataLoader workers by default, has no PSS, and has
no cgroup memory limit, so it cannot reproduce the difference.

## Candidates

Each run in the matrix below disables one candidate. A candidate is a cause if
disabling it closes the gap to `main`.

1. **`unique()` rewrites the dataset.** `RenderedDataset.__init__`
   (`src/prime_rl/trainer/sft/data/broker.py`) calls `Dataset.unique("__subset")` to
   number the sources. When subsets are interleaved, the dataset carries an indices
   mapping whose length differs from the base table, and `datasets` then runs
   `flatten_indices()`, a full rewrite of every column. The `add_column` calls in
   `setup_and_interleave_datasets` use random fingerprints, so every rank writes its
   own copy. Visible as `Flattening the indices` log lines, growing `hf_cache_delta`
   and `dirty`.
2. **Spawned workers.** The branch creates DataLoader workers with `spawn`, so each
   worker re-imports torch and the trainer and unpickles a private copy of the
   in-memory Arrow columns. `main` forks, which shares that memory copy-on-write.
   Scales with `ranks per node * num_workers`.
3. **In-memory shuffle.** Each worker shuffles with `keep_in_memory=True`, holding a
   private index of 8 bytes per row. `main` writes the index to a cache file that
   workers share through the page cache.

## Setup

Use two checkouts on the same Linux machine: this branch and `main`. Sync each
checkout's environment with `uv sync --all-extras`. The first run of each argument set
builds the synthetic data under `~/tmp/sft_oom_data/`; later runs reuse it. On
multiple nodes, pass a shared `--data-dir`.

## Experiment matrix

Run from the repo root of each checkout. Runs 1 to 3 go on both checkouts; runs 4 to 6
exist only on the branch.

| Run | Arguments | Disables |
|---|---|---|
| 1 | `--subsets 4 --num-workers 8` | nothing (baseline) |
| 2 | `--subsets 1 --num-workers 8` | 1 (no interleave, no indices mapping) |
| 3 | `--subsets 4 --num-workers 1` | most of 2 and 3 |
| 4 | `--subsets 4 --num-workers 8 --start-method fork` | 2 |
| 5 | `--subsets 4 --num-workers 8 --sources config` | 1 |
| 6 | `--subsets 4 --num-workers 8 --shuffle disk` | 3 |

```bash
uv run torchrun --nproc-per-node 8 sft_oom_repro.py --subsets 4 --num-workers 8
```

`--sources config` replaces `Dataset.unique` with a lookup on the base Arrow table,
which skips the rewrite. `--shuffle disk` forces `keep_in_memory=False`. Both are
monkeypatches applied in the main process and, through environment variables, in
spawned workers; the branch code is unchanged.

Scale with `--rows` and `--payload-chars` until the baseline gap is clear. Memory costs
of candidates 2 and 3 grow with row count, so a small dataset can hide them. The
defaults are 200k rows of 4000 characters.

## Reading the output

- `[mem] tick` lines print every 5 seconds from local rank 0, so the trend survives an
  OOM kill. `pss` is summed over the `torchrun` agent and all descendants, including
  DataLoader workers.
- `[mem] begin` and `end` lines bracket each phase: `build_data`, `load_dataset`,
  `build_loader`, `first_batch`, `step_N`. A jump in `build_loader` points to
  candidate 1; a jump in `first_batch` points to worker startup (candidates 2 and 3).
- `dirty`, `shmem`, and `hf_cache_delta` track page-cache writes, tmpfs usage, and
  cache files written by `datasets`. Rewrites count against cgroup memory limits even
  though they do not appear in PSS.
- `[mem] PEAK` gives the overall maximum and the phase where it occurred.
- `[rank N] <phase>: <seconds>` lines show per-rank phase timings.

Record the `PEAK` line and the per-phase `end` lines for each run and checkout.
