# RL image preprocessing: inline vs worker processes vs #3907 thread

Mock benchmark run 2026-10-07 on one exclusive H200 node (SLURM job 3358, prime-nebius-puku-h200-gpu-039),
all runs serial on the same node.

## Setup

- Trainer-only (`torchrun --standalone --nproc-per-node 4 -m prime_rl.trainer.rl.train`), no inference or
  orchestrator. Qwen3.5-9B (untied embeddings), FSDP over 4 GPUs, seq_len 16384, 8 micro batches per rank per
  step, 12 steps. No offloading (`optim_cpu_offload = false`, `fsdp_cpu_offload = false`, no `ac_offloading`),
  no checkpoints. Dtypes left at defaults. Config: `mock-trainer.toml`.
- Data: local, uncommitted mock (`mock_mm.py`, hooked into `FakeDataLoader` by `apply_mock.py`). Each sample is
  64 prompt tokens, 2 photo-like JPEG images (drawn from a fixed pool of 16), and 256 completion tokens, packed
  to fill the micro batch. Image side set by `PRL_MOCK_MM_IMAGE_PX`. Real `materialize_mm_refs` and the real
  Qwen3.5 adapter run on these, and the adapter's placeholder-length check passes.
- Per micro batch (login-node CPU, 1 thread): 512 px: 38 images, 9728 image tokens, 858 ms decode.
  1024 px: 12 images, 12288 image tokens, 1092 ms. 1440 px: 6 images, 12150 image tokens, 1103 ms.
- Arms, each from its own frozen worktree with the mock applied:
  - base: `51879014f` (main at the branch point), inline materialization.
  - branch: `9ea1f9c3e` (feat/rl-data-workers incl. the generic `prepare` refactor), `data.num_workers` 0/2/4.
  - 3907: `e80e22db3` = `51879014f` + cherry-pick of #3907's `3449123bb` (background thread, pinned memory,
    `non_blocking` copies).
- Driver: `run_matrix.sh`; summary: `compare_mock.py` (full output in `mock-results.txt`).

## Results

Median `time/forward_backward` in seconds over steps 3 to 12 (step 1 includes compile), lower is better.
`time/step` is 0.06 to 0.08 s higher in every run.

| Arm | 512 px | 1024 px | 1440 px |
|---|---|---|---|
| base, inline | 24.52 | 24.79 | 24.25 |
| branch, num_workers 0 | 24.64 | 24.91 | 24.29 |
| branch, num_workers 2 | 19.97 | 19.02 | 18.48 |
| branch, num_workers 4 | 20.12 | 19.22 | 18.70 |
| #3907 thread | 19.47 | 18.40 | 17.82 |

- Noise floor: a repeat of base at 1024 px gave 24.84 s vs 24.79 s (0.05 s). Within-run spread is about 0.3 s.
- Overlapping image work with the GPU cuts step time by 19 to 27%. At 1024 px: 24.79 s to 18.40 s with the
  thread (26% less time per step), to 19.02 s with 2 workers (23% less).
- The thread is 0.50 to 0.66 s per step faster than the best process setting at every size, about 10 times the
  noise floor. GIL contention from decoding on a thread does not show up at this load, even at 1440 px.
- 2 workers beat 4 by 0.15 to 0.22 s at every size.
- num_workers 0 is within 0.12 s of base (inline path, as intended).
- Peak memory is identical across arms at each size (54.0 to 55.1 GiB).

Parity: at 1024 and 1440 px every arm matches base bit for bit in `loss/mean` and `optim/grad_norm` at all 12
steps, including the base repeat. At 512 px all arms, including num_workers 0, differ from base by at most
8e-9 in loss and 2e-4 in grad norm. That points to GPU nondeterminism with many small images rather than the
workers. There is no 512 px base repeat to confirm this.

## Not measured

- Which part of the thread's lead comes from no pickling or `/dev/shm` round trip, and which from pinned memory
  with `non_blocking` copies.
- Copy-on-write memory, `/dev/shm` and fd growth over long runs, forkserver, kill -9 robustness (GPU test plan
  experiments 2, 3, 5).
- Lighter image loads, CP/EP layouts, and MoE models.

## Other findings from the session (main, not this branch)

- Single-node RL with N>1 inference GPUs and NCCL broadcast needs `--inference.vllm.data-parallel-size N`, or
  the startup broadcast group is sized wrong (PR #3841 closed unmerged).
- Tied-embedding Qwen3.5 (0.8B/2B/4B): the trainer never loads `lm_head`, and NCCL then overwrites vLLM's tied
  embeddings with it. Garrett plans to remove weight tying. Evidence in `~/tmp/bcast-bug/`.
- Nightly CI has not actually run since at least 2026-09-30 (jobs queue without a runner, then are cancelled).
- `multimodal_color_codeword.toml` as written gets no training signal with Qwen3.5 (thinking on, 64 tokens).

# Second matrix: the thread-pool rework (2026-10-08)

One exclusive node (SLURM job 3386, prime-nebius-puku-h200-gpu-020), all runs serial, 1024 px mock, same setup
as above. New arms: `thread` = `be65f5dd5` (WorkerMap on a ThreadPoolExecutor, whole micro batches through
`prepare_micro_batch`; later renamed `WorkerPool` with call-time `map_fn` in `39ecd0711`, same logic) and
`3907-nopin` = `e80e22db3` with `.pin_memory()` and `non_blocking=True` removed (uncommitted local edit).
Driver `run_matrix2.sh`, summary `compare_mock2.py`, output `mock2-results.txt`.

Median `time/forward_backward` in seconds over steps 3 to 12, lower is better:

| Arm | offload off | offload on |
|---|---|---|
| base, inline | 24.98 (repeat 24.99) | 26.56 |
| thread pool, num_workers 1 | 18.86 | 20.24 |
| thread pool, num_workers 2 | 18.93 | |
| #3907 without pinning | 18.88 | |
| #3907 with pinning | 18.63 | |

- The thread pool cuts step time by 24.5% (offload off) and 23.8% (offload on). Saving is 6.1 to 6.3 s either
  way, so preprocessing on a thread does not contend with CPU-offloaded optimizer work.
- Thread pool matches #3907 without pinning (18.86 vs 18.88 s). Pinning alone is worth 0.25 s per step.
- num_workers 2 is no faster than 1.
- Every arm, offload on or off, is bit-identical to base in loss/mean and grad_norm at all 12 steps.
- Host RSS per rank over 30 steps (`run_rss.sh`, sampled every 10 s): base 3.18 to 3.22 GiB, thread pool
  3.49 to 3.52 GiB, both flat. The constant 0.3 GiB is the image tensors of the prefetched micro batches.
- The first 30-step RSS attempt sampled the wrong processes (ranks rename themselves via set_proc_title);
  the rerun samples torchrun's children.
