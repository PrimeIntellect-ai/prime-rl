# Plan: async trainer checkpoints with pinned staging (replaces #4331's design)

## Handoff notes for the implementer
- This file was committed so the plan can be pulled on the cluster. Remove `PLAN.md` in a final commit before the branch goes to review.
- Do Phase 0 first and report its numbers to Garrett before implementing anything; he decides go/no-go and the target interval from them.
- Ask Garrett for the production run's W&B link; it is private and stays out of every commit, PR description, and committed file.
- Never push without Garrett's explicit approval in the conversation, and open PRs only as drafts.

## Context
We want more frequent trainer checkpoints in production RL runs, which run with `model.offload = "none"`. PR #4331 (`feat/async-ckpt-offload`, head `7194fd1`) only makes saves async when CPU optimizer offload is on, because its design hands the CPU-resident optimizer state to the writer by reference instead of copying it; with no offload it silently falls back to the sync save. Async checkpointing does not need offload: the standard design (torchtitan `async` / `async_with_pinned_mem`, and the closed #3874) stages a CPU snapshot and writes it in the background. This branch implements that design on top of current `main`, fixes the synchronization gaps found in review, and gives #4372 (`feat/ckpt-remote`, remote upload, currently stacked on #4331) a base to be restacked onto later.

References read for this plan: prime-rl `origin/main` @ `cbd3702` (torch pinned at 2.13.0), torchtitan `upstream/main` @ `eabd60d` (`torchtitan/components/checkpointer/dcp.py`, `training_engine.py`), torch `v2.13.0` (`torch/distributed/checkpoint/staging.py`, `_state_dict_stager.py`, `state_dict_saver.py`, `filesystem.py`, `_async_thread_executor.py`).

## Settled decisions
- Branch `feat/async-ckpt` off `main`. #4331 is not built on. #4372 is restacked onto this branch later, out of scope here.
- Staging: blocking copy into reused pinned CPU buffers, on the compute stream, then a background writer thread. A side-stream copy (torchtitan `async_with_pinned_mem`) is a possible follow-up only.
- Stage everything, in every offload mode: no by-reference special case for CPU-resident state.
- Our own small stager, not torch's `DefaultStager` (#3874 hit `RuntimeError: Attempted to call copy_() on an invalid python storage` with it on fused params).
- Config: `ckpt.async_save: bool = False` on the trainer `CheckpointConfig`, applying to all `model.offload` modes.

## Why this design, and what to measure first
Model: per save, sync blocks for `t_copy + t_write`; async with a blocking pinned copy blocks for `t_copy` plus any slowdown the background write causes to the steps it overlaps. The write time bounds checkpoint frequency: `interval * T_step > t_write`. Pinned host RAM per node is the sum of the local ranks' staged bytes, held for the run.

A current production multi-node RL run (`ckpt.interval = 5`, striped parallel FS) logs `time/save_ckpt` of about 97 to 135 s per sync save, against `time/step` of roughly 1500 to 3900 s (sampled history; confirm units and which process logs each key). If that holds, sync saves cost about 1% of step time at interval 5 and about 4% at interval 1, so the go/no-go depends on Phase 0. The run is private: Garrett supplies its W&B link out of band, and it must not be named or described in detail in commits, PR descriptions, or files under `benchmarks/`; report only derived, anonymized numbers (sizes, bandwidths, timings).

Phase 0 (before implementing), on the cluster, recorded under `benchmarks/scripts/async-ckpt/`:
- From the production run: model size and parallelism, `model.offload`, optimizer state bytes per param, number of trainer nodes, `time/save_ckpt`, `time/step` and `time/wait_for_batch` over the full history (not the sampled one), and the checkpoint size on disk.
- Whether the trainer is on the critical path: if `time/wait_for_batch` dominates, the trainer idles waiting for rollouts and sync saves are already nearly free end to end.
- Derived: checkpoint bytes per rank `S`, aggregate write bandwidth `S_total / t_save`, and pinned RAM per node per offload mode against the trainer nodes' `MemAvailable`.
- Microbenchmarks on one trainer node: pinned vs pageable GPU-to-CPU copy bandwidth per GPU with all 8 GPUs copying at once; time to allocate and pin `S` bytes (first-save cost).
- Interference: step time of a few steps with a background DCP write of `S` per rank running, vs without (GIL, host memory bandwidth, storage traffic on the fabric).
- Decide from these whether async is worth it, and at what checkpoint interval.

## Design
Save path when `async_save` is on (`CheckpointManager.save`, `src/prime_rl/trainer/ckpt.py`):
1. `self.wait()`: block until the previous background write is done, raising its error.
2. `maybe_clean()` runs here, after `wait()` and before the new write, so disk holds at most `keep_last` complete checkpoints plus one in flight.
3. Master mkdir, removes any existing `step_N/trainer/.metadata` (a re-save into an existing step must not leave an old `.metadata` beside half-overwritten shards), barrier; `_save_mapping`, dataloader state: otherwise unchanged from the sync path.
4. Build `{"app": app_state.state_dict()}` exactly as the sync path does today (including the existing `prepare_checkpoint_save` / `finish_checkpoint_save` handling for offload), so the state dict contents are identical to a sync save.
5. `dcp.async_save(state_dict, storage_writer=FileSystemWriter(path, per_thread_copy_ahead=0), process_group=self._async_write_group, async_stager=self._stager)` with the default thread executor. `per_thread_copy_ahead=0` selects `_SerialCpuLoader`, so the writer thread makes no CUDA calls (the default `_OverlappingCpuLoader` calls `current_stream()` / `synchronize()` from a thread that never set its device, which would create a context on GPU 0 for every local rank).
6. Record `(step, future)` as pending. `wait()` is the only place that adds the step to `ckpt_steps` and the only place that reports a save to the controller (`mark_saved`), after the future succeeds.
7. Each training step polls `future.done()` cheaply and raises a failed write's error then, instead of at the next save.

The stager (new `PinnedStager(AsyncStager)`, `_synchronize_after_execute = False`):
- Walks the state dict with paths (`torch.utils._pytree.tree_flatten_with_path` or equivalent). Every tensor leaf is copied: GPU sources into pinned buffers, CPU sources (offload state, `step` counters) into pageable buffers, since pinning gains nothing for a CPU-to-CPU copy and offload runs are the memory-tight ones.
- Pinned buffers are views into one flat pinned slab per rank, sized on the first save and reallocated only if the total grows. This avoids the caching host allocator rounding each pinned allocation up to a power of two (torch 2.13 default), which could double the pinned footprint. Buffers are keyed by path, so tensors `get_state_dict` builds fresh every save (the split packed params) reuse their views.
- Before allocating, compare the bytes to stage with `MemAvailable` and fail with a clear error if they do not fit. Log pinned and pageable bytes per rank.
- DTensors: stage `_local_tensor` and rewrap with the same `_spec` and `requires_grad=False`. GPU copies: `buffer.copy_(src, non_blocking=True)` on the current (compute) stream; CPU copies: synchronous `copy_`.
- Non-tensor leaves (scheduler state, `param_groups`, progress) are `copy.deepcopy`'d, because DCP pickles them later on the writer thread. Containers are rebuilt with their original types (tuples stay tuples).
- Ends with `torch.cuda.current_stream().synchronize()` when any GPU copy was issued.

## Intended invariants and where each is enforced
1. The copy reads post-optimizer values: GPU copies are enqueued on the compute stream after `optimizer.step()`, so stream order guarantees it. (Torch's `DefaultStager` copies on a side stream with no `wait_stream`/`wait_event` on the compute stream; we avoid that gap by construction. A future side-stream variant must record an event on the compute stream and `wait_event` it before copying.)
2. Training does not mutate staged state before the copy finishes: `stage()` synchronizes the compute stream before returning, CPU copies are synchronous, and non-tensor leaves are deep-copied.
3. Staging buffers are not overwritten while the writer reads them: `save()` calls `wait()` before staging.
4. A save is visible only once written: `ckpt_steps`, `keep_last` cleanup, and `controller.mark_saved` (in-loop `checkpoint` action, pause, and final save in `src/prime_rl/trainer/rl/train.py` near L703 and L824, `src/prime_rl/trainer/train.py` near L546 and L683) all happen in or after `wait()`. Control-action and final saves call `wait()` immediately after `save()`. Any later remote upload (#4372) also chains on write completion.
5. Trainer and orchestrator resume from the same step, and only from a complete one. One shared predicate: step N is resumable if `step_N/trainer/.metadata` exists in the trainer's checkpoint root and, for RL, `step_N/orchestrator/progress.pt` exists. `src/prime_rl/entrypoints/rl.py` (near L583) resolves it once and passes an explicit `resume.step` to both sub-configs; standalone trainer (`trainer/rl/train.py:156`, `trainer/train.py:140`), `CheckpointManager.__init__`, `entrypoints/launcher.py:66`, and `orchestrator/orchestrator.py:1200` use the same helper. `get_all_ckpt_steps` stays unchanged because broadcast dirs use it too. Verify how the orchestrator's checkpoint root relates to `trainer.ckpt.output_dir` when that is set.
6. Process groups are created once, on the main thread, in `CheckpointManager.__init__`, on every rank in the same order, both with `timeout=DCP_TIMEOUT`: one for main-thread barriers, sync save and load; one used only by the background writer (two threads must not run collectives on one group concurrently). The load path also moves to the cached group. `get_dcp_group()`'s other caller (`src/prime_rl/trainer/model.py:743`) is left alone and flagged.
7. Errors surface: a failed write raises at the next step's poll or at `wait()`. A crash mid-write leaves a step without `.metadata`, which invariant 5 skips.
8. Incomplete checkpoints do not outlive a resume: when resuming, the run removes checkpoint step directories after the resume step that fail the resumable predicate, by extending the existing future-step cleanup (`clean_future_steps`, `src/prime_rl/utils/pathing.py` near L429) to the checkpoint root(s). This keeps the SLURM grace loop (`multi_node_rl.sbatch.j2` near L612-631, which waits for `.metadata` in the highest `step_*/trainer`) from stalling on a stale directory.

## Known gaps
- A crash or preemption loses up to two intervals of progress (the in-flight write plus the steps since).
- If one rank dies mid-write, the others can block in the writer group's collectives for up to `DCP_TIMEOUT` (4 h).
- With async on, `time/save_ckpt` measures only the blocking part; the write time is logged separately.
- `get_dcp_group()` creating a new group per call remains for `model.py`.

## Files
- `packages/prime-rl-configs/src/prime_rl/configs/trainer.py`: `async_save` field with a one-sentence docstring matching the file's style.
- `src/prime_rl/trainer/ckpt.py`: `PinnedStager`, async path in `save_to_path` / `save`, `wait()`, per-step poll, group creation, `mark_saved` handoff, complete-step filtering in `__init__`.
- `src/prime_rl/utils/pathing.py`: the resumable-step helper; callers listed in invariant 5. `src/prime_rl/entrypoints/rl.py`: resolve once and pass down.
- `src/prime_rl/trainer/rl/train.py`, `src/prime_rl/trainer/train.py`: `wait()` placement per invariant 4, per-step poll, `maybe_clean` moved per design step 2; `time/save_ckpt` keeps measuring the blocking part; log the background write time when it completes.
- `docs/training.md`: checkpoint section: what `async_save` does, host RAM cost, disk peak of `keep_last` plus one in flight, and the known gaps a user would hit.

Reuse from #4331 where it fits (read with `git show 7194fd1:src/prime_rl/trainer/ckpt.py`): the `wait()` / pending-step bookkeeping, the DTensor rewrap, and the final-save `wait()` calls. Do not port its offload-specific pieces (`_NoCopyStager`, `cpu_state_dict`, `hold_state_updates`, `_copy_cpu_state_split_on_sharded_dim_to_gpu`).

## Out of scope
Side-stream copy, process-based writer, #4372's remote upload, #4331's by-reference staging, changes to `optimization_dtype` / `reduce_dtype`.

## Verification
Run everything inside a SLURM allocation, never on a login node. Follow `~/.claude/skills/prime-rl/e2e.md` for e2e hygiene and commit every script and config behind reported numbers under `benchmarks/scripts/async-ckpt/`.
- Unit: extend `tests/unit/utils/test_pathing.py` for the resumable-step helper (missing `trainer/.metadata` or missing `orchestrator/progress.pt` is skipped). No other new tests unless clearly necessary.
- A test-only slow writer (a `FileSystemWriter` subclass that sleeps before `write_data`, enabled by a temporary patch) makes write windows long and deterministic for the checks below.
- Bitwise correctness, Qwen3-0.6B reverse-text, 2 GPUs, `ckpt.interval = 2`, each of `model.offload` = `none`, `optimizer`, `full`, SFT and RL: a temporary patch does a sync reference save of the same state right before each async save; compare every entry, tensors and non-tensor state (scheduler, `param_groups`, progress), at steps 2 and 4. Expect 0 mismatches.
- Negative controls, each must make a comparison fail: (a) with the slow writer, skip the pre-staging `wait()` (using a separate executor so writes do not share the writer group) and compare step N after step N+interval has staged; (b) remove the `synchronize()` at the end of `stage()` and compare the current step.
- Control API: `checkpoint` and pause actions reach "committed" only after `.metadata` exists (slow writer on).
- Crash mid-write, full RL stack (slow writer on, kill the trainer during the write): trainer and orchestrator resume from the same complete step, no hang, and the incomplete step directory is removed.
- Failure during a write: raise in step N+1 while write N is in flight; the process exits in bounded time and resume picks a complete step.
- `keep_last = 1`: at most two checkpoints on disk at any time, and a complete one always exists.
- Writer thread makes no CUDA calls: `nvidia-smi` shows no extra processes on GPU 0 during a multi-GPU async write.
- Multi-node, MoE with EP and packed fused params, offload `none`: bitwise check once, then timing: blocking time per save, background write time, pinned bytes per rank and per node, step-time impact during writes, sync vs async.
- `uv run pytest` on the touched unit tests, `ruff check`, `ruff format`.
