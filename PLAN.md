# Gather-form MoE combine: implement, then A/B against PR #4074 on GPU

## Context

PR #4074 (https://github.com/PrimeIntellect-ai/prime-rl/pull/4074, branch `fix/deterministic-moe-combine`,
head `d45d8194d17b6aea0eb9ed2dd288e249af9a55e1`, author hubert-marek) makes the trainer's MoE combine
deterministic. Before the PR (main at the merge base `4487ef3`), `_scatter_routed_output` in
`src/prime_rl/trainer/distributed/token_dispatcher.py` summed each token's `top_k` expert outputs with a BF16
`scatter_add`. Its CUDA atomics add in a random order and round to BF16 after every add, so repeated frozen
forwards gave different logprobs.

The PR (function now `_combine_routed_output`) does this instead:
1. `_local_reorder` keeps the undivided `argsort` result `route_indices_experts_sorted`. Entry `i` is the flat
   route slot `p = token * top_k + k` that expert-sorted row `i` came from, where `k` is the position among the
   token's selected experts. Dispatch gathers inputs with `x[route_indices_experts_sorted // top_k]`.
2. Combine writes each expert-sorted output row back to its slot in a fresh buffer, with every slot written once
   and no additions: `output = torch.empty_like(routed_output); output[route_indices_experts_sorted] = routed_output`.
3. `output.view(num_tokens, top_k, dim).sum(dim=1, dtype=float32).to(bf16)`: a fixed-order FP32 sum with one
   final rounding.

The PR's history includes a reverted attempt (`5a9e4b6ac`) that used a scoped deterministic `scatter_add`
(torchtitan's approach). `d45d8194d` restored the route-ordered FP32 form. The author also removed the PR's unit
tests from the diff (`5a437a2b3`). `token_dispatcher.py` is identical at `0fbfb1f` and `d45d8194d`.

A review checked the PR against vLLM 0.31.0 (pinned in `uv.lock`). On H200, vLLM's BF16 MoE uses Triton. Its
`moe_sum` kernel sums the `top_k` slots in slot order with an FP32 accumulator and one final rounding. The PR
matches that structure, so its numerics are as faithful as the trainer can get without fusing the router score
into the expert GEMM (out of scope).

The cost: step 2 materializes a `num_tokens * top_k * hidden_dim` BF16 buffer, and an in-place index write cannot
be fused by inductor into the following reduction. The PR description reports compiled peak extra memory of
13.1 MiB before and 117.6 MiB after (2,231 tokens, top_k 8, hidden 3,072), but those numbers may come from an
intermediate version, so re-measure them rather than quoting them.

**Alternative (this branch):** the same arithmetic as a gather, wrapped in a small autograd function so both
directions are gathers:
- **Forward:** each slot pulls its row through the inverse permutation, so inductor can fold the indirect load
  into the reduction and skip the buffer. A CPU inductor check (torch 2.13.0) showed no `T*K x D` buffer, with
  output bitwise identical to the PR's.
- **Why the custom backward:** plain autograd through a gather backpropagates with `index_put(accumulate=True)`.
  On sm90 and newer, inductor lowers that to a zero-filled `T*K x D` BF16 buffer plus BF16 atomic adds
  (`needs_fallback_due_to_atomic_add_limitations` in torch `_inductor/utils.py`, v2.13.0). The result is exact,
  since each slot is written once, but it costs a zero-fill and atomic traffic. A backward that gathers through
  the forward permutation avoids both.

**Goal:** implement the gather form on a child branch of the PR branch. Then compare three arms on GPU, **main**,
**PR** and **gather**, on bitwise parity, speed, memory and recompilation behavior. If the gather form wins, open
a draft PR from this branch into `fix/deterministic-moe-combine`.

## Setup

- You are on a Linux GPU machine, in a worktree of branch `fix/deterministic-moe-combine-gather`. That branch forks
  from `fix/deterministic-moe-combine` at `d45d8194d`. Its only commit adds this `PLAN.md`.
- Check the setup first:
  - `git log --oneline -3` should show `d45d8194d` directly under the PLAN.md commit.
  - `git submodule status` should show the submodules at their recorded commits.

  If either check fails, stop and ask the user.
- Build the environment with `uv sync --all-extras --all-packages`. Run all code with `uv run`.
- **Never `git push` without explicit approval from the user in the conversation, every time.** Committing locally
  is fine.
- Before any PR into the parent branch, delete `PLAN.md` in its own commit.
- Repo rules (`AGENTS.md`) and the user's rules:
  - Commit messages are conventional with a scope.
  - No em-dashes in any file.
  - Minimal comments.
  - Every script that produces reported numbers is committed under `benchmarks/scripts/moe_combine/`.

## Phase 1: implement and check

All production edits are in `src/prime_rl/trainer/distributed/token_dispatcher.py`. Nothing else in the repo
reads the symbols changed below. `deepep.py` has its own combine, so leave it untouched.

1. **Row permutation with gather backward.** Add a small `torch.autograd.Function`. The names below are
   suggestions:
   ```python
   class _PermuteRows(torch.autograd.Function):
       @staticmethod
       def forward(ctx, rows, gather_index, inverse_index):
           ctx.save_for_backward(inverse_index)
           return rows[gather_index]

       @staticmethod
       def backward(ctx, grad):
           (inverse_index,) = ctx.saved_tensors
           return grad[inverse_index], None, None
   ```
   Called as `_PermuteRows.apply(routed_output, sorted_row_of_slot, route_indices_experts_sorted)`:
   - **Forward:** `out[p] = routed_output[sorted_row_of_slot[p]]`.
   - **Backward:** `grad_routed[i] = grad_out[route_indices_experts_sorted[i]]`. This is exact, because
     `route_indices_experts_sorted[i] = p` exactly when `sorted_row_of_slot[p] = i`.
2. **`_local_reorder(...)`:**
   - After `route_indices_experts_sorted = torch.argsort(flattened_experts, stable=True)`, compute the inverse
     with an integer scatter under `torch.no_grad()`, with no extra sort:
     `sorted_row_of_slot = torch.empty_like(route_indices_experts_sorted).scatter_(0, route_indices_experts_sorted,
     torch.arange(route_indices_experts_sorted.numel(), device=route_indices_experts_sorted.device))`.
   - Return both index tensors. The input gather stays `x[route_indices_experts_sorted // top_k]`.
   - Keep returning `top_scores_experts_sorted`: the `score_before_experts=True` path multiplies routed inputs by
     it in dispatch.
3. **`LocalDispatchState`** (inherited by `TorchDispatchState`):
   - Hold `route_indices_experts_sorted`, `sorted_row_of_slot` and `top_k: int`.
   - `scores_after_experts` holds the **slot-order** scores `top_scores` (shape `[num_tokens, top_k]`) when scores
     apply after the experts, and `None` otherwise.
   - Update `LocalTokenDispatcher.dispatch` and `TorchTokenDispatcher.dispatch`; both already have `self.top_k`.
   - `MXFP8TorchTokenDispatcher` inherits and needs no change.
4. **`_combine_routed_output(routed_output, *, num_tokens, top_k, route_indices_experts_sorted, sorted_row_of_slot,
   scores_after_experts)`:**
   ```python
   hidden_dim = routed_output.shape[-1]
   restored = _PermuteRows.apply(routed_output, sorted_row_of_slot, route_indices_experts_sorted)
   restored = restored.view(num_tokens, top_k, hidden_dim)
   if scores_after_experts is not None:
       restored = (restored.float() * scores_after_experts.unsqueeze(-1)).to(routed_output.dtype)
   accumulation_dtype = torch.float32 if restored.dtype in (torch.bfloat16, torch.float16) else restored.dtype
   return restored.sum(dim=1, dtype=accumulation_dtype).to(routed_output.dtype)
   ```
   - **The rounding points match the PR exactly.** Each contribution is multiplied in FP32 and rounded to the
     routed dtype. All `top_k` contributions are then summed in FP32 and rounded once. Router scores are FP32
     (`src/prime_rl/trainer/models/layers/moe.py`), so the promotion is the same as in the PR.
   - **Use the explicit `hidden_dim`, not `-1`.** With `-1`, `view(0, top_k, -1)` raises on empty input. With the
     explicit size, the PR's `if num_tokens == 0` early return can go. Remove it only if the empty-input test
     passes.
   - **Comments:** drop the PR's comments about the scatter form. Add at most one short comment, saying that the
     gathers let the compiler fuse the lookup into the sum and keep the backward free of atomics.
5. **Validation tests:** the author removed `tests/unit/train/test_token_dispatcher.py` from the PR on purpose,
   so do not re-add it under `tests/`. Restore it as `benchmarks/scripts/moe_combine/test_combine.py` from
   `git show 0fbfb1f:tests/unit/train/test_token_dispatcher.py`, then adapt it as follows.
   - **Test 1 (`test_combine_restores_routes_and_gradients`):**
     - Build `sorted_contributions` with the forward permutation, recomputed as
       `torch.argsort(selected.reshape(-1), stable=True)` or taken from the new return value.
     - Pass both index tensors, `top_k=`, and the slot-order `scores` (not `sorted_scores`) as
       `scores_after_experts`.
   - **Test 2 (BF16 cancellation, `256, 1, -256, 1` must sum to 2):**
     - `order` is the forward permutation.
     - Pass `sorted_row_of_slot=torch.argsort(order)`, `route_indices_experts_sorted=order` and `top_k=4`.
   - **Test 3 (empty input and gradcheck):**
     - Pass `top_k` (any value for the empty case; 3 for the gradcheck, which has 6 rows and 2 tokens).
     - Pass the inverse of `[3, 0, 5, 1, 4, 2]` alongside it.
   - **Check that the tests can fail.** Gradcheck passes even if the two permutations are swapped, because any
     permutation is linear. Tests 1 and 2 catch a swap. Confirm this by temporarily swapping them and seeing
     tests 1 and 2 fail.
   - Keep every assertion exact (`rtol=0, atol=0`).
   - Run with `uv run pytest benchmarks/scripts/moe_combine/test_combine.py`, plus
     `uv run pytest tests/unit -k "moe or dispatcher"`.
6. **Three-arm parity script** (`benchmarks/scripts/moe_combine/parity.py`):
   - Load each arm's `token_dispatcher.py` as a separate module via `importlib`, from
     `git show <sha>:src/prime_rl/trainer/distributed/token_dispatcher.py`:
     - M at `4487ef3`, where `_scatter_routed_output` takes token indices (route index `// top_k`).
     - A at `d45d8194d`.
     - B from the working tree.
   - From one routing draw, compare A and B over random BF16 inputs: `top_k` in {1, 2, 8}, token counts in
     {0, 1, 7, 257, 2231}, hidden 3072, CPU and CUDA, eager and compiled.
   - Use `torch.equal` on the forward output, the input gradient and the score gradient. All three must be
     bitwise equal between A and B. Report M's maximum absolute difference from A.
7. **Compile check** (`benchmarks/scripts/moe_combine/compile_check.py`):
   - **The block:** compile `_local_reorder`, an all-to-all stand-in and the combine with
     `torch.compile(fullgraph=True)`, with `torch._dynamo.config.capture_scalar_outputs = True`.
   - **The stand-in:** it must mimic `_all_to_all_single` in `src/prime_rl/trainer/distributed/collectives.py`. Use
     a `torch.library.custom_op` that returns a clone in eager. Its `register_fake` returns
     `x.new_empty((torch.library.get_ctx().new_dynamic_size(), *x.shape[1:]))`, which makes the row count unknown
     at trace time, as in the real EP path. Give it identity autograd, and apply it on both dispatch and combine.
   - **The run:** token counts 64, 96, 130, 257, 3, 2, 1, 0, 50, forward and backward, for all three arms, under
     `aot_eager` and under `inductor` on CUDA.
   - **Pass criteria:**
     - No errors, and the same compiled-frame count for every arm. On CPU, M and A both compiled 4 frames: static,
       dynamic, and the 1- and 0-size specializations.
     - Inspect B's generated code with `torch._inductor.utils.run_and_get_code` on forward plus `.backward()`. The
       forward must allocate no `T*K x D` buffer. The backward must contain no `atomic_add` and no zero-filled
       `T*K x D` buffer.
8. **Commit** as `perf(trainer): combine MoE routes with fusable gathers`, ending with the attribution trailer the
   session's system reminder specifies. Report the Phase 1 results to the user, then continue to Phase 2. Arm B
   is this commit's SHA, used from a local detached worktree, so no push is needed.

## Phase 2: GPU A/B

**Before designing or launching anything in this phase, invoke the user's `/prime-rl` and `/profiling` skills**
(`~/.claude/skills/prime-rl/`, `~/.claude/skills/profiling/`) and follow them.
- **What they govern:** A/B run hygiene, frozen arms, noise floors, harness-in-history, trace capture, and the
  analysis scripts named below.
- **What to read:** at least `prime-rl/SKILL.md`, `prime-rl/e2e.md`, `profiling/SKILL.md` and
  `profiling/projects/prime-rl.md`.
- **Known stale entry in the skills:** `e2e.md`'s offload settings. The current config has a single
  `model.offload` field (`packages/prime-rl-configs/src/prime_rl/configs/trainer.py`), so use
  `model.offload = "none"` and `model.ac_offloading = None`.
- **Conflicts:** where this plan and the skills disagree on anything else, follow the skills and tell the user.
- **Missing skills:** if the skills are not installed on this machine, stop and ask the user to install them
  rather than improvising a harness.

**Arms:**
- **M = main** at `4487ef3`, the PR's merge base, which has the BF16 atomic `scatter_add`. Pinning main there
  instead of at the current `main` tip means M and A differ only in the combine. `4487ef3..d45d8194d` changes only
  `token_dispatcher.py`, with the same `deps` and `uv.lock`.
- **A = PR** at `d45d8194d` (scatter form).
- **B = this branch** at the Phase 1 commit (gather form).

**How to run:**
- Run each arm from its own detached worktree at its SHA, with a fresh `TORCHINDUCTOR_CACHE_DIR` and
  `TRITON_CACHE_DIR` per run, serially on the same exclusive node.
- Record all three SHAs, the GPU model, and the torch and vLLM versions from `uv.lock`.
- Before launching anything long, post the full run matrix to the user and get confirmation. It covers
  workloads, arms, layouts and repeats, plus concrete values for `data.seq_len`, `data.micro_batch_size`,
  `data.batch_size`, GPU count, `model.ep` (the EP used in practice, likely 8), GLM's `model.debug.num_layers`,
  and the logprob tool's model and `--tokens-per-gpu`. Pick `--tokens-per-gpu` to match one microbenchmark
  shape, so op-level and end-to-end results connect.

1. **Op microbenchmark** (`profiling/scripts/bench_ops.py`, with a cases file in `benchmarks/scripts/moe_combine/`):
   - **Loading the arms:** load each arm's module from its SHA as in Phase 1 step 6, so one process holds all
     three. List A first, so `bench_ops.py`'s reference and speedup columns are relative to the PR.
   - **What to time:** dispatch index preparation (`_local_reorder`, which in B includes the inverse scatter) plus
     the combine, with "after experts" scores (every MoE model in the tree sets `score_before_experts=False`).
     Time forward and forward+backward as separate cases, in eager and compiled mode.
   - **Compiled cases:** run them through the all-to-all stand-in from Phase 1 step 7, with `dynamic=True`, so the
     row count is unbacked and the custom op acts as a fusion barrier, as in the real EP path. Add one case
     without the stand-in that includes `unpermute_from_grouped_gemm` before the combine (the EP=1
     `LocalTokenDispatcher` path), where neighboring gathers can fuse.
   - **Shapes:**
     - `top_k` 8, with hidden dims 2048 (Qwen3-30B-A3B), 3072 (Qwen3.5-122B-A10B) and 4096 (GLM-4.5-Air); verify
       each against the model's HF `config.json`.
     - Expert counts from the same configs, with random top-k routing.
     - Tokens per rank of 2231, 4096, 8192 and 16384. At 16384 x 8 x 4096, A's buffer is about 1 GiB per layer.
   - **What to report:** median, p20 and p80 time, plus peak additional allocated memory (`--memory`), per arm.
   - **Bitwise check:** `bench_ops.py --check` uses tolerances, so do not rely on it for bitwise claims. Check bits
     in the cases file with `torch.equal` on outputs and on input and score gradients.
   - **M against itself:** run N repeated calls and count those that differ from the first. Report the count as an
     observation; zero is a valid result, since atomics often repeat at small sizes.
   - **If B's backward is slow:** inspect the kernels with `bench_ops.py --trace` and `top_kernels.py` before
     anything else.
2. **Logprob parity on real weights** (`profiling/scripts/prime_rl_logprob_ab.py`, usage in its docstring):
   - **What it measures:** only step 1's forward logprobs, at a fixed `--tokens-per-gpu`. It cannot test gradients
     or dynamic shapes; steps 1 and 4 cover those.
   - **Setup:** dump every arm twice, with the same world size and tokens per GPU, compile on, on an MoE model with
     EP > 1 so the Torch all-to-all path runs (`TorchMoEDispatchConfig` is the default whenever `ep > 1`).
   - **Expected:** 0 differing tokens for A vs A, B vs B and A vs B. M vs M may differ; report M vs M and M vs A as
     counts and mean |d|, and repeat the M dumps 3 times if they differ.
   - **Memory:** take the allocated-memory peak from this tool (`peak_alloc_gib`), alongside step time.
3. **End-to-end SFT** (per `prime-rl/e2e.md`):
   - **Workloads:**
     - Qwen3-30B-A3B with `model.debug.force_balanced_routing`, on fake data.
     - GLM-4.5-Air truncated with `model.debug.num_layers`, on `PrimeIntellect/INTELLECT-3-SFT-10K` (renderer
       `glm-4.5`).
   - **Settings:** compile on (the default), `model.ep` explicit, `model.offload = "none"`,
     `model.ac_offloading = None`, no checkpoints, about 20 steps.
   - **Arms:** run M, A and B, and repeat A once with identical code as the noise floor.
   - **What to report:** median `time/step` with min and max over steady-state steps, MFU, throughput per GPU, peak
     memory, and per-step loss and grad norm (`profiling/scripts/run_metrics.py --summarize`). The SFT
     peak-memory metric is `max_memory_reserved`; label it as reserved.
   - **Expected:** A and B match within the noise floor. M may drift slightly from its BF16 accumulation, but
     should not show a gap that grows over steps.
4. **Recompilation and graph breaks under dynamic shapes:**
   - **Where the token count varies:**
     - With real data, the GLM runs vary: `CatDataset` emits a short pack when the next sample would overflow,
       padding only to a multiple of `cp` (`src/prime_rl/trainer/sft/data.py`).
     - Qwen's fake data is fixed-length unless `data.length = "variable"`. Add a Qwen run per arm with that
       setting.
   - **Logging:** launch these runs with `TORCH_LOGS=recompiles,graph_breaks`. Summarize recompiles with
     `profiling/scripts/recompile_reasons.py <trainer log> --rank 0 --step-regex 'Step (\d+) \|'`, and count graph
     breaks per arm from the same log.
   - **Fullgraph run:** also run each arm once with `model.compile.fullgraph = true`. It defaults to false, which
     turns data-dependent-shape errors into silent graph breaks. SFT resolves the router dtype to bf16, which
     fullgraph requires. If M itself fails under fullgraph for unrelated reasons, record that and rely on the
     graph-break counts.
   - **Pass criterion:** neither A nor B has recompiles, graph breaks or fullgraph errors that M lacks.

## Reporting

- **`RESULTS.md`:** write `benchmarks/scripts/moe_combine/RESULTS.md` with the tables, the commands that produced
  them, all three SHAs and the hardware, and commit it. Keep tables to about 120 characters per row.
- **Summary for the user:**
  - Does B match A bitwise, in forward and gradients?
  - What do A and B each cost relative to M, op-level and end to end, in time and allocated memory?
  - Do recompiles or graph breaks differ?
- **PR:** do not push or open a PR without the user's approval. If approved, delete `PLAN.md` in its own commit,
  ask the user whether the benchmark scripts should stay or be deleted, then
  `gh pr create --draft --base fix/deterministic-moe-combine` and link the commit that adds the results.

## Critical files

- `src/prime_rl/trainer/distributed/token_dispatcher.py`: `_local_reorder`, `_combine_routed_output`,
  `LocalDispatchState`, `TorchDispatchState`, `LocalTokenDispatcher`, `TorchTokenDispatcher`, the new
  `_PermuteRows`
- `src/prime_rl/trainer/distributed/collectives.py`: the `_all_to_all_single` fake, to mirror in the stand-in
- `src/prime_rl/trainer/model.py`: `apply_compile` compiles each decoder layer whole, so the combine is traced
- `benchmarks/scripts/moe_combine/` (new): `test_combine.py`, `parity.py`, `compile_check.py`, the bench cases
  file, `RESULTS.md`
