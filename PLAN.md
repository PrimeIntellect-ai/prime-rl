# Handoff: measure and finish `fix/fla-compile-custom-ops`

You are picking up a prime-rl branch on a machine with H200 nodes. You have no access to the conversation that produced it; everything you need is here, in the branch history, and in `benchmarks/scripts/fla-compile-ab/README.md`. Read `AGENTS.md` and the user's global instructions first. In particular: no `git push` without the user's explicit approval each time, no code comments without asking, no em or en dashes in any file, and nothing private in public text. prime-rl is a public repo, so commits, PR text and benchmark files never name internal models, run names, hostnames, local paths or private links.

## What the branch does

GLM5-next (`src/prime_rl/trainer/models/glm5_next/`), Qwen3.5 (`src/prime_rl/trainer/models/layers/gated_delta_net.py`) and Qwen3.8 flash next (same gated delta net layer) never compiled their decoder layers. Under activation checkpointing, `apply_compile` in `src/prime_rl/trainer/model.py` compiles each checkpointed decoder layer. FLA's causal conv1d, `chunk_gated_delta_rule`, `chunk_kda` and gated RMSNorm are `torch.compiler.disable`d (FLA 0.6.0), and a graph break inside a checkpointed block makes Dynamo run the whole layer eagerly.

The branch wraps those ops as `prime_rl::` custom ops in `src/prime_rl/trainer/models/layers/fla_ops.py`, each with its backward as a second custom op calling FLA's internal fwd and bwd functions. It then fixes what compiling the layers exposed. Commits, oldest first, on main `fc5556cd4`:

- `2692743ed` passes the Qwen3.5 and Qwen3.8 decoder-layer args by keyword and marks their seq dims dynamic (`DYNAMIC_TENSOR_ARG_DIMS` in `trainer/model.py`). The user wants this as its own PR. Split it out with `git tree split` before review, as the parent of this branch, after asking the user.
- `1ef73f6cb` adds the custom ops and swaps the non-CP call sites. The CP branches are unchanged; a CP follow-up PR is undecided, so don't start it.
- `2176c4ab1` adds a GPU parity test (each op bitwise equal to FLA's eager function); `19601b5b8` removes it. Link `2176c4ab1` from the PR description.
- `0947dbbb7`, `3b7025941`, `4ee1b4306`, `bafed317a` fix compile-only bugs:
  - Inductor's no-op removal copied an activation-checkpoint tag onto `pool_start_tokens + 0` (AOT partition failure).
  - `F.pad(value=<symbolic token count>)` pinned the count and recompiled every row length (Qwen3.8 step 19 s vs 1.8 s).
  - GLM5-next's `PoolLayout` sizes were static.
- `5d4005966` builds FLA chunk indices once per packed row outside the layers. This removed about 6 host syncs per layer.
- `988e5a8d2` adds comments and renames `FLA_CHUNK_SIZE`.
- `ab6c5ff81` adds the A/B harness `benchmarks/scripts/fla-compile-ab/`.

A commit that made compiled checkpointed layers skip the activation-offload hooks was deliberately dropped. The user trains with activation checkpointing ON and offloading OFF, so offload hooks are not in play.

## What is known (one 8x B200 node, single runs, so treat them as directional)

All earlier runs used the trainer defaults: full activation checkpointing plus activation offloading and optimizer offloading. That is NOT the user's setup, so these numbers are not final:

- **Graph breaks:** the fix arms have zero graph breaks in the layer frame and one compile per layer type, with no recompiles on sequence length. Base runs the layers eagerly.
- **GLM5-next at seq_len 8192:** steady step 3.27 s base vs 3.43 s fix with inputs offloaded, and 3.03 s with them kept on GPU. Peak memory 98.6 GiB base, 103.2 GiB fix with inputs offloaded, 104.4 GiB fix with inputs on GPU.
- **Qwen3.5 at 16384:** peak 26.0 GiB in both arms; steady steps too noisy to compare.
- **Step 1:** grows by the added compile time, 20 to 60 s.
- **Numerics:** compiled versus eager, loss agrees to 4 to 6 significant digits and grad norm to about 3. With compile off, the fix arm was bitwise equal to main.
- **Profiling ceiling:** kernel-time savings from compiling these layers is about 3.5% of step time (hard bound 5 to 6%). Host launches fall by about 30%.

**Open question.** With activation checkpointing on and no offloading, GLM5-next's compiled layers are expected to need about 5.8 GiB more peak memory than eager:
- about 1.2 GiB is the layer inputs staying on GPU;
- about 4.6 GiB is unexplained. Suspects are the compiled backward saving tensors that eager recomputes, custom-op intermediates, and Inductor buffer reuse.

The user relies on activation checkpointing for memory, so this is the main thing to nail down.

## Next steps

1. **Setup.** Use this machine's prime-rl venv (or `uv sync --all-extras` in this worktree, inside an allocation, never on a login node). Then run `benchmarks/scripts/fla-compile-ab/stage_models.sh MODELS_DIR` (config and tokenizer files only, no weights) and `python benchmarks/scripts/fla-compile-ab/make_vlm_data.py DATA_DIR/vlm-synth`. Model weights aren't needed (random init). Ask the user before allocating nodes.
2. **A/B, one 8-GPU H200 node.** Use base = `fc5556cd4` and fix = `988e5a8d2` (code head; the harness commit only adds scripts), one run per arm, 8 steps: `run.sh MODEL base fc5556cd4` and `run.sh MODEL fix 988e5a8d2` for MODEL in `glm5_next`, `glm5_next_8k`, `qwen3_5`, `qwen3_8_flash_next`. `configs/common.toml` already sets activation checkpointing on (default full) and `ac_offloading`/`offload` off. Report per arm:
   - graph breaks and compiles (via `analyze.py`);
   - compile seconds;
   - step-1 time;
   - steady step (median of steps 4 to 8);
   - per-step peak memory, max over ranks (`peaks/rank*.jsonl`);
   - loss and grad norm for steps 1 to 8.

   If H200 memory (141 GiB) is short for `glm5_next_8k`, say so rather than changing the config silently.
3. **Memory.** If fix peaks above base, record a rank-0 CUDA memory history over exactly one step after compile (step 3), for base and fix of `glm5_next_8k`, and diff them by allocation site. The user's profiling skill has a hook (`scripts/memory_history/` with `MEMORY_HISTORY_START_CALL=3`, `MEMORY_HISTORY_STOP_CALL=4`) and `scripts/memory_peak.py --diff`. Explain the extra memory. If it is an implementation artifact (for example the partitioner saving activations the checkpoint should recompute), fix it on this branch and measure again. Before calling anything a limit, try two genuinely different designs.
4. **PR.** Rewrite the PR description (the draft PR for this branch) with the H200 table, the memory explanation, and links to `2176c4ab1` (parity test) and `ab6c5ff81` (harness). Keep it to one screen. Then remove `benchmarks/scripts/fla-compile-ab/` and this `PLAN.md` in final commits. Ask the user before every push.

Out of scope unless the user asks: the CP path (`build_cp_context` and the CP conv in the same two layer files), changing activation checkpointing defaults, and the dropped offload-hook commit.
