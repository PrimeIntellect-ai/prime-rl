# Restore torch.compile for MoE blocks with an fp32 router: implement two variants, then benchmark

Repo: prime-rl. All paths below are relative to the repo root. Target: PyTorch 2.13 (`torch>=2.13.0` in
`pyproject.toml`). The work happens on a GPU machine. Follow `AGENTS.md`, use `uv run` for everything, and never
`git push` without the user's explicit approval each time.

## Context: why this work exists

### What is broken
- **Where the fp32 router unit comes from.** With `model.moe_router_dtype="float32"`, the MoE router becomes its
  own FSDP2 unit nested inside each transformer block's unit. This is the RL default: `"auto"` resolves to
  `float32` for RL and to `bfloat16` for SFT. The code is `setup_fsdp` in `src/prime_rl/trainer/model.py`, around
  line 593:

  ```python
  fully_shard(block_mlp.router, mp_policy=MixedPrecisionPolicy(param_dtype=fp32, reduce_dtype=fp32), ...)
  ```

  The unit is needed because FSDP2's `MixedPrecisionPolicy` is per unit. In PyTorch 2.13 one unit cannot mix
  parameter dtypes: `_init_mp_dtypes` asserts uniform dtypes.
- **Why that blocks compile.** FSDP2 wraps all of its hooks in `torch._dynamo.disable`
  (`torch/distributed/fsdp/_fully_shard/_fsdp_common.py:25-34` in 2.13). Traceable FSDP2 was removed upstream. So
  the router's FSDP pre-forward hook is always a Dynamo graph break.
- **Why the whole block goes eager.** The current order is compile(AC(block)): `setup_model` (model.py ~998)
  runs `apply_ac`, then `apply_compile`, then `setup_fsdp`.
  - `apply_ac` (~815) wraps every `ac.freq`-th block in `checkpoint_wrapper` (NO_REENTRANT, with a selective-AC
    `context_fn`; see `src/prime_rl/trainer/activation_checkpointing.py:119-130`).
  - `apply_compile` (~832) calls `.compile(fullgraph=..., mode=...)` in place on each `layers[i]`, i.e. on the
    `CheckpointWrapper`.
  - Under compile, the checkpoint becomes the `tag_activation_checkpoint` higher-order op. Graph breaks inside a
    higher-order op body are forbidden: `torch/_dynamo/symbolic_convert.py:3795`, "Do not allow nested graph breaks
    in HOPs".
  - The checkpoint op keeps the default fallback-to-eager behavior. `torch.utils.checkpoint.checkpoint` is
    `@torch._disable_dynamo` (recursive; `torch/utils/checkpoint.py:354`).
  - Net effect: when the router hook breaks the graph, the whole MoE block runs uncompiled. PyTorch's
    `test/dynamo/test_activation_checkpointing.py::test_fallback` demonstrates this. With `fullgraph=True` it
    raises instead, which is why `setup_fsdp` already raises a `ValueError` for fullgraph plus an fp32 router.
- **Precedent.** Commit `d4a11f237` (#3753) fixed the same problem for EP experts by keeping them in the block's
  unit through `_expert_shard_placement_fn`. It reported a large speedup: Qwen3-235B SFT went from 6.98 to 4.40
  s/step. The fp32 router unit was left as is.

### Goal
Compile the code on both sides of the router's graph break with `fullgraph=False`, while keeping the router as an
fp32 FSDP unit. Fullgraph with an fp32 router stays unsupported, and the `ValueError` stays.

### Long-term fix (not this work)
PyTorch #196626 adds per-parameter mixed precision, `MixedPrecisionPolicy(param_dtype_override_fn=...)`. It landed
on main on 2026-10-02 and is not in 2.13 or 2.14.1. It will let the router stay inside the block unit, removing the
nested unit altogether. Whichever variant wins here is a stopgap until then.

### Why two variants
It isn't clear in advance which is better, so implement both and measure.
- **Variant 1** keeps block-level AC but makes it eager, so the compiler no longer owns recompute.
- **Variant 2** keeps AC traced by the compiler, but over smaller regions, which leaves some activations
  uncheckpointed.

## Variant 1: AC(compile(block)), reversed order

### Change (`src/prime_rl/trainer/model.py` only)
- `apply_compile` gets a flag from `setup_model`: compile inside AC when `config.moe_router_dtype == "float32"`.
  The value is already resolved from `"auto"` there; `setup_fsdp` keys on the same check.
- For each layer: if the flag is set, the layer is a `CheckpointWrapper`, and the inner block's MLP is a `MoE`
  (`prime_rl.trainer.models.layers.moe.MoE`, found with the same `getattr(block, "mlp", None)` lookup `setup_fsdp`
  uses at ~581), call `layer._checkpoint_wrapped_module.compile(fullgraph=..., mode=...)`. Otherwise keep
  `layer.compile(...)`. Layers skipped by `ac.freq > 1` are not wrapped and compile as today.
- Update the `# the right order is AC -> Compile -> FSDP` comment in `setup_model` and the `apply_compile` log line
  to reflect the exception.

### Why it should work (from reading PyTorch 2.13; not yet run)
- **Compiled code still runs compiled inside the eager checkpoint.** A compiled function installs its own
  eval-frame callback when called, even inside a `_disable_dynamo` region (`torch/_dynamo/eval_frame.py:1084,1159`).
- **Where the graphs split.** Dynamo traces `block.forward` as an ordinary frame. With the default
  `nested_graph_breaks=False`, the router hook's break moves up to the `self.mlp(x)` call site. `mlp.forward` then
  compiles as its own frame and breaks again at `self.router(...)`. Result: several graphs per block instead of an
  eager block.
- **Selective AC semantics are preserved under default Inductor config** (`wrap_inductor_compiled_regions=False`).
  The SAC dispatch mode still sees the ops that compiled code dispatches: fallback ops, custom ops, and extern GEMMs
  as `.out` overloads.
  - `aten.topk` is an Inductor fallback, unless `TORCHINDUCTOR_DECOMPOSE_SORT_OPS=1`, so routing stays MUST_SAVE.
  - DeepEP runs eagerly, and `prime_rl::record_moe_routing_statistics` is a custom op. Both stay saved.
- **Expected costs:**
  - Recompute reruns the whole compiled forward, with no compiler pruning or fusion of recompute.
  - The per-op Python SAC dispatch overhead returns for the dispatched ops (#3753 credited its removal for part of
    its win).
  - In selective mode, compiled matmuls show up as `aten.mm.out`, which doesn't match the `aten::mm` save target,
    so they get recomputed.

## Variant 2: compile(AC(region)) with regions that exclude the router

### Shape
For MoE blocks with an fp32 router, remove the block-level checkpoint and checkpoint two submodule regions
instead:
1. the block's attention module, i.e. `self_attn` and/or `linear_attn` (whichever exists; see the list below),
2. the post-router MoE work.

The block is then compiled as today with `fullgraph=False`. Dynamo traces each checkpoint as a higher-order op,
graph-breaks at the router hook (which is outside both regions), and compiles the second checkpoint in the resume
function. Norms, residual adds, and the router itself stay compiled but are not checkpointed.

### Changes
- **`src/prime_rl/trainer/models/layers/moe.py`.** Split `MoE.forward` (~357-416) into two methods:
  - `route(x, routed_experts)` covers the reshape, the router call, and `record_moe_routing_statistics`.
  - `compute(...)` covers `token_dispatcher.run`, the shared expert, `token_dispatcher.synchronize()`,
    `prepare_expert_output`, the add, and the reshape.
  - `forward` becomes `route` then `compute`, so behavior is unchanged for everyone else.
  - Nothing in `MoE.forward` runs before the router call today.
  - Subclasses: `DeepseekV4MoE.forward` (`src/prime_rl/trainer/models/deepseek_v4/moe.py:99-120`) does a
    `tid2eid[input_ids]` hash-routing lookup before `super().forward`; adapt it so the split still applies.
  - `NemotronHMoE` and `SigmoidOutputGatedMoE` override `prepare_expert_input/output` or other hooks; check that
    they still compose.
- **`apply_ac` in model.py.** Add a branch for MoE blocks when `moe_router_dtype == "float32"`.
  - Wrap `block.self_attn` / `block.linear_attn` with the same wrapper `get_activation_checkpoint_wrapper(ac_config)`
    returns, via `register_module`.
  - Make `mlp.compute` run under `torch.utils.checkpoint.checkpoint(..., use_reentrant=False,
    context_fn=<same SAC context_fn>)`.
  - Reference for method-level checkpointing, removed in #3419: `git show
    866c3e143^:src/prime_rl/trainer/models/layers/checkpointing.py`.
  - Refactor `get_activation_checkpoint_wrapper` minimally so the policy and `context_fn` can be reused.
  - Keep `ac.freq` semantics.
- **Attention attribute names:** `self_attn` in qwen3_moe, glm4_moe, glm_moe_dsa, minimax_m2, laguna, afmoe,
  gpt_oss, deepseek_v4 and nemotron_h (attention layers only). qwen3_5 and qwen3_8_flash_next use `self_attn` or
  `linear_attn` per layer. Some nemotron_h MoE layers have no attention, so wrap nothing there.

### Things to verify
- **Keys and attribute lookups.** `setup_fsdp`'s `getattr(transformer_block, "mlp")` and the EP prefetch wiring
  (model.py ~657-707, which references `block.mlp.router`) must still find modules once the block is no longer
  wrapped.
- **`_checkpoint_wrapped_module` inside submodule FQNs.** Check that state-dict keys still resolve for HF weight
  loading, DCP checkpoints, and weight broadcast to inference. Today the prefix only appears at the block level.
  Grep for `_checkpoint_wrapped_module` handling.
- **Backward ordering.** The router is no longer recomputed in backward, so its FSDP unshard/reduce order in
  backward changes. #3419 notes the router is prefetched "so FSDP collectives occur in the same order as the saved
  forward". Check the EP backward prefetch list.
- **Expected costs.** More activation memory: norm inputs, residual stream, and the router's fp32 `x.float()`
  `(T, dim)` plus `(T, E)` scores stay resident per MoE layer. Possibly different DeepEP overlap.

## Commits
- Branch from `main`.
- Commit variant 1 and variant 2 separately: variant 1 first, variant 2 as a second commit that reverts variant 1's
  `apply_compile` change if needed. Each arm is then a frozen commit.
- Use conventional commits with a scope, e.g. `perf(trainer): ...`.
- Don't add new test files unless a change clearly needs one (`AGENTS.md`). Make sure existing tests pass:
  - `uv run pytest tests/unit/train/models/test_checkpointing.py tests/unit/train/models/test_moe.py`
  - `uv run pytest tests/unit -m "not gpu"`
- `tests/unit/train/models/test_checkpointing.py::test_checkpoint_records_moe_routing_once` (a tiny `MoE` under
  both AC modes) is the natural template if a compile-inside-checkpoint check turns out to be needed.

## Validation before any timing
1. **Graph breaks.** On 2 GPUs, run with `TORCH_LOGS=graph_breaks,recompiles`, compile and AC on, and
   `--model.moe-router-dtype float32`:

   ```
   uv run sft @ configs/debug/fake/sft.toml --model.name samsja/mini-glm-moe --model.compile --model.ac \
     --model.moe-router-dtype float32 --max-steps 5
   ```

   - Use torchrun with 2+ GPUs so the nested FSDP unit is real. Add `--model.ep 2` for the EP path.
   - Confirm main shows the checkpoint fallback, and that each variant compiles graphs on both sides of the router
     with no eager fallback of a checkpoint region.
2. **Numerics.** Same seed, ~20 steps, arms main / variant 1 / variant 2. Loss and grad-norm should track step by
   step. There should be no `CheckpointError`; recomputed routing diverging from the forward would raise one.
3. **Full and selective.** Repeat for `ac.mode=full` and `ac.mode=selective`, and with EP on and off.
4. **What the AC policy sees (variant 1).**
   - **Why.** Under AC(compile(block)) the selective-AC policy runs at runtime and only sees ops that compiled code
     still dispatches. Its rules match on `op.name()`, which includes the overload suffix. Two consequences:
     - Correctness: `aten.topk`, `prime_rl::record_moe_routing_statistics`, and the `deepep` ops must still reach
       the policy and come back MUST_SAVE. Otherwise routing is recomputed, which can diverge from the forward or
       double-count the routing statistics.
     - Performance: Inductor's external GEMMs dispatch as `.out` overloads (`aten.mm.out`, `aten.addmm.out`,
       `aten.bmm.out`). These don't match selective targets like `aten::mm`, so they are recomputed instead of saved.
   - **How.** Add temporary, uncommitted instrumentation: wrap the policy built by
     `get_activation_checkpoint_wrapper` in `src/prime_rl/trainer/activation_checkpointing.py` so it counts
     `(op.name(), decision)` per rank and logs the counts at the end of a step. Run the 2-GPU smoke command from
     step 1 for 3 steps, for main and variant 1, in both `ac.mode=full` and `ac.mode=selective`, with EP on and
     off.
   - **Pass criteria.** In every variant 1 run, `aten::topk`, `prime_rl::record_moe_routing_statistics` and the
     `deepep::*` ops (when EP uses DeepEP) appear with MUST_SAVE, with call counts consistent with main. If any is
     missing, stop and report it before benchmarking.
   - **Record for the report.** Which `.out` GEMM overloads appear and how often, and how the selective-mode
     saved and recomputed sets differ from main.
   - **Possible follow-up fix (only if the selective-mode cost turns out significant).** Match selective targets
     by `op._schema.name`, the op name without the overload suffix, but only for non-`.out` overloads. Returning
     a cached tensor for an `.out` op skips the write into its `out` buffer, so `.out` ops must never be saved
     this way. Propose this change to the user rather than committing it unprompted.

## Benchmark and profiling
Use the user's `profiling` skill: read `SKILL.md` and `projects/prime-rl.md`. Key rules:
- **Hygiene:** freeze each arm in its own worktree at its commit, with fresh compile caches per arm
  (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TILELANG_CACHE_DIR` per run, outside the run directory).
- **Timing:** median `time/step` over steady-state steps of untraced runs, from `<run dir>/monitors/file/metrics.jsonl`.
- **Attribution:** traces only, via `--trace-path ~/tmp/profiling/<investigation>/traces/<run>` with
  `--max-steps < 10`. Analyze the last step with the skill's scripts (`trim_last_step.py`, `top_kernels.py`).

Arms:
- **A0:** main, fp32 router (the eager MoE-block baseline).
- **A1:** variant 1.
- **A2:** variant 2.
- **R:** main with `--model.moe-router-dtype bfloat16` and `--model.compile.fullgraph`. This is the upper bound
  only; its numerics differ.

Workloads:
- **Proxy first.** Qwen3-30B-A3B, one node, SFT with `--model.moe-router-dtype float32`, fake data,
  `--model.debug.force-balanced-routing`. Both AC modes, and EP on and off.
- **Then the #3753 setup**, if the proxy shows a winner: Qwen3-235B-A22B-Thinking-2507 SFT, 4 nodes x 8 GPUs, ep=8,
  seq 16384, full AC, 20 steps, steady state = steps 5 to 20.
- **Optional harness:** `benchmarks/scripts/run_single_benchmark.py` computes mean MFU, throughput, step time and
  peak memory. It has no passthrough for `moe_router_dtype`, so the flag would have to be appended in
  `build_command` locally; don't commit that.

Report:
- A results table: median step time with min/max, MFU, tokens/s/GPU, and peak memory per arm and workload, with the
  exact commits and configs.
- A trace-based explanation of where A1 and A2 differ: recompute cost, SAC dispatch overhead, memory.
- Use the profiling skill's figure conventions where the mechanism is visual.
- Write it to `~/tmp/profiling/<investigation>/NOTES.md`, then recommend one variant to keep. Don't open a PR until
  the user decides.
