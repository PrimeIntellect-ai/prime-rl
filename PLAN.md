# Status: dist_moe as a prime-rl MoE backend (with torch 2.14 bump)

This file is the handoff from the H200 implementation pass (agent 1) to whoever continues on Blackwell (agent 2). It assumes no access to any prior conversation. Branch: `feat/dist-moe` (parent `main` at `7812237dd`). Nothing on this branch is pushed or reviewed upstream yet.

## Context

meta-pytorch/dist_moe (https://github.com/meta-pytorch/dist_moe, pinned at `caf454c`, v0.1.0 alpha, explainer https://meta-pytorch.org/dist_moe/previews/visual-explainer/) fuses MoE expert-parallel dispatch, the W13 -> SwiGLU -> W2 grouped GEMMs, and combine into communication-compute pipelines over torch symmetric memory (NVLink peer HBM, no NVSHMEM, no InfiniBand). Routing metadata is computed on device with no host sync, tokens are never dropped, and a device-side planner decides per layer whether to save or recompute activations. torchtitan integrated it on 2026-10-03 (torchtitan PRs #4541-#4543, code under `torchtitan/models/common/dist_moe/`). This branch adds an equivalent prime-rl backend in the fused-dispatch slot created by prime-rl PR #3651 (Mega MoE).

### Hardware limitation

- dist_moe runs only on Blackwell (SM100+): its CuTe DSL kernels use tcgen05/TMEM. The prime-rl SLURM cluster where agent 1 worked is H200 (SM90) only, so no dist_moe kernel has ever run on this branch.
- The EP group must sit inside one NVLink domain (one HGX node, EP <= 8, or one NVL72 domain).
- The rebuilt prime-kernels (`flash_moe`, sm_100a/sm_103a) and Mega MoE `mxfp8` are also SM100-only and were only import-tested.

### Version prerequisites

- dist_moe requires `torch>=2.14`, `triton>=3.8,<3.10`, `nvidia-cutlass-dsl[cu13]>=4.6.0.dev0,<=4.8.0`, `cuda-python>=13.1,<14`, `apache-tvm-ffi>=0.1.10,<0.2`. It is pure Python with JIT kernels. Its README says it was validated on CUTLASS DSL 4.6.1, Linux ARM64, cuda-python 13.4.1. Our lock has torch 2.14.0+cu130, triton 3.8.0, cutlass-dsl 4.8.0, cuda-python 13.3.1, apache-tvm-ffi 0.1.11.
- vLLM 0.32.0 requires `torch==2.13.0`, and vLLM 0.33.0 was not released as of 2026-10-21. The branch therefore pins the vLLM nightly `0.32.1rc1.dev224+gf143f65d7` from `https://wheels.vllm.ai/f143f65d7c74b59072ac0f085577aafd75fa7d25/` (commit `315d72672`). This must be replaced by the 0.33.0 release pin once it ships.

## Key design decisions

1. **Slot**: a `MoEDispatchConfig` variant `type="dist_moe"` (`DistMoeMoEDispatchConfig`) plus `DistMoEExpertCompute`, a `FusedDispatchExpertCompute` paired with `FusedTokenDispatcher`, exactly as Mega MoE does. Routing stats, checkpoint/resume, weight broadcast, inference and optimizer special-casing need no change because parameter layout is unchanged.
2. **Compile**: dist_moe is wrapped in opaque custom ops `prime_rl::dist_moe_forward`, `prime_rl::dist_moe_backward` and `prime_rl::dist_moe_eval_forward` (in `src/prime_rl/trainer/models/layers/dist_moe_experts.py`) whose impls call `torch.ops.dist_moe.bf16_forward` / `bf16_backward` eagerly, with `register_autograd` (Mega MoE pattern). The context is reached through an int key into a module-level registry, mapped to `context.context_id` inside the impl.
3. **Activation checkpointing**: `prime_rl::dist_moe_forward` is in `MANDATORY_SAVE_OPERATIONS` (`src/prime_rl/trainer/activation_checkpointing.py`). dist_moe's planner is per-slot LIFO and the inner forward always pushes, so an AC recompute would double-push. This entry is load-bearing: a graph trace on torch 2.14 showed that without it, AC `full` recomputes the forward op in backward (the AOT partitioner does not protect custom ops inside a checkpointed region, contrary to what the original plan assumed).
4. **Weights**: prime-rl's `gate_up_proj [E, 2H, D]` (gate first) and `down_proj [E, D, H]` match dist_moe's W13 and W2. `to_local()` of the EP `Shard(0)` DTensor is enough, so there is no per-step copy. The `gate_up` fusion is required.
5. **Token count**: every EP rank must pass the same `T` or the kernel device-traps. The first pass requires `pad_to_seq_len = True` (RL `orchestrator.pad_to_seq_len`, SFT `data.pad_to_seq_len`). `max_tokens_per_rank` is per rank after CP: RL `seq_len // cp`, SFT `micro_batch_size * seq_len // cp`. RL also requires `trainer.model.seq_len == orchestrator.seq_len`, and SFT requires validation micro batches to match the training size. An explicit `max_tokens_per_rank` that differs from the derived value is rejected. At runtime, a token count mismatch raises `NotImplementedError` behind a `# TODO:` describing the padding follow-up.
6. **Context lifetime**: one `dist_moe.Context` per rank, created in `configure_moe_runtime` (`src/prime_rl/trainer/moe_runtime.py`) before `apply_compile`, shared by all MoE layers, with `max_moe_layers_per_activation_slot = len(moe_layers)`. `DistMoEExpertCompute.destroy()` exists but, like Mega MoE's, is not called at teardown.
7. **FSDP2 and dtypes**: unsharded expert params are bf16 via `MixedPrecisionPolicy`. wgrad is returned functionally in bf16. The compute casts `x` to bf16, ids to int64, scores to fp32 (router scores are bf16 in prime-rl, the fp32 cast follows Mega MoE), and the output back to `x.dtype`.
8. **Models**: only `Silu` activation, no expert biases, gated experts, `score_before_experts=False`, `hidden % 64 == 0`. DeepSeek V4 (different clamped SwiGLU), GPT-OSS (expert biases), Nemotron-H (non-gated) and LoRA experts are rejected. Expected to pass: Qwen3-MoE, Qwen3.5 MoE, GLM-4.x MoE, GLM-5 DSA, MiniMax-M2, Laguna, qwen3_8_flash_next.
9. **Memory outside `plan_memory`**: every MoE layer keeps dist_moe's 9 routing-state tensors through autograd, including three int64 tensors of length up to `num_experts * T`, so about `24 * E * T` bytes per layer (an upper bound). These are plain `torch.empty` allocations, so they do show in caching-allocator peaks, but `plan_memory` does not count them. The estimate is logged at startup next to `plan_memory(...).explain()`.
10. **Overflow detection**: `begin_overflow_check()` runs after the last backward and `finish_overflow_check()` after grad clipping, before `optimizer.step`, in both `rl/train.py` and `sft/train.py`. It raises on overflow (dist_moe raises `ValueError`) and returns `dist_moe/*_used` peaks maxed over the EP group as metrics. These peaks are run-wide high-water marks because nothing calls `context.reset()`.
11. **Precision scope**: bf16 only. MXFP8 is a later follow-up.
12. **`ac_offloading`, LoRA, `compile.mode` in `reduce-overhead`/`max-autotune`** are rejected with `dist_moe` at config time. Offloading is rejected because the saved routing-state tensors may alias dist_moe-owned storage.

## What was done

### Commits by unit

Commits are grouped so each unit can later become its own stacked branch and PR. The last commit (`57c7caf2b`) belongs to unit 1 and was found by the e2e runs after units 2-4 were applied.

| Unit | SHA | Subject |
| --- | --- | --- |
| 1 | `315d72672` | chore(deps): pin vLLM nightly 0.32.1rc1.dev224 for torch 2.14 |
| 1 | `b5b64618e` | chore(deps): move flash-attn and mamba-ssm builds to torch 2.14 |
| 1 | `7c0af09cc` | chore(deps): drop vLLM patches fixed upstream and adapt the rest to the nightly |
| 1 | `d0a017fd9` | chore(deps): pass hash_block_size to AuxOutputSchedulerConnector in the P/D test |
| 1 | `57c7caf2b` | chore(deps): import vLLM serve CLI args from their new module |
| 2 | `ebea79466` | feat(deps): add dist_moe as an optional extra |
| 3 | `058e9438b` | refactor(moe): move reserve_sms_for_comm to expert_compute |
| 3 | `4394459b8` | feat(moe): add dist_moe dispatch config and validators |
| 3 | `923661fed` | feat(moe): add the dist_moe fused dispatch backend |
| 3 | `eb0775067` | docs(moe): document the dist_moe dispatch backend |
| 3 | `ebfcf5197` | fix(moe): log only the used dist_moe scratch peaks |
| 4 | `efae027bc` | feat(bench): add a dist_moe parity and timing benchmark |

### Unit 1 details

- Pins: torch 2.14.0 override, torchvision 0.29.0 and triton 3.8.0 via vLLM, flashinfer-python/cubin/jit-cache 0.7.2, `nvidia-cudnn-frontend>=1.31.0`, flash-attn 2.8.3 cu130torch2.14 wheels (mjun0812 v0.10.0 x86_64, v0.10.1 aarch64), mamba-ssm built against torch 2.14.
- vLLM patches: vllm#59709 (DeepSeek V4 C128 boundary) and vllm#57797 (DeepSeek V4 attn_sink loading) are in the nightly, so their patches were removed. The GPT-OSS patch was adapted to vllm#57797's weight-loader change, and the Qwen3.8 patch to the removal of `compute_ple_shard_overlap`. The vllm#62406 MXFP8 reload patch is kept. The copied `LoRAModel.from_local_checkpoint` is unchanged upstream.
- vLLM's binary extension is now `vllm._C_stable_libtorch` (there is no `vllm._C`).
- deep-ep, deep-gemm, prime-kernels and flash-mla still point at their torch 2.13 release wheels in `pyproject.toml`, which are ABI-incompatible with torch 2.14. See "What remains".

### Units 2-4 details

- Unit 2 added only the `dist-moe` package to `uv.lock`. No other version moved. `dist-moe` is not in the `all` extra.
- Unit 3 deviations from the original plan: `reserve_sms_for_comm` moved to `layers/expert_compute.py` (not `moe_runtime.py`) to avoid a circular import. The backward drops index 6 (`gain_center`) of `_registered_rmsnorm_args(None)`, not index 5 as the plan said (index 5 is `require_bitwise`). The single-node EP check uses `rank // LOCAL_WORLD_SIZE`, which assumes torchrun's contiguous rank layout. `offload = "full"` is not rejected, so with it the overflow check fires after the in-memory update but still before any checkpoint save.
- Unit 4 times forward and backward inside each fwd+bwd iteration, because a grad-enabled forward without its backward would leave planner state unpopped. Parity is printed from rank 0 only, as in `bench_mega_moe.py`.

### H200 results

The full run artifacts (configs, run dirs, logs, summary README) are on the H200 cluster under `~/tmp/dist-moe-e2e/`, and are not needed to continue.

- Unit tests at the combined branch: 1094 passed, 2 skipped, 5 failed. The 5 failures are `tests/unit/train/models/test_fp8_utils.py::test_token_cast_matches_vllm_cuda_quant[*]`: vllm#61343 changed vLLM's per-token-group FP8 quant from `x / s` to `x * (1/s)`, so our `per_token_cast_to_fp8_triton` no longer matches bit for bit (34 of 65536 bytes differ at 256x256, scales identical). Not fixed, because the fix changes trainer FP8 numerics and needs a decision from the maintainer.
- Integration tests on the bump: all pass except `test_dashboard`, which needs root to install playwright on that cluster.
- Selecting `dist_moe` on SM90 fails with `RuntimeError: dist_moe dispatch requires an SM100+ GPU, but this device is SM90.` `import dist_moe` works and `plan_memory` runs host-side.
- Graph-structure check on SM90 with a fake process group and stub kernels (a tiny 2-layer Qwen3-MoE through the real `configure_moe_runtime`, `apply_ac` and per-layer compile): for AC none, full and selective, with and without fullgraph, and under inductor, each forward graph has exactly one `dist_moe_forward`, `dist_moe_backward` appears only in backward, there are no graph breaks, and the forward op runs once per layer per step. Inductor marks it `MUST_SAVE`.

Trainer A/B, main `7812237dd` vs bump, 20 steps, step time in seconds (lower is better), median over steps 2-20 (after the last recompile):

| Workload | main | bump | Peak memory (both) |
| --- | --- | --- | --- |
| SFT Qwen3-30B-A3B, fake data, balanced routing, ep 8 | 5.933 s | 5.934 s | 76.71 GiB |
| SFT GLM-4.5-Air 8 layers, real data, ep 8 | 2.994 s | 3.009 s | 55.55 GiB |
| SFT GLM-4.5-Air 8 layers, cp 2, fullgraph off | 3.378 s | 3.386 s | 52.77 GiB |

Same-code reruns reproduce each number to within 0.01 s. Loss differences between arms are within same-code rerun noise and step-1 losses are bitwise identical. Recompile counts are identical between arms (0 per rank for Qwen, 3 per rank in step 1 for GLM with and without cp 2).

RL A/B on Qwen3-30B-A3B, 4 trainer and 4 inference GPUs, 20 steps, bump at `57c7caf2b`:

| Arm | Mismatch KL median (range) | Step time median, steps 3-20 | Peak memory |
| --- | --- | --- | --- |
| main | 8.88e-4 (7.95e-4 to 1.14e-3) | 55.5 s | 120.0 GiB |
| bump | 9.15e-4 (8.08e-4 to 1.05e-3) | 57.2 s | 119.9 GiB |

The KL ranges overlap almost entirely. The 3% step-time gap is within the noise of rollout-dominated RL steps on different nodes.

Smokes on the bump: a DeepSeek V4 SFT trainer (4 layers, real weights, default `cudnn_flashmla`) and a GLM-5.3 SFT trainer (6 layers, step-10 loss 6.0079 vs main 6.0077) pass. DeepSeek V4 RL (4 layers, real weights, flash-mla) passes 10 steps with mismatch KL median 1.71e-3. It needed `[orchestrator.train.algo] type = "debug"` because the truncated model scores 0 on every rollout and all groups are otherwise dropped.

### dist_moe memory sizing (from `plan_memory`, ep 8)

"Default slot" is the all-recompute minimum (each layer's saved input). "Full save" is `activation_slot_capacity_factor = 1.0`. "Routing state" is decision 9's estimate, which `plan_memory` does not count.

| Model, tokens per rank | Default slot / scratch / total | Full-save total | Routing state |
| --- | --- | --- | --- |
| Qwen3-30B-A3B, 8192 | 1.5 / 0.97 / 2.47 GiB | 33.97 GiB | 1.21 GB |
| Qwen3-30B-A3B, 16384 | 3.0 / 1.94 / 4.94 GiB | 67.94 GiB | 2.42 GB |
| Qwen3-30B-A3B, 65536 | 12.0 / 7.75 / 19.75 GiB | 271.75 GiB | 9.66 GB |
| GLM-4.5-Air, 16384 | 5.625 / 3.72 / 9.34 GiB | 124.66 GiB | 2.26 GB |

Full save is about 22x the default slot, so real runs need an `activation_slot_capacity_factor` well below 1.

## What remains

### External prerequisites (not blocking dist_moe work on Blackwell, but blocking a merge)

- vLLM 0.33.0 release, to replace the nightly pin in `315d72672`. Re-check every vLLM patch in `src/prime_rl/inference/patches.py` and the `vllm.entrypoints` imports against it.
- A prime-kernels release built with `TORCH_VERSION: 2.14.0+cu130` (`deps/prime-kernels/.github/workflows/build_kernels.yaml`) for prime-kernels, deep-ep and deep-gemm, then the matching submodule and `pyproject.toml` bumps. DeepGEMM's `setup.py` hardcodes `-std=c++17`, and torch 2.14 ATen headers require C++20, so the CI recipe needs that change.
- A torch 2.14 flash-mla wheel (the current one is NovaSky skyrl-wheels `cu13torch2.13-flashmla-b7643bd`). It is needed by the default `dsa_backend = "cudnn_flashmla"` for GLM-5 and DeepSeek V4.
- A torch 2.14 rebuild of the closed-source prime-mega-moe `deep_gemm`, or Mega MoE breaks.
- `skills/kernels/SKILL.md` references `cu130torch2.13.0` and should say `cu130torch2.14.0` (and could note the C++20 requirement). Skills are only edited with maintainer approval.
- Decide whether to follow vLLM's `x * (1/s)` FP8 quant in `src/prime_rl/trainer/models/kernels/fp8_utils.py` (the 5 failing unit tests).

### Blackwell work (unit 5 and everything below)

1. dist_moe's own GPU tests (`pytest tests/test_dist_moe.py` in the dist_moe repo) in our resolved environment. Gate on them passing with CUTLASS DSL 4.8.0 on x86_64, and repeat on every later cutlass-dsl bump (our `nvidia-cutlass-dsl==4.8.0` override hides dist_moe's `<=4.8.0` cap).
2. `benchmarks/dist_moe/bench_dist_moe.py` parity and timing, force-balanced and random routing.
3. Verify the dist_moe API points that were matched by reading source only: the 18-argument `bf16_forward` and 15-argument `bf16_backward` calls, the private `activation_buffer.activation_slot_id_1` and `_num_moe_layers_in_selected_slot` fields, the private `_fake_registered_bf16_forward` fake, and that SAC, `register_autograd` and Dynamo accept the `(Tensor, list[Tensor])` forward output with real kernels.
4. Short SFT matrix {AC none, full, selective} x {compile off, default, fullgraph} (ac_offloading off), loss vs torch dispatch, tlparse inspection, FSDP prefetch hang check, overflow-check hooks firing.
5. e2e A/B at ep 8 on Qwen3-30B-A3B and GLM-4.5-Air vs `torch` and `deepep` dispatch. Report peak memory including the routing-state estimate.
6. Memory check at real RL `seq_len // cp`.
7. Real-routing imbalance stress to tune `scratch_capacity_factor` from the logged `dist_moe/*_used` peaks, and exercise VMM (`vmm_capacity_factor`).
8. A `model.cp = 2` arm on GLM-4.5-Air real data.
9. Hang check under concurrent FSDP NCCL with and without `num_reserved_sms` plus `NCCL_MAX_CTAS`.
10. Unit 5: commit the A/B harness (configs, drivers, results) under `benchmarks/scripts/dist_moe/` as `chore(bench)` commits that stay in history.

### Follow-ups after the first pass

- In-kernel padding to lift the `pad_to_seq_len` requirement (the `# TODO:` in `DistMoEExpertCompute.__call__`): pad to `max_tokens_per_rank` with zero rows, score 0 and expert id `-1`, then slice the output back (torchtitan does this in `routed_experts.py`).
- Lift the `ac_offloading` reject once the aliasing of the saved routing-state tensors is checked.
- An overflow on one rank raises before the EP all-reduce in `finish_overflow_check`, so its peers hang until torchrun tears the job down. All-reducing an overflow flag first would make every rank raise.
- Upstream request to dist_moe for a `+1`-free, alpha 1 clamped SwiGLU to unlock DeepSeek V4.
- Whether `dist-moe` joins the `all` extra, and whether units 2 and 3 become one PR.

## How to start on Blackwell

1. Clone and check out `feat/dist-moe`, then `git submodule update --init --recursive`.
2. `uv sync --all-extras --all-packages` (dist_moe is in the `dist-moe` extra, which `--all-extras` includes). Then replace the four torch-2.13 wheels (deep-ep, deep-gemm, prime-kernels, flash-mla) with torch 2.14 builds, either from published releases if they exist by then or built locally: prime-kernels from `deps/prime-kernels` with `version = "0.1.0+cu130torch2.14.0"` and `PRIME_KERNELS_REQUIRE=1 uv build --wheel --no-build-isolation`, deep-ep with `deps/prime-kernels/scripts/install_ep_kernels.sh` (the build host needs the rdma-core headers, for example `infiniband/mlx5dv.h` from `libibverbs-dev` and `librdmacm-dev`, because the NVSHMEM headers include them), deep-gemm with `deps/prime-kernels/scripts/install_deep_gemm.sh` plus `-std=c++20` in its `setup.py`, flash-mla from deepseek-ai/FlashMLA `nv_dev` at `b7643bd`. Install them with `uv pip install --no-deps <wheels>` and use `uv run --no-sync` afterwards, because `uv sync` reinstalls the pinned torch 2.13 wheels. On aarch64 (for example GB200), skip flash-mla: the `gpu` extra only requests it on x86_64. Without a torch 2.14 prime-kernels, the trainer does not import (`trainer/distributed/collectives.py` imports it unconditionally).
3. Smoke: `uv run --no-sync python -c "import torch, dist_moe, vllm._C_stable_libtorch, prime_kernels, deep_ep, deep_gemm, flash_mla"` (import `torch` first).
4. Run dist_moe's own tests (item 1 above). They live in the dist_moe repo, not in the installed package: clone https://github.com/meta-pytorch/dist_moe at `caf454c` and run `uv run --no-sync pytest -q <clone>/tests/test_dist_moe.py` from the prime-rl worktree so they use our resolved environment (they need at least two Blackwell GPUs, and the VMM tests are in `tests/test_dist_moe_vmm.py`). Then `NCCL_MAX_CTAS=16 torchrun --nproc_per_node=8 benchmarks/dist_moe/bench_dist_moe.py --num-experts 128 --top-k 8 --hidden 2048 --intermediate 768 --tokens-per-rank 8192`. It runs a force-balanced and a random routing case. Then run the SFT matrix.
5. To enable the backend in an RL config: `[trainer.model.moe.dispatch] type = "dist_moe"`, `trainer.model.ac_offloading = "None"`, an explicit `trainer.model.ep` (at most 8 within one node), and `orchestrator.pad_to_seq_len = true` (SFT: `data.pad_to_seq_len = true`). The dist_moe part of the "Expert Parallelism" section in `docs/scaling.md` has a full example.
