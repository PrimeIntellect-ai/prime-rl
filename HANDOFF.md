# Handoff: taking over PR #4109 (online-MXFP8 MoE kernel across weight reloads)

This file carries findings from a GPU-less machine to a B300 machine. Remove it with a removal commit (not a history rewrite) before the follow-up PR into samsja's branch is marked ready.

## The PR

- URL: https://github.com/PrimeIntellect-ai/prime-rl/pull/4109
- Title: "fix(inference): keep online-MXFP8 MoE kernel across weight reloads"
- Author: samsja. Draft PR against `main`.
- Branch: `fix/mxfp8-moe-reload-cudagraph`, in the main `PrimeIntellect-ai/prime-rl` repo.
- Contents (2026-10-16): one commit, `d8d453117`, adding 41 lines to `src/prime_rl/inference/patches.py`. Its base is `d61df5690`, 2 commits behind `main` at the time.

Garrett is taking the PR over, at least temporarily. We must not push to samsja's branch, so our work goes on `fix/mxfp8-moe-reload-followup`, which will be PR'd into theirs.

## Summary

Bug: with online MXFP8 MoE inference (`quantization = "mxfp8"`), vLLM's `Mxfp8OnlineMoEMethod._setup_kernel` rebuilds `moe_quant_config` and `moe_kernel` on every weight reload. Decode replays CUDA graphs captured against the boot-time kernel. Once that kernel is dropped, the graphs read freed memory and decode sometimes samples garbage tokens. Eager prefill and the trainer score those tokens at logprob -10 to -30, and they push RL mismatch KL up about 10x (DeepSeek-V4-Flash: ~0.04 vs ~0.003). NCCL broadcast makes it worse because it churns more GPU memory.

Fix: one monkey patch, `monkey_patch_online_mxfp8_moe_keep_kernel_on_reload` in `src/prime_rl/inference/patches.py`, registered in `apply_shared_vllm_patches`. At boot (`moe_kernel` is None) it calls the original `_setup_kernel`. On reload it runs `convert_to_fp8_moe_kernel_format`, installs the w13/w2 weights and scales with `replace_parameter`, and keeps the existing kernel. This relies on layerwise reload copying new values into the original parameter storage, which the boot-time kernel already references.

Evidence from the PR description (DeepSeek-V4-Flash, TP8/EP, FLASHINFER_TRTLLM MoE):

- Decode-vs-prefill KL (generate, then re-score the same tokens with `prompt_logprobs`): after a reload, 0.0065 unpatched vs 0.0028 patched (0.0025 after 10 reloads). Fresh servers sit at ~0.0027, and `enforce_eager` unpatched stays at 0.0027.
- 60-step RL (lr 1e-6, router replay, NCCL broadcast): mean mismatch KL 0.0437 to 0.0038, steps 1-20 0.0613 to 0.0014.
- Stale-weights check: layer 2's routed-expert `w2` is zeroed and loaded by reload into a server booted on base weights. That server matches a server booted directly on the zeroed checkpoint (KL 0.0026, vs ~0.0027 for two servers on identical weights). So the patch serves the new weights, not stale boot-time ones.

## Diagrams

Branch layout:

```
origin/fix/mxfp8-moe-reload-cudagraph   (samsja's PR #4109, base: main)
        │  git pull --ff-only only, never commit here
        ▼
fix/mxfp8-moe-reload-cudagraph          local mirror, git-tree root (not under main)
  d8d453117  the 41-line patch
        │
        ▼
fix/mxfp8-moe-reload-followup           our commits
        │  push only with Garrett's approval
        ▼
  draft PR  ──► base: fix/mxfp8-moe-reload-cudagraph
```

The bug (unpatched), as the PR describes it:

```
BOOT
  _setup_kernel ──► kernel K0 ──refs──► params P (w13, w2, scales)
                         │      └─refs─► tensors held only by K0
                         ▼
  capture decode CUDA graph G   (G bakes in raw addresses used by K0)

WEIGHT RELOAD (each RL step)
  layerwise reload copies new values into P in place       ✓ addresses unchanged
  _setup_kernel runs again ──► builds K1, new quant config
                               K0 dropped ──► K0-only tensors freed  ✗

DECODE
  replay G ──► reads K0's freed addresses ──► reused memory, garbage values
           ──► sporadic junk token sampled

PREFILL (eager) / TRAINER
  uses K1 or the real weights ──► scores the junk token at -10..-30
                              ──► mismatch KL ~0.04 instead of ~0.003
```

The fix:

```
Mxfp8OnlineMoEMethod._setup_kernel (patched)
        │
   self.moe_kernel is None?
        │
   yes ─┴─ no
    │       │
    │       ▼  reload path
    │     convert_to_fp8_moe_kernel_format(new w13, w2, scales)
    │     replace_parameter(w13_weight, w2_weight, w13_/w2_ scales)
    │     keep K0  (no kernel or quant config rebuild)
    ▼
  boot path: original _setup_kernel ──► builds K0

result:  G ──► K0 ──► P (holding the new values)   nothing freed, new weights served
```

## Open for the critical review

- The fix assumes reload writes into the memory K0 already points to. The zeroed-`w2` test is the PR's evidence for this.
- The fix assumes K0 holds no state that needs rebuilding when weights change.
- Backend dependence: is the patch correct for MXFP8 MoE backends other than FLASHINFER_TRTLLM? See the hardware section.

## Hardware: why B300

MXFP8 MoE does not strictly need Blackwell. `select_mxfp8_moe_backend` in `vllm/model_executor/layers/fused_moe/oracle/mxfp8.py` auto-selects the first supported backend in this order: FLASHINFER_TRTLLM, DEEPGEMM, MARLIN, XPU, AITER_MXFP8, HUMMING, TRITON_MXFP8, EMULATION. Hopper therefore falls through to a later backend.

The PR's results all use FLASHINFER_TRTLLM, which only runs on datacenter Blackwell: `_supports_current_device` in `fused_moe/experts/trtllm_fp8_moe.py:165` requires `is_device_capability_family(100)` and FlashInfer's TRT-LLM fused MoE. The backend matters for this bug. `convert_to_fp8_moe_kernel_format` (`fused_moe/oracle/fp8.py`, around line 640) sends TRT-LLM through `prepare_fp8_moe_layer_for_fi`, which for MXFP8 does a W31 swap, reorder and shuffle (`quantization/utils/flashinfer_utils.py:470`). EMULATION and TRITON_MXFP8 skip any load-time layout change. A Hopper run would exercise a different reload path and would not reproduce the PR's case.

Line numbers refer to the vLLM installed in `prime-rl-main/.venv` on 2026-10-16.

## Rebuilding the worktree layout on a new machine

From the `main` worktree (`~/github/PrimeIntellect-ai/prime-rl-main`):

```bash
git fetch origin
git branch --track fix/mxfp8-moe-reload-cudagraph origin/fix/mxfp8-moe-reload-cudagraph
git tree branch ~/github/PrimeIntellect-ai/prime-rl-fix-mxfp8-moe-reload-cudagraph fix/mxfp8-moe-reload-cudagraph --parent main
(cd ~/github/PrimeIntellect-ai/prime-rl-fix-mxfp8-moe-reload-cudagraph && git tree detach -y)
git branch fix/mxfp8-moe-reload-followup origin/fix/mxfp8-moe-reload-followup
git tree branch ~/github/PrimeIntellect-ai/prime-rl-fix-mxfp8-moe-reload-followup fix/mxfp8-moe-reload-followup --parent fix/mxfp8-moe-reload-cudagraph
(cd ~/github/PrimeIntellect-ai/prime-rl-fix-mxfp8-moe-reload-cudagraph && uv sync --all-extras --all-packages)
(cd ~/github/PrimeIntellect-ai/prime-rl-fix-mxfp8-moe-reload-followup && uv sync --all-extras --all-packages)
git tree --json
```

`git tree detach` makes the mirror its own git-tree root. If it stayed under `main`, `git tree propagate main` would rebase it locally onto newer `main`. Its SHAs would then diverge from `origin`, the follow-up PR would show rebase noise, and pushing the mirror would overwrite samsja's work. `git tree branch` initializes submodules itself. Expect `git tree --json` to show `fix/mxfp8-moe-reload-cudagraph` as a root with `fix/mxfp8-moe-reload-followup` as its only child.

## Working rules

- Never commit in the `prime-rl-fix-mxfp8-moe-reload-cudagraph` worktree. Update it only with `git pull --ff-only` there, then run `git tree propagate fix/mxfp8-moe-reload-cudagraph` to rebase the follow-up onto samsja's new tip.
- If samsja force-pushes (the pull is not a fast-forward), tell Garrett before resetting the mirror to `origin` and re-propagating.
- Never run `git tree push` from `fix/mxfp8-moe-reload-cudagraph`, because it pushes descendants too. Push only `fix/mxfp8-moe-reload-followup`, and only after Garrett explicitly approves each push. Open it as a draft PR with base `fix/mxfp8-moe-reload-cudagraph`.

## Next step

On the B300 machine, in plan mode with Garrett: do the critical review of the patch (the open questions above), then plan what the follow-up branch adds. Nothing gets implemented before that discussion.
