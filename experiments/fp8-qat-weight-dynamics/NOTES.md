# FP8 weight-shift dynamics on Qwen3-30B-A3B-Base

Plan: `PLAN.md` in this worktree. Runs write to `/home/garrett/fp8wd/runs/<run name>`.

## Starts

| start | path |
|---|---|
| offgrid | `/home/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` |
| ongrid-ue8m0 | `/home/garrett/fp8wd/snapshots/Qwen3-30B-A3B-Base-ue8m0-ongrid` |

`make_on_grid.py` on the off-grid snapshot (report in `<ongrid>/on_grid_report.json`):

| tensors | changed | relative RMS change | round-trip failures | untouched mismatches | small-amax blocks |
|---|---|---|---|---|---|
| 18624 | 92.7% | 2.66% | 0 | 0 | 0 |

## Smokes (20 steps, 8k x 16, code at `1eee457cf`, 2026-10-14)

Each smoke ran on its own fresh H200 node. Step time is the mean over steps 11 to 20, lower is better.

| run | start | train | loss step 1 | loss step 20 | step time (s) | peak mem (GiB) |
|---|---|---|---|---|---|---|
| 0a | offgrid | bf16 | 0.6239 | 0.5753 | 5.35 | 59.1 |
| 0a, tracker off | offgrid | bf16 | 0.6236 | 0.5761 | 5.33 | 58.8 |
| 0b | ongrid-ue8m0 | qat-ue8m0 | 0.6249 | 0.6025 | 9.17 | 59.6 |
| 0c | offgrid | qat-amax | 0.6248 | 0.5761 | 9.07 | 59.6 |

- Every run's start-weight check matched its intended snapshot.
- FP8 replaced 240 linear modules (q, k, v, fused qkv and o per layer) and skipped the 48 routers and `lm_head`.
- `probe_weight_ue8m0.py`: with `PRIME_RL_FP8_WEIGHT_UE8M0=1`, the four weight casts use power-of-two scales. Activation, gradient and wgrad casts keep `amax/448` scales.
- Weight-update tracker overhead is within run-to-run noise (5.35 vs 5.33 s/step).
- FP8 step time alternates between about 5.7 s and 10 to 20 s through step 20, with the same pattern in 0b and 0c. The slow time is in forward/backward. The DeepGEMM JIT cache (`DG_JIT_CACHE_DIR`) is node-local under `/tmp`, so new grouped-GEMM shapes compile on each fresh node. This is the likely cause and is not yet verified.
- 0b falls behind 0a in loss by 0.007 at step 5, 0.018 at step 10 and 0.027 at step 20. 0c stays within 0.002 of 0a. Master-weight drift (`delta_rms`) at step 20 agrees across 0a, 0b and 0c within about 5%, so the masters move alike and the gap comes from the forward weights. On-grid weights must move half a bin before their quantized value changes, so the qat-ue8m0 forward from the on-grid start barely changes in 20 steps.
