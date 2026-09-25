# Handoff: DeepSeek V4 Flash FP8 trainer/inference mismatch work (as of 2026-09-22)

You are picking up finished investigation work; the likely next asks are cleanup, small feature tests, or performance
benchmarking. This file gives the context; the detailed chronological record is `MISMATCH_NIGHT_LOG.md` (long; read
its "Correction", "Attribution" and "Morning summary" sections first). The original problem statement is
`MISMATCH_HANDOFF.md` (untracked); ignore its DeepGEMM hypothesis, which was wrong.

## The problem and what was found

DeepSeek V4 Flash (`PrimeIntellect/DeepSeek-V4-Flash-0731-bf16`, 131k SWE RL on 8 trainer + 8 inference H200 nodes)
served with online blockwise FP8 in vLLM showed trainer/inference mismatch KL of 0.025 at step 1, growing to 0.12 and
collapsing near step 240, versus 0.0005 to 0.001 with bf16 serving. Two independent causes, both confirmed:

1. **A vLLM bug, not DeepGEMM.** Below 128 tokens (every decode step) vLLM 0.29.0 falls back to `TritonExperts`,
   whose fused SiLU-and-quantize fast path (`fused_moe/experts/triton_moe.py`, `ops.silu_and_mul_per_block_quant`)
   drops the model's `swiglu_limit = 10.0`. Tokens whose gate/up pre-activation exceeds 10 get a wrong expert output,
   producing deterministic junk-token argmaxes (`)Skip` etc., 79% of the early mismatch). GLM-4.5-Air has no clamp
   and was unaffected. Fixed in prime-rl by `monkey_patch_triton_moe_swiglu_clamp` (`src/prime_rl/inference/patches.py`,
   always on) and validated on TP 8 + EP servers (0 of 30 glitch positions, was 17 of 30). An upstream vLLM PR is in
   progress in `/home/garrett/github/garrett361/vllm-fix-triton-moe-swiglu-clamp` (branch `fix/triton-moe-swiglu-clamp`,
   own git-tree rooted at `vllm-main`; one-line gate `and self.activation_config.clamp_limit is None`).
2. **Pinned served weights.** The bf16 release is a dequantized FP8 checkpoint, so every weight sits at an e4m3 bin
   centre. With round-to-nearest online quantization the served FP8 weights do not change until a weight has moved
   half a bin (about 1000 steps at lr 1e-6), so the served policy stays at step 0 while the trainer moves and the
   mismatch grows like an ever-increasing rollout lag (job 991: 0.0015 at step 1, 0.025 by step 118). Two remedies
   were each shown to flatten the curve for 75 to 100 steps: server-side stochastic rounding of the broadcast weights
   (job 1047, flat at 0.0021 to 0.0023) and FP8 on the trainer side too with the same `amax / 448` recipe (job 1115,
   flat at 0.0026 to 0.0028). A trainer-side dither of the masters was tried and made things far worse (do not reuse).

Ruled out: weight drift off the e4m3 grid (measured at step 40), o_proj activation quantization, position, batch
load, turn index. Trainer floor contributors not yet tested (bf16 RoPE tables, bf16 logits before log-softmax).

## Branches and worktrees (git-tree stack, root `/home/garrett/github/PrimeIntellect-ai/prime-rl-main`)

`main` -> `fix/fp8-quant-parity` -> `feat/ds-v4-bf16-rl` -> `feat/ds-v4-fp8-rl` (this worktree) -> `exp/ds-v4-fp8-dither`.
Worktrees are flat siblings named `prime-rl-<branch with / replaced by ->`. `git tree --json` shows the state.

- `feat/ds-v4-fp8-rl` (this branch): the investigation, the swiglu-clamp patch, `inference.fp8_ue8m0_weight_scales`
  (exact power-of-two weight scales for on-grid checkpoints), the DeepSeek configs, all logs. Rebased onto the fix
  branch on 2026-09-22 and NOT yet force-pushed (origin is behind).
- `fix/fp8-quant-parity` (`4dc088220`): trainer per-token FP8 activation quantizer made bit-identical to vLLM's
  production CUDA op (amax floor 1e-10, `div_rn` scale, output clamp). Its own PR to `main`; `PLAN.md` and
  `PR_DESCRIPTION.md` sit untracked in its worktree. Before/after lr = 0 probes read 0.0089 / 0.0091 (no KL change
  expected; the point is parity).
- `exp/ds-v4-fp8-dither`: NOT rebased since it forked (143 commits pending). Holds the rejected dither flag and the
  stochastic-rounding feature (`1bc6e8d32`: `inference.fp8_stochastic_weight_rounding`, env
  `PRIME_FP8_STOCHASTIC_WEIGHT_ROUNDING`, `stochastic_round_fp8` in `patches.py`; run config `swe-fp8-ue8m0-sr.toml`
  in `4b8ca3ad2`). Promotion into `feat/ds-v4-fp8-rl` was recommended but not done: cherry-pick those two commits,
  not the whole branch.
- Other branches off `main` created by other sessions: `feat/fp8-grouped-linear`, `feat/fp8-ue8m0-weight-scales`,
  `feat/ds-v4-e2e-sft`. Not touched here.

## Configs (`configs/advanced/deepseek-v4-flash/`)

- `swe.toml`: production FP8-serving SWE run; now `fp8_ue8m0_weight_scales = true`, `VLLM_USE_DEEP_GEMM_E8M0 = "0"`,
  indexer ignored, bf16 trainer. Growth from cause 2 still applies to this config unless stochastic rounding is added.
- `swe-bf16.toml`: bf16 serving control. `swe-fp8-ue8m0.toml`: the fix-test variant (job 991).
- `swe-fp8-fp8.toml`: FP8 on both sides (trainer `quantization.type = "fp8"` + `moe.compute.type = "deepgemm_fp8"`,
  eleven ignore patterns incl. `o_a_proj`, `indexer\.`, `compressor\.`; inference `fp8_ue8m0_weight_scales = false`
  so both sides use `amax / 448`). 100 steps run (job 1115). `rl_fp8_fp8.toml`: its 5-node lr = 0 probe.
- `rl.toml`, `rl_fp8.toml`, `rl_math.toml`, `rl_fp8_math.toml`: 5-node lr = 0 mismatch profiles (bf16 / FP8 serving).
- Checkpointing is off everywhere (3 TB per save). Launch: `PRL_OUTPUT_DIR=/home/garrett/prl_output_dir uv run rl @
  <toml>` (the env var is mandatory in non-interactive shells); dry-run first with `--dry-run` and re-validate the
  three resolved JSONs (`SWE_RUN_HANDOFF.md:72-91`). wandb project `primeintellect/deepseek-v4-flash`.

## Runs to compare against (run dirs under `/home/garrett/prl_output_dir/`, metrics in `monitors/file/metrics.jsonl`)

| run dir | serving | trainer | 20-step mismatch means |
|---|---|---|---|
| `dsv4-swe-131k` | FP8, fp32 scales, no clamp patch (glitch) | bf16 | 0.025 flat, then 0.05 to 0.12, collapsed |
| `dsv4-swe-131k-bf16` | bf16 | bf16 | 0.0006, 0.0008, 0.0009, 0.0010, 0.0010 (killed at 106) |
| `glm45air-swe-131k` | FP8 (GLM-4.5-Air, native bf16 ckpt) | bf16 | 0.004 to 0.007 flat, 351 steps |
| `dsv4-swe-131k-fp8-ue8m0` (job 991) | FP8, exact scales, E8M0=1 | bf16 | 0.0022, 0.0033, 0.0043, 0.0066, 0.0119, 0.0184 |
| `dsv4-swe-131k-fp8-ue8m0-sr` (job 1047) | as 991 + stochastic rounding | bf16 | 0.0018, 0.0021, 0.0022, 0.0023 |
| `dsv4-swe-131k-fp8-fp8` (job 1115) | FP8, amax/448 | FP8 | 0.0027, 0.0027, 0.0026, 0.0028, 0.0026 |
| `dsv4-swe-131k-fp8-ue8m0-dither` (job 1045) | as 991 | bf16 masters dithered | 0.0025 -> 0.031 by step 20 (rejected) |

## Tooling (outside the repo)

- `~/tmp/mismatch_evidence/`: `monitor_961.py` (per-step table; `MON_RUN=<run dir> MON_CTRL=<control run dir>`),
  `glitch_check_run.py <run dir>` (trace scan for glitch ids and large gaps), `growth_analysis*.py` (mismatch by lag
  and step), `grid_drift.py`, `glm_grid_check.py`, `fp8_parity/parity_check.py` (trainer vs vLLM quantizer bytes).
- `~/tmp/fp8diag/bisect/`: single-node vLLM server rotation (`start_server.sh`, `stop_server.sh`), `prefill_topk.py`,
  `decode_probe.py`, `bisect_report.py`, the 50-item scoring set, results of rounds r1 to r6.
- `~/tmp/fp8diag/deepgemm_repro/`: standalone kernel reproducer for the clamp bug (`clamp_test.py`, `patch_validate.py`).
- `~/tmp/vllm_fix/`: the vLLM one-line patch and bug-report draft.

## Open items

- Gradient norm in the fp8/fp8 run was 3 to 5x below bf16-trainer runs throughout (0.002 to 0.018 vs 0.03 to 0.09);
  unexplained; check FP8 vs bf16 backward on one batch (per-layer norm and cosine) before trusting trainer FP8.
- Promote stochastic rounding into `feat/ds-v4-fp8-rl` (cherry-pick from `exp/ds-v4-fp8-dither`), then decide the
  recommended DeepSeek FP8 serving config (exact weight scales + clamp patch + stochastic rounding, E8M0 off).
- `o_a_proj` has no trainer FP8 path (block-diagonal `DeepseekV4GroupedLinear`); it is bf16 on the trainer and FP8
  (`wo_a`, `is_bmm`) on vLLM. `feat/fp8-grouped-linear` may be addressing it.
- Trainer floor fixes never run: fp32 RoPE tables (PR 3584) and fp32 logits in `layers/lm_head.py:177`.
- Force pushes pending for `feat/ds-v4-bf16-rl` and `feat/ds-v4-fp8-rl`; never push without Garrett's explicit OK.

## Working rules that applied

One 16-node training run at a time, under 20 nodes total, `sinfo` before submitting, release allocations when idle;
one commit per logical change with conventional-commit scopes; log runs in `MISMATCH_NIGHT_LOG.md`; never edit
`.venv`; `uv run --no-sync` when the environment is known good (GitHub 504s have broken plain `uv run`); ad-hoc
vLLM servers take about 10 min on a warm node and over 70 min on a cold one (router timeout 4200 s).
