# Grid-dither experiment, night of 2026-09-21 (branch `exp/ds-v4-fp8-dither`)

Purpose: test the pinned-served-policy explanation for FP8 mismatch growth. Control: run `dsv4-swe-131k-fp8-ue8m0`
(job 991, 2026-09-20, identical config, 20-step means 0.0022, 0.0033, 0.0043, 0.0066, 0.0119, 0.0184). Treatment:
`trainer.model.fp8_grid_dither = true`, which once at start moves every attention-projection, routed-expert and
shared-expert master weight to a random position inside its e4m3 bin (mean |delta|/|w| 0.021, max 0.0585), with
elements at the top mantissa step dithered only toward zero so no block scale changes. CPU check on three tensors:
100.000% identical FP8 bytes and 0 changed block scales after quantization with power-of-two scales, so the served
FP8 model at step 0 is bit-identical to the control's.

Prediction if the mechanism is right: a higher step-1 floor (about 0.004-0.008, the trainer now differs from the
served grid points by up to half a quantum) that stays flat instead of doubling every 20 steps; on-policy (lag 0)
KL flat across step buckets; zero glitch tokens; reward comparable to the control's 0.4-0.8. If it climbs like the
control, the growth is activation-driven and the pinning explanation is wrong or incomplete.

Commits: `a6dc380b0` feat(trainer) flag, `9bae3231a` fix(trainer) toward-zero for saturating mantissas, `b59bbc860`
feat(configs) run config `swe-fp8-ue8m0-dither.toml` (`max_steps = 100`, checkpointing off, wandb
`swe-scaleswe-131k-fp8-ue8m0-dither-adamw1e-6-bs64g8-8t8i`). Check script `~/tmp/mismatch_evidence/dither_check.py`.

## Timeline (UTC)

- 00:05 (approx) Job 1045 submitted, PENDING (Resources). Run dir `/home/garrett/prl_output_dir/dsv4-swe-131k-fp8-ue8m0-dither`
  (attempt_3; attempts 1-2 were dry runs). Expected trainer log lines at start: "Dithering FP8-scope master weights
  within their e4m3 bins" then "Dithered N elements across M FP8-scope parameters (mean |delta| / |w| = ~2.1e-02)".

## Results

(pending)

---

# Mismatch investigation, night of 2026-09-20

Plan: `~/.claude/plans/read-mismatch-handoff-md-and-summarize-radiant-lighthouse.md`. Background: `MISMATCH_HANDOFF.md`.

Ground rules: one 16-node training run at a time (job 961, the bf16 control, until killed); under 20 nodes total;
checkpointing off in new launches; wandb project `primeintellect/deepseek-v4-flash`; one commit per logical change.

## Correction (2026-09-21, 15:00): the glitch is a vLLM bug, not a DeepGEMM bug

The night-1 attribution "DeepGEMM's small-M grouped FP8 GEMM misreads fp32 activation scales" was wrong. Root-caused
on 2026-09-21 by two independent code audits and a one-node kernel reproducer (`~/tmp/fp8diag/deepgemm_repro/`,
`~/tmp/mismatch_evidence/vllm_deepgemm_moe_caller.md`, `dsv4_vs_glm_moe_path.md`, `deepgemm_891d57b_audit.md`):

- With `VLLM_USE_DEEP_GEMM_E8M0=0`, vLLM 0.29.0 does not call DeepGEMM for routed experts when the batch has fewer
  than 128 tokens: `TritonOrDeepGemmExperts._select_experts_impl` (`fused_moe/experts/triton_deep_gemm_moe.py:83`)
  falls back to `TritonExperts` unless `_valid_deep_gemm_shape` (`experts/deep_gemm_moe.py:53-55`, needs
  `128 <= M` on SM90) passes. That is the 111-versus-135 boundary and why every decode step was affected.
- Inside `TritonExperts.apply` (`experts/triton_moe.py:474-486`) the fused SiLU-and-quantize fast path
  `ops.silu_and_mul_per_block_quant` has no clamp argument, so DeepSeek V4 Flash's `swiglu_limit = 10.0`
  (`gate.clamp(max=10)`, `up.clamp(-10, 10)`, as the trainer applies in `deepseek_v4/moe.py:30-31`) is dropped. Every
  other path honours it: the else branch (`self.activation`, `silu_and_mul_with_clamp`), DeepGEMM
  (`silu_mul_per_token_group_quant_fp8_colmajor(clamp_limit=...)`), the E8M0 packed kernel, and bf16.
- Reproducer at T = 64 with real layer-15 expert weights: the fast path's output matches the UNCLAMPED reference to
  FP8 noise (0.045) and is off by up to 7.4x the token norm against the clamped reference on tokens whose gate or up
  pre-activation exceeds 10 (12 of 15 such tokens catastrophic, error monotone in the peak); in-limit tokens 0.045.
  Forcing the else branch, T = 256 (DeepGEMM), or E8M0=1 at T = 64 are all clean (0.06). DeepGEMM called directly at
  every M from 1 to 256, with fp32 or power-of-two activation scales, NaN-filled or zeroed padding, and TMA-aligned
  scale layouts, is clean; on SM90 `disable_ue8m0_cast` is a no-op inside DeepGEMM and the contiguous alignment is
  fixed at 128, so tiles cannot straddle experts.
- Why GLM-4.5-Air was immune: identical module class, quant method, kernel dispatch and CUDA-graph sizes; it has no
  `swiglu_limit`, so the unclamped fast path is correct for it.
- Why `VLLM_USE_DEEP_GEMM_E8M0=1` "fixed" it: it forces the DeepGEMM experts path at every M, routing around the
  buggy Triton branch. The activation-scale-format story was a coincidence of that routing. The shipped fix-test
  config still works and its measured numbers stand; its mechanism was misdescribed.
- Proper fix (one line, vLLM): gate the fast path on `self.activation_config.clamp_limit is None`. Patch and bug
  report under `~/tmp/vllm_fix/`. Alternatively give `silu_and_mul_per_block_quant` a clamp argument.
- 15:30: prime-rl carries the fix as `monkey_patch_triton_moe_swiglu_clamp` (commit 7ee8090a5): when a clamp is
  configured, `TritonExperts.apply` runs with the module-level `is_deep_gemm_e8m0_used` bound to True, which is
  its only use inside `apply` and routes the call to the clamped `self.activation` branch. Validated on one H200
  with the layer-15 reproducer (`~/tmp/fp8diag/deepgemm_repro/patch_validate.py`): at T = 64 the impl is still
  `TritonExperts`, over-limit tokens go from 7.36 / 2.62 / 12-of-15 catastrophic (max / median / count) to
  0.061 / 0.047 / 0, the fused op is not called; with no clamp the fused fast path is still taken and output is
  bit-identical to unpatched. Not yet exercised under expert parallelism (`expert_map`), so a two-node decode probe
  on a TP 8 + EP server is the remaining check before this is treated as fully validated.
- Config decision: DeepSeek V4 runs go back to `VLLM_USE_DEEP_GEMM_E8M0 = "0"` (the patch removes the reason for
  "1", which worked only by forcing the DeepGEMM path and also coarsens activation scales) and keep
  `fp8_ue8m0_weight_scales = true`, now also set in `swe.toml`. The upstream vLLM PR is being prepared in
  `/home/garrett/github/garrett361/vllm-fix-triton-moe-swiglu-clamp` (branch `fix/triton-moe-swiglu-clamp`).
- 17:30 Round 6 (`~/tmp/fp8diag/bisect/results/r6.decode.report.txt`): on two real TP 8 + EP servers with
  `VLLM_USE_DEEP_GEMM_E8M0 = "0"`, S12 (original production config plus the patch) and S11 (patch plus
  `fp8_ue8m0_weight_scales = true`, the committed production config) both read 0 of 30 glitch positions on the
  decode probe (baseline without the patch: 20 in the top-20, 17 argmax), with the ordinary `.` / `,` argmax at
  every item. S11 bulk prefill versus the bf16 reference: KL 4.0e-3, IPO masked 0.0000, |lr| p90 0.0835, mean lr
  +0.0008 (S3 with the same weights: 0.085 / +0.0009; production fp32-scale config: 0.116 / -0.0016). The
  expert-parallel branch of the patched path is therefore validated too. Operational note: cold `/tmp/garrett`
  JIT caches made the first server launch exceed the router's fixed 4200 s startup timeout (over 70 min versus
  10 min warm); prime-rl exposes no knob for that timeout.

## Morning summary (final, 11:25; no jobs running)

1. **The FP8 mismatch had two causes, both now attributed.** (a) A vLLM bug (see the Correction above; the DeepGEMM
   attribution written here on night 1 was wrong): below 128 tokens vLLM's Triton MoE fallback drops DeepSeek V4's
   swiglu clamp, producing the glitch tokens (`)Skip` etc.) that carried 79% of the FP8 run's step-1 mismatch. Never
   visible in prefill scoring, which is why the trainer and FP8 prefill agreed. (b) A resolution floor: the bf16
   release is a dequantized FP8 checkpoint sitting exactly on the e4m3 grid, so at lr 1e-6 the weight updates are
   far below half an e4m3 quantum and the served FP8 weights stay pinned at step 0 while the trainer moves. The
   mismatch therefore grows like an ever-increasing rollout lag. This is the growth and, plausibly, the collapse.
   GLM's FP8 run never grew because its checkpoint is native bf16 (weights dither across bin boundaries).
2. **Shipped** (commits on `feat/ds-v4-fp8-rl`, nothing pushed): `VLLM_USE_DEEP_GEMM_E8M0=1` avoids the faulty path;
   new `inference.fp8_ue8m0_weight_scales` quantizes with exact power-of-two scales (lowest bulk error of every FP8
   config measured); run config `swe-fp8-ue8m0.toml`; a fake-quant ignore regex; a logger fix.
3. **Fix test (job 991, 16 nodes, running since 05:43):** mismatch KL 0.0015 at step 1 versus 0.025 for the FP8
   baseline (17x lower; max 1.5 versus 370), zero glitch tokens in 585k trained tokens, then an accelerating climb to 0.0135
   by step 100 (20-step means 0.0022, 0.0033, 0.0043, 0.0066, 0.0119; bf16 control: 0.0005 to 0.0010 over the same
   steps), doubling every 20 steps and on course to pass the FP8 baseline's flat 0.025 around step 115 from the pinned-policy drift plus lag. Reward
   and entropy healthy. Leave it running or kill it; it is the only job I hold.
4. **Not done:** trainer floor fixes (fp32 RoPE from PR 3584, fp32 logits) for lack of nodes under the 20-node rule;
   any remedy for the pinning (stochastic rounding at broadcast, larger lr, or bf16 serving). The bf16 control (job
   961) was killed at step 106 and its checkpoints deleted as agreed.
5. Details, per-round tables and file pointers follow. Tooling is under `~/tmp/fp8diag/bisect/` and
   `~/tmp/mismatch_evidence/`; a stray gitignored `outputs/dsv4-swe-131k-fp8-e8m0/` in the worktree is yours to delete.

## Timeline (UTC)

- 03:22 Start. Job 961 at step 63, mismatch KL 0.0008 at step 63. 5 idle nodes. Checkpoints `step_40`, `step_60`
  exist (job runs off its pre-edit resolved config; `keep_last = 2`). Both to be deleted when 961 is killed.
- 03:53 S0 (bf16, node 004, job 978) and S1 (FP8 production, node 007, job 979) healthy; scoring set built
  (50 items, 30 glitch prefixes from steps 1-10, 20 bulk sequences 3k-69k tokens, all glitch joins verified).
- 03:57 B2 done (below). Round 1 scoring S0 vs S1 started.
- 04:12 Round 1 done: glitch not reproduced in prefill on the FP8 server. Round 2 (decode path) started.
- 04:27 Round 2 done: glitch REPRODUCED on the FP8 server's decode path, deterministic argmax, M = 1. Round 3 started.
- 04:30 Glitch position-structure analysis done (below). Control at step 92, steps 81-92 mean 0.00109.
- 04:58 Round 3 done: ffn FP8 required, o_proj exonerated. Round 4 (S7 routed-only, S9 E8M0=1) started.
- 05:12 Round 4 done: routed-experts DeepGEMM GEMM is the culprit; E8M0=1 removes the glitch (0/30).
- 05:15 Killed job 961 (bf16 control) at step 106: fix candidate in hand and the control past 100 steps. Its
  checkpoints (`step_80`, `step_100`, 6.3 TB) deleted (background rm, log `~/tmp/rm_bf16_ckpts.log`).
- 05:20 Submitted job 988: `swe-fp8-e8m0.toml`, 16 nodes, FP8 with UE8M0 scales, checkpointing off, wandb
  `swe-scaleswe-131k-fp8-e8m0-adamw1e-6-bs64g8-8t8i`. Run dir `/home/garrett/prl_output_dir/dsv4-swe-131k-fp8-e8m0`.
- 05:17 Round 5 started (S10, S3).
- 05:38 Round 5 done: activation-scale misread confirmed; S10 (E8M0=1 + exact power-of-two weight scales) is the
  best config. Bisection nodes released (jobs 978, 979 gone).
- 05:39 Killed job 988 after 90 s: it tested E8M0=1 alone, and the run should test the recommended config.
- 05:45 Promoted the weight-scale diagnostic to `inference.fp8_ue8m0_weight_scales`; relaunching as
  `swe-fp8-ue8m0.toml` (run `dsv4-swe-131k-fp8-ue8m0`).
- 05:43 Submitted job 991: `swe-fp8-ue8m0.toml`, 16 nodes, FP8 with `VLLM_USE_DEEP_GEMM_E8M0=1` and
  `fp8_ue8m0_weight_scales = true`, checkpointing off, wandb `swe-scaleswe-131k-fp8-ue8m0-adamw1e-6-bs64g8-8t8i`.
  Run dir `/home/garrett/prl_output_dir/dsv4-swe-131k-fp8-ue8m0` (attempt_2; attempt_1 was the dry run).
- 05:50 Verified on inference node 007 that `PRIME_FP8_UE8M0_WEIGHT_SCALES=1`, `VLLM_USE_DEEP_GEMM_E8M0=1` and
  `VLLM_USE_DEEP_GEMM=1` are in the EngineCore and worker process environments, and the log shows "DeepGEMM E8M0
  enabled". The patch's own INFO line was swallowed by a non-`vllm.*` logger name; fixed in a follow-up commit
  (does not affect the running job's behaviour, only its logging).
- 06:26 Job 991 step 1: mismatch KL 0.0015, max 1.48 (FP8 baseline 0.025 / 370; bf16 0.00054 / 0.77).
- 06:58 Job 991 steps 1-7 mean 0.00176, max <= 3.4; trace scan: zero glitch tokens in 585k trained tokens.
- 07:34 Job 991 step 20: 0.00315, climbing monotonically since step 12. Growth analysis started.
- 11:23 Job 991 step 120 at 0.0255 (level with the FP8 baseline), is_masked 0.0098: killed at step 130. Final
  trace scan: zero glitch tokens across the run. No jobs running.
- 10:34 Job 991 step 100: steps 81-100 mean 0.01187, doubling per 20-step window; is_masked up to 3.6e-3; reward healthy.
- 09:45 Job 991 step 80: steps 61-80 mean 0.00656, accelerating; is_masked 2.5e-3; reward healthy. Left running.
- 09:05 Job 991 step 60: steps 41-60 mean 0.00432; the 20-step means are linear in step count; reward 0.36-0.89.
- 08:25 Job 991 step 40: steps 21-40 mean 0.00327 (+0.00107 over steps 1-20), max <= 4.4, reward 0.3-0.7, no
  inference errors; the climb continues at about 0.00005 per step as the pinned-policy mechanism predicts.
- 07:48 Growth analysis done: lag plus an FP8-specific drift explained by sub-quantum weight updates never reaching
  the served FP8 weights (the served policy is pinned at step 0).
- 03:25 Launched: A0 scoring-set builder, A1 server infrastructure (2 nodes: S0 bf16 reference, S1 FP8
  production), B2 e4m3 grid-departure analysis on `step_40`.

## Attribution (as of 05:55)

| cause | evidence | magnitude | fix | status |
|---|---|---|---|---|
| vLLM `TritonExperts` fused SiLU-quant fast path drops the swiglu clamp; used below 128 tokens when E8M0 is off (night-1 text blamed DeepGEMM; corrected 2026-09-21) | rounds 2-5: glitch reproduces only in decode / short tails on FP8 servers; routed-experts-only FP8 reproduces; attention-only and E8M0=1 do not; power-of-two weight scales alone do not remove it | 79% of the FP8 run's step-1 mismatch KL (glitch tokens), 25% late; `mismatch_kl/all/max` 48-3851; the tokens are IPO-masked so they add no gradient but pollute trajectories | `VLLM_USE_DEEP_GEMM_E8M0=1` | validated on 30 positions; under training in job 991 |
| Online FP8 weight rounding with `amax / 448` scales rotates the checkpoint's power-of-two e4m3 grid | F27/F28; round 5: S3/S10 halve p90 log-ratio vs fp32 scales | bulk prefill KL 5.6e-3 -> 3.5e-3, IPO masked 0.0002 -> 0.0000 | `inference.fp8_ue8m0_weight_scales = true` (new field) | in job 991 |
| Residual FP8 activation quantization (UE8M0 per-128 groups) | S10 vs bf16 reference: KL 3.5e-3, p90 0.086 | about 4-6x the bf16 floor | none tonight; bf16 serving if unacceptable | measured |
| Weight drift off the e4m3 grid (handoff hypothesis 1) | B2: 87-93% of FP8-scope weights bit-identical at step 40, rest one bf16 ULP; > 1000 coherent steps to move a quantum | cannot explain growth at step 150 | none needed | ruled out |
| o_proj UE8M0 activation quant (hypothesis 3) | S4 reproduces the glitch at production strength | not the glitch; bulk effect not separately measured | none | exonerated for the glitch |
| FP8 serving cannot represent sub-quantum weight updates: the served policy stays pinned at the initial checkpoint | growth analysis: on-policy FP8 KL climbs +0.044e-3 per step while bf16 on-policy is flat; B2: 0.000% of routed-expert weights changed FP8 value by step 40; GLM (off-grid bf16 checkpoint) never grew | the diffuse climb (old run 0.005 -> 0.056 tail-excluded; fix-test 0.0015 -> 0.0031 by step 20) | stochastic rounding at broadcast, larger lr, or bf16 serving | explained, remedy untested |
| Rollout lag (staleness) | KL vs lag rises in both runs, 2x FP8 / 2.6x bf16 from lag 0 to 9+, mostly inside `<think>` | +0.0007 of the fix-test's step 1-20 climb | lower `max_off_policy_steps` or more inference capacity | measured |
| Lightning Indexer top-k flips as amplifier (hypotheses 4, 8) | bf16 self-noise above 2048 tokens reaches several nats on single tokens; bf16 control drifted 0.0006 -> 0.0010 over 106 steps with no acceleration | a slow floor-level drift at most, explained by lag in bf16 | fp32 RoPE (PR 3584) would reduce indexer input divergence | low priority |
| Trainer bf16 logits before fp32 log-softmax (hypothesis 7) | code: `lm_head.py:177`; about 0.1 nat at logit 30 | part of the 5e-4 floor, cannot make a 44-nat gap | `torch.mm(..., out_dtype=float32)` | not run (node budget) |
| Trainer bf16 RoPE tables (hypothesis 2) | PR 3584 applies cleanly | floor contributor | cherry-pick PR 3584 | not run (node budget) |

Growth (resolved 07:48, see the job 991 growth analysis under Track C): what made the FP8 run's mismatch grow 5x after step 150 and collapse at 240. Grid drift is out. The
glitch rate per token fell over the run (35.7 -> 19.1 per 100k), so the growth was in the diffuse bulk (tail-excluded
KL 0.005 -> 0.056), i.e. the policy drifted into states where the misread-scale kernel error is larger, or another
mechanism. Job 991 with the fix is the test: if its mismatch stays flat past step 150 the growth was FP8-kernel-driven.

## Track A: glitch-token and module bisection

Infrastructure: two exclusive one-node `salloc` holds (jobs 978 slot 0 node 004, 979 slot 1 node 007, 8 h, expire
about 11:26). Scripts under `~/tmp/fp8diag/bisect/`: `start_server.sh`, `stop_server.sh`, `release.sh`,
`prefill_topk.py` (top-10 prompt logprobs per position), `bisect_report.py`. Server startup is 22-23 min each
(weights 330-560 s, engine init 550-770 s). vLLM module names for ignore regexes:
`model.layers.N.attn.{fused_wqa_wkv,wq_b,wo_a,wo_b,compressor}`, `model.layers.N.ffn.{gate,shared_experts,experts}`.

| id | serving | env / ignore | status |
|---|---|---|---|
| S0 | bf16 | | up 03:53, stopped 04:28 (prefill results saved in r1.bf16.json) |
| S1 | FP8 production | ignore `.*indexer.*`, E8M0=0, deep_gemm | up 03:52, stopped 04:28 |
| S2 | bf16 + fake-quant weights | `PRIME_DIAG_FAKE_QUANT_WEIGHTS=1 PRIME_DIAG_FAKE_QUANT_IGNORE=.*indexer.*` | not needed (glitch is a kernel effect, not weight rounding) |
| S4 | S1 + bf16 o_proj | `PRIME_DIAG_BF16_OPROJ=1` | up 04:44, reproduces 17/30, stopped 05:00 |
| S5 | attention-only FP8 | `servers/s5-attn-only.toml` (ignore `.*ffn.*`) | up 04:39, clean 0/30, stopped 05:00 |
| S7 | routed-experts-only FP8 | `servers/s7-routed-only.toml` (ignore `.*attn.*`, `.*shared_experts.*`) | up 05:10, reproduces 13-14/30, stopped 05:17 |
| S9 | S1 + UE8M0 scales | `VLLM_USE_DEEP_GEMM_E8M0=1` | up 05:10, CLEAN 0/30, stopped 05:17 |
| S10 | S9 + power-of-two weight scales | `VLLM_USE_DEEP_GEMM_E8M0=1 PRIME_DIAG_UE8M0_WEIGHTS=1` | up 05:30, CLEAN 0/30, best bulk, stopped 05:38 |
| S3 | S1 + power-of-two weight scales | `PRIME_DIAG_UE8M0_WEIGHTS=1` | up 05:29, reproduces 17/30, stopped 05:38 |
| S6 | ffn-only FP8 | `servers/s6-ffn-only.toml` (ignore `.*attn.*`) | superseded by S7 |

Round 1 (S0 vs S1, 50 items, 822k tokens): running since 04:05.

### Round 1 result (04:12): static FP8 prefill does NOT produce the glitch

Report `~/tmp/fp8diag/bisect/results/r1.report.txt`. 50/50 items scored on both servers, one request in flight.

- At all 30 glitch positions S1 (FP8 production config) scores the glitch token at -40 to -50, the same as S0 and
  the trainer (median: S0 -44.8, S1 -44.1, production trainer -44.9, production inference -1.07). Reproduced
  (S1 lp > -3): 0 of 30. `)Skip` is never the argmax at any of 822k positions on either server; its best S1
  logprob anywhere is -35. S0's top-1 token is S1's argmax at 29/30 glitch positions.
- Bulk non-glitch tokens (203k): S0-vs-S1 KL 5.6e-3, |lr| p50 0.000 / p90 0.116, mean signed lr -0.0016. The
  production trainer-vs-inference numbers on the same tokens: KL 1.2e-2, p90 0.100, mean signed lr -0.0126.
  So static FP8 prefill explains about half the bulk production gap and none of the glitch.
- bf16 self-noise is not bitwise: re-scoring items 0-4 on S0 gave |dlp| p50 5e-5, p90 0.06, max 12 nats over 71k
  tokens (515 tokens over 1 nat). F9's "bitwise at concurrency 1" was measured at <= 2048 tokens, where the
  indexer is in its shortcut regime; above 2048 the top-512 selection is live and any rounding difference can
  flip membership. Consequence: compare |lr| quantiles, not KL means, on small subsets.
- Interpretation: production inference scored the glitch tokens while DECODING. Prefill on the same weights
  gives the trainer's answer. Suspects: FP8 decode kernels (F4: the FP8 linear branches at M < 32), the indexer's
  decode path (`fp8_paged_mqa_logits` plus decode top-k selectors), FlashMLA sparse decode, or batch composition.
  Round 2 tests decode at the glitch position, single request and under a 40-request decode load, on S0 and S1.

### Round 2 result (04:27): the glitch reproduces on S1 through the DECODE path, single request

Report `~/tmp/fp8diag/bisect/results/r2.report.txt`, script `decode_probe.py`. Probe: prompt = path tokens up to
g-1, two greedy tokens with `logprobs 20`, so position g is computed in a real decode step (M = 1).

- S1 (FP8 production): glitch token in the top-20 at 20/30 positions, logprob > -3 at 18/30, `)Skip` is the greedy
  ARGMAX at 17/30 (logprob as high as -0.04). S1 decode logprobs match production inference to within about 0.1
  nat (item 11: -0.07 vs -0.11; item 15: -0.11 vs -0.07; item 27: -0.07 vs -0.04; item 20: -7.69 vs -7.33). 8 of
  the 12 non-reproducing items had a different greedy token at g-1 than the sampled original.
- S0 (bf16): 0/30 in the top-20 (floors -6 to -16), argmax is the ordinary `.` / `,` / ` (`.
- S1 decode agrees with S1 prefill at position g-1 (|dlp| p50 0.03, max 0.14) and diverges by 40-50 nats at g. So
  the FP8 server has a decode-path-specific failure, not a generic quantization error. Production trainer logprobs
  match S0/S1 prefill; production inference logprobs match S1 decode.
- Load is irrelevant: with 40 concurrent sampling requests (M about 41) S1 reproduces 16/30, S0 0/30.
- M-dependence: with the prefix cache warm and only the tail past the last 256-token block computed as a small
  prefill chunk, S1 reproduces 8/30, all with tails of 19-111 tokens; none of the 17 items with tails >= 135
  tokens reproduced. The trigger is a token-count branch somewhere near 128, not F4's M < 32.
- S0 decode vs prefill self-consistency: argmax |dlp| p50 0.02 / p90 0.07 / max 0.13.
- Round 3 (04:28): S0 and S1 stopped; slot 0 -> S4 (FP8 + `PRIME_DIAG_BF16_OPROJ=1`), slot 1 -> S5 (attention-only
  FP8). Decode probe on both. Meanwhile reading vLLM for the decode/prefill threshold and FP8-only decode kernels.

### Round 3 result (04:58): the MoE (ffn) FP8 path is required; o_proj is exonerated

Report `~/tmp/fp8diag/bisect/results/r3.report.txt`. Decode probe counts (d2: top-20 / lp > -3 / argmax = glitch):

| server | d2 | p1 (small tail prefill) |
|---|---|---|
| S1 FP8 production | 20 / 18 / 17 | 8 |
| S4 FP8 + bf16 o_proj | 19 / 17 / 15 | 10 |
| S5 attention-only FP8 (ffn bf16) | 0 / 0 / 0 | 0 |

- S4 reproduces as strongly as production, so the o_proj path (`fused_inv_rope_fp8_quant` UE8M0 activation quant
  plus the `wo_a` FP8 einsum, hypothesis 3) is not the mechanism.
- S5 is clean at every position with the ordinary `.` / `,` argmax, so attention FP8 does not suffice and the ffn
  FP8 path (DeepGEMM `m_grouped_fp8_gemm_nt_contiguous` for routed experts and/or plain `fp8_gemm_nt` for the shared
  experts) is required.
- Code reading: vLLM classifies decode as query_len <= 1 for attention, compressor and indexer, so the 111-vs-135
  boundary is not an attention decode/prefill switch. No Python-level M branch exists in the FP8 linear or MoE
  path; M-dependent kernel selection lives inside DeepGEMM's compiled library (BLOCK_M 64 vs 128, 1D1D vs 1D2D),
  whose default contiguous-layout alignment is 128. The kernels exclusive to FP8 servers are the FP8 GEMMs and
  their activation-quant kernels; none is decode-exclusive. In vLLM 0.29.0 the DSv4 indexer has no
  `fp8_paged_mqa_logits` path; top-k comes from the compressor path.
- Round 4 (05:00): slot 0 -> S7 routed-experts-only FP8; slot 1 -> S9 production FP8 with `VLLM_USE_DEEP_GEMM_E8M0=1`
  (power-of-two scales). Hypothesis: the small-M DeepGEMM kernel misreads fp32 scales, which UE8M0 avoids.

### Round 4 result (05:12): routed experts carry the glitch; `VLLM_USE_DEEP_GEMM_E8M0=1` removes it

Report `~/tmp/fp8diag/bisect/results/r4.decode.report.txt`, `r4.s9.prefill.report.txt`.

| server | d2 top-20 / lp > -3 / argmax | p1 top-20 / lp > -3 / argmax |
|---|---|---|
| S1 FP8 production | 20 / 18 / 17 | 8 / 8 / - |
| S7 routed-experts-only FP8 | 19 / 13 / 14 | 10 / 7 / 5 |
| S9 production + E8M0=1 | 0 / 0 / 0 | 0 / 0 / 0 |

- S7 (only `ffn.experts` quantized; attention, shared experts, indexer bf16) reproduces at production magnitude, so
  the glitch lives in DeepGEMM's `m_grouped_fp8_gemm_nt_contiguous` for routed experts. Shared experts and the plain
  `fp8_gemm_nt` linears are not needed.
- S9 (production scope, UE8M0 power-of-two scales for weights and activations) is clean at every position; the
  glitch token never enters the top-20 anywhere (floors -7 to -15); the argmax is the ordinary `.` / `,` as on bf16.
- Mechanism (consistent, not proven at the CUDA level since DeepGEMM 2.5.0+891d57b is compiled-only here): the
  kernel variant DeepGEMM selects for M below its 128 BLOCK_M reads block scales in a layout that only matches
  power-of-two scales, so fp32 scales are misread in decode steps and small prefill tails while full prefill
  chunks (M >= 128) are correct. Fits the 111-vs-135 boundary, decode-only production glitches, and the trainer
  (prefill-like) agreeing with FP8 prefill.
- Cost of E8M0=1 on bulk prefill (203k tokens vs the bf16 reference): KL 8.5e-3 vs 5.6e-3 for the fp32-scale
  config, IPO masked 0.0005 vs 0.0002, |lr| p90 0.144 vs 0.116, mean signed lr -0.0001 vs -0.0016. Production
  trainer-vs-inference on the same tokens: KL 1.2e-2, p90 0.100, mean lr -0.0126. So the catastrophic tail is
  gone and the diffuse penalty is about 25% higher than fp32 scales (double rounding of weight scales; round 5
  tests bit-exact power-of-two weight scales via `PRIME_DIAG_UE8M0_WEIGHTS=1` to recover that).
- Round 5 (05:17): S10 = E8M0=1 + `PRIME_DIAG_UE8M0_WEIGHTS=1`; S3 = `PRIME_DIAG_UE8M0_WEIGHTS=1` alone (fp32-format
  activation scales) to tell whether the misread is on weight or activation scales. Then release both nodes.

### Round 5 result (05:38): the misread is on ACTIVATION scales; exact weight scales lower the bulk penalty

Report `~/tmp/fp8diag/bisect/results/r5.decode.report.txt`, `r5.*.prefill.report.txt`. Both nodes released 05:38.

| server | glitch d2 lp > -3 | bulk KL | IPO masked | lr p90 | lr max | mean lr |
|---|---|---|---|---|---|---|
| S1 FP8 production (fp32 scales) | 18/30 | 5.65e-3 | 0.0002 | 0.116 | 5.60 | -0.0016 |
| S9 E8M0=1 (requantized weights) | 0/30 | 8.48e-3 | 0.0005 | 0.144 | 4.68 | -0.0001 |
| S10 E8M0=1 + power-of-two weight scales | 0/30 | 3.46e-3 | 0.0000 | 0.086 | 3.74 | +0.0010 |
| S3 fp32 act scales + power-of-two weight scales | 17/30 | 5.40e-3 | 0.0001 | 0.085 | 6.04 | +0.0009 |
| production trainer vs inference (same tokens) | | 1.23e-2 | 0.0002 | 0.100 | 34.6 | -0.0126 |

- S3 keeps the glitch at production strength while S9/S10 remove it, so the small-M DeepGEMM grouped GEMM misreads
  the fp32 per-token-group ACTIVATION scales (`per_token_group_quant_fp8` with `disable_ue8m0_cast=True`), not the
  weight scales. The large-M path used by full prefill reads them correctly.
- Power-of-two weight scales (bit-exact for this checkpoint) cut the diffuse penalty: S10 has the lowest KL, p90
  and masked fraction of every FP8 config measured (61% of production's KL, below the production trainer-vs-inference
  p90 of 0.100). S9's higher penalty came from vLLM's re-quantization of already-rounded weights.
- Recommended production change: `VLLM_USE_DEEP_GEMM_E8M0=1` plus power-of-two weight scales, promoted from the
  `PRIME_DIAG_UE8M0_WEIGHTS` diagnostic to the config field `inference.fp8_ue8m0_weight_scales` (commit below).

### Glitch position structure in production traces (04:30; `~/tmp/mismatch_evidence/glitch_position_structure.md`)

330 occurrences in 83 of 128 traces versus 200k ordinary sampled tokens.
- Context, not position: 95.5% inside `<think>` (baseline 48.5%); 51.5% immediately followed by `.` or `.\n\n`
  (in-think baseline 4.8%), preceded by sentence closers (` pass`, ` complete`, ` works`, ` fine`). The token after
  the glitch is scored consistently by both sides (2% with gap > 1), so only the glitch token is mis-scored and the
  KV state after it is fine.
- No alignment with `pos mod 256/128/4`, `(pos // 4) mod 512` or 2048-token windows (chi-square p 0.3-0.96).
- Mild recency effect: offsets 4-10 tokens after the last prefill are 2.9x enriched relative to in-think tokens,
  11-50 are 2x, but 64% of glitches occur past offset 50 and 14% past 1000. Zero at offset 0 (the first sampled
  token, whose logits come from the prefill pass), consistent with decode-only.
- No load effect: concurrent requests per server at glitch time equal the ordinary value (steps 1-10 median 16
  vs 16; steps 100-110 52 vs 52). Not clustered: 248 of 282 glitch-holding nodes hold exactly one.
- Rate per 100k trained tokens falls over the run: 35.7 (steps 1-10), 24.7 (100-110), 23.5 (180-190), 19.1 (230-239).
- Production serving from the logs: vLLM 0.29.0, `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` for dense FP8
  linears, DeepGEMM FP8 MoE, FP8 indexer cache, DSA indexer decode path `use_flattening=False supports_varlen=False
  next_n=1`. Warning `DeepseekV4ScalingRotaryEmbedding: Failed to load weights` on every weight reload (3531 times
  on node 0). Benign: `reload/layerwise.py:268` warns when a layer has buffers (the cos/sin cache) but the state
  dict carries no weights for it, then restores the original kernel tensors.

## Track B: bf16 control (job 961) and weight drift

| step | kl_mean | kl_max | is_masked | grad_norm | note |
|---|---|---|---|---|---|
| 1-20 | 0.00051-0.00084 | <= 3.4 | 0 | 0.03-0.09 | from handoff-time extraction |
| 21-45 | 0.00052-0.00159 | <= 3.0 | <= 1.2e-4 | 0.02-0.13 | mean 0.00087, 1.4x steps 1-20 |
| 46-60 | 0.00063-0.00128 | <= 3.9 | <= 4.9e-5 | 0.005-0.073 | mean 0.00090 |
| 61-80 | 0.00058-0.00143 | <= 6.6 | <= 1.5e-5 | 0.02-0.09 | mean 0.00096 |
| 81-100 | 0.0023-0.0031, mean 0.00261 | <= 7.7 | | | | 0.01187 (81-100) | 0.00101 (81-100) |
| 81-84 | 0.00083-0.00143 | <= 2.5 | <= 8.7e-6 | 0.04-0.08 | mean 0.00105; slow creep, about +0.0001 per 20 steps |
| 81-100 | 0.00062-0.00165 | <= 9.8 | <= 3.2e-5 | 0.02-0.09 | mean 0.00101 |
| 101-106 | 0.00074-0.00117 | <= 4.7 | <= 5.0e-6 | 0.004-0.045 | mean 0.00095; killed at step 106 (05:15) |

Verdict on the control: the bf16 mismatch drifted from 0.00061 (steps 1-20) to about 0.00100 (steps 81-106), a
1.6x rise over 100 steps with no acceleration, versus the FP8 run's 0.025 flat to step 150 then 5x by step 239. Not
the FP8 shape; a slow DeepSeek-specific drift (hypothesis 8) remains plausible at a much smaller scale.

### B2: e4m3 grid departure at step 40 (done 03:57; `~/tmp/mismatch_evidence/grid_drift.md`)

Hypothesis 1 (weights leave the e4m3 grid, so online re-quantization error grows) is ruled out on timescale.

- Checkpoint `step_40/trainer/` is DCP over 64 ranks, fp32 masters plus AdamW state. Sampled 77 tensors, 1.69e9
  elements across attention projections, routed and shared experts, compressor, indexer, gates, norms, mHC, embed,
  lm_head, layers 0/2/15/30/42.
- FP8-scope families (attention / routed experts / shared expert): off-grid fraction 0.000% at step 0, 6.7% / 12.0% /
  9.8% at step 40, and every off-grid element moved by exactly one bf16 ULP, which is 1/16 of an e4m3 quantum.
  Elements that moved at least half a quantum: 0.41% / 0.000% / 0.59%. Max |dw| over all sampled tensors 3.4e-5,
  under the coherent bound 40 * lr = 4e-5.
- AdamW normalized update at step 40 is p50 0.15, p99 0.53 (routed experts p50 0.067). One bf16 ULP at median |w|
  needs about 60-120 fully coherent steps; one e4m3 quantum needs 1000-2000 coherent steps (6500-13000 at the
  observed rate). Extrapolated to step 150 even fully coherently: 15-33% one ULP off, 0.8-1.2% a full quantum
  (routed experts, 98% of FP8-scope parameters: 0.000%).
- Corollary from F27/F28: the release is on the grid only under its own power-of-two block scales. Production
  uses amax/448 scales, which already inject full e4m3 rounding error at step 0. Power-of-two scales
  (`PRIME_DIAG_UE8M0_WEIGHTS=1`) are bit-exact at step 0 and stay within 1/16 quantum through step 150.
- Side facts: all 43 layers are MoE (no dense MLP; the shared expert is the analog); the indexer is frozen in the
  trainer (no optimizer state, zero elements moved) and ignored by vLLM's quantization; 59% of embedding rows never
  updated by step 40.
- Consequence: FP8 mismatch growth must come from something other than weight drift off the grid. Hypothesis 8
  (DeepSeek-specific growth, FP8 as amplifier) gains weight, but the bf16 control's plateau at 0.0009 through step
  72 is not showing that growth yet.

## Track C: fixes

### C2 fix-test config prepared (05:05, not submitted)

`configs/advanced/deepseek-v4-flash/swe-fp8-e8m0.toml`: copy of `swe.toml` with `VLLM_USE_DEEP_GEMM_E8M0 = "1"`,
run name `dsv4-swe-131k-fp8-e8m0`, wandb `swe-scaleswe-131k-fp8-e8m0-adamw1e-6-bs64g8-8t8i`, tags include
`e8m0`, `fix-test`. Dry run clean, all three resolved JSONs re-validate OK, no ckpt. Launch:
`PRL_OUTPUT_DIR=/home/garrett/prl_output_dir uv run rl @ configs/advanced/deepseek-v4-flash/swe-fp8-e8m0.toml`
(non-interactive shells do not source `~/.localrc`, so the variable must be explicit). Attempt 1 of that run dir
was consumed by the dry run. A stray gitignored `outputs/dsv4-swe-131k-fp8-e8m0/` in the worktree came from a
dry run without the variable; left for Garrett to delete.

### C2 launched (05:43): job 991, the recommended FP8 config under real training

Success criteria: mismatch KL at steps 1-40 well below the FP8 baseline band 0.020-0.032 (bf16 control 0.0005-0.0008
at the same steps), `mismatch_kl/all/max` well below 48 (bf16 max about 3), no glitch ids sampled with a large gap.

### C2 results (job 991)

| step | fix-test kl_mean | kl_max | is_masked | FP8 baseline kl_mean (max) | bf16 control kl_mean (max) |
|---|---|---|---|---|---|
| 1 | 0.00149 | 1.48 | 2.7e-6 | 0.0249 (370) | 0.00054 (0.77) |
| 2-7 | 0.00153-0.00201, mean 0.00176 | <= 3.43 | <= 3.2e-5 | 0.0232-0.0318 | 0.00051-0.00061 |
| 8-14 | 0.00182-0.00246 | <= 3.58 | <= 8.0e-5 | 0.0229-0.0285 | 0.00056-0.00064 |
| 15-20 | 0.00266-0.00315, monotone | <= 3.92 | <= 1.0e-4 | 0.0226-0.0269 | 0.00059-0.00071 |
| 21-40 | 0.00236-0.00403, mean 0.00327 | <= 4.41 | <= 3.6e-4 | 0.0198-0.0294 | 0.00052-0.00159 |
| 41-60 | 0.00336-0.00508, mean 0.00432 | <= 12.9 (one spike at 42), else <= 7.8 | <= 6.4e-4 | 0.0226-0.0297 | 0.00063-0.00128 |
| 61-80 | 0.0026-0.0030, mean 0.00278 | <= 8.0 | | | | 0.00656 (61-80) | 0.00096 (61-80) |
| 61-80 | 0.00520-0.00866, mean 0.00656 | <= 26 (76), 16 (67), 14 (79) | <= 2.5e-3 | 0.0216-0.0313 | 0.00058-0.00143 |
| 81-100 | 0.00887-0.01403, mean 0.01187 | <= 57 (95), 52 (87), 47 (100) | 1.8e-3 to 3.6e-3 | 0.0221-0.0338 | 0.00062-0.00165 |
| 101-120 | 0.01252-0.02548, mean 0.01839 | <= 82 (108), 63 (101), 61 (109) | 2.7e-3 to 9.8e-3 | 0.0197-0.0302 | (control killed at 106) |

The 20-step means 0.00220 / 0.00327 / 0.00432 are linear in the step count (+0.00107 per 20 steps, i.e. +0.000054
per step, matching the +0.044e-3 per step from the per-token fit plus the lag term). That linearity is the pinned-policy
signature: the served FP8 weights do not move, so the trainer's distance from them grows with the cumulative update.
Steps 61-80 break the linearity upward (+0.00224 over the previous window versus +0.00107 before), with the IPO masked
fraction rising 6e-4 -> 2.5e-3 and single-step maxima of 12-26. Under the pinning mechanism this is expected once
drift dominates lag: KL between nearby policies scales with the square of the parameter distance, and the trainer's
distance from the frozen served weights grows linearly with steps. The old FP8 run collapsed with `is_masked` near 0.02;
this run is at 0.0025 with reward 0.37-0.83 and entropy 0.21-0.41, so it is left running as evidence of the trajectory.
Steps 81-100 (10:34): mean 0.01187, so the 20-step means now double per window (0.0066 -> 0.0119); step 100 reads 0.0135,
half the FP8 baseline's flat 0.025, and extrapolates past it around step 115. Reward 0.36-0.86, entropy 0.28-0.44,
grad norm 0.007-0.088, no inference errors. Kill rule while unattended: `is_masked/mean` above 0.01 or reward falling
below 0.3 for five consecutive steps; otherwise the run stays up so the trajectory is on record. Without the glitch
floor this run makes the growth mechanism plain: with the served FP8 policy pinned at step 0, the trainer-vs-inference
KL tracks the trainer's distance from the initial checkpoint, which is why the old run also collapsed once the
importance ratios became meaningless.

**Killed job 991 at step 130 (11:23).** Step 120 read 0.0255, level with the FP8 baseline (predicted crossing at
step 115, observed 118), with `is_masked/mean` 0.0098 at the kill threshold; reward was still 0.40-0.71. Continuing would
only have replayed the old run's collapse on 16 nodes. No jobs remain.

Final trace scan (240 traces, 2.15M trained tokens, steps 1-130; `glitch_check_run.py`): glitch tokens 0 at every step
bucket; tokens with |gap| > 20: 0; |gap| > 5: 14 in total (0.7 per 100k; the old FP8 run had 118 per 100k). Per-bucket
mismatch KL 0.0018 (steps 1-10) -> 0.0046 (51-60) -> 0.0109 (81-90) -> 0.0218 (111-120) -> 0.0292 (121-130), with the
p90 |lr| rising 0.074 -> 0.304 and the IPO masked fraction 6e-6 -> 7.4e-3. Inside `<think>` KL is about twice the
all-token value at every bucket; outside `<think>` it climbs 0.0008 -> 0.0038 by steps 71-80. The growth is diffuse
across all tokens, not a tail effect: the kernel glitch is gone for good, and what remains is the served policy being
frozen at step 0.

Step 1 landed at 06:26 after a 14 min first step. The mean is 17x below the FP8 baseline and 2.8x above the bf16
control; the max is 250x below the baseline. Monitor: `uv run python ~/tmp/mismatch_evidence/monitor_961.py <since>`
(now pointed at this run). Trace glitch check: `uv run python ~/tmp/mismatch_evidence/glitch_check_run.py
/home/garrett/prl_output_dir/dsv4-swe-131k-fp8-ue8m0` (validated: FP8 run 31.9 glitches per 100k trained tokens,
KL 0.031; bf16 control 0 glitches, KL 0.0010).

Trace scan at 06:58 (80 traces, 584,665 trained tokens, steps 1-10): glitch tokens 0, tokens with |gap| > 5: 0,
|gap| > 20: 0; mean |lr| 0.0216 (bf16 control 0.0179, FP8 run 0.0688); mismatch KL 0.00178; IPO masked 5e-6.
Inside `<think>` mean |lr| 0.037 (bf16 control's think-region KL was 0.0015 vs this run's 0.0029); outside 0.009.
The FP8 run's same-step numbers were 32 glitches and 118 large-gap tokens per 100k. Steps take about 4-5 min
while the rollout pipeline ramps (off_policy 0-1.6).

Steps 8-20 (07:34): the mean climbs steadily from 0.0020 to 0.0031, a doubling since step 1, while off_policy lag
ramps 1.2 -> 5.2 and `is_masked/mean` rises 1e-5 -> 1e-4. The FP8 baseline was flat at 0.025 over the same steps
because glitch tokens set its floor; its tail-excluded bulk KL grew 0.005 (steps 1-10) -> 0.056 (230-239). The bf16
control did not climb over steps 1-20 (0.0005-0.0008) at similar lag, and B2 showed bf16 broadcast weights are
mostly bit-identical to initial at step 40, so weight-version lag alone should not move bf16 logprobs; an FP8-specific
sensitivity to small weight changes, or an episode-mix effect as longer episodes start completing, are the candidates.
This is the late-growth question from the handoff showing up early without the glitch floor. Analysis of KL versus
lag, episode length and think fraction on this run's traces started 07:36.

### Growth analysis on job 991 (07:48; `~/tmp/mismatch_evidence/growth_analysis.md`)

Per-token lag from the `watcher/policy_version` gauge matched to each call's start time (lag = ship step - 1 -
version). FP8 fix-test: steps 1-22, 11.2M trained tokens; bf16 control: steps 1-30, 15.2M tokens. Token-weighted
per-step KL reproduces the trainer's `mismatch_kl/all/mean` to the last digit.

| lag | FP8 fix-test KL | bf16 control KL |
|---|---|---|
| 0 | 0.00173 | 0.00051 |
| 1 | 0.00189 | 0.00055 |
| 2 | 0.00217 | 0.00058 |
| 3-4 | 0.00254 | 0.00064 |
| 5-8 | 0.00311 | 0.00087 |
| 9+ | 0.00370 | 0.00131 |

Within lag 1, by step bucket (1-5 / 6-10 / 11-15 / 16-20 / 21+): FP8 0.00160 / 0.00194 / 0.00199 / 0.00231 / 0.00240;
bf16 0.00055 / 0.00055 / 0.00054 / 0.00054 / 0.00054. On-policy (lag 0) FP8 climbs 0.00151 -> 0.00270. Outside
`<think>`, where lag has no effect, FP8 climbs 0.00078 (step 1) -> 0.00135 (step 20) while bf16 sits at 0.00027-0.00033
for 30 steps. Joint fit (KL x 1e-3): FP8 lag +0.124 per lag step and +0.044 per training step; bf16 +0.066 per lag
step and -0.003 per training step. Length and position add nothing once lag is held fixed; think fraction is noise.

**Mechanism for the FP8-specific per-step drift (the handoff's growth question).** The checkpoint is a dequantized
FP8 release, so every FP8-scope weight sits exactly at the centre of an e4m3 bin. Online quantization maps a weight
to a different FP8 value only after it moves at least half a quantum (8 bf16 ULPs at the median |w|, about 1000-2000
coherent AdamW steps at lr 1e-6). B2 measured the fraction that had done so by step 40: 0.41% of attention weights,
0.59% of shared-expert weights, 0.000% of routed-expert weights (98% of FP8-scope parameters). So the FP8 server keeps
serving the initial policy while the trainer's bf16 weights move (7-12% of elements by step 40). The trainer scores
rollouts from a policy that is effectively frozen at step 0: mismatch grows like a lag equal to the step count, which
is what the on-policy FP8 column shows (+0.044e-3 per step, the same order as the measured per-lag-step slope), and
it is diffuse over all tokens, not glitch-structured. It predicts the old FP8 run's trajectory (glitch floor 0.025
plus a slow diffuse climb reaching 0.045-0.05 by step 200, then the collapse from training on ever-staler rollouts,
with `is_masked` rising 25x), and it predicts GLM's flat curve: GLM-4.5-Air is a genuine bf16 checkpoint whose
weights sit at random offsets inside their bins, so a fraction proportional to the drift flips each step and the
served model tracks the trainer in expectation. bf16 serving has no quantum and tracks exactly. Checked 07:50
(`~/tmp/mismatch_evidence/glm_grid_check.py`): 21 GLM-4.5-Air tensors including routed experts have 93.7% of elements
with nonzero low-4 mantissa bits and about 2600 distinct values per 128x128 block, versus 0% and about 20 for the
DeepSeek release, so GLM's checkpoint is native bf16 and starts off-grid as the mechanism requires. Power-of-two weight
scales make the pinning exact; amax/448 scales rotate the grid so a little dithering occurs, at the cost of full
rounding error at step 0.

Consequences: (1) online FP8 serving of an on-grid checkpoint at lr 1e-6 is off-policy by construction, and the
mismatch it produces is not a numerics bug but a resolution floor; (2) shipped fixes (UE8M0 scales, exact weight
scales) remove the catastrophic glitch and lower the step-0 gap 17x, but do not stop the drift; (3) candidate
remedies, none tested tonight: stochastic (unbiased) rounding of weights at each broadcast so the served model
tracks the trainer in expectation (raises the per-step noise floor to roughly a generic FP8 checkpoint's level, GLM's
0.004-0.007, but removes growth), a larger learning rate, or bf16 serving. Direct test: save one checkpoint at
step N, quantize it with the production path, and count elements whose FP8 value differs from the initial FP8
weights (prediction: about 0 for routed experts); and two-server KL of bf16(step N) vs FP8(step N) should equal
bf16(step 0) vs bf16(step N).

### C1 (trainer floor fixes, fp32 RoPE from PR 3584 and fp32 logits in the LM head): not run

The 5-node lr=0 profile would put the total at 21 nodes with job 991 running, above the under-20 rule, and the
glitch finding made the inference-side fix the priority for the single 16-node slot. Both remain one-line-ish
changes with the plan and code pointers in the handoff; they address the both-runs floor (0.0005-0.001), not the
FP8 gap.

## Commits made tonight

- `feat(configs): add the FP8 SWE run variant with UE8M0 DeepGEMM scales` (05:20).
- `feat(inference): add fp8_ue8m0_weight_scales for power-of-two online FP8 weight scales` (05:42). Promotes the
  `PRIME_DIAG_UE8M0_WEIGHTS` diagnostic to `inference.fp8_ue8m0_weight_scales` (env `PRIME_FP8_UE8M0_WEIGHT_SCALES`).
- `feat(configs): test the FP8 SWE run with UE8M0 scales and exact weight scales` (05:42). Renames the run config
  to `swe-fp8-ue8m0.toml` and turns the new field on.
- `fix(inference): log the UE8M0 weight-scale patch under the vllm logger namespace` (05:52).
- `docs(mismatch): ...` log commits after each round.

- `feat(inference): add PRIME_DIAG_FAKE_QUANT_IGNORE regex to the fake-quant diagnostic` (03:45). Needed so S2
  can skip `.*indexer.*` like production, and so weight-only rounding can be bisected by module family.

## fp8/fp8 (trainer FP8 too) preparation, 2026-09-21 evening

- Configs committed on `feat/ds-v4-fp8-rl`: `rl_fp8_fp8.toml` (5-node lr = 0 probe, o_a bf16 on both sides) `0c62bfec7`
  and `swe-fp8-fp8.toml` (16-node run) `36092f0c1`; trainer `quantization.type = "fp8"` with eleven ignore patterns
  (defaults plus `o_a_proj`, `indexer\.`, `compressor\.`) and `moe.compute.type = "deepgemm_fp8"`; inference
  `fp8_ue8m0_weight_scales = false`, E8M0 off. Plan of record: `~/.claude/plans/read-mismatch-handoff-md-and-summarize-radiant-lighthouse.md`.
- Quantizer parity measurement (`~/tmp/mismatch_evidence/fp8_parity/results.md`, one GPU, 4 min): weights are already
  bit-identical between the trainer's `per_block_cast_to_fp8_triton` and vLLM's `per_block_cast_to_fp8` on four real
  tensors (100% scales and bytes; minimum block amax 0.0625, so the 1e-4 floor placement never triggers). Activations:
  the trainer matches vLLM's Triton kernel exactly in the normal regime, but production uses the CUDA op
  `torch.ops._C.per_token_group_fp8_quant`, which computes the scale with correctly rounded IEEE division; the trainer
  (and vLLM's own Triton kernel, and torch eager) compute `amax * (1/448)`, 1 fp32 ULP off in about 59% of groups,
  flipping about 0.1% of activation elements by one e4m3 step. Dequantization error is identical to 1e-6 between
  recipes. Consequence for the `fix/fp8-quant-parity` branch: leave the weight kernels alone (they already match);
  in the two per-token activation kernels clamp amax at 1e-4 like vLLM, compute the scale with correctly rounded
  division (`tl.math.div_rn`), and clamp the output to [-448, 448], so the trainer matches the production CUDA op.

### fp8/fp8 probe (job 1103, `rl_fp8_fp8.toml`, 5 nodes, 21:17-22:19), BEFORE the quantizer-parity fix

Trainer FP8 on DeepSeek V4 Flash works: "Replaced 301 linear layers with FP8 blockwise linear (skipped 253 by name,
0 by 128-divisibility)" and "Configured 43/43 MoE layers with compute=deepgemm_fp8", first forward clean (both sides
bf16 for o_a in this probe). 20 lr = 0 steps on reverse-text at 2048 tokens: mismatch KL per step 0.0054, 0.0095,
0.0053, 0.0107, 0.0097, 0.0078, 0.0081, 0.0072, 0.0061, 0.0056, 0.0091, 0.0126, 0.0062, 0.0105, 0.0103, 0.0114, 0.0138,
0.0103, 0.0098, 0.0096; mean 0.0089, median 0.0096. Baselines on the same profile (`FP8_MISMATCH_RESULTS.md`): FP8
serving with a bf16 trainer 0.0307, bf16 on both sides 0.0015. So fp8/fp8 removes about 70% of the old FP8 gap but
sits about 6x above the bf16 floor; candidates for the residual are the server's Triton FP8 experts below 128 tokens
versus the trainer's DeepGEMM contiguous kernel (decode-heavy reverse-text), the 0.1% one-step activation-scale flips
the parity fix addresses, and the indexer's top-k discontinuity amplifying any kernel difference. Rank 0 logged a CUDA
allocator mapping failure with 5 MB free (the 4-trainer-node lr = 0 configs are memory-tight; steps continued). This
is the "before" for `fix/fp8-quant-parity`; re-run the same config after the fix for the "after".

### fp8/fp8 16-node run (job 1115, `swe-fp8-fp8.toml`, started 22:34)

Startup: "Replaced 301 linear layers with FP8 blockwise linear (skipped 253 by name, 0 by 128-divisibility)",
"Configured 43/43 MoE layers with compute=deepgemm_fp8"; inference `fp8_per_block`, E8M0 off, no power-of-two
weight-scale patch (both sides on `amax / 448`), clamp patch active, `wo_a` FP8 on the server only (the TODO).

| step | fp8/fp8 kl_mean | kl_max | is_masked | reward | peak mem | control (job 991) | bf16 control |
|---|---|---|---|---|---|---|---|
| 1 | 0.00212 | 1.22 | 2.1e-5 | 0.580 | 74.0 GiB | 0.00149 | 0.00054 |
| 2-20 | 0.0021-0.0038, mean 0.00266 (1-20) | <= 22.7 (15), else <= 8.9 | <= 3.0e-5 | 0.32-0.84 | | 0.00220 (1-20) | 0.00061 (1-20) |
| 21-40 | 0.0023-0.0032, mean 0.00265 | <= 12.2 (36), else <= 4.3 | <= 2.3e-5 | 0.30-0.74 | | 0.00327 (21-40) | 0.00084 (21-40) |
| 41-60 | 0.0023-0.0030, mean 0.00263 | <= 20.1 | | | | 0.00432 (41-60) | 0.00090 (41-60) |

Steps 2-20 (00:08): flat within noise; steps 17-20 read 0.0026-0.0028 where the control read 0.0027-0.0031 and
then kept climbing. Anomaly to investigate separately: the trainer's `optim/grad_norm` is 0.004-0.018 in this run
versus 0.03-0.09 in the control at the same steps, about 5x lower. The FP8 backward (dgrad and wgrad via the
`(1,1,128)` recipe in `fp8_linear.py` / `fp8_grouped_gemm.py`) is the only trainer-side difference; Adam's
normalisation keeps update sizes unaffected, but gradient direction quality could differ. Offline check to do: one
batch, FP8 versus bf16 backward, per-layer gradient norm and cosine.

Steps 21-40 (01:04): mean 0.00265, indistinguishable from steps 1-20 (0.00266); the control's same window was
0.00327 and its next 0.00432. Both-sides-FP8 with the rotated `amax / 448` grid is tracking as predicted so far.

Step 1 at 23:13 after a 20 min first step. Grad norm 0.006 (control 0.04 at step 1); worth watching. Comparison targets:
job 991's 20-step means 0.0022, 0.0033, 0.0043, 0.0066, 0.0119 (pinned served weights, growth) and job 1047's
0.0018, 0.0021, 0.0022, 0.0023 (stochastic rounding, flat). Prediction for fp8/fp8: flat, since both sides quantize
the same bf16 weights with the same recipe and the rotated `amax / 448` grid lets both track the masters.

### Quantizer-parity fix landed (02:30, `fix/fp8-quant-parity` 4dc088220)

`fix(trainer): match vLLM's per-token FP8 activation quantization bit for bit` (23 lines in
`src/prime_rl/trainer/models/kernels/fp8_utils.py`, 22 in `tests/unit/train/models/test_fp8_utils.py`), implemented in
Garrett's `git-tree-plan-impl` session from the worktree's `PLAN.md`. Stack now: `main` -> `fix/fp8-quant-parity` ->
`feat/ds-v4-bf16-rl` (rebased onto it 02:35, clean) -> `feat/ds-v4-fp8-rl` (rebase deferred until job 1115 ends,
because that job runs from this worktree and the fix changes a Triton kernel source) -> `exp/ds-v4-fp8-dither` (still
frozen). The "after" lr = 0 probe (`rl_fp8_fp8.toml`, expect the 0.0089 "before" to move by the 0.1% one-step
activation flips at most) runs from this worktree once rebased. Separate branches `feat/fp8-grouped-linear` and
`feat/fp8-ue8m0-weight-scales` off main have appeared from other sessions and are not touched here.

**fp8/fp8 result (job 1115 completed 100 steps, 03:22).** 20-step means 0.00266, 0.00265, 0.00263, 0.00278, 0.00261: flat
to three digits for 100 steps, versus 0.00220 -> 0.01187 for the FP8-serving / bf16-trainer control (job 991) and the
bf16 control's 0.00061 -> 0.00101. With both sides quantizing the same bf16 weights with the same `amax / 448` recipe,
the served and trained forwards agree to a constant floor and track together as the masters move; the pinning-driven
growth is gone, as with stochastic rounding (job 1047: 0.0018, 0.0021, 0.0022, 0.0023). Floor about 1.2x the control's
step-1 value (kernel differences: Triton experts below 128 tokens on the server, `wo_a` FP8 on the server only, the
0.1% activation-scale flips the parity fix addresses). Reward 0.30-0.84 and entropy 0.28-0.47 like the controls; the
trainer's gradient norm stayed 3-5x below the bf16-trainer runs throughout (open item). Peak trainer memory 74 GiB
versus about 90 GiB in bf16.

### fp8/fp8 probe AFTER the parity fix (job 1181, `rl_fp8_fp8.toml`, 03:26-03:56)

Same config and code as the "before" probe plus `fix/fp8-quant-parity` 4dc088220. 20 lr = 0 steps: 0.0045, 0.0109,
0.0093, 0.0098, 0.0073, 0.0170, 0.0055, 0.0072, 0.0084, 0.0061, 0.0080, 0.0095, 0.0092, 0.0063, 0.0165, 0.0101, 0.0105,
0.0058, 0.0101, 0.0093; mean 0.0091 (before 0.0089), median about 0.0093 (before 0.0096). No measurable change, as
expected: the fix moves about 0.1% of activation elements by one e4m3 step, well below this profile's per-step noise.
The fix's value is bit-level agreement with vLLM's production per-token quantizer (verified by byte identity in the
fix session), not a KL shift; the 0.009 fp8/fp8 floor on this profile is dominated by kernel-level differences
(Triton experts below 128 tokens on the server versus DeepGEMM contiguous in the trainer, and whatever the indexer's
top-k discontinuity amplifies), not by quantizer arithmetic.
