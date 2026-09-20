# Mismatch investigation, night of 2026-09-20

Plan: `~/.claude/plans/read-mismatch-handoff-md-and-summarize-radiant-lighthouse.md`. Background: `MISMATCH_HANDOFF.md`.

Ground rules: one 16-node training run at a time (job 961, the bf16 control, until killed); under 20 nodes total;
checkpointing off in new launches; wandb project `primeintellect/deepseek-v4-flash`; one commit per logical change.

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
- 03:25 Launched: A0 scoring-set builder, A1 server infrastructure (2 nodes: S0 bf16 reference, S1 FP8
  production), B2 e4m3 grid-departure analysis on `step_40`.

## Attribution (as of 05:55)

| cause | evidence | magnitude | fix | status |
|---|---|---|---|---|
| DeepGEMM grouped FP8 MoE GEMM misreads fp32 activation scales at small M (decode steps, prefill tails < ~128 tokens) | rounds 2-5: glitch reproduces only in decode / short tails on FP8 servers; routed-experts-only FP8 reproduces; attention-only and E8M0=1 do not; power-of-two weight scales alone do not remove it | 79% of the FP8 run's step-1 mismatch KL (glitch tokens), 25% late; `mismatch_kl/all/max` 48-3851; the tokens are IPO-masked so they add no gradient but pollute trajectories | `VLLM_USE_DEEP_GEMM_E8M0=1` | validated on 30 positions; under training in job 991 |
| Online FP8 weight rounding with `amax / 448` scales rotates the checkpoint's power-of-two e4m3 grid | F27/F28; round 5: S3/S10 halve p90 log-ratio vs fp32 scales | bulk prefill KL 5.6e-3 -> 3.5e-3, IPO masked 0.0002 -> 0.0000 | `inference.fp8_ue8m0_weight_scales = true` (new field) | in job 991 |
| Residual FP8 activation quantization (UE8M0 per-128 groups) | S10 vs bf16 reference: KL 3.5e-3, p90 0.086 | about 4-6x the bf16 floor | none tonight; bf16 serving if unacceptable | measured |
| Weight drift off the e4m3 grid (handoff hypothesis 1) | B2: 87-93% of FP8-scope weights bit-identical at step 40, rest one bf16 ULP; > 1000 coherent steps to move a quantum | cannot explain growth at step 150 | none needed | ruled out |
| o_proj UE8M0 activation quant (hypothesis 3) | S4 reproduces the glitch at production strength | not the glitch; bulk effect not separately measured | none | exonerated for the glitch |
| Lightning Indexer top-k flips as amplifier (hypotheses 4, 8) | bf16 self-noise above 2048 tokens reaches several nats on single tokens; bf16 control drifted 0.0006 -> 0.0010 over 106 steps with no acceleration | a slow floor-level drift at most; FP8 growth mechanism still unassigned | fp32 RoPE (PR 3584) would reduce indexer input divergence | open |
| Trainer bf16 logits before fp32 log-softmax (hypothesis 7) | code: `lm_head.py:177`; about 0.1 nat at logit 30 | part of the 5e-4 floor, cannot make a 44-nat gap | `torch.mm(..., out_dtype=float32)` | not run (node budget) |
| Trainer bf16 RoPE tables (hypothesis 2) | PR 3584 applies cleanly | floor contributor | cherry-pick PR 3584 | not run (node budget) |

Open question: what made the FP8 run's mismatch grow 5x after step 150 and collapse at 240. Grid drift is out. The
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
