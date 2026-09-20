# Mismatch investigation, night of 2026-09-20

Plan: `~/.claude/plans/read-mismatch-handoff-md-and-summarize-radiant-lighthouse.md`. Background: `MISMATCH_HANDOFF.md`.

Ground rules: one 16-node training run at a time (job 961, the bf16 control, until killed); under 20 nodes total;
checkpointing off in new launches; wandb project `primeintellect/deepseek-v4-flash`; one commit per logical change.

## Timeline (UTC)

- 03:22 Start. Job 961 at step 63, mismatch KL 0.0008 at step 63. 5 idle nodes. Checkpoints `step_40`, `step_60`
  exist (job runs off its pre-edit resolved config; `keep_last = 2`). Both to be deleted when 961 is killed.
- 03:53 S0 (bf16, node 004, job 978) and S1 (FP8 production, node 007, job 979) healthy; scoring set built
  (50 items, 30 glitch prefixes from steps 1-10, 20 bulk sequences 3k-69k tokens, all glitch joins verified).
- 04:05 B2 done (below). Round 1 scoring S0 vs S1 started.
- 04:35 Round 1 done: glitch not reproduced in prefill on the FP8 server. Round 2 (decode path) started.
- 05:05 Round 2 done: glitch REPRODUCED on the FP8 server's decode path, deterministic argmax, M = 1. Round 3 started.
- 03:25 Launched: A0 scoring-set builder, A1 server infrastructure (2 nodes: S0 bf16 reference, S1 FP8
  production), B2 e4m3 grid-departure analysis on `step_40`.

## Track A: glitch-token and module bisection

Infrastructure: two exclusive one-node `salloc` holds (jobs 978 slot 0 node 004, 979 slot 1 node 007, 8 h, expire
about 11:26). Scripts under `~/tmp/fp8diag/bisect/`: `start_server.sh`, `stop_server.sh`, `release.sh`,
`prefill_topk.py` (top-10 prompt logprobs per position), `bisect_report.py`. Server startup is 22-23 min each
(weights 330-560 s, engine init 550-770 s). vLLM module names for ignore regexes:
`model.layers.N.attn.{fused_wqa_wkv,wq_b,wo_a,wo_b,compressor}`, `model.layers.N.ffn.{gate,shared_experts,experts}`.

| id | serving | env / ignore | status |
|---|---|---|---|
| S0 | bf16 | | up 03:53, stopped 05:10 (prefill results saved in r1.bf16.json) |
| S1 | FP8 production | ignore `.*indexer.*`, E8M0=0, deep_gemm | up 03:52, stopped 05:10 |
| S2 | bf16 + fake-quant weights | `PRIME_DIAG_FAKE_QUANT_WEIGHTS=1 PRIME_DIAG_FAKE_QUANT_IGNORE=.*indexer.*` | queued |
| S3 | S1 + power-of-two weight scales | `PRIME_DIAG_UE8M0_WEIGHTS=1` | queued |
| S4 | S1 + bf16 o_proj | `PRIME_DIAG_BF16_OPROJ=1` | starting 05:10, slot 0 |
| S5 | attention-only FP8 | `servers/s5-attn-only.toml` (ignore `.*ffn.*`) | starting 05:10, slot 1 |
| S6 | ffn-only FP8 | `servers/s6-ffn-only.toml` (ignore `.*attn.*`) | queued |

Round 1 (S0 vs S1, 50 items, 822k tokens): running since 04:05.

### Round 1 result (04:35): static FP8 prefill does NOT produce the glitch

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

### Round 2 result (05:05): the glitch reproduces on S1 through the DECODE path, single request

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
- Round 3 (05:10): S0 and S1 stopped; slot 0 -> S4 (FP8 + `PRIME_DIAG_BF16_OPROJ=1`), slot 1 -> S5 (attention-only
  FP8). Decode probe on both. Meanwhile reading vLLM for the decode/prefill threshold and FP8-only decode kernels.

## Track B: bf16 control (job 961) and weight drift

| step | kl_mean | kl_max | is_masked | grad_norm | note |
|---|---|---|---|---|---|
| 1-20 | 0.00051-0.00084 | <= 3.4 | 0 | 0.03-0.09 | from handoff-time extraction |
| 21-45 | 0.00052-0.00159 | <= 3.0 | <= 1.2e-4 | 0.02-0.13 | mean 0.00087, 1.4x steps 1-20 |
| 46-60 | 0.00063-0.00128 | <= 3.9 | <= 4.9e-5 | 0.005-0.073 | mean 0.00090 |
| 61-80 | 0.00058-0.00143 | <= 6.6 | <= 1.5e-5 | 0.02-0.09 | mean 0.00096 |
| 81-84 | 0.00083-0.00143 | <= 2.5 | <= 8.7e-6 | 0.04-0.08 | mean 0.00105 (04:45); slow creep, about +0.0001 per 20 steps |

### B2: e4m3 grid departure at step 40 (done 04:05; `~/tmp/mismatch_evidence/grid_drift.md`)

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

(pending)

## Commits made tonight

- `feat(inference): add PRIME_DIAG_FAKE_QUANT_IGNORE regex to the fake-quant diagnostic` (03:45). Needed so S2
  can skip `.*indexer.*` like production, and so weight-only rounding can be bisected by module family.
