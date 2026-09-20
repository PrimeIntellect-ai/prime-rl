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
- 03:25 Launched: A0 scoring-set builder, A1 server infrastructure (2 nodes: S0 bf16 reference, S1 FP8
  production), B2 e4m3 grid-departure analysis on `step_40`.

## Track A: glitch-token and module bisection

(pending)

## Track B: bf16 control (job 961) and weight drift

| step | kl_mean | kl_max | is_masked | grad_norm | note |
|---|---|---|---|---|---|
| 1-20 | 0.00051-0.00084 | <= 3.4 | 0 | 0.03-0.09 | from handoff-time extraction |
| 21-45 | 0.00052-0.00159 | <= 3.0 | <= 1.2e-4 | 0.02-0.13 | mean 0.00087, 1.4x steps 1-20 |
| 46-60 | 0.00063-0.00128 | <= 3.9 | <= 4.9e-5 | 0.005-0.073 | mean 0.00090 |
| 61-65 | 0.00067-0.00083 | <= 2.9 | <= 9.6e-5 | 0.04-0.07 | mean 0.00078; plateau, not a climb (03:35) |

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
