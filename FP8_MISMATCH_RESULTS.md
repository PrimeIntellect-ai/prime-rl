# Why online FP8 costs DeepSeek V4 Flash its mismatch KL: results

Answers the investigation set out in `FP8_MISMATCH_PLAN.md`. Experiments ran on 8xH200 single
nodes against `PrimeIntellect/DeepSeek-V4-Flash-0731-bf16`, from repo state `d42a8ea0d` plus two
measurement-only commits (`5a89723c0`, `c6e1e0e1f`) that are env-gated and inert by default.
Raw notes and every number: `~/tmp/fp8diag/findings.md`.

## Answers to the definition of done

### 1. A1: the reload path is exonerated

Scoring identical token sequences on a bf16 server and an FP8 server, with no trainer, no NCCL
broadcast and no weight reload anywhere in the loop, reproduces the production mismatch. On the
math environment it reproduces it almost exactly:

| statistic | bf16 vs FP8 servers | production FP8 math (lag 2) |
|---|---|---|
| mean mismatch KL | 0.00491 | 0.00681 |
| floor (unmasked mean) | 0.00486 | 0.00591 |
| masked fraction | 0.0132% | 0.0173% |

48 sequences, 15188 completion tokens, sampled from the FP8 server at temperature 1.0 and scored on
both at concurrency 1. `FP8_MISMATCH_PLAN.md` pre-committed to "near `0.03` exonerates reload, near
`0.001` implicates it". Quantization alone accounts for the whole number, so **reload is
exonerated**. Two independent lines agree: the reload path structurally cannot re-quantize an
already-quantized weight, because `restore_layer_on_meta` re-registers the boot-recorded meta bf16
parameters first; and the only accumulating operation, `requant_weight_ue8m0_inplace`, is gated off
by the `VLLM_USE_DEEP_GEMM_E8M0 = "0"` pin both configs set.

An audit of every run's inference log found `Failed to load weights` 336 times per run, but always
for `DeepseekV4ScalingRotaryEmbedding`, a rope cache with no learned weights, and identically in
bf16 and FP8 runs. No linear, MoE or attention layer ever failed to load.

### 2. No upstream reproduction is owed for reload

A1 did not implicate it, so this item is void.

### 3. A2: attribution, with per-component numbers

Each variant prefill-scored against the bf16 reference at concurrency 1, which is bitwise
deterministic (see "measurement" below).

| variant | reverse-text masked% | math masked% | math p90 |
|---|---|---|---|
| stock FP8 | 3.958% | 0.0132% | 3.990e-03 |
| **weights only** (bf16 kernels, fake-quant round trip) | 3.142% | 0.0066% | 2.915e-03 |
| FP8, bf16 `o_proj` | 3.678% | 0.0198% | 3.169e-03 |
| FP8, indexer excluded | 3.958% | 0.0132% | 3.897e-03 |
| FP8, `linear_backend=deep_gemm` | 3.958% (bitwise identical to stock) | - | - |
| FP8, `moe_backend=flashinfer_cutlass` | 4.047% | - | - |
| **FP8, power-of-two weight scales** | **2.952%** | **0.0000%** | **1.766e-03** |

Means are omitted deliberately: the per-token KL distribution is heavy-tailed enough that a single
token sets the average (see "measurement notes").

**Weight quantization is the dominant term.** Quantizing only the weights and serving them on the
completely untouched bf16 kernel stack reproduces most of the effect; activation quantization, the
linear kernel, the MoE kernel and the `o_proj` path are all second order, and the two levers
`FP8_MISMATCH_PLAN.md` hoped would separate the bundle both measure at zero effect.

That interacts with a known property of the checkpoint: `PrimeIntellect/DeepSeek-V4-Flash-0731-bf16`
is an upconversion of `deepseek-ai/DeepSeek-V4-Flash-0731` done to make training easier, so every
weight vLLM re-quantizes already sits exactly on the e4m3 grid. Measured: 100.00% of sampled
elements in quantized tensors carry only 4 significant mantissa bits, against 6.3% for
`embed`/`head`, which vLLM leaves alone and which are genuine bf16. The source repo mixes FP8 and
MXFP4 tensors, which does not matter here, since MXFP4 is e2m1 with power-of-two scales and e2m1
carries fewer mantissa bits than e4m3, so both families land on the e4m3 grid.

The consequence is that vLLM's scale `amax / 448` is not a power of two, so it rotates the grid and
injects rounding error on values that would otherwise round-trip bit-exactly (confirmed: ue8m0
round-trip error is exactly 0.0, `amax/448` is ~2.7% relative, on both attention and expert
tensors).

Forcing power-of-two weight scales is accordingly the best variant measured, halving math p90 and
taking the masked-token count to zero. It is **not** the full collapse that exact round-tripping
would predict (residual ~45% of p90 rather than ~4%), and the reason is now established rather than
guessed: the ue8m0 round-trip is bit-exact (max abs error exactly 0.0) on routed-expert weights as
well as linears, and all three call sites use the patched module-global with `use_ue8m0` passed by
keyword, so the server really was holding bit-exact weights. The residual is activation
quantization plus discrete-selection instability.

**That falsifies reading the ablation table as additive shares.** Making the weights exact does not
remove the share the weights-only variant appeared to carry. When the dominant mechanism is
near-tied discrete selections, removing one perturbation source mostly relocates which ties fall
which way. Read each row as "this variant's total distance from bf16", not as a contribution that
sums.

Note also that power-of-two scales are only free while weights sit on the grid; once training moves
them off it, they cost range rather than saving it.

### 4. Recommendation

**Online `fp8_per_block` is fit for DeepSeek V4 RL serving.** The FP8 math run passed the 0.015
merge bar on all 20 steps (mean 0.00625, max 0.01388) with reward indistinguishable from bf16
(0.9875). Reverse-text fails the bar, but it is not a representative gate: its *bf16* floor is
already 3.3x higher than math's, and in direct measurement its catastrophic-token rate is 300x
math's (3.96% vs 0.0132%). Gate FP8 on a realistic environment.

**Do not pursue A3 (trainer-side `quantization`).** The gap is not a systematic weight-precision
offset that matching precision would cancel; it is discrete selection instability (below). Matching
precision on both sides would not address the mechanism.

**Config changes worth making:**
- Nothing for the current workloads. `linear_backend` and `moe_backend` are measured no-ops.
- For RL with sequences beyond ~2048 tokens, add
  `[inference.vllm] quantization_config = {ignore = ["re:.*indexer.*"]}`. See defect 2.
- Power-of-two weight scales are worth following up as an upstream option, but not shipping on
  these numbers: the benefit is specific to a checkpoint whose weights are still on the e4m3 grid,
  which stops being true after the first optimizer step.

**Next experiment, if this continues.** Nothing here was validated end to end in an RL run, because
no config-level fix survived measurement. The one worth an RL run is the indexer exclusion, and
only on a long-context environment where it is not a no-op.

## Per-step mismatch KL, in the format of the bf16 PR (#3543)

Same model and topology as #3543: 4 trainer nodes + 1 inference node, 20 steps, `lr = 0`, full
depth, router replay on. The bf16 columns reproduce #3543's own measurements (it reported max
0.00040 on math with replay on, and 0.0016 mean / 0.0027 max on reverse-text), which makes these
runs directly comparable to it.

### math, `batch_size = 64`

| step | bf16 | online FP8 | lag bf16 / FP8 |
|---|---|---|---|
| 1 | 0.00032 | 0.00667 | 0 / 0 |
| 2 | 0.00037 | 0.00551 | 1 / 1 |
| 3 | 0.00039 | 0.01388 | 2 / 2 |
| 4 | 0.00032 | 0.00232 | 3 / 3 |
| 5 | 0.00028 | 0.00795 | 2 / 2 |
| 6 | 0.00028 | 0.00633 | 2 / 2 |
| 7 | 0.00035 | 0.00357 | 3 / 2 |
| 8 | 0.00032 | 0.00194 | 2 / 2 |
| 9 | 0.00028 | 0.01246 | 2 / 2 |
| 10 | 0.00038 | 0.00490 | 3 / 3 |
| 11 | 0.00029 | 0.00449 | 3 / 3 |
| 12 | 0.00040 | 0.00798 | 3 / 2 |
| 13 | 0.00026 | 0.00333 | 2 / 3 |
| 14 | 0.00033 | 0.00352 | 2 / 2 |
| 15 | 0.00031 | 0.01098 | 3 / 2 |
| 16 | 0.00046 | 0.01148 | 3 / 3 |
| 17 | 0.00036 | 0.00889 | 4 / 3 |
| 18 | 0.00032 | 0.00246 | 3 / 3 |
| 19 | 0.00032 | 0.00214 | 2 / 2 |
| 20 | 0.00034 | 0.00414 | 2 / 2 |

All 40 entries under the 0.015 bar: max 0.00046 bf16, **0.01388 online FP8**. FP8 costs 19x the
bf16 mismatch and leaves 8% headroom against the bar.

### reverse-text, `batch_size = 32`

| step | bf16 | online FP8 | lag bf16 / FP8 |
|---|---|---|---|
| 1 | 0.00061 | 0.03801 | 0 / 0 |
| 2 | 0.00106 | 0.02151 | 1 / 1 |
| 3 | 0.00114 | 0.02435 | 2 / 2 |
| 4 | 0.00157 | 0.03105 | 3 / 3 |
| 5 | 0.00180 | 0.02843 | 4 / 1 |
| 6 | 0.00210 | 0.03454 | 5 / 2 |
| 7 | 0.00072 | 0.01865 | 2 / 2 |
| 8 | 0.00133 | 0.02574 | 2 / 2 |
| 9 | 0.00140 | 0.02873 | 3 / 2 |
| 10 | 0.00100 | 0.04190 | 4 / 2 |
| 11 | 0.00112 | 0.03550 | 3 / 2 |
| 12 | 0.00201 | 0.04578 | 4 / 2 |
| 13 | 0.00154 | 0.03430 | 5 / 2 |
| 14 | 0.00174 | 0.02528 | 4 / 2 |
| 15 | 0.00132 | 0.03099 | 4 / 2 |
| 16 | 0.00326 | 0.02057 | 5 / 2 |
| 17 | 0.00180 | 0.03565 | 3 / 2 |
| 18 | 0.00089 | 0.02995 | 3 / 2 |
| 19 | 0.00141 | 0.02867 | 4 / 2 |
| 20 | 0.00220 | 0.03390 | 4 / 2 |

**All 20 FP8 steps are over the bar.**

### summary

| configuration | mismatch_kl | median | min | max | over 0.015 | is_masked |
|---|---|---|---|---|---|---|
| math, bf16 | 0.00033 | 0.00032 | 0.00026 | 0.00046 | 0 of 20 | 0.0000% |
| math, online FP8 | 0.00625 | 0.00520 | 0.00194 | 0.01388 | 0 of 20 | 0.0120% |
| reverse-text, bf16 | 0.00150 | 0.00140 | 0.00061 | 0.00326 | 0 of 20 | 0.0249% |
| reverse-text, online FP8 | 0.03068 | 0.03047 | 0.01865 | 0.04578 | **20 of 20** | 1.9395% |

Mismatch KL depends on off-policy lag, so the lag columns are included. Lag-matched at lag 2, the
best-populated bucket, FP8 costs 21.2x on math (n=11) and 29.0x on reverse-text (n=16), and the
3.2x gap between the environments is present in the bf16 runs too (0.00106 vs 0.00032), i.e. it is
environment-intrinsic rather than an FP8 effect.

Reward is unaffected: reverse-text 0.8644 FP8 against 0.8576 bf16, math 0.9875 FP8.

## The mechanism, which is not what the plan assumed

The framing "which part of FP8 is lossy" understates what is happening. **DeepSeek V4 Flash's
forward pass is dense with near-tied discrete selections, and any numerical perturbation reshuffles
them.** Four independent measurements, none of which vary precision:

- Swapping only the **MoE kernel** between two FP8 servers (identical weights, identical
  quantization) reroutes **33.7%** of (token, layer) expert selections. Quantization reroutes 40.9%,
  barely more.
- Two **bf16** servers with identical weights, differing only in which GEMM implements the mHC
  pre-norm, disagree on **1.29%** of completion tokens by more than 0.3 in probability.
- **Greedy decoding is not reproducible across batch sizes** on any variant: 31/48 sequences diverge
  on bf16, 35/48 on FP8, 39/48 on FP8 with the M<32 branch removed. Rollout content is a function of
  server load.
- **Request length changes the logprobs of earlier positions**, up to 24% of prefix positions and
  3.5 nats, despite attention being causal. Two servers running identical kernels and differing only
  in weight values gave 76% and 100% agreement on the same test, so it is a knife-edge effect rather
  than a property of one code path.

prime-rl already neutralises the largest channel: `trainer.enable_router_replay = true` makes the
trainer reuse inference's expert choices, cancelling the routing divergence entirely. That is why
production's 0.031 on reverse-text sits below the 0.066 floor of an unreplayed two-server
comparison, and it is the single most important thing the config gets right.

## Defects found

1. **`o_proj`'s activation quantizer ignores a config pin.**
   `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:110-121` computes
   `scales = exp2(ceil(log2(block_absmax / fp8_max)))` unconditionally, so it uses power-of-two
   scales regardless of `VLLM_USE_DEEP_GEMM_E8M0 = "0"`, which both DS-V4 configs set and every
   other linear honors. DeepSeek-V4-only code path. Real bug; measured cost near zero, so file it
   without urgency.
2. **The Lightning Indexer is a genuine error source past ~2048 tokens.** Stock FP8 versus FP8 with
   `indexer.wq_b` excluded is **bitwise identical** for every position below 2048 and then steps to
   0.31-0.36% masked tokens, worst KL 47. The boundary is the short-context shortcut
   (`vllm/models/deepseek_v4/attention.py:925-949`) applied per chunked-prefill chunk. Harmless for
   the current configs, which never reach that length; a one-line fix for long-context RL.
3. **`moe_backend = "flashinfer_cutlass"` OOMs** during CUDA graph capture at
   `gpu_memory_utilization = 0.85`, which DEEPGEMM tolerates. Worked at 0.70.
4. **vLLM's quantization `ignore` list silently accepts globs that never match.**
   `_is_equal_or_regex_match` supports exact strings or `re:`-prefixed regexes only, so
   `{"ignore":["*indexer*"]}` matches nothing and produces a run identical to no-ignore, with no
   warning.
5. **`prefill_logprobs` is fragile but currently correct.** `src/prime_rl/orchestrator/clients.py:547`
   takes `next(iter(entry.values()))`, which is id-blind, and vLLM returns two entries whenever the
   rank-1 token differs from the prompt token. Measured over 1794 positions, 34% of them
   multi-entry, the prompt token was first **100.00%** of the time, so nothing is wrong today.
   Selecting by token id would remove the dependence on dict ordering. It feeds on-policy
   distillation (`orchestrator/algo/opd.py:43`, `opsd.py:76`).

## Measurement notes

Scoring is bitwise reproducible only at concurrency 1. The same server scored twice at concurrency 4
gives mean KL 2.7e-3 (max 0.109), which is larger than the entire bf16 production floor; at
concurrency 1 it is exactly 0.0. Every comparison above uses concurrency 1.

Means are unusable on this data: the per-token KL distribution is heavy-tailed enough that one token
at KL 2.3e5 sets the average, and because the IPO mask keys on the absolute probability difference,
a token going from probability 1e-10 to 1e-16 counts as "unmasked" while contributing a log-ratio of
14. Report the masked fraction, the floor, and the median or p90.

Tooling, all outside the repo: `~/tmp/fp8diag/score_servers.py` (two-server scorer reusing the
trainer's own `compute_importance_ratio_and_mismatch_kl`), `routing_diff.py`, `long_score.py`,
`truncation_test.py`, `quant_error_survey.py`, `run_forensics.py`.
