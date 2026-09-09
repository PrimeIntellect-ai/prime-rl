# Mismatch reproduction progress

The target follows the [mismatch article](https://kiddyboots216.github.io/mismatch/):
FP32 selected-token logprob bytes must agree across trainer scoring, serving
prefill, and decode. A rounded zero K3 value alone is insufficient. The audits
also require generation, shipment, and training to use the same policy version.

Exact-zero sampled-logprob checks have passed all six learning arms: reverse
text and single-turn GSM8K for Qwen3-8B, Qwen3-30B-A3B, and GLM-4.5-Air. The
requested numerical reproduction is **complete**. All three models also passed
frozen short- and long-input checks with live trainer/prefill/decode equality.
The six learning runs cover412,696 sampled tokens; all twelve passing runs cover
533,161. `outputs/mismatch/completion-audit.json` joins their final audits,
resolved configs, update evidence, sampling checks, and scheduler completion
records. These are 20-step numerical
diagnostics, not converged learning curves.
Runs retain FP32 optimization/reduction defaults and use at most one trainer
node plus one inference node.

The six learning runs use temperature1, top-p1, top-k disabled, and min-p0.
`outputs/mismatch/learning-sampling-audit.json` checks these actual request
parameters and one model call per consumed training trace: all3,360 records
pass, covering all20 training steps in each of the six learning runs. Serving
uses `generation_config="vllm"` without checkpoint sampling overrides.

## Completed exact-zero audits

Every row has zero FP32 bit mismatches, zero absolute logprob error, and a passing
generation/shipment/trainer policy-version audit. Counts cover sampled tokens
actually scored in training, rather than every possible token or sequence.

| Model / task | Job | Steps | Sampled tokens | Maximum sequence |
| --- | --- | --- | --- | --- |
| Qwen3-8B frozen reverse | 269 | 3 | 12,536 | 326 |
| Qwen3-8B reverse learning, LR 1e-6 | 271 | 20 | 62,444 | 326 |
| Qwen3-8B GSM8K learning, LR 1e-6 | 273 | 20 | 152,381 | 626 |
| Qwen3-8B frozen long reverse | 277 | 3 | 45,402 | 4,220 |
| Qwen3-30B-A3B frozen reverse | 280 | 3 | 4,637 | 134 |
| Qwen3-30B-A3B reverse learning, LR 1e-6 | 285 | 20 | 26,833 | 302 |
| Qwen3-30B-A3B uniform GSM8K learning, LR 1e-6 | 300 | 20 | 119,124 | 521 |
| Qwen3-30B-A3B frozen long reverse | 301 | 3 | 40,947 | 4,220 |
| GLM-4.5-Air frozen reverse, full CPU offload | 313 | 3 | 4,655 | 124 |
| GLM-4.5-Air reverse learning, LR 1e-6, full CPU offload | 321 | 20 | 27,653 | 151 |
| GLM-4.5-Air uniform GSM8K learning, LR 1e-6, full CPU offload | 322 | 20 | 24,261 | 416 |
| GLM-4.5-Air frozen long reverse, full CPU offload | 323 | 3 | 12,288 | 4,137 |

Job313 completed with a passing full audit and all96 matching policy records.
Four live prefill replays covered107 sampled tokens with exact trainer/prefill/
decode equality. All frozen router probes showed unchanged finite weights.
Trainer GPU peak was53.7GiB; post-step host availability remained around305–311GiB.
This validates the full CPU-offload configuration on short frozen sequences.
GLM reverse learning job321 completed all20 steps with exact logprobs and all640
matching policy records. Router changes were observed at every optimizer step;
steps2–20 verified equality after updates. Math322 also passed its20-step
numerical audit and completed successfully (exit0). Long-input323 completed on
the same022/030 allocation pair with exit0 in59m33s.

Long-input323 completed all three frozen steps:12,288 sampled tokens, all at
positions593 or later, match exactly with24 matching policy records. Maximum
sequence length is4,137; all frozen-router probes are complete, finite, and
unchanged. Four
live prefill replays of1,550-token traces cover2,048 sampled tokens with zero
bit mismatches, absolute error, nonfinite pairs, and stable K3 across all three
trainer/prefill/decode comparisons. All three batches were100% completion-truncated,
so this establishes numerical stress coverage rather than task accuracy. Trainer
GPU peak was54.9GiB. The trainer, orchestrator, and final audit finished normally;
no reproduction jobs remain allocated.

Math322's complete20-step audit covers24,261 sampled tokens and160 matching
policy records with zero raw FP32 bit mismatches or absolute error. The router
changed at steps1–20, and steps2–20 verify exact scoring after updates. Maximum
sequence length is416; the separate completed long-input check supplies longer
position coverage.
Initial evaluation scored62/64 with
thinking disabled; all64 evaluation records used policy0 throughout generation.
The step5, step10, and step15 evaluations scored60/64,58/64, and63/64; all64
records in each stayed on the corresponding policy, without runtime errors
or truncation. Final evaluation scored62/64, equal to the initial score; all64
records stayed on policy20 and ended normally. These small evaluations do not
establish an accuracy improvement. Job322 completed in3h51m39s, before its4h
limit; no time extension was applied.


Job300 completed successfully with a passing launcher audit. All640 policy
records matched. Nonzero gradients occurred at steps2
and15; the router probe observed parameter changes at steps2–20, with exact
subsequent-policy scoring at steps3–20. Other steps can change weights through
Adam momentum despite zero current gradients. The run sampled GSM8K uniformly
with seed42 and disabled thinking in both training and evaluation. It did not
cover positions593 or later; the separate frozen long-sequence job301 passed.
Final evaluation was61/64 correct, equal to its initial evaluation; no accuracy
improvement claim is made. Intermediate evaluations at steps5,10,15 were63/64,
61/64,62/64 respectively.

Job301's96 policy records matched, and all40,947 sampled tokens were at positions
593 or later. Its four live replays covered630 sampled tokens at sequence
lengths3,757/1,194/3,889/1,334; trainer, prefill, and decode agreed bitwise in all
three comparisons. This was a heavily truncated numerical stress check with
frozen weights. It completed successfully before GLM frozen job302 started.

Frozen live prefill replays also matched trainer and decode logprobs: 114 sampled
tokens for job269, 751 for job277, and 98 for job280. Job273's final held-out
GSM8K evaluation was 61/64 correct; no improvement claim is made. Its evaluation
used the server's default thinking-enabled chat template, while training disabled
thinking through the renderer. The same evaluation setting is retained in the
already-launched Qwen MoE math job286. The current math overlay explicitly disables
evaluation thinking for subsequent runs; these evaluation scores are not directly
comparable across the two settings.

The following sections record the investigation and failed candidates. Later
measurements supersede earlier pending-status statements.

## Frozen-weight measurements

| Run / SLURM job | Task | Completed batches | Stable K3 range | Bit mismatch fraction |
| --- | --- | --- | --- | --- |
| `qwen8b-reverse-frozen-drain` / 251 | reverse text | 3 | 0.00139–0.00343 | 81.8–83.9% |
| `qwen8b-reverse-frozen-invariant-nccl` / 253 | reverse text, serving batch invariance | 3 | 0.00120–0.00257 | 83.9–87.5% |
| `qwen8b-math-frozen-sync` / 254 | GSM8K | 1, then starvation failure | 0.000310 | 62.9% |
| `qwen8b-math-frozen-all` / 258 | GSM8K, retain zero-advantage samples | 3 | 0.000198–0.000260 | 52.2–58.1% |
| `qwen8b-reverse-swiglu-sync` / 260 | reverse text, inference SwiGLU, retain zero-advantage samples | 3 | 0.000791–0.00206 | 74.4–84.1% |
| `qwen8b-reverse-dense-align-local` / 267 | reverse text, shared dense forward, BF16 head | 3 | 2.64e-15–3.87e-15 | 26.1–32.7% |
| `qwen8b-reverse-dense-align-fp32-head` / 269 | reverse text, shared dense forward, FP32 head and V2 sampler | 3 | **0 exactly** | **0 exactly** |

All these measurements have zero non-finite logprob pairs. These are small
numerical diagnostics, not learning curves or held-out accuracy results. The
sampling population differs when zero-advantage filtering changes, and separate
runs generated different traces; the table is not a paired effect estimate.

Job 258's audit joins all 96 trained traces to their generation policy spans,
shipment steps, and trainer policy versions. Every record agrees with `step - 1`.
Job 254's 32 trained traces pass the same audit. Earlier reverse-text jobs have
matching generation spans and shipment steps, but lack explicit trainer-version
annotations, so the stricter audit reports missing evidence for those jobs.
Job 258 covered 24,745 sampled tokens, but its longest trained sequence was only
519 tokens. The 8,192-token configuration cap is not evidence of long-position
coverage; positions beyond 593 still require explicit validation.

Four job-254 traces were replayed through the same live frozen-weight server.
Their trainer-versus-prefill K3 ranges from 0.000257 to 0.000578, and their
prefill-versus-decode K3 ranges from 0.000102 to 0.000177. Thus both forward-path
differences remain when optimizer drift is removed. These K3 terms do not add.

## First operator alignment

H200 probe 257 compared identical BF16 tensors. For 3,158,016 SwiGLU elements,
eager PyTorch matched the vLLM CUDA forward, while compiled PyTorch differed on
27.47%. Probe 259 verified that the opt-in inference SwiGLU forward matches
vLLM exactly on those inputs, and both input gradients match eager PyTorch
exactly. This verifies one operator on the tested inputs, not the full model.

The probes also detect shape-dependent matrix-multiplication differences and
small RMSNorm differences. RoPE, attention, and the vocabulary reduction still
need alignment and full-model validation. Qwen30B MoE weights are cached;
GLM4.5-Air has not been run yet. Job 260 completed and its 96 trained traces
pass the policy-version audit. The full model still has nonzero mismatch.
Qwen30B reverse-text job 261 generated its first batch but stalled before the
trainer completed step 1. It was cancelled after 18 minutes with no trainer
mismatch measurement. Probe 262 subsequently completed successfully.

The opt-in `dense-alignment.toml` candidate shares linear, RMSNorm, RoPE,
SwiGLU, FA4, and full-vocabulary log-softmax arithmetic with serving. It disables
trainer compilation and uses eager inference. This candidate is experimental:
configuration validation and GPU gradient checks passed. Probe 262 found exact
FA4 prefill/decode outputs for the last token at lengths 257, 1,024, and 8,192
on identical Q/K/V tensors; its attention gradients passed comparison with an
FP32 PyTorch reference at the probe's BF16 tolerances. These gradient checks
do not require bit equality. Jobs 263 and 264 were stopped during startup after
finding incompatible serving settings: the default FP32-output head bypassed
the shared projection, and vLLM silently downgraded FA4 to FA2 under its batch
invariance flag. The corrected candidate explicitly uses BF16-output heads on
both sides and a Hopper-only FA4 selector override. Probe 265 passed exact
serving RoPE comparisons (41,943,040 query/key elements) and paged attention
comparisons at lengths 257, 1,024, and 8,192. Job 266 confirmed that all eight
serving workers selected FA4, but trainer forward failed on a missing compiled
kernel in the shared Triton cache. Job 267 completed with private per-rank
compiler caches. All 96 trained traces passed the policy-version audit, but
its maximum logprob error was 9.54e-7 and its bit mismatch remained nonzero.

The candidate's BF16-output head is an initial alignment variant; the article
uses an FP32-accumulator head. Matching forward bits in this variant would not
establish reproduction of every numerical choice in the article.

Probe 268 validated the shared FP32-accumulator head: full logits and logprobs
matched exactly across 1, 17, and 128 rows, and both gradients matched the FP32
PyTorch reference. The selected-logprob patch now also covers vLLM's V2 model
runner; its original sampler patch alone did not cover that path. Job 269
(`qwen8b-reverse-dense-align-fp32-head`) completed with exact zero in all three
batches. Independent trace auditing verified 12,536 sampled tokens and all 96
policy-version records. Four live replays covering 114 sampled tokens also
matched exactly between trainer scoring, serving prefill, and decode. The
longest trained sequence was 326 tokens, so this is a frozen, short-sequence
result, not proof of the remaining learning/math/MoE arms.

Learning job 270 kept exact zero for two completed steps at LR 1e-6, then two
trainer ranks received SIGSEGV while the timed diagnostic stack dump was
printing. Job 271 removes that diagnostic hook and retries the same 20-step
experiment. Its scheduler success requires the complete raw-bit/policy audit,
not just a zero process exit code from training.

Job 271 (`qwen8b-reverse-learning-aligned-local`) completed all 20 learning
steps at LR 1e-6. The independent audit verified 62,444 sampled tokens and 640
trained traces: zero FP32 bit mismatches, zero absolute error, zero non-finite
pairs, and matching generation/shipment/trainer policy versions at every step.
The maximum sequence length remained 326. Job 273 runs the same aligned
learning configuration on single-turn GSM8K; job 277 is its dependent frozen
long-input reverse-text check. Neither pending result establishes MoE alignment.

The Qwen30B candidate uses EP1 training and TP1 serving to avoid distributed
projection/expert reductions. Its shared router explicitly uses IEEE FP32 dot
products; a launch-level Triton precision default proved insufficient, and
the generated IR revealed TF32. The corrected GPU probe on job 273 passed
exact router/route-weight and expert dispatch/reduction comparisons across
1, 17, and 128 rows. Router and grouped-GEMM gradient comparisons also passed,
including an empty expert. Router maximum error against FP64 was 5.28e-6.
Job 280 (`qwen30b-reverse-frozen-aligned`) is queued after job 277 for three
full-model frozen batches with a 64-token completion cap. Its result is pending.

Job 273's trainer completed all 20 single-turn GSM8K learning steps. Its raw
audit verified 152,381 sampled tokens, zero FP32 bit mismatches, zero absolute
error, zero non-finite pairs, and all 640 matching policy-version traces. The
maximum sequence length was 626, with 81 sampled tokens at positions 593 or
later. The final held-out evaluation was still draining at audit time.

The extended MoE probe (`moe-glm-probe-273.json`) also passed GLM-sized sigmoid
routing comparisons across batch shapes, preservation of the unrotated half
of GLM's RoPE and its gradients, and exact router/expert GEMMs with UVA views
of pinned CPU weights. A GLM alignment/offload config is staged; the full
model's fit, forward agreement, and learning remain unvalidated.

Job 277's long-input dense audit verified all three frozen steps: 45,402
sampled tokens, all at positions593 or later, zero FP32 bit mismatches or
non-finite pairs, and all96 matching policy-version traces. Maximum sequence
length was4,220. Two live replays (sequence lengths1,381 and4,107) covered751
sampled tokens with exact agreement between trainer scoring, serving prefill,
and decode. This is a numerical stress test with heavy truncation, not a
reverse-text learning result.

The serial queue continues with Qwen30B jobs280 (frozen),281 (reverse learning),
282 (math learning), then GLM job283 (frozen, full CPU optimizer offload and
150GiB CPU serving-weight offload per TP1 replica). All successors require
the preceding exact-zero audit. The CPU Adam kernel build passed on node030;
the full GLM model's GPU/host fit still needs validation.

Job 280 (`qwen30b-reverse-frozen-aligned`) completed successfully. Its audit
verified all three steps, 4,637 sampled tokens, and 96 matching policy-version
traces, with zero FP32 bit mismatches, non-finite pairs, or absolute error.
The longest sequence was 134 tokens. Four live replays covering 98 sampled
tokens also matched trainer scoring, serving prefill, and decode exactly.
Peak trainer memory was 61.4 GiB per GPU; this EP1/TP1 configuration fits the
two H200 nodes. Job 281 has started the 20-step reverse-text learning arm.
The existing CPU optimizer correctness suite passed 3 tests on node030,
including AdamW agreement and preservation of per-parameter compute dtypes.

The long RoPE probe found identical GPU-generated cosine/sine tables between
the baseline trainer and serving at positions 0–8,191 on these inputs. Eager
rotation differed on about 26.2% of query/key elements, while compiled rotation
matched serving. The coordinated candidate uses the same CUDA rotation on both
sides and explicitly shares CPU-generated tables.

Run artifacts live under `outputs/mismatch/<run_name>/`, including
`mismatch-summary.json`, `policy-audit.json`, and `prefill-replay.json` where
available. GPU probe outputs are `outputs/mismatch/operator-probe-<job>.json`.
Jobs use reduced scheduling priority and a watchdog that yields our allocation
when another eligible job is pending. Dependent experiment stages run serially.

Job281 failed after the first weight update: step2 mismatch KL was0.0230
and step3 was0.0551. Independent auditing found3,866 bit mismatches among6,396
sampled tokens, with max absolute logprob error8.9978456497. All96 policy
records matched, ruling out asynchronous weight lag for this failure.
The trainer broadcast downcast learned FP32 router gates to BF16; the opt-in
alignment now preserves their FP32 transfer through the existing model
callback. Replacement job285 tests this correction; no updated-weight MoE
exact-zero claim is made until that run passes. Job286 depends on its audit.

GLM actual safetensors headers total220,937,637,888 BF16 bytes plus23,552 FP32
bytes, including MTP weights. Full FP32 Adam storage with resident gradients
would exceed node host RAM. The staged GLM config enables Linux gradient-page
reclamation after native optimizer consumption. This changes storage lifetime,
not optimization/reduction dtypes. Full-model fit remains unverified.

Job285’s live audit after steps1–2 found2,796 sampled tokens, zero bit
mismatches or absolute error, and all64 valid policy traces. This includes
the first updated-weight comparison; the full learning result is pending.
The FSDP/native CPU offload probe passed3steps with2accumulated microbatches
per step: raw FP32 bits of masters and both Adam moments were identical with
and without gradient-page reclamation. GLM job287 is pending after math286.

## 2026-09-09 06:56 UTC

GLM meta-device checkpoint validation passed: all780 trainer tensors match
both HF conversion and cached PrimeRL shapes; all extra HF tensors are the
ignored MTP layer46. Cached selection biases are FP32. Actual trainer parameter
count is106,852,245,504, excluding MTP: full masters/moments/gradients need
1592.22GiB before overhead, still above host RAM. Masters plus Adam
moments need1194.17GiB; consumed-gradient reclamation remains necessary.
Evidence: outputs/mismatch/glm-checkpoint-shapes.json.

## 2026-09-09 06:59 UTC

The extended GPU probe passed updates through UVA CPU-backed weight views:
FP32 router and BF16 expert storage bits match the update source, and subsequent
router/expert forward bits match resident GPU weights. This closes the small
operator-level offload-update check; full GLM startup and learning still need
validation. Evidence: moe-glm-updated-uva-probe-285.json.

Job285 (`qwen30b-reverse-learning-fp32-transfer`) completed successfully with
its final independent audit: all20 learning steps, 26,833 sampled tokens, zero
FP32 bit mismatches/nonfinite pairs/absolute error, and all640 matching policy
records. Maximum sequence length was302. Nonzero gradient norms occurred on
all20 steps. Preserving FP32 router transfer resolved the updated-weight
failure seen in281. Math job286 has started; GLM frozen287 depends on its audit.

Math286 live audit through steps1–5 found31,888 sampled tokens and160 valid
policy records, with zero FP32 bit mismatches or absolute error. Maximum trained
sequence495. All five gradients were zero; actual updated-policy math coverage
is pending. New aligned-MoE runs additionally log a sampled router parameter’s
pre/post-optimizer bit changes, validated through real FSDP/native CPU offload.
The audit reports subsequent steps scored after those observed changes.

## 2026-09-09 08:25 UTC

Job286 yielded at08:18 to pending224 (BadConstraints). Final audit covers
15/20 steps:91,998 sampled tokens,480 matching policy records, zero FP32 bit
mismatches/nonfinite pairs/absolute error, max sequence559. Nonzero gradients
at steps6,9,13,14 were followed by exact updated-policy comparisons. The full
audit correctly reports verified=false because steps16–20 are absent.
Dependent jobs290/287 cancelled without starting. No checkpoint was saved.

Pending224 explicitly excludes both030 and039; both nodes were verified idle.
The watchdog now rechecks excluded nodes and continues only when all of our
allocation is excluded, retaining conservative yielding otherwise. It also
checks explicit unfulfilled dependencies independently of transient reasons.
No other user's job was modified.

Fresh math job291 is running on030/039, with20 steps,LR1e-6,FP32 optimization
and reduction, max_off_policy_steps=0, plus the actual router-update probe.
Evaluation now explicitly disables thinking (old286 used default thinking-on
evaluation); training remains thinking-disabled. Long frozen292 depends on291,
and GLM frozen293 depends on292. All jobs are named mismatch, nice10000, and
use at most two nodes total.

## 2026-09-09 09:29 UTC

Math291 finished all20 trainer steps. Independent full-precision audit covers
119,702 sampled tokens, zero FP32 bit mismatches/nonfinite pairs/absolute error,
all640 matching policy records, max sequence590. All20 router-probe records
are finite and unchanged; all20 gradient norms are zero. The audit therefore
correctly reports verified=false solely because no updated policy was scored.
This is a complete unchanged-policy numerical result, not a successful learning
arm. Final64-example evaluation is still running; await terminal SLURM state
before starting the prepared harder-math run.

After final exit,292/293 will be invalidated by their afterok dependencies.
Next intended run is qwen30b-math-harder-learning-math500-prepared, on the same
two nodes, with a4-hour submission time allowance for longer responses and the
priority watchdog retained. Resubmit long frozen and GLM frozen serially after
its successful audit. No new GPU allocation has been made yet.

## 2026-09-09 09:43 UTC

Job295 failed before sampling: math-env/math500 declare network restrictions,
and their tasks are incompatible with subprocess. Episode-level errors (no
traces) identified the cause;320 immediate failures tripped the no-output guard.
Jobs296/297 cancelled. No model comparison or learning result is claimed.

CPU-only SLURM probes298/299 verified Docker is absent on039 and user namespace
creation is disallowed. No isolation checks were removed. math-harder.toml now
requires Docker; its already-prepared subprocess launchers must not be submitted.

Selected the supported GSM8K environment with the existing difficulty-pool
sampler, all weights1 and seed42, giving uniform sampling across the full7473
training tasks rather than the default source-order cycling.128 sampled tasks
spanned indices4–7454, all128 were unique and passed resolve_runtime_config
with subprocess. Evidence: qwen30b-math-uniform-learning/runtime-sampler-check.json.

Submitted300 on030/039 with nice10000 and4h time override. Numerical settings,
LR1e-6,20steps, thinking-disabled single-turn sampling and zero policy lag are
unchanged. Long frozen301 depends on300; GLM frozen302 depends on301. All use
the priority watchdog and require preceding audits. No extra GPUs allocated.

## 2026-09-09 10:02 UTC

The expanded GPU attention probe passed Qwen MoE32q/4kv and GLM96q/8kv layouts
on node030 within existing job300. For each layout, prefill-vs-decode and
prefill-vs-paged-decode had zero BF16 bit mismatches and absolute error at
lengths257,1024,8192 (six comparisons per layout). Gradient reference assertions
also passed. Artifact: moe-attention-layouts-300.json/.log. Compiler caches were
private to the probe; no extra node was allocated. This strengthens the operator
evidence but does not certify GLM full-model fit or equality.

## 2026-09-09 10:10 UTC

GLM selection-bias cache check passed: all45 non-MTP MoE layers,5,760 FP32
elements, exactly match the original HF checkpoint bits; all are finite and
nonzero. Artifact: glm-selection-bias-cache-check.json. The active training
cache is /home/huggingface/hub, as shown by the trainer's DCP load log; the
separate default /home/sami/.cache/huggingface cache does not contain its Prime
conversion. Original and converted GLM biases were compared in the active cache.

Source inspection found selection_bias is a persistent buffer and no optimizer
step hook updates it in the current implementation. Full-model frozen GLM and
live prefill replay remain pending; these cache/operator checks do not replace
those tests. Math300 step8 completed with zero reported mismatch.

## 2026-09-09 11:19 UTC

GLM302 cancelled during serving model construction before any samples. Trainer initialized full CPU optimizer state; serving host039 reached1,446,876,100KiB Shmem with124,743,304KiB available and became unresponsive. SLURM subsequently marked039 DOWN/DRAIN/NOT_RESPONDING. All302 job/step handles are terminal (batch/extern ended11:10:25); no allocation remains there. SSH banner times out; local kubectl has no cluster context. User recovery request is pending separately.

Confirmed pinned allocator rounding: probe310 failed only because CUDA was not explicitly initialized before reading stats. Corrected probe311 COMPLETED0 in19s. Actual GLM expert sizes2.75GiB+1.375GiB requested4,429,185,024bytes but default allocator reserved6,442,450,944bytes. PYTORCH_ALLOC_CONF=expandable_segments:False,pinned_max_round_threshold_mb:1 gives exact4,429,185,024bytes. Same legacy variable is set consistently. No dtype/weight/numerical setting changed. Source config, README, and monitoring skill updated.

Fresh frozen/reverse-learning/uniform-math-learning -unrounded launchers dry-run validated and prepared ONCE. Job313 now RUNNING on030trainer/054inference after old job steps ended;054 was idle and explicitly excluded by pending224, preserving priority and two allocated nodes. Same150GiB logical serving offload per replica. Live prefill watcher PID2147796 waits for313 first metric. GLM learning launchers remain unsubmitted pending frozen/reverse gates. Artifacts:pinned-memory-default.json,pinned-memory-unrounded.json,pinned-allocator-probe-311.log,memory-startup.log in old302 run.

## 2026-09-09 12:01 UTC

Job314 yielded to Matej pending job318 at11:54:58; all steps terminal by11:55:00. Dependent315/316 cancelled without starting;316 watcher ended accordingly. No completed trainer steps or numerical result for314. Separately, batch log shows one serving replica CUDA allocation failure on first decode at11:52:59, before cancellation. Host guards did not trigger. Matej319 is now running on054 (user matej is identifiable in SLURM); we have no active allocations.

Prepared fresh GLM reverse-learning, uniform math-learning, and long-frozen idle-headroom candidates. Serving GPU-memory utilization reduced0.9→0.75; context cap8192 instead of model default131072. All forward/optimizer/reduction dtypes, exact-zero gates, CPU-offload settings, and policy synchronization remain unchanged. This GPU capacity adjustment still needs live verification. Plan uses idle022/030, both explicitly excluded by pending224;054 is left to Matej.

## 2026-09-09 12:01 UTC

Submitted321 GLM reverse learning,322 uniform GSM8K afterok321,323 frozen long reverse afterok322. All named mismatch, fixed022/030, nice10000,4h maximum; only321 is allocated. Inference022, trainer030. Matej continues on054 and061, which our runs do not use. Long323 live-prefill watcher installed; serial numerical/update audit gates retained. Resolved configs and bash syntax validated, including FP32 optimization/reduction, zero policy lag,150GiB logical CPU serving offload and0.75 GPU utilization.

## 2026-09-09 12:20 UTC

Job321 first trainer step completed at12:17:55. Independent live audit confirms1,389 sampled tokens, zero raw FP32 bit mismatches/absolute error/nonfinite pairs, all32 matching policy records, maxsequence151. The actual optimizer changed520,188 finite router elements atstep1. Updated-policy scoring and full20-step audit remain pending; overall verified=false is expected at1/20. Initial32-episode batch had no errors/truncation and max policy lag0.

All8 serving replicas completed warmup at12:13:00. Actual KV53.83GiB each; nvidia-smi measured~34.8GiB free per GPU during warmup. No serving CUDA error has recurred through the first batch. Step1 trainer peak53.5GiB; host availability~300GiB serving/~306GiB trainer. Second batch is collecting underpolicyv1; dispatcher paused at shipment until updated policy applied.

## 2026-09-09 12:22 UTC

GLM321 independent audit throughsteps1–2 confirms2,997 sampled tokens, zero raw FP32 bit mismatches/absolute error/nonfinite pairs, all64 policy records matching, maxsequence151. Router changed520,188 finite elements atstep1 and519,875 atstep2; step2 scores the genuinely updated policy exactly. updated_policy_scored=true. Overall verified=false solely reflects incomplete20-step coverage at this point. This is the first full-model GLM updated-weight equality evidence, not completion of reverse/math/long arms.

Step2 orchestrator shipped32 trainable episodes with max off-policy0 and error0; surplus episodes were cancelled under the zero-lag rule. The independent policy audit found no violations in consumed traces. Step3 collecting;322math and323long remain pending afterok dependencies. No other GPU allocation.

## 2026-09-09 12:34 UTC

Job321 independent audit through steps1–5:7,126 sampled tokens, zero raw FP32 bit mismatches, absolute error, or nonfinite pairs; all160 policy records match. Router updates at1–5; exact updated-policy scoring at2–5. Max sequence151; full20-step and GLMmath/long checks remain pending. Step5 completed at12:33:03, trainer peak53.7GiB. Host RAM remains near300GiB available on both nodes. Job321 RUNNING on022/030;322 and323 remain serial dependencies.

## 2026-09-09 12:48 UTC

Job321 independent audit through steps1–9:12,908 sampled tokens, zero raw FP32 bit mismatches/absolute error/nonfinite pairs,288 matching policy records. Router updates at1–9, exact subsequent-policy scoring at2–9; max sequence151. Overall gate remains incomplete at9/20. Step9 completed at12:47:15. Job321 RUNNING on022/030;322math and323long pending. README now documents the measured serving GPU reserve and8192 context cap. Node039 remains unavailable;323 watcher2198022 confirmed alive.

## 2026-09-09 13:04 UTC

Job321 independent audit through steps1–13:18,409 sampled tokens, zero raw FP32 bit mismatches/absolute error/nonfinite pairs,416 matching policy records. Router updates at1–13, exact subsequent-policy scoring at2–13; max sequence151. Overall gate remains incomplete at13/20. Step13 completed at13:02:33. Job321 RUNNING on022/030;322math and323long pending.

## 2026-09-09 13:18 UTC

Job321 independent audit through steps1–17:23,750 sampled tokens, zero raw FP32 bit mismatches/absolute error/nonfinite pairs,544 matching policy records. Router updates at1–17, exact subsequent-policy scoring at2–17; max sequence151. Overall gate remains incomplete at17/20. Step17 completed at13:16:43. Job321 RUNNING on022/030;322math and323long pending.

## 2026-09-09 13:28 UTC

GLM reverse job321 COMPLETED0:0 after1h27m30s. Final launcher zero-audit.json verified=true:20/20 steps,27,653 sampled tokens, zero raw FP32 bit mismatches/absolute error/nonfinite pairs, all640 matching policy records. Router changed at every step1–20; exact updated-policy scoring at2–20. Maximum sequence151; no sampled positions593+. Peak trainer GPU53.7GiB. Final update, broadcast, trainer/orchestrator shutdown, and all SLURM steps completed cleanly. This completes the short reverse-text numerical learning arm, not GLM math or long-input validation.

Math322 is now RUNNING on022inference/030trainer,20steps LR1e-6,batch8,seq8192,seed42 uniform GSM8K, thinking disabled for training and evaluation. The same full CPU offload, GPU/pinned-memory settings, priority and host-memory guards, and strict complete exact-zero/updated-policy audit are retained. Long323 remains pending afterok322; its live-prefill watcher remains planned within that allocation. No extra nodes allocated.
