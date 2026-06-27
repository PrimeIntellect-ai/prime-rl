# Sampling Kernel Goal, Part 2

Date: 2026-06-25

This file continues `docs/plans/sampling-kernel/goal.md`.

## Current State

We have a real opt-in sampler candidate for the current PrimeRL/Qwen debate
shape:

```text
processed_logprobs
max_num_logprobs = 1
temperature = 1.0
top_k = 20
top_p = 0.95
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
frequency_penalty = 0.0
TP = 4
LoRA rank = 64
max_num_seqs = 256
max_inflight_rollouts = 2300
```

The winning implementation shape is:

```text
native vLLM logits processors when needed
or Prime dense-presence specialization when safe
  -> FlashInfer top_k(values, ids)
  -> sort K values
  -> Triton K-tail top-p/sample/logprob columns
  -> LogprobsTensors with sampled-only columns for max_num_logprobs=0
     or sampled+top1 columns for max_num_logprobs=1
```

The important result from Part 1:

```text
R45/R46 broad high-pressure:   1.164x to 1.189x
R45/R46 strict decode-heavy:   1.263x to 1.277x
R47/R48 broad high-pressure:   1.131x to 1.156x
R47/R48 strict decode-heavy:   about 1.249x over 3 patched rows
R68/R69 broad high-pressure:   1.240x
R68/R69 strict decode-heavy:   1.209x over 6 patched / 5 native rows

Current production estimate:
  broad serving throughput:    about 1.13x to 1.24x
  decode-heavy throughput:     about 1.21x to 1.28x
```

Do not round this into "1.28x production speedup". The stronger claim is:

```text
The sampler patch is probably positive for the current debate workload.
The exact effect size still needs a preregistered repeated A/B.
```

## What We Learned

### Top-k Is Not The Next Kernel

The custom top-k direction is mostly done for now.

R36 collapsed the visible native sampler/logprob stack from about `33.66%` of
CUDA kernel time to about `4.06%` for FlashInfer top-k/sampler. That means a
perfect replacement for the remaining FlashInfer top-k bucket has only a small
local roofline.

The next kernel work should not be another value/index top-k kernel unless new
profiling contradicts R36.

### The Remaining Kernel-ish Work Is Preprocessing

After the sampler patch, residual time moved into:

```text
broad ATen elementwise work
generic logits preprocessing
penalty processing
TP/cross-rank communication
startup/warmup/JIT/cuda-graph control
```

The most plausible next kernel win is a policy-specific preprocessing path:

```text
presence-only generated-token penalty
temperature = 1.0 skip
finite top_k/top_p sampled-logprob path
```

A fused or semi-fused path should be evaluated only if the preregistered A/B
confirms the sampler patch is worth productionizing.

## Overfit Audit

### Semantic Overfit

The patch is semantically narrow by design. It is exact for the current debate
policy shape and falls back outside that shape.

Known fallback or non-goal cases:

```text
mixed top_k
mixed top_p
top_k > 64
disabled top_k
greedy or mixed greedy/random rows
per-request RNG generators
explicit logprob_token_ids
requested top alternatives
active logit-bias processors
active min-token processors
non-unit repetition penalty
non-zero frequency penalty
non-CUDA logits
non-processed-logprobs modes
```

Known semantic caveats:

```text
FlashInfer top_k returns exactly K ids.
vLLM native top-k masking can keep more than K on boundary ties.

The Triton tail samples with distribution-equivalent uniforms.
It does not preserve vLLM's exact per-row RNG bitstream.

For max_num_logprobs=0, the fast path returns the sampled-token column and the
exact selected-token rank inside the retained top-p/top-k support.

For max_num_logprobs=1, it returns vLLM-shaped sampled+top1 columns and the
same selected-token rank. It is still not a general arbitrary-top-logprobs API.
```

Verdict: semantic overfit is real but mostly controlled. It becomes a bug only if
we advertise this as a general vLLM sampler replacement.

### Measurement Overfit

The evidence is positive but still thin:

```text
two production-shaped A/B orderings
short throughput windows
filtered rows
slight prefix-cache/KV/prompt drift
only three strict patched decode rows in R47/R48
R57 throughput is supportive only, not a clean production-pressure A/B
R68/R69 is a useful fresh pair, but R69's top-level launcher exited 137 and
the internal Slurm step had to be salvaged from logs
R68/R69 both logged JIT sidecar events during measured traffic, so this is not
a clean init/JIT pass
```

Verdict: effect direction is likely positive. The exact point estimate is not
locked.

## Part 2 Objective

Turn the sampler candidate into a production decision, then decide whether the
next optimization axis is:

```text
A. productionize this narrow sampled-logprob path;
B. invest in preprocessing/penalty fusion;
C. switch effort back to topology/scheduler/prefix-cache;
D. stop because the remaining gains are below the cost of complexity.
```

The first objective is not another run. It is a no-new-compute observability
read that tells us whether the sampler patch has already pushed rollout time to
the trainer/rollout crossover. Only after that read should we spend allocation
time on the production A/B.

The concrete launch/audit goal for this sub-ladder now lives in:

```text
docs/plans/sampling-kernel/goal-init-prod-ab.md
```

Keep this file as the broader Part 2 ladder. Use `goal-init-prod-ab.md` when
actually launching or judging the init objective, two-node production-pressure
A/B, or full production A/B.

2026-06-26 W&B refresh:

```text
Current 4t12i / LoRA-64 W&B runs are more balanced than the older 1t7i
baseline. The current trainer wait_for_batch fraction is about 0.30-0.44
across the most relevant 4t12i rows, not 0.74-0.88 as in the older 1t7i
telemetry. Forward/backward is about 168-221s/step and broadcast is about
9-20s/step.

Therefore a 1.13x-1.24x serving-only win roofs out at roughly:
  wait=0.30 -> 1.04x-1.06x E2E
  wait=0.44 -> 1.05x-1.09x E2E

So the full production A/B is still required, but the correct prior is now:
the sampler feature probably improves serving, while E2E may land below the
1.08x success gate if trainer compute and orchestration now take enough of the
step. That would not falsify the serving win.
```

Refresh command:

```bash
uv run --no-sync python scripts/analyze_wandb_production_gate.py \
  simul=b32544b2d5db411b929dd6496da34f61 \
  pcd4=9ba4977c913d46b3a927ced63d907cf0 \
  seq4=60ddf022c6e4417ea6b2934680291aea \
  mallopt=db474e36863a4faaabb3c03f2ccecdce
```

Production A/B comparison command:

```bash
uv run --no-sync python scripts/analyze_wandb_production_gate.py \
  native=<native-wandb-run-id> \
  patched=<patched-wandb-run-id> \
  --skip-trainer-rows 1 \
  --max-trainer-rows <same-startup-excluded-row-count> \
  --compare native,patched \
  --e2e-pass-ratio 1.08 \
  --e2e-fail-ratio 1.03 \
  --serving-pass-ratio 1.10 \
  --min-trainer-rows 2 \
  --min-active-inference-rows 10 \
  --min-running-cap 1024 \
  --min-waiting-positive-fraction 0.5
```

The comparison table includes `decision = pass | weak_positive | mixed | fail |
missing`. Use the same `--skip-trainer-rows` and `--max-trainer-rows` values for
both arms; otherwise the comparison is not a same-window A/B. The row minimums
make the scorer fail closed if either arm has too little trainer or inference
evidence. The running-cap and waiting-positive fraction gates prevent an
underfed candidate from passing on E2E alone. W&B `inference/agg/*` metrics are
summed aggregate telemetry, so the full-production scorer uses a conservative
aggregate running floor of 1024 rather than the single-replica cap of 256. Use
`--min-running-cap 256` only for explicit single-replica/two-node serving lanes.
Still use vLLM per-replica logs for the high-pressure row gate.

W&B analyzer logic coverage:

```bash
uv run --no-sync pytest tests/unit/inference/test_wandb_production_gate.py
```

## Execution Goal: Init Objective + Production A/B

Goal: decide whether the finite-top-k sampled-logprob path should become a
production feature, without confusing a serving-speed win for an end-to-end RL
win.

Important distinction:

```text
two-node available lane:
  production-length, production-vLLM-shape, single-replica pressure A/B
  answers: does the serving hot path still win at deployment size?
  does not answer: final bs512 / 12-replica trainer step speedup

full production lane:
  bs512, 4 train nodes, 12 inference replicas, max_inflight_rollouts=2300
  answers: does the patch improve end-to-end RL wall-clock?
```

Initial objective:

```text
Use existing W&B and run logs before launching anything.

Question:
  Is the patched system still rollout/inference-bound, or did the sampler patch
  already move the bottleneck to trainer compute / broadcast / orchestration?

Required evidence:
  time/step
  time/wait_for_batch
  time/forward_backward
  time/broadcast_weights
  perf/throughput
  perf/mfu
  inference/agg/running_requests
  inference/agg/waiting_requests
  inference/agg/kv_cache_usage_max
  inference/agg/avg_queue_time_seconds
  vLLM log rows: generation tok/s, Running, Waiting, KV, prefix hit rate
  rollout lengths: completion p50/p90/max if available

Decision:
  wait_for_batch high + running saturated      -> run production-pressure A/B
  wait_for_batch low + forward_backward high   -> pivot trainer/overlap
  running below cap + waiting low              -> pivot supply/orchestrator
  running saturated + KV high                  -> pivot KV/concurrency/prefix
```

Production-pressure A/B on the current two-node lane:

```text
Scope:
  This is a single-replica serving decision, not final E2E production proof.

Use deployment sizes:
  model = Qwen/Qwen3.5-35B-A3B
  seq_len = 32768
  max_model_len = 32768
  max_completion_tokens = 16384
  thinking_token_budget = 8192
  tp = 4
  LoRA rank = 64
  enable_prefix_caching = true
  max_num_seqs = 256
  max_num_batched_tokens = 131072
  sampling: temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5

Pressure rule:
  max_inflight_rollouts must be high enough that vLLM reaches sustained
  Running ~= 256 with nonzero Waiting. If it does not, the run is underfed and
  cannot adjudicate the kernel path.

Current allocation rule:
  if running under job 5379916, use only nid011175,nid011195.
  Do not use nid011153,nid011166; those belong to the other lane.

Arms:
  native:
    no PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB

  patched:
    PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL=triton
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL=1000
    PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK=1

Order:
  native -> patched -> patched -> native, unless allocation time only allows
  one matched pair.

Primary two-node metric:
  steady-state generation tokens/s on rows with Running ~= 256.

Secondary two-node metrics:
  startup time excluded
  fallback rate on learner traffic
  token-export / trainer health
  prefix-cache and KV bands
  queue pressure matched between arms

Two-node promotion gate:
  patched/native >= 1.10x broad high-pressure generation throughput
  and patched/native >= 1.20x strict decode-heavy throughput if enough rows exist
  and fallback rate <= 1% on learner traffic
  and no token-export / non-finite / response-shape failure

Two-node failure gate:
  patched/native < 1.05x broad high-pressure
  or the run cannot sustain Running ~= 256
  or fallback rate > 1%
  or any correctness/training-health issue appears
```

Full production A/B after the two-node gate:

```text
Scope:
  This is the real production decision.

Use:
  bs512
  group_size = 16
  target_lag = 3
  max_inflight_rollouts = 2300
  max_num_seqs = 256
  4 train nodes + 12 inference replicas, or the closest exact production lane
  available at launch time

Primary metric:
  end-to-end trainer/orchestrator step wall-clock after startup is excluded.

Supporting metrics:
  time/wait_for_batch
  time/forward_backward
  time/broadcast_weights
  generation tok/s under high pressure
  inference running/waiting/KV/prefix bands

Success:
  patched/native >= 1.08x end-to-end step speed
  and patched/native >= 1.10x broad high-pressure serving throughput
  and no correctness/training-health regression

Strong success:
  patched/native >= 1.12x end-to-end step speed

Failure:
  patched/native < 1.03x end-to-end step speed
  or trainer_wait_fraction is already small enough that the serving win is masked
  or fallback/correctness failures appear
```

## Rung 0: Initial Observability Objective

Purpose: decide whether faster decode still has end-to-end value.

The sampler patch improves the inference decode window, but end-to-end RL step
time only improves while rollout production is the bottleneck. If rollout time
has already fallen to the trainer compute/broadcast time, more decode work is
masked by the trainer and the next optimization should move to trainer overlap,
weight broadcast, or closed-loop scheduling.

Use existing W&B / run logs first. Do not launch a new Slurm run for this rung.

Primary questions:

```text
Q1. Is the trainer still waiting for rollout material after the patched sampler?
Q2. Is inference still decode/cap-bound, KV-bound, or underfed?
Q3. Does current serving pressure actually hold max_num_seqs=256 running?
Q4. Are completion lengths closer to the 8k regime where extra concurrency helps,
    or the 32k regime where KV/capacity dominates?
Q5. Is the training step now dominated by forward/backward, wait_for_batch,
    broadcast_weights, or checkpoint/export?
```

Metrics to read:

```text
trainer:
  time/wait_for_batch
  time/forward_backward
  time/broadcast_weights
  time/step
  perf/throughput
  perf/mfu

orchestrator / inference:
  inference/agg/running_requests
  inference/agg/waiting_requests
  inference/agg/kv_cache_usage_mean
  inference/agg/kv_cache_usage_max
  inference/agg/avg_queue_time_seconds
  train batch progress / buffered groups from orchestrator logs

serving logs:
  Avg prompt throughput
  Avg generation throughput
  Running
  Waiting
  GPU KV cache usage
  Prefix cache hit rate

rollout artifacts if available:
  completion token length distribution
  prompt token length distribution
  trainable group yield
  failed rollout reasons
```

Derived quantities:

```text
trainer_wait_fraction = time/wait_for_batch / time/step
trainer_compute_fraction = time/forward_backward / time/step
broadcast_fraction = time/broadcast_weights / time/step
rollout_vs_train_gap = rollout_step_time - trainer_forward_backward_time
running_headroom = 256 - inference/agg/running_requests
kv_headroom = 1.0 - inference/agg/kv_cache_usage_max
```

Interpretation:

```text
Case A: wait_for_batch remains high and running_requests ~= 256
  Inference is still the gating side. Run the production A/B, then consider
  preprocessing/penalty fusion or scheduler/concurrency.

Case B: wait_for_batch is near zero and forward_backward dominates
  The sampler patch likely reached the rollout/trainer crossover. Do not chase
  more decode kernels for E2E. Run trainer Nsight or optimize broadcast/overlap.

Case C: running_requests < 256 and waiting_requests low
  Decode kernels are not the next bottleneck. Investigate orchestrator/env/judge
  supply, scheduler, or group-completion dynamics.

Case D: running_requests ~= 256 and kv_cache_usage_max is high
  Concurrency is KV-bound. Consider lower max_num_seqs, fp8-KV only if sequence
  length/capacity requires it, or prefix/cache improvements. Do not blindly raise
  max_num_seqs.

Case E: running_requests ~= 256 and kv_cache_usage_max has headroom
  Concurrency/scheduler sweep is plausible after the A/B.
```

Acceptance artifact:

```text
Append a short table here before the production A/B:

run id
arm
time/step
time/wait_for_batch
time/forward_backward
time/broadcast_weights
trainer_wait_fraction
inference/agg/running_requests
inference/agg/waiting_requests
inference/agg/kv_cache_usage_max
completion length p50/p90/max
decision: run A/B | pivot trainer | pivot scheduler/env | pivot KV/concurrency
```

Decision gate:

```text
Run the production A/B only if existing data says rollout/inference is still a
material contributor to end-to-end step time.

If trainer_wait_fraction is already small and forward_backward dominates, skip
the production A/B for now and profile/optimize trainer-side work instead.
```

### 2026-06-26 Rung 0 Read

Current allocation:

```text
job: 5379916
occupied GPU lane: nid011153,nid011166
free GPU lane for this work: nid011175,nid011195
```

W&B full-production read, project
`jvelja-private/gpqa-openended-debate-calibration`:

```text
run 653564de94f04c6f8e0034eb061f3693
name qwen35-a3b__qwen9b-or__simul
state crashed, but has useful steady metrics

recent trainer rows:
  step 1657: time/step=357.1s, wait=167.1s, fwd_bwd=188.4s,
             broadcast=9.291s, wait_frac=0.468
  step 1720: time/step=282.2s, wait=134.7s, fwd_bwd=146.4s,
             broadcast=8.649s, wait_frac=0.477
  step 1774: time/step=225.6s, wait=71.73s, fwd_bwd=152.6s,
             broadcast=8.516s, wait_frac=0.318

recent inference aggregate, last 50 rows:
  running_requests mean=1243, p50=1199, range=1163..1383
  waiting_requests mean=790.1, p50=787, range=661.9..899.6
  kv_cache_usage_max mean=0.3797, p50=0.3645, range=0.3418..0.4397
  throughput mean=55.07k tok/s
  avg_queue_time_seconds mean=158.2
```

Interpretation:

```text
The current production-shaped system is not cleanly trainer-bound:
wait_for_batch is still 32%-48% on recent complete rows.

It is also not a simple "all slots saturated" story:
aggregate running_requests is materially below the naive 12*256 slot cap, while
waiting_requests and queue time are nonzero and KV max has headroom.

Decision: rollout/inference is still material, so run the two-node
production-pressure A/B if allocation time allows. The run must validate
single-replica pressure with Running ~= 256 and Waiting > 0; otherwise it is
underfed and does not adjudicate the sampler path.
```

Existing two-node pressure A/B read, `max_inflight_rollouts=2300`,
`max_num_seqs=256`, bs512:

```text
broad high-pressure rows:
  R45 native  all-matching mean=11425.4 gen tok/s
  R46 patched all-matching mean=13301.1 gen tok/s  => 1.164x
  R48 native  all-matching mean=11357.3 gen tok/s
  R47 patched all-matching mean=12843.0 gen tok/s  => 1.131x

strict decode-only rows:
  R45 native  all-matching mean=11386.3 gen tok/s
  R46 patched all-matching mean=14384.7 gen tok/s  => 1.263x
  R48 native  all-matching mean=11640.5 gen tok/s
  R47 patched all-matching mean=14789.8 gen tok/s  => 1.271x
```

Health / semantics evidence:

```text
R57/R62/R65 b16 token-export runs reached "RL trainer finished".
R57 final sampler stats included row_hit_rate=0.999787 with
fallback_reason_hist={'max_num_logprobs_not_width1': 2}; those are the known
warmup/profiling fallbacks, not learner-traffic fallback.
```

Practical launch correction:

```text
Use docs/plans/sampling-kernel/rl_probe_openended_debate_2node.toml for the
two-node production-pressure A/B. It is bs512 with max_inflight_rollouts=2300.

Do not use the b16 text-only token-export config for pressure adjudication
unless batch/max_inflight behavior is explicitly overridden and verified; its
native max_inflight_rollouts=64 is intentionally under pressure for serving.
```

### 2026-06-26 Concrete Goal: Init Objective + Production A/B

Objective: make one production decision about the narrow finite-top-k
sampled-logprob path, without upgrading a serving-only win into an end-to-end
RL claim.

The ladder is:

```text
0. Init objective:
   Read existing production telemetry.
   Decide whether rollout/inference is still material enough to justify A/B.

1. Two-node production-pressure A/B:
   Use deployment lengths/sizes on one clean inference replica.
   Decide whether the serving hot path still wins under real pressure.

2. Full production A/B:
   Use the full bs512 / 12-replica lane.
   Decide whether the serving win moves end-to-end RL wall-clock.
```

Rung 0 acceptance:

```text
Append a table with trainer wait, trainer compute, broadcast, inference
running/waiting/KV, and rollout length evidence.

Decision must be one of:
  run two-node production-pressure A/B
  pivot trainer/overlap
  pivot scheduler/env supply
  pivot KV/concurrency/prefix
```

Two-node A/B acceptance:

```text
Required shape:
  Qwen/Qwen3.5-35B-A3B
  max_model_len = 32768
  max_completion_tokens = 16384
  thinking_token_budget = 8192
  tp = 4
  LoRA rank = 64
  max_num_seqs = 256
  max_num_batched_tokens = 131072
  max_inflight_rollouts = 2300
  sampling = temp 1.0, top_k 20, top_p 0.95, presence_penalty 1.5

Pressure gate:
  sustained Running == 256
  Waiting > 0
  high-pressure row filters reported, not hand-picked screenshots

Serving promotion:
  patched/native >= 1.10x broad high-pressure generation tok/s
  fallback row rate <= 1% on learner traffic
  no response-shape, token-export, or non-finite failure

Non-claim:
  This does not prove production E2E speed. It only promotes the patch to a
  full production A/B candidate.
```

Full production A/B acceptance:

```text
Required shape:
  bs512
  group_size = 16
  target_lag = 3
  max_inflight_rollouts = 2300
  max_num_seqs = 256
  production train/inference topology, ideally 4 train nodes + 12 inference
  replicas

Primary metric:
  startup-excluded end-to-end trainer/orchestrator step wall-clock

Supporting metrics:
  time/wait_for_batch
  time/forward_backward
  time/broadcast_weights
  generation tok/s under high pressure
  inference running/waiting/KV/prefix bands
  sampler fast/fallback stats

Production success:
  patched/native >= 1.08x end-to-end step speed
  and patched/native >= 1.10x broad serving throughput
  and no training-health regression

Strong production success:
  patched/native >= 1.12x end-to-end step speed
```

Current two-node result, R66 native vs R67 patched on job `5379916`,
nodes `nid011175,nid011195`:

```text
native:
  output_dir = outputs/sampling-kernel/r66-native-pt2-ab-n175n195
  lane_tag = r66-native-n175n195
  server ready = 18:33:16 UTC
  engine init = 264.28s, compilation = 82.80s

patched:
  output_dir = outputs/sampling-kernel/r67-patched-pt2-ab-n175n195
  lane_tag = r67-patched-n175n195
  server ready = 18:48:30 UTC
  engine init = 98.25s, compilation = 5.96s
  final sampler stats seen:
    calls=23000
    fast_calls=22998
    fallback_calls=2
    row_hit_rate=0.999913
    fallback_reason_hist={'max_num_logprobs_not_width1': 2}
```

Matched serving rows after cancellation:

```text
window                         native rows / mean     patched rows / mean    ratio
broad all pressure             46 / 7402.0 tok/s      46 / 8505.6 tok/s      1.149x
broad 15%-45% KV               39 / 7603.8 tok/s      46 / 8505.6 tok/s      1.119x
broad 35%-50% KV               35 / 6526.9 tok/s      39 / 7620.4 tok/s      1.168x
strict decode 15%-45% KV        5 / 10038.3 tok/s      4 / 13919.9 tok/s     1.387x
```

2026-06-26 recomputation from the persisted R66/R67 logs:

```text
broad 15%-45% KV, first 12 rows:
  native  = 10775.2 tok/s
  patched = 12253.5 tok/s
  ratio   = 1.137x

broad 15%-45% KV, all matching rows:
  native  = 7603.8 tok/s over 39 rows
  patched = 8505.6 tok/s over 46 rows
  ratio   = 1.119x

strict decode 15%-45% KV:
  native  = 10038.3 tok/s over 5 rows
  patched = 13919.9 tok/s over 4 rows
  ratio   = 1.387x
```

2026-06-26 fresh R68/R69 repeat on `nid011175,nid011195`:

```text
patched:
  output_dir = outputs/sampling-kernel/r68-patched-repeat-n175n195
  learner_row_hit_rate = 1.0
  fallback rows on learner traffic = 0
  JIT sidecars = 100 events across 4 workers

native:
  output_dir = outputs/sampling-kernel/r69-native-repeat-n175n195
  top-level rl process exited 137 after spawning the internal Slurm step
  step 5379916.43 continued on the correct lane and was salvaged from logs
  no finite_topk_sampler_stats sidecar, as expected for native vLLM
  JIT sidecars = 92 events across 4 workers

broad 15%-45% KV, first 12 rows:
  native  = 10856.0 tok/s
  patched = 13465.5 tok/s
  ratio   = 1.240x

strict decode 15%-45% KV:
  native  = 11944.0 tok/s over 5 rows
  patched = 14439.6 tok/s over 6 rows
  ratio   = 1.209x
```

Init/JIT caveat:

```text
R67 wrote jit_monitor sidecars with 25 events per worker. The events include
the patched K-tail kernel around first traffic and MoE/LoRA kernels around the
early high-pressure window.

R68/R69 also wrote JIT sidecars during measured traffic: patched had 100 total
events and native had 92 total events.

R70 enabled the sampler-tail precompile probe and confirmed it runs before JIT
monitor activation, but real batched traffic still produced 96 sidecar events,
including `_k_tail_uniform_kernel` 4 times.

R73 enabled batch-shaped sampler-tail precompile with batches 1,128,256 and
again logged precompile before JIT monitor activation on all 4 workers. It
still produced 80 post-ready sidecar events, including `_k_tail_uniform_kernel`
4 times. So the easy precompile-list fix did not close the init/JIT rung.

R74 warmed a broader observed-shape list, 1,47,98,128,241,256. It still produced
68 post-ready sidecar events, including `_k_tail_uniform_kernel` 4 times.
The first learner hits were 1,1,9,54,176,250,256,256, and the K-tail JIT warning
appeared on the first batch=1 learner hit despite batch=1 precompile.

So R66/R67 and R68/R69 are valid serving-pressure evidence, but they do not
prove the stronger init objective "no post-ready JIT during measured windows".
Treat post-ready JIT coverage as still open under Rung 8 /
goal-init-prod-ab.md.
```

Interpretation:

```text
Rung 0 passed: production telemetry still shows material wait_for_batch, so
rollout/inference remains worth testing.

R66/R67 pass the two-node serving gate:
  broad production-pressure serving win is about 1.12x-1.17x by KV band
  broad all-pressure win is 1.149x
  strict decode is positive but under-sampled
  patched prefix-cache hit rate was lower than native, so prefix cache does not
  explain the win
  fallback rate is far below 1%, with only known warmup/profiling fallback

R68/R69 pass the two-node serving gate again:
  broad production-pressure serving win is 1.240x over the first 12 rows
  strict decode is 1.209x over 6 patched / 5 native rows
  R69's recovered native run is not a clean launcher-success artifact, but the
  internal step produced usable pressure rows on the correct lane

R66/R67 do not pass the full production gate because they were stopped during
the long-output pressure phase at Train batch 0/512. That is fine for the
serving-hotpath rung, but it leaves end-to-end step speed unproven.

Decision:
  Promote the patch to a full production A/B candidate.
  Do not claim a production E2E speedup until the full lane passes.
```

## Rung 1: Freeze New Top-k Work

Status target: policy decision, not code.

Rule:

```text
Do not write another top-k selection kernel unless a fresh profile shows
FlashInfer top_k/sampler above 10% of steady decode CUDA kernel time.
```

Allowed work:

```text
bug fixes in the current K-tail path
metrics
fallback logging
production cleanup
preprocessing/penalty probes
```

Acceptance:

```text
goal-pt2.md records any exception to this rule with profiler evidence.
```

## Rung 2: Preregistered Production A/B

Purpose: decide whether the current sampler candidate is worth production
cleanup. On a two-node lane this is a production-size serving A/B. On a full
production lane this becomes an end-to-end RL wall-clock A/B.

Prerequisite: Rung 0 says rollout/inference is still material. If Rung 0 says the
patched system is trainer-bound, do not spend allocation time on this A/B yet.

Run matrix:

```text
same free node pair for all arms
current 2026-06-26 lane under job 5379916: nid011175,nid011195 only
same model/checkpoint
same stable compile-cache namespace
same max_num_seqs = 256
same pressure target: sustained Running ~= 256 with nonzero Waiting
full production confirmation uses max_inflight_rollouts = 2300
same generated config artifacts

arms:
  native
  patched: PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
           PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL=triton
           PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1

order:
  randomized or alternated
  at least native -> patched -> patched -> native if only four runs fit
```

Primary metric:

```text
two-node lane: steady generation tokens/s after startup is excluded
full production lane: end-to-end trainer/orchestrator step wall-clock after
startup is excluded
```

Primary supporting metric:

```text
generation tokens/s in high-pressure rows
```

Primary row filter:

```text
Running == 256
Waiting > 0 on a two-node lane; Waiting >= 3000 for full production logs
15% <= GPU KV cache usage <= 45%
0 <= prompt tokens/s <= 10000
```

Secondary stricter filter:

```text
Running == 256
Waiting > 0 on a two-node lane; Waiting >= 3000 for full production logs
15% <= GPU KV cache usage <= 45%
prompt tokens/s == 0
```

Minimum useful data:

```text
at least 3 native and 3 patched completed steps if allocation allows
at least 2 native and 2 patched completed steps as the minimum interpretable run
at least 10 matching high-pressure rows per arm
at least 6 strict decode rows per arm if available
record time/wait_for_batch, time/forward_backward, time/broadcast_weights
record prefix-cache hit rate bands
record KV bands
record prompt throughput bands
record sampler fast/fallback stats for patched arms
```

Success gate:

```text
patched/native improves end-to-end step time by >= 1.08x
and patched/native >= 1.10x broad high-pressure generation throughput
and no learner-traffic fallback reason except known warmup/profiling calls
and no throughput regression under strict decode filter
```

Stronger promotion gate:

```text
patched/native improves end-to-end step time by >= 1.12x
or patched/native >= 1.15x broad high-pressure
or patched/native >= 1.20x strict decode-heavy
with no correctness red flags
```

Failure gate:

```text
patched/native < 1.03x end-to-end step time
or patched/native < 1.05x broad high-pressure
or fallback row rate > 1% on learner traffic
or new non-finite/token-export/training-quality issue
```

Output artifact:

```text
append a table here:
  run id
  arm
  order
  nodes
  startup times
  time/step
  time/wait_for_batch
  time/forward_backward
  time/broadcast_weights
  matched row counts
  broad mean
  strict mean
  prefix/KV bands
  sampler fallback summary
```

## Rung 3: Training-Quality Canary

Purpose: check that distribution-equivalent sampling plus scalar logprob path is
safe enough for the current RL loop.

This is not a bitwise equivalence test.

Canary shape:

```text
deployment sequence lengths
return_token_ids = true
token export enabled
no checkpoint writes unless required
same sampling params as production debate
native baseline and patched arm both captured
```

Checks:

```text
RL trainer finished
no non-finite token logprobs
no NaN/Inf loss
no post-batch filters dropping all rows
mismatch KL in the same rough band as native
entropy in the same rough band as native
loss in the same rough band as native
sampler row hit rate near 1.0 after warmup
fallback reasons explained
```

Acceptance:

```text
patched is not worse than native on token-export numerical health
patched does not produce new response-shape errors
patched does not create a new off-policy / mismatch-KL pathology
```

If this fails, stop productionization and debug semantics before any more
throughput work.

### 2026-06-26 Rung 3 Canary Read

No new GPU run was needed for this gate. The existing R56/R57 pair already
matches the canary shape:

```text
deployment sequence lengths:
  seq_len = 32768
  max_model_len = 32768
  max_completion_tokens = 16384
  thinking_token_budget = 8192

training shape:
  batch_size = 16
  group_size = 16
  token export enabled
  two train steps
  no response-shape failure

sampler comparison:
  R56 = native vLLM sampler baseline
        directory label says "patched" only because launcher/trainer patches
        were present; no FlashInfer sampled-logprob env was active
  R57 = same b16 token-export canary with FlashInfer sampled-logprob enabled
```

Trainer log evidence:

```text
R56 native:
  Step 0 | Loss 0.0021 | Entropy 0.4590 | Mismatch KL 0.0568
  Step 1 | Loss 0.0006 | Entropy 0.4298 | Mismatch KL 0.0368
  RL trainer finished

R57 patched:
  Step 0 | Loss 0.0004 | Entropy 0.4437 | Mismatch KL 0.0493
  Step 1 | Loss 0.0007 | Entropy 0.4418 | Mismatch KL 0.0372
  RL trainer finished
```

Token-export JSONL check, recomputed from current artifacts:

```text
run          files/STABLE  rows  tokens   loss tokens  shape mismatches  bad numeric
R56 native   4 / 2         32    546123   473697       0                 0
R57 patched  4 / 2         32    538785   475905       0                 0
```

Loss-masked token metrics:

```text
metric                  R56 native      R57 patched
logIR mean              -0.0657         -0.0617
logIR p95                0.1808          0.1786
logIR p99                0.6741          0.6613
importance ratio p99     1.9622          1.9373
mismatch KL mean         0.0462          0.0431
mismatch KL p99          0.7530          0.7063
entropy mean             0.4435          0.4427
```

Loss-unmasked token metrics:

```text
metric                  R56 native      R57 patched
logIR mean              -0.0523         -0.0525
logIR p95                0.1518          0.1489
logIR p99                0.5964          0.5800
importance ratio p99     1.8155          1.7860
mismatch KL mean         0.0371          0.0358
mismatch KL p99          0.7008          0.6858
entropy mean             0.4202          0.4221
```

Sampler activation in R57:

```text
final observed stats:
  calls = 33000
  fast_calls = 32998
  fallback_calls = 2
  row_hit_rate = 0.999788
  fallback_reason_hist = {'max_num_logprobs_not_width1': 2}
```

Supporting patched canaries:

```text
R62 text-only patched canary:
  trainer finished
  token exports: 4 jsonl + 2 STABLE
  rows = 32, tokens = 571987, bad numeric = 0

R65 patched JIT-monitor canary:
  trainer finished
  token exports: 4 jsonl + 2 STABLE
  rows = 20, tokens = 338881, bad numeric = 0
```

Interpretation:

```text
Rung 3 passes for the current narrow sampler path.

The patched canary is not worse than native on the headline token-export
numerics. It has no non-finite exported values, no JSONL array shape mismatch,
no response-shape failure, and no new trainer-health failure.

This is not bitwise equivalence and not a full production E2E speed claim.
It is enough to allow Rung 4 production cleanup to start behind the default-off
feature gate.
```

## Rung 4: Production Cleanup

Only start this after Rung 2 and Rung 3 pass.

Tasks:

```text
rename the feature from "FlashInfer sampler" to "finite-top-k sampled-logprob"
keep env-gated default-off posture
add request/runtime toggle if many more A/Bs are expected
export sampler stats into structured run metrics
separate warmup/profiling fallbacks from learner-traffic fallbacks
document exact semantics and fallback cases
add version guard for vLLM and FlashInfer
add a short operator note for enabling/disabling the path
```

Do not make this default-on until a longer training run passes.

### 2026-06-26 Rung 4 Cleanup Status

Completed cleanup:

```text
feature name:
  user-facing module docstring and runtime logs now say
  "finite-top-k sampled-logprob" instead of presenting the feature as a general
  "FlashInfer sampler" replacement.

default-off gate:
  new primary enable knob:
    PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
  disabling remains unset or 0.

compatibility:
  existing PRIME_RL_ENABLE_FLASHINFER_SAMPLED_LOGPROB and
  PRIME_RL_FLASHINFER_SAMPLER_* experiment env names remain aliases, so old
  run scripts do not break mid-allocation.

version guards:
  vLLM remains pinned to sampler_perf.SUPPORTED_VLLM.
  FlashInfer is now pinned to 0.6.11.post2 before the patch installs.

operator note:
  docs/inference.md now documents the contract, enable/disable envs, fallback
  cases, and legacy aliases.
```

Additional cleanup completed:

```text
structured run metrics:
  sampler stats now write JSONL sidecars under logs/inference:
    finite_topk_sampler_stats_*.jsonl
  These records include calls, rows, hit rates, batch histograms, fallback
  reasons, runtime identity, and fallback traffic classification.

warmup/profiling fallback accounting:
  periodic stats now include learner_row_hit_rate plus
  warmup_or_profiling_fallback_{calls,rows}.
  The known vLLM warmup/profiling fallback is classified separately when
  max_num_logprobs is absent and there is no live learner InputBatch.

runtime toggle:
  env-gated startup toggle exists. A request/runtime toggle is not needed yet;
  build it only if later A/Bs need same-process arm switching.

verification:
  uv run --no-sync ruff check src/prime_rl/inference/vllm/flashinfer_sampler.py
  uv run --no-sync python -m py_compile src/prime_rl/inference/vllm/flashinfer_sampler.py
  local JSONL stats probe: PASS
```

Rung 4 status:

```text
Complete for default-off production cleanup.

Do not make the feature default-on until a longer production training run
passes. That is a release/default decision, not remaining Rung 4 cleanup.
```

## Rung 5: Preprocessing / Penalty Fusion

Only start after the production A/B confirms that sampler-side work remains
worth engineering time.

Hypothesis:

```text
The next material sampler-adjacent win is reducing full-vocab preprocessing and
generic penalty overhead, not replacing FlashInfer top_k.
```

Candidate paths:

### Dense Presence Path, Current

Already implemented behind:

```text
PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1
legacy alias: PRIME_RL_FLASHINFER_SAMPLER_DENSE_PRESENCE=1
```

Mechanism:

```text
build output-token seen mask
logits -= presence_penalty * seen_mask
skip temperature when temperature == 1.0
run finite-top-k sampled-logprob path
```

Keep this if it is correct and fast enough.

### 2026-06-26 Rung 5 Read

Current status:

```text
Dense presence is already in the patched production candidate used by R66/R67.
It is not a separate future kernel project.

The live holder-aware gate showed dense_presence=True on production-shaped
requests with thinking_token_budget=8192 and batch=256.

The latest two-node A/B therefore already tests:
  finite-top-k sampled-logprob
  Triton K-tail
  unit-temperature skip
  dense presence-only preprocessing when safe
```

Sparse presence remains deferred:

```text
The useful live output-token count evidence has unique_counts around hundreds:
  p50 about 586
  p90 about 820
  max about 1009

This is not a "tiny sparse set" regime. Sparse can still win, but it needs a
real GPU microbench at the live |S| bands and then a live profile showing the
broad ATen/preprocessing bucket actually moves.
```

Decision:

```text
Do not write more Rung 5 code until the full production A/B says sampler-side
work remains worth engineering time.

If Rung 5 reopens, the next evidence step is:
  1. rerun scripts/bench_presence_penalty_paths.py on a free node at B=256,
     V=248320, and unique counts near 586, 820, 1009;
  2. only implement sparse/fused preprocessing if it beats dense at those
     bands and a live profile still shows preprocessing as a material bucket.
```

### Sparse Presence Path

Try only if output-token unique counts are small enough.

Mechanism:

```text
for each row, update only generated token ids seen in that row
avoid full [B,V] dense mask construction when |S| << V
deduplicate ids to preserve presence, not frequency
```

Risk:

```text
dedup and scatter overhead can beat the win
duplicate handling must be exact
```

Acceptance:

```text
microbench beats dense presence at deployment B,V and measured |S|
live profile reduces broad ATen elementwise bucket
no scalar-logprob parity failure
```

### Fused Preprocess + K-tail

Try only after sparse/dense evidence says launch overhead or memory traffic is
still material.

Possible contract:

```text
input logits [B,V]
generated-token ids / seen representation
presence penalty
top_k <= 64
top_p scalar or uniform

output sampled ids [B]
output sampled logprobs [B]
```

This is harder because top-k selection still needs a full row scan. Prefer
FlashInfer top_k as the value/index primitive unless profiling says otherwise.

## Rung 6: Topology Recheck

The sampler patch changes the roofline. Re-evaluate TP/EP after the sampler path
is enabled, not on old native-sampler profiles.

Question:

```text
With native sampler/logprob mostly removed, is TP=4 communication again the best
remaining lever?
```

Arms:

```text
TP4 native
TP4 patched
EP candidate native if available
EP candidate patched if available
```

Constraints:

```text
same LoRA behavior
same max_num_seqs / max_inflight
same prompt/decode mix
same output_dir discipline
no config drift in sampling params
```

Metrics:

```text
generation tokens/s
train tokens/s
end-to-end step time
inference queue health
NCCL / all-gather profile slices
fallback stats
```

Decision:

```text
If EP improves per-token comm but loses deployable throughput, stay TP4.
If EP wins end-to-end under patched sampler, make it a separate topology PR.
```

## Rung 7: Scheduler / Concurrency / Prefix Cache

Do not assume `max_num_seqs=256` is optimal.

Sweep:

```text
max_num_seqs: 128, 256, 384, 512
max_inflight_rollouts paired with each value
```

Hold constant:

```text
model
sampling params
nodes
compile cache namespace
prompt/decode workload
```

Measure:

```text
generation tokens/s
KV cache usage
Running/Waiting
prefix-cache hit rate
request latency tails if available
trainer starvation
off-policy lag
```

Stop condition:

```text
If KV pressure, latency tails, or trainer lag worsen faster than throughput
improves, do not raise concurrency further.
```

## Rung 8: Warm Engine / JIT / CUDA Graph Control

R65 made first-real-traffic Triton JITs machine-readable. R70 showed that a
batch=1 sampler-tail precompile is not enough to eliminate real-traffic JIT for
the patched path.

Use:

```text
logs/inference/jit_monitor_*.jsonl
```

Known residual kernels from R65:

```text
_causal_conv1d_fwd_kernel
_compute_slot_mapping_kernel
_fused_moe_lora_one_shot_kernel
_fused_post_conv_kernel
_topk_topp_kernel
_zero_kv_blocks_kernel
batch_memcpy_kernel
fused_moe_kernel
```

R70 precompile probe:

```text
PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL=1
top_k = 20
top_p = 0.95
vocab = 248320

result:
  precompile logged before JIT monitor activation on all 4 workers
  engine init = 86.50s, compilation = 7.00s
  96 post-ready JIT sidecar events across 4 workers
  _k_tail_uniform_kernel still appeared 4 times
  learner_row_hit_rate = 1.0
```

Batch-shaped precompile update:

```text
PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES now accepts a
comma-separated batch list while preserving the old PRECOMPILE_BATCH fallback.

Local constructor probe passed:
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES=1,128,256
  logged batches=[1, 128, 256]

R71/R72 live launch attempts are tainted:
  they produced or coincided with one-node bash steps on nid011166
  even though the inner lane step was pinned to nid011175,nid011195
  do not use them for speed/JIT evidence

R73 valid lane probe:
  output_dir = outputs/sampling-kernel/r73-patched-precompile-batches-overlap-n175n195
  nodes = nid011175,nid011195
  precompile logged batches=[1, 128, 256] before monitor activation
  engine init = 84.86s, compilation = 6.78s
  80 post-ready sidecar events across 4 workers
  _k_tail_uniform_kernel still appeared 4 times
  learner_row_hit_rate = 1.0

R74 valid lane probe:
  output_dir = outputs/sampling-kernel/r74-patched-precompile-observed-batches-n175n195
  nodes = nid011175,nid011195
  precompile logged batches=[1, 47, 98, 128, 241, 256] before monitor activation
  engine init = 84.41s, compilation = 5.90s
  first learner batches logged = 1,1,9,54,176,250,256,256
  68 post-ready sidecar events across 4 workers
  _k_tail_uniform_kernel still appeared 4 times
  learner_row_hit_rate = 1.0 at calls=1000

R75 one-GPU same-process diagnostic on nid011175:
  shape:
    no vLLM server, no model load, one Python process, one visible CUDA device
  result:
    precompile K=20 top_p=0.95 batch=1
    activate a Triton/vLLM-style post-compile hook
    repeat K=20 top_p=0.95 batch=1 -> no hook events
    call K=20 top_p=0.95 batch=9 -> no hook events
    call K=21 top_p=0.95 batch=1 -> _k_tail_uniform_kernel hook event
    call K=20 top_p=0.90 batch=1 -> _k_tail_uniform_kernel hook event
  interpretation:
    the hook is capable of seeing real new constexpr compiles, and the isolated
    precompile path really does suppress same-meta K-tail recompilation. R74 is
    therefore not explained by "batch list incomplete" or "precompile function
    is a no-op" in a plain process.

JIT sidecar enrichment:
  code:
    src/prime_rl/inference/vllm/jit_monitor.py now writes a `details` object
    alongside the old sorted `kwargs` list. It captures sanitized values for
    Triton hook fields `key`, `repr`, `already_compiled`, `is_manual_warmup`,
    and `compile`.
  tests:
    uv run --no-sync pytest tests/unit/inference/test_jit_monitor.py
    uv run --no-sync ruff check src/prime_rl/inference/vllm/jit_monitor.py tests/unit/inference/test_jit_monitor.py
    uv run --no-sync python -m py_compile src/prime_rl/inference/vllm/jit_monitor.py tests/unit/inference/test_jit_monitor.py
  result:
    all passed
  purpose:
    the next production-shaped JIT probe should compare the recorded Triton
    compile `details` for `_k_tail_uniform_kernel` against the isolated R75
    shape, rather than rerunning another blind precompile-list experiment.

Conclusion:
  env-list sampler-tail precompile is not enough to eliminate first-real-traffic
  sampler-tail JIT. The R74 failure is stronger than a missed-batch result:
  _k_tail_uniform_kernel still JITed on the first batch=1 learner hit after
  batch=1 precompile. R75 says same-meta precompile works in isolation, so the
  remaining issue is likely production vLLM lifecycle, specialization drift, or
  warmup integration. Do not run more precompile-list probes. Either debug that
  integration path, or stop treating this as a blocker and use startup-excluded
  production A/B metrics.
```

Next work:

```text
do not rerun precompile-list probes on the two-node lane
if JIT cleanliness is release-blocking, debug why production vLLM still compiles
_k_tail_uniform_kernel after an isolated same-process precompile stays hot
use enriched `jit_monitor_*.jsonl` details to inspect Triton compile keys before
writing another warmup fix
otherwise move to the full production E2E A/B and exclude startup from the
primary metric
keep JSONL monitor as acceptance evidence
avoid fresh vLLM process restarts for sampler-only A/B when possible
investigate runtime toggle or persistent engine harness
```

Acceptance:

```text
hot-cache startup remains about 30s engine init or better
first learner requests do not trigger new sampler/MoE JIT warnings
JIT JSONL event count drops for covered shapes
```

## Rung 9: Gemma / K64 Generalization

Do not claim Gemma support from Qwen K20 runs.

Gemma default sampler shape appears compatible only if the actual serving path
requests processed sampled-token logprob and has finite explicit `top_k=64`.

Required checks:

```text
official/source sampling params recorded
rendered config has top_k = 64 or request body proves it
logprobs mode is processed_logprobs
max_num_logprobs is width-1
no explicit top-logprob alternatives
no active min-token/logit-bias processors
fallback histogram clean
K64 microbench and live probe both pass
```

Expected effect:

```text
K64 speedup is likely smaller than K20.
Do not extrapolate Qwen K20 1.13x-1.28x to Gemma.
```

## Decision Table

Use this table after Rung 0, then again after Rung 2 and Rung 3.

```text
Evidence                                      Decision
--------------------------------------------  -------------------------------
Rung 0 says trainer-bound                     skip prod A/B; profile trainer
Rung 0 says inference-bound                   run prod A/B
Rung 0 says underfed scheduler/env            debug supply path before kernels
Rung 0 says KV-bound                          scheduler/KV/prefix workstream
AB >= 1.12x E2E and canary clean              cleanup + opt-in production PR
AB 1.03x-1.12x E2E and canary clean           keep as experiment toggle
AB < 1.03x E2E                                stop sampler productionization
fallback > 1% learner rows                    debug semantics, no speed claim
token-export/training health regresses        debug semantics, no speed claim
R36-like profile shows top_k > 10% again       reconsider top-k kernel work
profile dominated by TP comm                  topology workstream
profile dominated by ATen preprocessing       preprocessing/penalty fusion
startup dominates iteration cost              warm engine / runtime toggle
```

## Trainer Profile Fallback

Use this only after the production A/B is weak, failed, or masked enough that
trainer-side buckets become the next decision point. Do not use a two-node
trainer profile to claim production trainer percentages; it is only a hook/write
smoke. The production trainer profile needs the same 4-train + 12-inference
shape as the A/B, because FSDP/CP, LoRA, dataloader pressure, and broadcast wait
change with topology.

Supported hook:

```text
src/prime_rl/templates/multi_node_rl.sbatch.j2
PRIME_RL_NSYS_TRAINER=1 profiles TRAIN_NODE_RANK=0 with process-tree sampling
and writes $OUTPUT_DIR/nsys/trainer_node0.nsys-rep.
```

Preflight:

```bash
module load cuda/12.6
NSYS=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/profilers/Nsight_Systems/bin/nsys
"$NSYS" --version
```

Production trainer profile command, after the sampler production A/B indicates
trainer/broadcast/orchestrator may be masking the serving win:

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main
set -a
source .env
set +a

BASE=configs/calibration/gpqa_openended_debate_50step_bs512_g16_4t12i_simul_r64.toml
PROD_HOSTS=<comma-separated-16-node-lane>
PORT_BASE=<port-base>

export PRIME_RL_NSYS_TRAINER=1
export PRIME_RL_NSYS_BIN=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/profilers/Nsight_Systems/bin/nsys
export PRIME_RL_NSYS_DELAY=260
export PRIME_RL_NSYS_DURATION=120
unset PRIME_RL_NSYS

PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ "$BASE" \
  --max-steps 8 \
  --output-dir outputs/sampling-kernel/prod-trainer-nsys-<stamp> \
  --wandb.name prod-trainer-nsys-<stamp> \
  --deployment.hosts "$PROD_HOSTS" \
  --deployment.port-base "$PORT_BASE" \
  --deployment.lane-tag prod-trainer-nsys-<stamp>
```

Two-node hook smoke, allowed only to check the trace path on the free lane:

```bash
BASE=docs/plans/sampling-kernel/rl_probe_openended_debate_2node.toml
export PRIME_RL_NSYS_TRAINER=1
export PRIME_RL_NSYS_BIN=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/profilers/Nsight_Systems/bin/nsys
export PRIME_RL_NSYS_DELAY=260
export PRIME_RL_NSYS_DURATION=60
unset PRIME_RL_NSYS

uv run --no-sync rl @ "$BASE" \
  --max-steps 2 \
  --output-dir outputs/sampling-kernel/two-node-trainer-nsys-smoke-<stamp> \
  --deployment.hosts nid011175,nid011195 \
  --deployment.port-base <port-base> \
  --deployment.lane-tag two-node-trainer-nsys-smoke-<stamp>
```

Do not score the smoke if any new Slurm step lands outside
`nid011175,nid011195`.

Postprocess:

```bash
"$PRIME_RL_NSYS_BIN" stats --force-export=true \
  --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum,nvtx_sum \
  outputs/sampling-kernel/prod-trainer-nsys-<stamp>/*/nsys/trainer_node0.nsys-rep
```

Evidence to extract:

```text
trainer CUDA kernels: attention, MLP, fused lm head, optimizer, memcpy
CUDA API wait/sync: cudaEventSynchronize, cudaStreamSynchronize, graph launch
NVTX ranges: forward_backward, optimizer, data loading, broadcast if present
OS runtime: filesystem wait, futex/IPC wait, CPU scheduling
W&B side-by-side: time/wait_for_batch, time/forward_backward,
time/broadcast_weights, perf/mfu, perf/throughput
```

Decision:

```text
forward/backward dominates with high MFU       -> decode work is masked; trainer math is next only if worth the cost
broadcast dominates                            -> weight broadcast / overlap workstream
OS/filesystem wait dominates                   -> orchestration/checkpoint/export/data path workstream
CUDA API sync dominates outside useful kernels -> launch/graph/synchronization workstream
wait_for_batch still dominates                 -> inference/scheduler/orchestrator supply remains the target
```

## Immediate Next Command Sketch

Do not run the A/B blindly. Fill in concrete W&B run ids, output dirs, and lane
tags.

Initial observability read:

```bash
# First answer from existing data:
#   Is patched serving still rollout/inference-bound, or did it reach the
#   trainer crossover?
#
# Pull these from W&B or local run summaries:
#   time/wait_for_batch
#   time/forward_backward
#   time/broadcast_weights
#   time/step
#   perf/throughput
#   perf/mfu
#   inference/agg/running_requests
#   inference/agg/waiting_requests
#   inference/agg/kv_cache_usage_max
#   inference/agg/avg_queue_time_seconds
#
# Then append the Rung 0 decision table above.
```

Production A/B only if Rung 0 says inference is still material:

```bash
# native arm
unset PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB
unset PRIME_RL_ENABLE_FLASHINFER_SAMPLED_LOGPROB
unset PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL
unset PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE
unset PRIME_RL_FLASHINFER_SAMPLER_TAIL
unset PRIME_RL_FLASHINFER_SAMPLER_DENSE_PRESENCE
SLURM_JOB_ID=<job-id> SLURM_JOBID=<job-id> \
PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ \
  docs/plans/sampling-kernel/rl_probe_openended_debate_2node.toml \
  --deployment.hosts <free-node-a>,<free-node-b> \
  --deployment.port-base <port> \
  --deployment.lane-tag <native-tag> \
  --orchestrator.batch-size 512 \
  --orchestrator.max-inflight-rollouts 2300 \
  --output-dir outputs/sampling-kernel/<native-dir>

# patched arm
export PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL=triton
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL=1000
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_HIT_LOG_LIMIT=6
export PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK=1
SLURM_JOB_ID=<job-id> SLURM_JOBID=<job-id> \
PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ \
  docs/plans/sampling-kernel/rl_probe_openended_debate_2node.toml \
  --deployment.hosts <free-node-a>,<free-node-b> \
  --deployment.port-base <port> \
  --deployment.lane-tag <patched-tag> \
  --orchestrator.batch-size 512 \
  --orchestrator.max-inflight-rollouts 2300 \
  --output-dir outputs/sampling-kernel/<patched-dir>
```

## Summary

The Part 1 sampler work changed the roofline enough that further top-k tinkering
is now low-value. Part 2 should be a decision ladder:

```text
read existing observability to locate the current bottleneck
run cleaner production A/B only if inference still matters
prove training safety with canaries
productionize only if both pass
then move effort to preprocessing, topology, scheduler, or warm-engine control
based on the next profile
```

2026-06-26 21:24 UTC continuation audit:

```text
Do not spend more free-lane time on another sampler serving repeat unless a new
failure appears.

Rechecked current evidence:
  active allocation = 5379916
  current shell host = nid011153
  allowed free lane = nid011175,nid011195
  visible Slurm step = 5379916.batch on nid011153

R68/R69 serving repeat still passes:
  broad high-pressure = 13465.5 / 10856.0 = 1.240x patched/native
  strict decode      = 14439.6 / 11944.0 = 1.209x patched/native

R77 sampler-tail init subgoal still passes:
  4 jit_monitor sidecars, 60 total events, 0 _k_tail_uniform_kernel events
  learner_row_hit_rate = 1.0 at calls=2000 per worker

Current decision:
  two-node serving/JIT questions are answered enough for this ladder.
  The missing proof is full production startup-excluded E2E A/B, which cannot
  be proven on the two-node free lane.
```

## Completion Audit, 2026-06-26 21:27 UTC

This audit treats the Part 2 objective as unproven until each rung has direct
evidence. It is deliberately stricter than "the latest result looks good".

```text
rung 0 initial observability:
  status: proven enough to proceed
  evidence:
    W&B 4t12i/LoRA-64 wait_for_batch fractions are about 0.30-0.44 on
    relevant current rows; rollout/inference is still material.
  caveat:
    the current roofline predicts only about 1.04x-1.09x E2E from a
    1.13x-1.24x serving-only win, so the full A/B may miss the 1.08x gate.

rung 1 freeze new top-k work:
  status: policy passed
  evidence:
    R36 already collapsed visible native sampler/logprob CUDA time from
    33.66% to 4.06%; no fresh profile has reopened top-k as >10%.
  decision:
    do not write another top-k selection kernel without new profiler evidence.

rung 2 production A/B:
  status: partial
  two-node serving evidence:
    R68/R69 broad high-pressure patched/native = 1.240x
    R68/R69 strict decode patched/native = 1.209x
    both used nid011175,nid011195 with deployment lengths and pressure.
  missing evidence:
    full production startup-excluded E2E A/B on the real train/inference
    topology has not run.
  decision:
    serving gate passes; production E2E gate is still missing.

rung 3 training-quality canary:
  status: proven enough for default-off continuation
  evidence:
    R56 native and R57 patched token-export canaries both reached
    "RL trainer finished"; token exports had 0 shape mismatches and 0 bad
    numeric values; R57 sampler activation was real.
  caveat:
    this is not bitwise equivalence and not a long training-quality proof.

rung 4 default-off production cleanup:
  status: passed for default-off feature cleanup
  evidence:
    feature is env-gated, renamed finite-top-k sampled-logprob, has structured
    JSONL stats, learner-vs-warmup fallback accounting, docs, aliases, and
    version guards.
  caveat:
    default-on release still depends on full production A/B plus longer train
    health.

rung 5 preprocessing / penalty fusion:
  status: deferred
  evidence:
    dense presence is already included in the current candidate; sparse/fused
    preprocessing should wait for full production A/B and a fresh profile.

rung 6 topology recheck:
  status: deferred
  evidence:
    no patched-sampler EP-vs-TP production A/B exists.

rung 7 scheduler / concurrency / prefix cache:
  status: deferred
  evidence:
    current W&B says aggregate running can be below naive slot cap even with
    waiting/queue pressure; this is a future workstream, not solved by the
    sampler two-node lane.

rung 8 warm engine / JIT:
  status: partial
  sampler-tail evidence:
    R77 sidecars had 4 files, 60 total events, and 0 _k_tail_uniform_kernel
    events after warming both top_p=0.95 and float32-rounded
    top_p=0.949999988079071.
    A0 hardening now removes top_p from the Triton constexpr key, so future
    precompile probes should no longer need the float32-rounded duplicate.
  remaining gap:
    unrelated vLLM/MoE/slot/KV post-ready JIT events still exist. Do not call
    all warm-engine work complete.

rung 9 Gemma / K64:
  status: not part of Qwen production decision
  evidence:
    no Gemma K64 live probe has passed. Do not extrapolate Qwen K20 speedups.

active-goal status:
  not complete. The two-node-safe ladder work has enough evidence to stop
  spending the free lane on sampler repeats, but the full production E2E A/B
  remains unproven and cannot be proven on only nid011175,nid011195.
```

Repeatable local audit command:

```bash
uv run --no-sync python scripts/audit_sampling_kernel_goal_pt2.py
```

Production preflight-only command for a future 16-node lane:

```bash
PROD_HOSTS=<comma-separated-16-node-lane>
uv run --no-sync python scripts/audit_sampling_kernel_goal_pt2.py \
  --preflight-only \
  --allocation-hosts "$PROD_HOSTS" \
  --allowed-hosts "$PROD_HOSTS" \
  --required-train-nodes 4 \
  --required-inference-replicas 12
```

On the current two-node-free lane this intentionally exits nonzero: config and
template gates pass, topology readiness fails.

Audit logic coverage:

```bash
uv run --no-sync pytest tests/unit/inference/test_sampling_kernel_goal_audit.py
```

Current output:

```text
| gate | status | key result |
|---|---|---|
| broad two-node serving | pass | 12 native rows, 12 patched rows, ratio 1.240x |
| strict decode two-node serving | pass | 5 native rows, 6 patched rows, ratio 1.209x |
| sampler-tail JIT | pass | 4 files, 60 events, _k_tail_uniform_kernel=0 |
| sampler stats | pass | 4 files, min learner row hit rate 1.000, learner fallback rows 0 |
| training canary token export | pass | native 4/2 files/STABLE, rows 32, bad numeric 0; patched 4/2 files/STABLE, rows 32, bad numeric 0 |
| production config shape | pass | configs/calibration/gpqa_openended_debate_50step_bs512_g16_4t12i_simul_r64.toml, checked 24 fields, mismatches 0 |
| trainer Nsight hook | pass | src/prime_rl/templates/multi_node_rl.sbatch.j2, checked 8 snippets, missing 0 |
| sampler-tail specialization | pass | constexprs=[10], top_p_values=[0.95], error=none |
| full production E2E | missing | requires real production topology; not provable from the two-node lane |
| full production topology preflight | missing | requires 16 nodes (4 train + 12 inference); allocation has 4, allowed lane has 2 |

local_two_node_gates_pass: true
full_production_ready: false
goal_complete: false
```
