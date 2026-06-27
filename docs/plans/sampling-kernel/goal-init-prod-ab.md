# Init Objective + Production A/B Goal

Date: 2026-06-26

This is the launchable goal for turning the finite-top-k sampled-logprob patch
into a production decision. It splits three things that are easy to conflate:

```text
init objective:
  prove startup/JIT/cache behavior is understood and not polluting steady-state
  measurements; read existing trainer/inference telemetry before spending more
  allocation time

two-node production-pressure A/B:
  production-length, production-shape serving decision on one inference replica

full production A/B:
  end-to-end RL wall-clock decision on the real train/inference topology
```

Do not claim production E2E speedup from the two-node lane. That lane answers
whether the serving hot path still wins at deployment size.

## One-Page Contract

Init objective:

```text
Input:
  existing W&B trainer/inference telemetry plus local JIT/stats sidecars

Questions:
  1. Is rollout/inference still a material part of step time?
  2. Are startup, cache, and sampler-tail JIT effects separated from the
     measured steady-state rows?
  3. Is the patched path actually active on learner traffic, with no learner
     fallback rows?

Pass:
  wait_for_batch is large enough that a serving win can matter;
  _k_tail_uniform_kernel does not appear after the JIT monitor is armed;
  finite-top-k sampler row hit rate is >= 0.999 on learner traffic;
  fallback rows, if any, are warmup/profiling only.

Fail:
  trainer is no longer materially waiting for inference;
  sampler-tail JIT appears in the measured window;
  row hit rate drops on learner traffic;
  startup/cache behavior dominates the measured rows.

Action:
  pass -> spend full production A/B;
  fail -> fix the named init problem, or pivot away from sampler work.
```

Two-node production-pressure A/B:

```text
Purpose:
  serving-only gate at deployment length/shape on one inference replica

Arms:
  native vLLM sampler/logprob
  finite-top-k sampled-logprob patch

Pinned lane:
  nid011175,nid011195 only, inside Slurm job 5379916

Primary metric:
  generation tok/s during high-pressure decode rows

High-pressure row:
  Running == 256, Waiting > 0, 15% <= KV max <= 45%

Pass:
  patched/native >= 1.10x on the broad high-pressure filter,
  with strict decode rows not contradicting it.

Current status:
  passed twice enough for promotion:
    R66/R67 broad = 1.137x
    R68/R69 broad = 1.240x

Action:
  do not spend more two-node serving repeats unless the full production A/B is
  blocked or a fresh profile invalidates this evidence.
```

Full production A/B:

```text
Purpose:
  decide whether the serving win moves end-to-end RL wall-clock

Topology:
  real production topology, not the two-node lane
  4 trainer nodes + 12 inference replicas

Arms:
  native = PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB unset/0
  patched = PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1

Primary metric:
  startup-excluded step wall-clock, same step window in both arms

Secondary metrics:
  wait_for_batch fraction
  forward_backward time
  broadcast_weights time
  inference generation tok/s
  running/waiting requests
  queue time
  KV usage
  prefix-cache hit rate
  completion length distribution
  sampler fallback stats
  token-export / logprob correctness canary

Pass:
  startup-excluded E2E step speed >= 1.08x,
  no correctness or training-health regression,
  sampler fallback on learner traffic remains zero or explainably negligible.

Weak positive:
  E2E step speed 1.03x-1.08x with serving still clearly faster.
  This is not a failed serving patch; it means the current production loop is
  partly trainer/broadcast/orchestrator limited.

Fail:
  E2E step speed < 1.03x,
  or serving no longer wins,
  or training-health/correctness gates fail.

Action:
  pass -> productionize default-off for this workload;
  weak positive -> keep opt-in and optimize the next largest bucket;
  fail -> stop sampler productionization unless a fresh profile reopens it.
```

## Goal Statement

Goal:

```text
Decide whether the finite-top-k sampled-logprob path is worth productionizing
for the current Qwen3.5-A3B GPQA open-ended debate workload.
```

Definition of done:

```text
1. Init objective says rollout/inference is still material, and startup/JIT/cache
   effects are controlled enough to interpret steady-state rows.

2. Two-node production-pressure A/B says whether patched serving still wins at
   deployment length and vLLM shape.

3. Full production A/B says whether that serving win moves end-to-end RL
   wall-clock after startup is excluded.

4. The final decision is one of:
     productionize as default-off for this workload;
     keep opt-in and pivot to trainer/broadcast/orchestrator/KV work;
     stop sampler productionization.
```

## Decision

Question:

```text
Should PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB become a production feature
for the current Qwen3.5-A3B GPQA open-ended debate workload?
```

Hypotheses:

```text
H1 serving:
  At deployment sizes, patched serving improves high-pressure generation
  throughput by >= 1.10x versus native vLLM sampling/logprob.

H2 E2E:
  If serving improves, the full RL loop improves startup-excluded step
  wall-clock by >= 1.08x. If not, the bottleneck has moved to trainer compute,
  broadcast, orchestration, env supply, or KV/prefix behavior.

H3 init:
  Hot-cache startup and post-ready JIT are controlled enough that they do not
  explain the observed throughput delta. If this fails, fix init/cache hygiene
  before interpreting another A/B.
```

Falsification:

```text
H1 false if patched/native < 1.05x broad high-pressure generation tok/s.
H2 false if patched/native < 1.03x startup-excluded E2E step speed.
H3 false if post-ready JIT appears in steady-state rows, cache namespaces drift,
or repeated hot-cache startup is still hundreds of seconds.
```

## Deployment Shape

All speed/prod arms must use deployment sizes:

```text
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
max_inflight_rollouts = 2300
sampling = temperature 1.0, top_k 20, top_p 0.95, presence_penalty 1.5
```

Use `docs/plans/sampling-kernel/rl_probe_openended_debate_2node.toml` for the
two-node pressure lane. Do not use the `b16` token-export configs for speed
adjudication unless their batch and inflight settings are explicitly overridden
and verified; they are semantic canaries, not production-pressure runs.

## Init Objective

Purpose:

```text
Before launching a long A/B, decide whether inference/rollout is still material
and whether startup/JIT/cache effects are controlled.
```

Read existing telemetry first:

```text
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
vLLM generation tok/s, Running, Waiting, KV, prefix hit rate
completion length p50/p90/max if available
```

Compute:

```text
trainer_wait_fraction = time/wait_for_batch / time/step
trainer_compute_fraction = time/forward_backward / time/step
broadcast_fraction = time/broadcast_weights / time/step
```

Decision table:

```text
wait high + Running ~= 256 + Waiting > 0      -> run two-node pressure A/B
wait low + forward_backward high              -> pivot trainer/overlap
Running below cap + Waiting low               -> pivot scheduler/env supply
Running saturated + KV high                   -> pivot KV/concurrency/prefix
post-ready JIT during measured rows           -> fix init/cache before A/B
hot-cache startup still hundreds of seconds   -> fix compile/cache hygiene
```

Init/cache hygiene:

```text
Use the same PRIME_RL_COMPILE_CACHE_NAMESPACE for comparable arms unless
intentionally running a cold-start test.

For cold-start diagnosis only:
  PRIME_RL_CLEAN_COMPILE_CACHE=1

For steady A/B:
  do not clean compile caches between arms unless both arms are cleaned and
  startup is explicitly excluded from the primary metric.

Preserve the launcher/template defaults:
  PRIME_RL_JIT_MONITOR_LOG_DIR=$OUTPUT_DIR/logs/inference
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_LOG_DIR=$OUTPUT_DIR/logs/inference

Do not manually export those two paths from a shell unless OUTPUT_DIR is set
for that exact run; the multi-node templates normally populate them.

Optional patched sampler-tail precompile probe:
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL=1
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_K=20
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P=0.95
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_VOCAB=248320
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES=1

After the A0 tail hardening, top_p is a runtime scalar argument rather than a
Triton constexpr key. The precompile log should therefore show only the
configured top_p:
  top_p_values=[0.95]

R70/R73/R74 showed that naive batch-list precompile still let
_k_tail_uniform_kernel appear after ready. R76 found the likely reason:
Triton's constexpr key was not batch-missing; it used the float32-rounded
TOP_P=0.949999988079071, while the old precompile only warmed Python 0.95.
Do not accept init/JIT cleanliness from a patched arm unless sidecars confirm
_k_tail_uniform_kernel is absent after ready under real traffic.
```

Init acceptance:

```text
Rung passes if:
  no post-ready Triton JIT events occur in measured steady-state windows
  hot-cache startup is stable enough to launch repeated arms
  rollout/inference remains material by wait/running/waiting evidence

Rung fails if:
  JIT sidecars show new compile events during measured rows
  hot-cache runs still behave like cold starts
  trainer wait is already small and forward/backward dominates

Rung is provisional if:
  sampler-tail JIT is gone but unrelated vLLM/MoE/cudagraph kernels still
  compile before the measured steady-state window. In that case, exclude startup
  and first-traffic rows, then record the exact sidecar window used.
```

## Launch Discipline

For `rl` in an existing allocation, do not wrap the launcher in an outer `srun`.
Run `rl` plainly and pin the lane through deployment hosts; the launcher creates
its own exact internal `srun` steps.

Raw one-off probes that are not `rl` may use explicit `srun --jobid ...` pinned
to the correct free nodes.

Agent-shell exception:

```text
If a plain rl launch from an agent shell creates an out-of-lane wrapper step,
do not score that run. For a narrow JIT probe only, render with --dry-run, patch
the generated script's internal lane srun to use --overlap and
--ntasks="$LANE_NNODES", then launch that generated script under a one-node
outer srun pinned to the allowed head node. Treat this as a workaround, not the
normal production launch path.
```

Template:

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main
set -a
source .env
set +a

BASE=docs/plans/sampling-kernel/rl_probe_openended_debate_2node.toml
```

Native arm:

```bash
unset PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB
unset PRIME_RL_ENABLE_FLASHINFER_SAMPLED_LOGPROB
unset PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL
unset PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE
unset PRIME_RL_FLASHINFER_SAMPLER_TAIL
unset PRIME_RL_FLASHINFER_SAMPLER_DENSE_PRESENCE

PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ "$BASE" \
  --output-dir outputs/sampling-kernel/<native-run> \
  --deployment.hosts <train-node>,<infer-node> \
  --deployment.port-base <port-base> \
  --deployment.lane-tag <native-lane-tag>
```

Patched arm:

```bash
export PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL=triton
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL=1000
export PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK=1

PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ "$BASE" \
  --output-dir outputs/sampling-kernel/<patched-run> \
  --deployment.hosts <train-node>,<infer-node> \
  --deployment.port-base <port-base> \
  --deployment.lane-tag <patched-lane-tag>
```

## Two-Node Production-Pressure A/B

Purpose:

```text
Decide whether the serving hot path still wins at deployment length and
production vLLM shape.
```

Arms:

```text
native:
  finite-top-k sampled-logprob env unset

patched:
  PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL=triton
  PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1
```

Order:

```text
preferred: native -> patched -> patched -> native
minimum interpretable: native -> patched, but mark as weak
```

Pressure gate:

```text
sustained Running ~= 256
Waiting > 0
KV band recorded
prefix-cache band recorded
at least 10 broad high-pressure rows per arm
at least 6 strict decode rows per arm if available
```

Primary metric:

```text
steady generation tokens/s after startup is excluded
```

Broad row filter:

```text
Running == 256
Waiting > 0
15% <= GPU KV cache usage <= 45%
0 <= prompt tokens/s <= 10000
```

Strict decode filter:

```text
Running == 256
Waiting > 0
15% <= GPU KV cache usage <= 45%
prompt tokens/s == 0
```

Summarize:

```bash
uv run --no-sync python scripts/summarize_vllm_throughput.py \
  native=<native-run>/logs/inference/node_0.log \
  patched=<patched-run>/logs/inference/node_0.log \
  --first-n 12 \
  --running 256 \
  --min-waiting 1 \
  --min-kv-cache-pct 15 \
  --max-kv-cache-pct 45 \
  --min-prompt-tokens-s 0 \
  --max-prompt-tokens-s 10000 \
  --min-matching-points 10 \
  --print-points
```

Strict decode summary:

```bash
uv run --no-sync python scripts/summarize_vllm_throughput.py \
  native=<native-run>/logs/inference/node_0.log \
  patched=<patched-run>/logs/inference/node_0.log \
  --first-n 12 \
  --running 256 \
  --min-waiting 1 \
  --min-kv-cache-pct 15 \
  --max-kv-cache-pct 45 \
  --prompt-tokens-s 0 \
  --print-points
```

Promotion gate:

```text
patched/native >= 1.10x broad high-pressure generation tok/s
learner fallback row rate <= 1%
no response-shape, token-export, or non-finite failure
no strict-decode regression if strict rows exist
```

Failure gate:

```text
patched/native < 1.05x broad high-pressure generation tok/s
or learner fallback row rate > 1%
or new correctness/training-health issue
```

Output table:

```text
run | arm | order | nodes | output_dir | startup excluded? | matching rows
broad mean tok/s | broad ratio | strict rows | strict mean tok/s | strict ratio
prefix % band | KV % band | sampler learner_row_hit_rate | fallback reasons
JIT sidecars after ready? | decision
```

## Full Production A/B

Purpose:

```text
Decide whether the serving win moves the real RL loop.
```

Required production shape:

```text
batch_size = 512
group_size = 16
target_lag = 3
max_inflight_rollouts = 2300
max_num_seqs = 256
production train/inference topology, ideally:
  4 train nodes + 12 inference replicas
```

Primary metric:

```text
startup-excluded end-to-end trainer/orchestrator step wall-clock
```

Supporting metrics:

```text
time/wait_for_batch
time/forward_backward
time/broadcast_weights
perf/throughput
perf/mfu
generation tok/s under high pressure
inference running/waiting/KV/prefix bands
sampler learner_row_hit_rate and fallback_reason_by_traffic
JIT monitor sidecars
completion length p50/p90/max
train health: finite losses/logprobs, no token-export shape regression
```

Preferred order:

```text
native -> patched -> patched -> native
```

Production config:

```text
configs/calibration/gpqa_openended_debate_50step_bs512_g16_4t12i_simul_r64.toml
```

Preflight:

```text
Do not launch this on the current two-node free lane. The production gate needs
16 nodes: 4 trainer nodes plus 12 one-node TP=4 inference replicas.
```

Concrete preflight command:

```bash
PROD_HOSTS=<comma-separated-16-node-lane>

uv run --no-sync python scripts/audit_sampling_kernel_goal_pt2.py \
  --preflight-only \
  --allocation-hosts "$PROD_HOSTS" \
  --allowed-hosts "$PROD_HOSTS" \
  --required-train-nodes 4 \
  --required-inference-replicas 12
```

This mode does not read historical R68/R69/R77 output logs. It checks only the
production config shape, trainer Nsight hook, and requested production topology,
and exits nonzero if `full_production_ready` is false.

Launch template:

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main
set -a
source .env
set +a

BASE=configs/calibration/gpqa_openended_debate_50step_bs512_g16_4t12i_simul_r64.toml
PROD_HOSTS=<comma-separated-16-node-lane>
PORT_BASE=<port-base>

# Native arm.
unset PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB
unset PRIME_RL_ENABLE_FLASHINFER_SAMPLED_LOGPROB
unset PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL
unset PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE
unset PRIME_RL_FLASHINFER_SAMPLER_TAIL
unset PRIME_RL_FLASHINFER_SAMPLER_DENSE_PRESENCE

PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ "$BASE" \
  --max-steps 8 \
  --output-dir outputs/sampling-kernel/prod-ab-native-<stamp> \
  --wandb.name prod-ab-native-finite-topk-off-<stamp> \
  --deployment.hosts "$PROD_HOSTS" \
  --deployment.port-base "$PORT_BASE" \
  --deployment.lane-tag prod-ab-native-<stamp>

# Patched arm.
export PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB=1
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL=triton
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE=1
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL=1000
export PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_HIT_LOG_LIMIT=6
export PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK=1

PRIME_RL_COMPILE_CACHE_NAMESPACE=qwen35-a3b-tp4-lora64-vllm022-fa3 \
uv run --no-sync rl @ "$BASE" \
  --max-steps 8 \
  --output-dir outputs/sampling-kernel/prod-ab-patched-<stamp> \
  --wandb.name prod-ab-patched-finite-topk-on-<stamp> \
  --deployment.hosts "$PROD_HOSTS" \
  --deployment.port-base "$PORT_BASE" \
  --deployment.lane-tag prod-ab-patched-<stamp>
```

Immediately after launch:

```bash
squeue --job "$SLURM_JOB_ID" --steps -o '%i %j %M %N'
```

Score only if every new non-batch step is confined to the intended 16-node
production lane and the two arms have distinct output dirs, W&B names, and
`lane_tag` values. Reuse `port_base` only for sequential arms; concurrent arms
must use distinct `port_base` values. Use `--max-steps 8` for the first
production A/B proof; a default-on decision still needs a longer health run.

Minimum interpretable full-production run:

```text
at least 2 completed startup-excluded train steps per arm
at least 10 high-pressure vLLM rows per arm per representative replica
clean sampler stats on patched arms
```

Success gate:

```text
patched/native >= 1.08x startup-excluded E2E step speed
and patched/native >= 1.10x broad high-pressure serving throughput
and no training-health regression
```

Strong success:

```text
patched/native >= 1.12x startup-excluded E2E step speed
```

Failure:

```text
patched/native < 1.03x startup-excluded E2E step speed
or serving win does not reproduce under production pressure
or any new correctness/training-health failure appears
```

Interpretation:

```text
serving pass + E2E pass:
  productionize the default-off feature, then decide separately whether to make
  it default-on for this workload

serving pass + E2E fail:
  keep the patch as opt-in, pivot to trainer/broadcast/orchestrator/KV profiling

serving fail:
  stop sampler productionization; do not write more top-k kernels

init fail:
  fix warm-engine/cache/JIT behavior before spending another A/B
```

Trainer-profile fallback:

```text
If the full production A/B lands in the weak-positive band or fails while
serving still wins, collect a production trainer Nsight trace before claiming
which non-serving bucket took over. Set PRIME_RL_NSYS_TRAINER=1, leave
PRIME_RL_NSYS unset unless intentionally collecting inference too, and score
only $OUTPUT_DIR/nsys/trainer_node0.nsys-rep from the real 4-train +
12-inference topology.

A two-node trainer Nsight run on nid011175,nid011195 is only a hook/write smoke,
not a production trainer profile.
```

## Non-Goals

```text
no new custom top-k kernel unless a fresh profile reopens that bucket
no FP8/KV experiment in this ladder
no EP vs TP topology claim from these A/Bs
no default-on release decision before full production A/B passes
no quality claim beyond the existing canary unless a longer training run passes
```

## Result Slot

Current filled result:

```text
init read:
  W&B run ids: none; R66/R67 were local output-only lanes
  output dirs:
    native  = outputs/sampling-kernel/r66-native-pt2-ab-n175n195/r66-native-n175n195
    patched = outputs/sampling-kernel/r67-patched-pt2-ab-n175n195/r67-patched-n175n195
  trainer bucket metrics:
    unavailable; both runs were stopped at Train batch 0/512
  inference running/waiting/KV:
    Running reached 256 with Waiting > 0 on both arms
    broad filter used 15% <= KV <= 45%
  JIT sidecar summary:
    R66 and R67 both wrote jit_monitor sidecars
    R67 has 25 JIT events per worker, including early-traffic K-tail and
    MoE/LoRA events around the first high-pressure rows
  decision:
    serving-pressure gate is positive but init/JIT cleanliness is not proven
    full E2E production gate remains unrun

two-node A/B:
  native output dirs:
    outputs/sampling-kernel/r66-native-pt2-ab-n175n195/r66-native-n175n195
  patched output dirs:
    outputs/sampling-kernel/r67-patched-pt2-ab-n175n195/r67-patched-n175n195
  broad ratio:
    first 12 broad 15%-45% KV rows = 12253.5 / 10775.2 = 1.137x
    all broad 15%-45% KV rows = 8505.6 / 7603.8 = 1.119x
  strict ratio:
    strict decode 15%-45% KV = 13919.9 / 10038.3 = 1.387x
    caveat: only 4 patched rows and 5 native rows
  learner fallback row rate:
    no JSONL sidecar yet in R67; text log final row_hit_rate = 0.999913
    fallback_reason_hist = {"max_num_logprobs_not_width1": 2}
  correctness notes:
    no response-shape/token-export/non-finite failure seen in these logs
    not a training-quality proof; run stopped before completing train batch 0
  decision:
    promote to full production A/B candidate
    do not claim production E2E speedup

patched repetition:
  output dir:
    outputs/sampling-kernel/r68-patched-repeat-n175n195/r68-patched-repeat-n175n195
  nodes:
    nid011175,nid011195
  broad result:
    first 12 broad 15%-45% KV rows = 13465.5 gen tok/s
  strict result:
    6 strict decode 15%-45% KV rows = 14439.6 gen tok/s
  learner fallback row rate:
    0; learner_row_hit_rate = 1.0
    only fallback reason was warmup/profiling max_num_logprobs_not_width1
  JIT sidecar summary:
    100 total sidecar events, 25 per worker
    first event 2026-06-26T19:35:22.757507+00:00
    last event 2026-06-26T19:37:37.734090+00:00
    kernels include fused_moe_kernel, _fused_moe_lora_one_shot_kernel,
    _zero_kv_blocks_kernel, _compute_slot_mapping_kernel, and _k_tail_uniform_kernel
  decision:
    good patched pressure/statistics repetition
    not a clean init/JIT pass; compare against a native repeat before using it
    as a fresh ratio

native repetition:
  output dir:
    outputs/sampling-kernel/r69-native-repeat-n175n195/r69-native-repeat-n175n195
  nodes:
    nid011175,nid011195
  launch note:
    top-level rl process exited with code 137 after spawning the internal Slurm
    step, but step 5379916.43 continued on the correct lane and was salvaged
    from logs; the step was cancelled after the stop rule passed
  broad result:
    first 12 broad 15%-45% KV rows = 10856.0 gen tok/s
  strict result:
    5 strict decode 15%-45% KV rows = 11944.0 gen tok/s
  sampler stats:
    no finite_topk_sampler_stats sidecar, as expected for native vLLM
  JIT sidecar summary:
    92 total sidecar events across 4 workers
    first event 2026-06-26T19:44:28.576188+00:00
    last event 2026-06-26T19:45:57.784396+00:00
    kernels include fused_moe_kernel, _fused_moe_lora_one_shot_kernel,
    _topk_topp_kernel, _zero_kv_blocks_kernel, and _compute_slot_mapping_kernel

fresh R68/R69 pair:
  broad ratio:
    first 12 broad 15%-45% KV rows = 13465.5 / 10856.0 = 1.240x
  strict ratio:
    strict decode 15%-45% KV rows = 14439.6 / 11944.0 = 1.209x
    caveat: only 6 patched rows and 5 native rows
  decision:
    serving-pressure gate passes again
    init/JIT cleanliness still does not pass; both arms logged JIT sidecar
    events during measured traffic
    full E2E production gate remains unrun

patched precompile/JIT probe:
  output dir:
    outputs/sampling-kernel/r70-patched-precompile-n175n195/r70-patched-precompile-n175n195
  nodes:
    nid011175,nid011195
  launch note:
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL=1 was enabled with
    top_k=20, top_p=0.95, vocab=248320
  startup:
    sampler tail precompiled before JIT monitor activation on all 4 workers
    JIT monitor activated at 19:58:10
    engine init took 86.50s, compilation 7.00s
  broad result:
    13 broad 15%-45% KV rows
    first 12 broad rows = 11958.8 gen tok/s
    all 13 broad rows = 11598.7 gen tok/s
  learner fallback row rate:
    0; learner_row_hit_rate = 1.0
    only fallback reason was warmup/profiling max_num_logprobs_not_width1
  JIT sidecar summary:
    96 total sidecar events across 4 workers
    first event 2026-06-26T19:58:54.665588+00:00
    last event 2026-06-26T20:01:09.320863+00:00
    kernels include fused_moe_kernel, _fused_moe_lora_one_shot_kernel,
    _compute_slot_mapping_kernel, _zero_kv_blocks_kernel, causal conv/post-conv,
    batch_memcpy_kernel, and _k_tail_uniform_kernel
  decision:
    batch=1 sampler-tail precompile is not enough for init cleanliness
    serving remains healthy, but post-ready JIT coverage is still open

patched batch-shaped precompile implementation:
  code:
    src/prime_rl/inference/vllm/flashinfer_sampler.py supports
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES, a comma-separated
    list that defaults back to the older single PRECOMPILE_BATCH knob
  local constructor probe:
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES=1,128,256
    successfully precompiled batches [1, 128, 256] before constructing Sampler
  launch attempts:
    R71/R72 are tainted and should not be used as speed or JIT evidence
    R71/R72 launch attempts produced or coincided with one-node bash steps on
    nid011166 even though the inner lane step was pinned to nid011175,nid011195
  decision:
    do not relaunch from this Codex shell unless launch isolation is proven
    next valid probe must show Slurm steps only on nid011175,nid011195, plus the
    unavoidable batch shell

patched batch-shaped precompile/JIT probe:
  output dir:
    outputs/sampling-kernel/r73-patched-precompile-batches-overlap-n175n195/r73-patched-precompile-batches-overlap-n175n195
  nodes:
    nid011175,nid011195
  launch note:
    R73 used a generated-script workaround because the agent shell could not
    launch a plain rl process without creating an out-of-lane wrapper step.
    The outer wrapper step was pinned to nid011175 and the internal lane step
    was pinned to nid011175,nid011195. The pre-existing Gemma4 step on
    nid011153,nid011166 was left untouched and is not part of this evidence.
  precompile:
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES=1,128,256
    all 4 workers logged precompile before JIT monitor activation
  startup:
    JIT monitor activated at 20:26:46
    engine init took 84.86s, compilation 6.78s
  broad result before stop:
    3 broad 15%-45% KV rows
    mean = 10473.1 gen tok/s
    rows = 10579.9, 11139.7, 9699.7
  learner fallback row rate:
    0; learner_row_hit_rate = 1.0 at calls=1000 and calls=2000
    only fallback reason was warmup/profiling max_num_logprobs_not_width1
  JIT sidecar summary:
    80 total sidecar events across 4 workers
    first event 2026-06-26T20:27:44.363901+00:00
    last event 2026-06-26T20:28:32.041509+00:00
    _k_tail_uniform_kernel still appeared 4 times
    other kernels included fused_moe_kernel, _fused_moe_lora_one_shot_kernel,
    _zero_kv_blocks_kernel, _compute_slot_mapping_kernel, causal conv/post-conv,
    and batch_memcpy_kernel
  decision:
    batch-shaped sampler-tail precompile with batches 1,128,256 is not enough
    to prove the init objective "no post-ready JIT during measured windows"
    do not spend another two-node pressure repeat on this unless the purpose is
    explicitly warmup/JIT coverage rather than serving speed
    full production E2E A/B remains the next real production proof

patched observed-batch precompile/JIT probe:
  output dir:
    outputs/sampling-kernel/r74-patched-precompile-observed-batches-n175n195/r74-patched-precompile-observed-batches-n175n195
  nodes:
    nid011175,nid011195
  launch note:
    Same generated-script workaround as R73; new R74 steps were confined to
    nid011175,nid011195. The pre-existing Gemma4 step on nid011153,nid011166
    was left untouched.
  precompile:
    PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES=1,47,98,128,241,256
    all 4 workers logged precompile before JIT monitor activation
  startup:
    JIT monitor activated at 20:39:33
    engine init took 84.41s, compilation 5.90s
  observed first fast-path batches:
    1, 1, 9, 54, 176, 250, 256, 256
  broad result before stop:
    3 broad 15%-45% KV rows
    mean = 10841.1 gen tok/s
    rows = 5779.0, 14448.7, 12295.7
  learner fallback row rate:
    0; learner_row_hit_rate = 1.0 at calls=1000
    only fallback reason was warmup/profiling max_num_logprobs_not_width1
  JIT sidecar summary:
    68 total sidecar events across 4 workers
    first event 2026-06-26T20:40:27.220600+00:00
    last event 2026-06-26T20:40:52.941202+00:00
    _k_tail_uniform_kernel still appeared 4 times
    other kernels included fused_moe_kernel, _fused_moe_lora_one_shot_kernel,
    _zero_kv_blocks_kernel, _compute_slot_mapping_kernel, causal conv/post-conv,
    and batch_memcpy_kernel
  decision:
    exact observed-batch precompile is not sufficient. The remaining sampler-tail
    JIT is not just "forgot to include batch 47/98/241"; it still appears on the
    first batch=1 learner hit after batch=1 was precompiled.
    do not run more two-node precompile-list probes
    if init/JIT cleanliness is release-blocking, the next useful work is inside
    the actual Triton/FlashInfer/vLLM warmup integration, not another env-list
    experiment
    otherwise use startup-excluded metrics and move to full production E2E A/B

same-process K-tail precompile diagnostic:
  node:
    nid011175
  shape:
    one Python process, one CUDA device visible, no vLLM server/model load
  result:
    precompiled K=20 top_p=0.95 batch=1 before installing the hook
    repeated K=20 top_p=0.95 batch=1 after installing the hook -> no hook event
    K=20 top_p=0.95 batch=9 -> no hook event
    K=21 top_p=0.95 batch=1 -> _k_tail_uniform_kernel hook event
    K=20 top_p=0.90 batch=1 -> _k_tail_uniform_kernel hook event
  decision:
    the local precompile function is not simply a no-op, and the hook can detect
    real new K-tail compiles. R74's post-ready K-tail JIT is therefore likely a
    production vLLM lifecycle/specialization/warmup integration issue, not a
    missing env-list batch size.

JIT sidecar enrichment:
  code:
    src/prime_rl/inference/vllm/jit_monitor.py now includes sanitized Triton
    hook details in each JSONL record: key, repr, already_compiled,
    is_manual_warmup, and compile when present.
  tests:
    uv run --no-sync pytest tests/unit/inference/test_jit_monitor.py
    uv run --no-sync ruff check src/prime_rl/inference/vllm/jit_monitor.py tests/unit/inference/test_jit_monitor.py
    uv run --no-sync python -m py_compile src/prime_rl/inference/vllm/jit_monitor.py tests/unit/inference/test_jit_monitor.py
  decision:
    the next JIT-cleanliness probe should inspect Triton compile details from
    production vLLM before attempting another warmup fix.

patched enriched-JIT root-cause probe:
  output dir:
    outputs/sampling-kernel/r76-enriched-jit-details-n175n195/r76-enriched-jit-details-n175n195
  nodes:
    nid011175,nid011195
  precompile:
    old precompile warmed TOP_P=0.95
  key observation:
    production traffic compiled _k_tail_uniform_kernel with Triton constexpr
    TOP_P=0.949999988079071
  interpretation:
    R70/R73/R74 did not prove that sampler-tail warmup is impossible. They
    mostly proved that warming Python 0.95 does not warm vLLM's float32-rounded
    top_p key.
  decision:
    historical workaround was to warm both configured top_p and runtime
    float32-rounded top_p. The A0 tail hardening supersedes that workaround by
    removing top_p from the Triton constexpr key.

patched fixed-top-p precompile probe:
  output dir:
    outputs/sampling-kernel/r77-fixed-top-p-precompile-n175n195/r77-fixed-top-p-precompile-n175n195
  nodes:
    nid011175,nid011195
  precompile:
    all workers logged
      top_p_values=[0.95, 0.949999988079071]
    This is historical R77 evidence. Current A0-hardened code should log only
    top_p_values=[0.95] because top_p is no longer constexpr-specialized.
  traffic:
    Running reached 256 with Waiting > 0
    strict decode rows included:
      15307.7 gen tok/s, Waiting=2560, KV=18.2%
      11912.3 gen tok/s, Waiting=3307, KV=16.1%
      13055.0 gen tok/s, Waiting=3724, KV=19.2%
  sampler stats:
    at calls=2000 per worker, learner_row_hit_rate=1.000000
    fallback_reason_hist={'max_num_logprobs_not_width1': 2}
    fallback traffic was warmup/profiling only
  JIT sidecar summary:
    4 files, 60 total post-ready events
    first event 2026-06-26T21:05:37.488467+00:00
    last event 2026-06-26T21:05:57.710713+00:00
    kernels:
      _zero_kv_blocks_kernel: 12
      _compute_slot_mapping_kernel: 12
      _causal_conv1d_fwd_kernel: 4
      _fused_post_conv_kernel: 4
      batch_memcpy_kernel: 4
      fused_moe_kernel: 16
      _fused_moe_lora_one_shot_kernel: 8
    _k_tail_uniform_kernel: 0
  status:
    sampler-tail init subgoal passes. The remaining post-ready JIT work is
    unrelated vLLM/MoE/slot/KV warmup, not the finite-top-k tail.
    R77 is not a production A/B arm; it was stopped after the JIT evidence
    window and should not be used as an E2E speed result.

fresh W&B init read:
  fetched:
    2026-06-26 from jvelja-private/gpqa-openended-debate-calibration
  refresh command:
    uv run --no-sync python scripts/analyze_wandb_production_gate.py \
      simul=b32544b2d5db411b929dd6496da34f61 \
      pcd4=9ba4977c913d46b3a927ced63d907cf0 \
      seq4=60ddf022c6e4417ea6b2934680291aea \
      mallopt=db474e36863a4faaabb3c03f2ccecdce
  production A/B comparison command:
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
  output:
    comparison rows include decision = pass | weak_positive | mixed | fail | missing
    W&B inference metrics are aggregate telemetry; vLLM per-replica logs still
    adjudicate high-pressure serving rows. The running-cap and waiting-positive
    fraction gates fail closed when an arm is underfed. The full-production W&B
    scorer uses an aggregate running floor of 1024; only explicit
    single-replica/two-node serving lanes should opt down to 256.
  analyzer logic coverage:
    uv run --no-sync pytest tests/unit/inference/test_wandb_production_gate.py
  pitfall:
    use unfiltered scan_history for current runs. A filtered scan_history(keys=...)
    probe returned 0 rows even when the run had history and summary keys.
  exact/current-ish 4t12i LoRA-64 rows:
    b32544b2 qwen35-bs512-g16-4t12i-50step-openended-simul-r64-sbatch
      trainer rows=20, wait_frac=0.296, fwd_mean=214s, broadcast_mean=15.7s
      inference cap=1536, running_med=1154, waiting_med=725, queue_med=163s,
      kv_mean_med=0.152, kv_max_med=0.397
    9ba4977c qwen35-bs512-g16-4t12i-50step-openended-pcd4-r64
      trainer rows=25, wait_frac=0.355, fwd_mean=190s, broadcast_mean=18.1s
      inference cap=1358, running_med=1156, waiting_med=3, queue_med=1.4s,
      kv_mean_med=0.137, kv_max_med=0.360
    90f47a25 qwen35-bs512-g16-4t12i-50step-openended-pcd4final-r64
      trainer rows=13, wait_frac=0.330, fwd_mean=221s, broadcast_mean=15.3s
      inference cap=1435, running_med=1245, waiting_med=12, queue_med=1.5s,
      kv_mean_med=0.145, kv_max_med=0.381
    3e14ae91 qwen35-bs512-g16-4t12i-50step-openended-sequential4-r64
      trainer rows=15, wait_frac=0.392, fwd_mean=186s, broadcast_mean=20.0s
      inference cap=1345, running_med=1146, waiting_med=2, queue_med=1.0s,
      kv_mean_med=0.135, kv_max_med=0.344
    60ddf022 qwen35-bs512-g16-4t12i-50step-openended-sequential4-r64
      trainer rows=17, wait_frac=0.439, fwd_mean=189s, broadcast_mean=14.5s
      inference cap=1407, running_med=1005, waiting_med=2, queue_med=0.5s,
      kv_mean_med=0.140, kv_max_med=0.352
  newer generated-name simul rows:
    653564de qwen35-a3b__qwen9b-or__simul
      trainer rows=27, wait_frac=0.438, fwd_mean=168s, broadcast_mean=9.2s
      inference cap=1523, running_med=1242, waiting_med=764, queue_med=167s,
      kv_mean_med=0.163, kv_max_med=0.446
    d823aad6 qwen35-a3b__qwen35-a3b-or__simul
      trainer rows=29, wait_frac=0.297, fwd_mean=209s, broadcast_mean=12.5s
      inference cap=1521, running_med=1213, waiting_med=797, queue_med=179s,
      kv_mean_med=0.154, kv_max_med=0.435
  high-concurrency mallopt reference:
    db474e36 oom-validate-mallopt-8step-simul-r2
      trainer rows=8, wait_frac=0.483, fwd_mean=181s, broadcast_mean=10.7s
      inference cap=3072, running_med=2382, waiting_med=320, queue_med=11s,
      kv_mean_med=0.247, kv_max_med=0.627
  interpretation:
    current 4t12i runs are no longer the old 1t7i regime where trainer
    wait_for_batch was 0.735-0.876. They are more balanced: wait_for_batch is
    about 0.30-0.44, forward/backward is about 168-221s, and broadcast is about
    9-20s.
  E2E roofline from serving-only speedup:
    if only wait_for_batch improves, E2E ratio =
      1 / ((1 - wait_frac) + wait_frac / serving_ratio)
    using the observed serving estimate 1.13x-1.24x:
      wait=0.296 -> 1.035x-1.061x
      wait=0.355 -> 1.043x-1.074x
      wait=0.392 -> 1.047x-1.082x
      wait=0.439 -> 1.053x-1.093x
      wait=0.483 -> 1.059x-1.103x
  decision:
    full production E2E A/B is still worth running, but a pass at the 1.08x
    gate is not guaranteed. If it lands around 1.04x-1.07x, that is consistent
    with the current trainer wait fraction and is not evidence that the serving
    patch failed.

full production A/B:
  native W&B/output:
  patched W&B/output:
  startup-excluded step ratio:
  serving ratio:
  trainer bucket shift:
  correctness/training-health notes:
  decision: missing; this is the next real production proof
```

## Current Free-Node Continuation Plan

As of 2026-06-26 21:06 UTC under job `5379916`:

```text
allocated nodes: nid011153,nid011166,nid011175,nid011195
active steps:   5379916.batch on nid011153
allowed lane:   nid011175,nid011195
lane status:    free after cancelling R77 steps 5379916.103 and 5379916.105
```

If spending the free lane on this ladder, use only:

```text
--deployment.hosts nid011175,nid011195
```

Normal operator launch:

```text
Run `rl` plainly from inside the allocation. Do not wrap `rl` in an outer
`srun`; the launcher creates exact internal `srun` steps pinned to the hosts
above.
```

Agent-shell exception:

```text
If a plain agent-shell launch creates an out-of-lane wrapper step, do not score
that run. For a narrow JIT/init probe, render with --dry-run, patch the
generated script's internal lane srun to use --overlap and
--ntasks="$LANE_NNODES", then launch the generated script under a one-node
outer srun pinned to nid011175. Score only if the resulting new Slurm steps are
confined to nid011175,nid011195.
```

Because this allocation is shared with other work, launch isolation is part of
the evidence:

```text
valid run:
  pre-existing out-of-lane steps are left untouched
  new Slurm steps for this ladder are only on nid011175,nid011195

tainted run:
  any new one-node or two-node step appears on nid011153 or nid011166
  any new long-running shell step appears outside nid011175,nid011195
```

Completed two-node repetition:

```text
R68 patched on nid011175,nid011195
R69 native on nid011175,nid011195
R70 patched precompile/JIT probe on nid011175,nid011195
R71/R72 batch-shaped precompile launch attempts are tainted; do not score them
R73 patched batch-shaped precompile/JIT probe on nid011175,nid011195
R74 patched observed-batch precompile/JIT probe on nid011175,nid011195
R76 patched enriched-JIT root-cause probe on nid011175,nid011195
R77 patched fixed-top-p precompile/JIT probe on nid011175,nid011195
```

Recommended next action:

```text
R77 removed _k_tail_uniform_kernel from post-ready sidecars. Close the
sampler-tail init subgoal and move to the full production E2E A/B, excluding
startup from the primary metric.

If broader init/JIT cleanliness becomes release-blocking, work on the remaining
vLLM/MoE/slot/KV warmup events. Do not reopen sampler-tail precompile-list
probing unless a future sidecar again shows _k_tail_uniform_kernel.

Do not spend more two-node serving repeats unless the full production A/B is
blocked; the serving gate has already passed twice.
```

2026-06-26 21:24 UTC continuation audit:

```text
current shell:
  hostname = nid011153
  SLURM_JOB_ID = 5379916
  SLURM_JOB_NODELIST = nid[011153,011166,011175,011195]

step audit:
  only 5379916.batch was visible on nid011153
  allowed free lane remains nid011175,nid011195

serving evidence rechecked from logs:
  R68/R69 broad 15%-45% KV, Running=256, Waiting>0:
    patched/native = 13465.5 / 10856.0 = 1.240x
  R68/R69 strict decode 15%-45% KV:
    patched/native = 14439.6 / 11944.0 = 1.209x

R77 init/JIT evidence rechecked:
  jit_monitor sidecars = 4 files, 60 events total
  _k_tail_uniform_kernel matches = 0
  learner_row_hit_rate = 1.0 at calls=2000 per worker
  fallback_reason_by_traffic = warmup_or_profiling:max_num_logprobs_not_width1 only

decision:
  do not launch another two-node sampler serving repeat from this shell.
  A plain agent-shell rl launch would originate on nid011153, and the serving
  gate is already satisfied. The remaining unproven gate is full production
  startup-excluded E2E A/B on the real topology.
```

Repeatable local audit:

```bash
uv run --no-sync python scripts/audit_sampling_kernel_goal_pt2.py
```

Audit logic coverage:

```bash
uv run --no-sync pytest tests/unit/inference/test_sampling_kernel_goal_audit.py
```

Expected current decision:

```text
local_two_node_gates_pass: true
training canary token export: pass
production config shape: pass
trainer Nsight hook: pass
full_production_ready: false
goal_complete: false
```

Stop rule for another two-node serving repetition:

```text
continue only until each arm has:
  >= 10 broad high-pressure rows
  or enough rows to show underfeeding / startup/JIT failure

Do not wait for full train-step completion on the two-node lane if the purpose
is only serving-pressure evidence; the full production lane is where E2E step
time must be measured.
```
