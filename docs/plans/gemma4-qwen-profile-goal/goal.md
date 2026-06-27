# Gemma4/Qwen Train+Infer Profiling Goal

Date: 2026-06-27

## Current Correction

We did start with Gemma4. The work then split into two separate evidence
threads:

- Gemma4 inference: attention-backend bound, with a measured mixed-backend win.
- Qwen inference: sampler/logprob bound, with a measured finite-top-k fast-path
  win.

Training is not comparably profiled yet. There is an old Qwen
`trainer_node0.nsys-rep`, but postprocessing showed no CUDA kernel/API trace.
It captured the trainer waiting for rollouts, not a forward/backward window.

## Gemma4 Draft/MTP Result

Gemma4 official assistant drafters do exist.

Primary sources:

- Google Gemma4 MTP blog:
  https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- Google AI Gemma4 MTP docs:
  https://ai.google.dev/gemma/docs/mtp/mtp
- vLLM Gemma4 recipe:
  https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#speculative-decoding-mtp
- vLLM speculative decoding docs:
  https://docs.vllm.ai/en/latest/features/speculative_decoding/
- HF checkpoint:
  https://huggingface.co/google/gemma-4-26B-A4B-it-assistant

The relevant vLLM nuance is important: Gemma4 `-assistant` checkpoints are MTP
speculators, not generic `draft_model` speculators. If startup logs show
`SpeculativeConfig(method='draft_model', ...)`, that runtime is taking the wrong
path or lacks Gemma4 MTP support.

Immediate Gemma4 MTP arm:

```toml
[inference.vllm_extra.speculative_config]
model = "google/gemma-4-26B-A4B-it-assistant"
num_speculative_tokens = 4
```

If vLLM requires explicit method in this version:

```toml
[inference.vllm_extra.speculative_config]
method = "mtp"
model = "google/gemma-4-26B-A4B-it-assistant"
num_speculative_tokens = 4
```

Success criteria:

- Startup logs show Gemma4 MTP, not generic draft-model speculation.
- Acceptance-rate telemetry exists or can be scraped.
- Same prompt/output distribution at `temperature=0` versus no-MTP.
- Saturated generation tok/s improves by at least 1.25x on the real debate
  shape before making it a production candidate.

Expected value:

- Google claims up to 3x in favorable settings.
- vLLM recipe recommends `num_speculative_tokens=4` for Gemma4 26B-A4B.
- Our prior Gemma4 profile was attention-bound; MTP amortizes the whole forward
  path, so it is a higher-ceiling lever than another attention-kernel pass.
- Confidence for a real debate-shape win: 45%. Confidence that the checkpoint
  exists and is the right next Gemma4 experiment: 95%.

## Pad Token Incident

This was not harmless serialization padding. Gemma4 was genuinely generating
pad token id `0` inside the trainable span.

Root cause:

- Gemma4 has `pad_token_id = 0`.
- The renderer/client path did not block pad-token generation.
- An initial attempt passed bad-token ids through `SamplingParams.extra_args`,
  but vLLM did not consume that field automatically for `/inference/v1/generate`.

Final fix shape:

- `deps/renderers/renderers/gemma4.py`: Gemma4 renderer exposes
  `get_bad_words_token_ids() == [[pad_token_id]]`.
- `deps/renderers/renderers/client.py`: renderer client forwards those ids via
  `sampling_params.extra_args.bad_words_token_ids`, requests `logprobs=0`, and
  validates returned token ids/logprobs.
- `src/prime_rl/inference/vllm/serving_tokens.py`: PrimeRL serving promotes
  `extra_args.bad_words_token_ids` into vLLM `SamplingParams.bad_words_token_ids`.

Status:

- This should become a renderers PR plus a PrimeRL companion PR.
- This is one of the highest-confidence landed fixes because it has narrow
  mechanism, unit tests, and a concrete production failure mode.

## Profile Matrix

| Model | Infer Profile | Train Profile |
|---|---|---|
| Gemma4 26B-A4B | Yes: attention bucket 37.55% -> about 12.4% with mixed backend | No valid trainer Nsight |
| Qwen3.5-35B-A3B | Yes: sampler/logprob 33.66% -> 4.06% with finite-top-k path | No valid trainer Nsight |

Important correction: `outputs/isambard/nsys_profile/nsys/trainer_node0.nsys-rep`
exists, but `nsys stats` reports no CUDA kernel/API data. The trainer log only
reaches `Starting training loop`; orchestrator logs show inference unavailable
for about 370s and train batch stuck at `0/64`. That trace is wait evidence,
not trainer-kernel evidence.

## Next Nsight Plan

### 1. Qwen production trainer profile

Run only when the topology is real enough to include 4 trainer nodes and 12
inference replicas, or label the result as a smoke.

Requirements:

- `PRIME_RL_NSYS_TRAINER=1`
- `PRIME_RL_NSYS` unset unless intentionally profiling inference too
- `max_steps >= 8`
- score only after at least one post-startup train step
- output must contain non-empty `cuda_gpu_kern_sum` and `cuda_api_sum`

Extract:

- CUDA kernels: attention, MoE/MLP, fused LM head, optimizer, memcpy
- CUDA API waits/syncs
- NVTX: forward/backward, optimizer, data loading, broadcast
- W&B: `time/wait_for_batch`, `time/forward_backward`,
  `time/broadcast_weights`, `perf/mfu`, `perf/throughput`

Decision:

- `wait_for_batch` dominates: inference/scheduler/orchestrator still bottleneck.
- `forward_backward` dominates with high MFU: trainer math becomes worth
  optimizing.
- `broadcast_weights` dominates: weight-transfer path A/B.
- OS/filesystem wait dominates: data/checkpoint/export path.

### 2. Trainer fake-data bench sweep

This is cheap and independent of inference:

```bash
uv run --no-sync trainer @ <cfg> --data.fake --bench
```

Sweep for Qwen MoE:

- `ep in {1, 4, 8}` with `ep_comm_backend = "torch"`
- current `cp = 2`
- `reshard_after_forward in {true, false}` if HBM allows
- selective AC versus full AC
- optimizer offload on/off if memory allows

Do not change `optimization_dtype` or `reduce_dtype`.

Goal: decide whether trainer EP can reduce comm/HBM enough to shed a trainer
node and convert it into inference capacity.

### 3. Gemma4 MTP serving A/B

Run Gemma4 26B-A4B with and without `google/gemma-4-26B-A4B-it-assistant` using
the same debate replay and production-ish concurrency. This is now higher value
than a new attention-kernel project.

Gate:

- startup path proves MTP
- acceptance rate is measured
- no pad-token recurrence
- no degraded token/logprob contract
- at least 1.25x generation tok/s at pressure

## PR Plan

Open draft PRs only for these high-confidence landed pieces:

1. Renderers Gemma4 pad-token/logprob contract
   - `deps/renderers/renderers/base.py`
   - `deps/renderers/renderers/client.py`
   - `deps/renderers/renderers/gemma4.py`
   - `deps/renderers/tests/test_client.py`
   - `deps/renderers/tests/test_gemma4.py`

2. PrimeRL bad-token promotion plus sampled-logprob contract
   - `src/prime_rl/inference/vllm/serving_tokens.py`
   - `tests/unit/inference/test_serving_tokens.py`
   - sampler fast-path width-0/width-1 logprob hardening
   - config-backed finite-top-k env setup

3. Production gate/audit hardening
   - `scripts/analyze_wandb_production_gate.py`
   - `scripts/audit_sampling_kernel_goal_pt2.py`
   - related tests and docs
   - calibration config enabling the config-backed sampler path

Do not PR scratch research references, local output artifacts, old `.nsys`
SQLite exports, or unvalidated experiment TOMLs unless a PR explicitly targets
research documentation.

## Ranked Easy Gains

1. Gemma4 MTP with official assistant drafter: potentially large; test now.
2. Valid Qwen trainer Nsight: zero algorithmic risk; resolves the missing train
   profile.
3. Qwen trainer fake-data EP/CP/FSDP bench sweep: likely fastest way to learn
   whether trainer-node shedding is real.
4. Production-shape hparams: keep BS/GS/inflight/max_num_seqs/max_num_batched
   aligned with calibration runs before judging throughput.
5. Gemma4 mixed attention backend: already measured; productionize if not
   already merged.
6. Qwen sampler path: do not add new kernels unless fresh profile reopens the
   bucket; current win is already in the right shape.
