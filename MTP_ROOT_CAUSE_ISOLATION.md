# MTP Root Cause Isolation

Investigation date: 2026-05-14 to 2026-05-15 UTC

Original feature branch: `pr-2406-mtp`

Isolation branch: `codex/mtp-root-cause-isolation-20260514`

Head: `a3ea6ca68a3f7aa5dcd4c2e3885a72299df0a00d`

`origin/main`: `fa55779272c6f61c8f27d1e709a51fa5f06137a6`

## Objective

Run narrow isolations until the root cause of the Qwen3.5 MTP sanity-run crash is firm. Persist intermediate findings in this markdown file on the isolation branch, not on the feature branch.

## Known Failure

Failed run output directory:

`outputs/qwen35-2b-hendrycks-sanity-mtp-non-thinking-8xh200-full-sanity64-a3ea6ca68-rebase-main-fa5577927`

Key failure timing:

- Trainer completed step 20.
- Trainer broadcasted step 21 weights.
- Orchestrator paused inference, updated weights, resumed inference.
- vLLM failed immediately after resume.

Root exception text:

```text
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [21,0,0], thread: [192,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
RuntimeError: Triton Error [CUDA]: device-side assert triggered
RuntimeError: Worker failed with error 'Triton Error [CUDA]: device-side assert triggered', please check the stack trace above for the root cause
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

Crash site:

```text
vllm/model_executor/layers/mamba/gdn_linear_attn.py:830
mixed_qkv_spec = causal_conv1d_update(...)
```

The failing scheduler dump had no new requests and only cached decode requests:

```text
scheduled_new_reqs=[]
scheduled_cached_reqs=CachedRequestData(...)
```

## Current Finding

The concrete failing value is now known: after a preserved-cache pause/update/resume, vLLM schedules cached Qwen3.5 MTP decode requests with speculative placeholder token id `-1`, and that `-1` reaches `Qwen3NextModel.embed_input_ids`.

The final token-debug run caught the invalid value before CUDA embedding:

```text
RuntimeError: PRIME_RL_TOKEN_DEBUG invalid input_ids before embedding:
  input_ids.shape=(56,)
  input_ids.dtype=torch.int32
  input_ids.device=cuda:3
  actual_vocab_size=248320
  org_vocab_size=248320
  padded_vocab_size=248320
  partition_vocab_size=248320
  min=-1
  max=35790
  invalid_count=28
  semantic_invalid_count=28
  sample_flat_positions=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55]
  sample_values=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  sample_unique_values=[-1]
```

The same worker also logged a variant where all 56 input IDs were `-1`.

The corresponding scheduler output had:

- `scheduled_new_reqs=[]`.
- 28 cached requests on DP3.
- `num_scheduled_tokens=2` for every cached request.
- `scheduled_spec_decode_tokens={...: [-1], ...}` for every cached request.
- `total_num_scheduled_tokens=56`.
- `resumed_req_ids=set()`.
- `kv_cache_usage=0.026730730188806895`.

So the GDN and embedding CUDA asserts are symptoms. The root cause is earlier: invalid speculative decode placeholder IDs are being treated as model input tokens for cached MTP decode requests after an in-flight weight update/resume.

## Current Working Hypothesis

This is not a trainer crash. It is an inference-side crash when Qwen3.5 MTP speculative decode resumes cached decode requests after an in-place full-weight update.

The important distinction for PipelineRL:

- PipelineRL wants to preserve in-flight rollouts and avoid wasting KV/cache work.
- The local synthetic isolations preserve that behavior and have not reproduced the crash, even with long-lived DP=4 requests, repeated reloads, and real prompt token IDs.
- So preserving KV/cache across in-flight updates is intended and is not, by itself, proven broken here.
- The crash still happens in the Qwen3.5 MTP/GDN speculative decode path, but it appears to need an additional trigger from the full RL workload.

Refined bug statement:

> Qwen3.5 MTP speculative rollout can leak `-1` speculative placeholder IDs into target-model input IDs for cached decode requests after PrimeRL's preserved-cache pause/update/resume workload. This is reached intermittently by the full RL scheduler/client lifecycle and manifests as embedding/GDN CUDA asserts.

## Upstream Code Facts

The behavior is present on `origin/main`; it was not introduced by this PR:

- `src/prime_rl/inference/vllm/server.py` hardcodes `/pause` to `pause_generation(mode="keep", clear_cache=False)`.
- `src/prime_rl/inference/vllm/server.py` `/update_weights` reloads weights and does not reset caches.
- `src/prime_rl/utils/client.py` calls `/pause` with `clear_cache=false`.
- `src/prime_rl/utils/client.py` says `/update_weights` resets prefix cache, but that is not implemented by the server endpoint.

## Isolation Plan

1. Re-read the orchestrator cancellation path and clarify whether cached requests are expected to survive a policy update.
2. Build the smallest runtime reproducer that can trigger `/update_weights` while active cached Qwen3.5 MTP requests exist, without running full RL.
3. Compare active-request update against idle update.
4. Compare MTP speculative rollout against no speculative rollout.
5. If needed, compare updating to the same weights versus updating to changed weights.

## Results

### Isolation 1: 1-GPU MTP active requests, base weights -> step 21

Command shape:

```text
CUDA_VISIBLE_DEVICES=0 uv run --extra flash-attn inference @ outputs/.../configs/inference.toml \
  --parallel.dp 1 \
  --api-server-count 1 \
  --deployment.gpus-per-node 1 \
  --server.port 8100 \
  --output-dir outputs/mtp_iso_server_mtp_active_update
```

Then launched 32 concurrent `/v1/chat/completions` requests with `max_tokens=4096`, `ignore_eos=true`, waited 45 seconds, and called:

```text
POST /pause?mode=keep&clear_cache=false
POST /update_weights {"weight_dir": ".../run_default/broadcasts/step_21"}
POST /resume
```

Result:

- Server stayed healthy.
- All 32 requests returned HTTP 200.
- At update time vLLM logged `Running: 32 reqs`.
- GPU KV cache usage was around 2.4-2.6%.
- Spec decode acceptance stayed healthy.

Interpretation:

- A single preserved-cache active update with Qwen3.5 MTP is not sufficient to reproduce the crash.
- This does not disprove the active-cache hypothesis, but it means the failing condition is narrower than "any active MTP request crosses any reload".
- Next isolation should mimic the original transition more closely: load step 20, run active requests, then update to step 21.

### Isolation 2: 1-GPU MTP active requests, step 20 -> step 21

Using the same 1-GPU server from Isolation 1:

1. With no active requests, called `/pause`, `/update_weights` to `step_20`, `/resume`.
2. Launched 32 concurrent long chat completions.
3. While vLLM logged `Running: 32 reqs`, called `/pause`, `/update_weights` to `step_21`, `/resume`.

Result:

- Server stayed healthy.
- All 32 requests returned HTTP 200.
- No CUDA assert.
- Spec decoding continued after the active update, with acceptance dropping slightly but remaining healthy.

Interpretation:

- The exact step 20 -> step 21 transition is not sufficient to reproduce on a 1-GPU, 1-API-server ordinary `/v1/chat/completions` workload.
- Remaining differences from the real failure:
  - Original run used DP=4 and four API servers.
  - Original run used PrimeRL token streaming (`use_token_client=true`) rather than plain chat completions.
  - Original run had many prior policy updates before step 21.
  - Original requests came from Hendrycks envs, with heterogeneous lengths near max context.

### Isolation 3: 1-GPU MTP token endpoint, step 20 -> step 21

Using the same 1-GPU server:

1. With no active requests, loaded `step_20`.
2. Tokenized 32 prompts through `/tokenize`.
3. Sent 32 concurrent requests to `/v1/chat/completions/tokens` with explicit `tokens`, `logprobs=true`, `return_token_ids=true`, `max_completion_tokens=4096`, `ignore_eos=true`.
4. While vLLM logged `Running: 32 reqs`, updated to `step_21` with preserved cache.

Result:

- Server stayed healthy.
- All 32 token-endpoint requests returned HTTP 200.
- Response bodies were much larger than plain chat completions because token ids/logprobs were returned.
- No CUDA assert.

Interpretation:

- The PrimeRL token endpoint alone is not sufficient to reproduce on one GPU.
- Remaining high-probability differences:
  - DP=4 / four API servers.
  - Repeated policy updates before the failing transition.
  - Real rollout length and cancellation/off-policy scheduling pattern.

### Isolation 4: DP=4 MTP token endpoint, step 20 -> step 21

Command shape:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra flash-attn inference @ outputs/.../configs/inference.toml \
  --parallel.dp 4 \
  --api-server-count 4 \
  --deployment.gpus-per-node 4 \
  --server.port 8101 \
  --output-dir outputs/mtp_iso_server_mtp_dp4_token_update
```

Steps:

1. With no active requests, loaded `step_20`.
2. Tokenized 128 prompts through `/tokenize`.
3. Sent 128 concurrent requests to `/v1/chat/completions/tokens`, routed evenly with `X-data-parallel-rank`.
4. While requests were active, called `/pause`, `/update_weights` to `step_21`, and `/resume`.

Result:

- Server stayed healthy after the active update.
- All 128 token-endpoint requests returned HTTP 200.
- Four API servers and four DP workers handled the requests successfully.
- GPUs 0-3 retained the vLLM allocation after completion, as expected for a live server.

Interpretation:

- DP=4 plus the PrimeRL token endpoint is still not sufficient to reproduce the crash.
- This further rules out a simple "MTP + preserved KV + in-flight full-weight update always fails" explanation.
- The likely missing trigger is now in the real rollout distribution: long heterogeneous sequences near the context limit, repeated prior updates, or PrimeRL's rollout cancellation/off-policy scheduling pattern.

### Isolation 5: DP=4 long-lived MTP token requests across repeated updates

Status: complete.

Reason:

- The original crash at step 21 happened only 23 seconds into orchestrator step 21, but the failing cached requests already had output lengths from about 1.9k to 8.1k tokens.
- That means the failed requests were likely long-lived off-policy rollouts preserved across prior policy updates, not only fresh step-21 requests.
- To mimic that shape without running full RL, keep 128 long token-endpoint requests alive, target near-8192 total sequence length, and apply multiple full-weight reloads while those requests remain cached.

Command shape:

```text
# Existing DP=4 server on port 8101.
# Load step_20, launch 128 token-endpoint requests with prompt lengths 88-112
# and max_completion_tokens=8050, then reload:
#   +45s: step_21
#   +75s: step_20
#   +105s: step_21
```

Result:

- First attempt used the wrong body shape for `/v1/chat/completions/tokens`: it sent `tokens` without `messages`, so the endpoint returned HTTP 400 for all requests. This was a test-script error, not an engine failure.
- Corrected attempt included both `messages` and `tokens`.
- The first active reload at +45s succeeded.
- The second active reload at +75s succeeded.
- The third active reload at +105s succeeded.
- All 128 long requests returned HTTP 200.
- Request wall times were about 123-147s with response bodies around 660-786 KB, consistent with very long completions.

Interpretation:

- Long-lived near-context requests crossing multiple preserved-cache full-weight reloads are not sufficient to reproduce the crash.
- This weakens the hypothesis that "preserved KV across in-flight updates" is itself the bug.
- The remaining likely missing ingredient is PrimeRL's real scheduler/client lifecycle around rollouts that are cancelled, finish, or are refilled while updates are happening, or a rare vLLM scheduler/spec-decode edge case only reached by the actual environment workload.

### Isolation 6: DP=4 replay of actual step-20 prompt token IDs

Status: complete.

Reason:

- Decoding `run_default/rollouts/step_20/train_rollouts.bin` showed 128 examples.
- Prompt lengths: min 72, median 110, p90 282, p99 449, max 449.
- Completion lengths: median 7807, p90 8096, max 8120.
- Total sequence lengths: median 8192, max 8192.
- Isolation 5 used synthetic prompts with lengths 88-112 and `max_completion_tokens=8050`, so it did not exactly match the real 8192-token cap behavior.

Plan:

- Replay the saved step-20 `prompt_ids` directly through `/v1/chat/completions/tokens`.
- Omit `max_completion_tokens`, matching server-side cap-to-context behavior.
- Keep DP=4, 128 concurrent requests, token ids/logprobs returned, and repeated preserved-cache reloads while requests are active.

Result:

- Replayed 128 saved `prompt_ids` from the step-20 training batch.
- Request body used the token endpoint with `messages`, `tokens`, `logprobs=true`, `return_token_ids=true`, `enable_thinking=false`, `top_k=-1`, and `min_p=0.0`.
- `max_completion_tokens` was omitted, so server-side context-cap defaulting was used.
- Reloads:
  - +45s: `step_21`
  - +75s: `step_20`
  - +105s: `step_21`
- All three reloads succeeded.
- All 128 requests returned HTTP 200.
- Many requests were still active across the reloads; wall times reached about 137s.

Interpretation:

- Real prompt-token lengths, real 8192-token cap behavior, DP=4, token-client response shape, and repeated preserved-cache reloads are still not sufficient to reproduce the CUDA assert.
- This makes the root cause narrower than the PipelineRL design itself.
- The missing ingredient is likely one of:
  - exact stochastic generated token content from the failed step-21 in-flight rollouts,
  - real scheduler lifecycle interactions around finishing, cancellation, refill, and update,
  - or a rare vLLM Qwen3.5 MTP/GDN speculative decode bug that only appears under the full RL workload.

### Isolation 7: full local RL retry through the real scheduler/trainer lifecycle

Status: complete; first attempt aborted and restarted with a closer filter match.

Reason:

- Synthetic HTTP replay preserved cached decode state across active updates but did not reproduce.
- The remaining untested surface is the real RL lifecycle: orchestrator refill/cancellation, environment workers, rollout transport, trainer step timing, and the exact update cadence.
- The previous failure happened just after trainer step 20 completed and step 21 weights were loaded into inference, so a 32-step run is enough to cross that point if the failure is reproducible.

Config:

- Throwaway ignored config: `outputs/mtp_rootcause_full_rl_retry.toml`.
- Output dir: `outputs/qwen35-2b-hendrycks-sanity-mtp-non-thinking-8xh200-rootcause-retry-filters`.
- 4 inference GPUs + 4 training GPUs on local 8xH200.
- `Qwen/Qwen3.5-2B`, seq/model len 8192.
- MTP training enabled and MTP rollout enabled with `num_speculative_tokens=1`.
- `batch_size=128`, `rollouts_per_example=8`, `max_inflight_rollouts=128`, `max_off_policy_steps=8`, `max_async_level=1`, `use_token_client=true`, seed 42.
- Filter behavior matched to the failed run: gibberish, repetition, and zero-advantage filters all monitor with `enforce=false`.
- `max_steps=32`.

Command:

```text
uv run rl @ outputs/mtp_rootcause_full_rl_retry.toml --clean-output-dir
```

Note:

- The first retry attempt was stopped after step 0 because it accidentally inherited the current default `zero_advantage` filter with `enforce=true`.
- That produced a 19-example step-0 batch, unlike the failed run where `zero_advantage` had `enforce=false`.
- The config was updated before restarting.

Progress:

- Corrected run started at 2026-05-14 22:01 UTC.
- Resolved orchestrator config confirms all filters are `enforce=false`.
- Step 0 used 128 rollouts and converted all 128 to training examples; it detected but did not enforce 52 filtered rollouts.
- Trainer completed all configured steps 0-31 and broadcasted `step_32` weights without an inference crash.
- `RL trainer finished!` was logged at 23:02:32 and `RL training finished!` was logged at 23:02:42.
- W&B run: `https://wandb.ai/primeintellect/mtp-qwen35-hendrycks/runs/2968242207534321bc6b53debd020307`.
- The old failure boundary was crossed successfully:
  - trainer step 20 completed at 22:43:03 and broadcasted `step_21` weights at 22:43:04-22:43:06;
  - orchestrator step 21 started at 22:42:43;
  - orchestrator loaded policy step 21, paused all inference engines, and resumed them at 22:43:06;
  - orchestrator step 21 completed at 22:44:19 with reward 0.4531, mean sequence length 5866.8 tokens/sample, and max off-policy level 1;
  - trainer step 21 completed at 22:44:39 and broadcasted `step_22`.
- The run continued through orchestrator/trainer step 31:
  - orchestrator step 31 completed at 23:02:06 with reward 0.4219, mean sequence length 6775.1 tokens/sample, and max off-policy level 1;
  - trainer step 31 completed at 23:02:29 with throughput 42062 tokens/s and peak memory 67.2 GiB.
- Final log scan found no `EngineDead`, `device-side assert`, `IndexKernel`, `Triton Error`, or `Worker failed` entries. The only `CUDA` matches were normal startup config/warning lines.

Interpretation:

- The full local scheduler/trainer lifecycle can cross the exact previous crash window and finish a 32-step run while preserving in-flight decode state across MTP weight updates.
- This makes the original failure look intermittent or dependent on a rarer generated-token/state pattern, rather than a deterministic consequence of PipelineRL-style KV preservation.
- A proper next isolation would capture per-request token ids and vLLM cached-request metadata at update boundaries in a future failing run, then replay exactly the failing cached request state if possible.

### Isolation 8: longer same-shape full local RL run

Status: reproduced a vLLM inference failure.

Reason:

- Isolation 7 was a clean 32-step full run, but the original crash may be intermittent.
- A longer run with the same shape gives many more preserved-cache MTP weight-update cycles without changing the workload surface.
- If it fails, collect the first vLLM/orchestrator/trainer exception verbatim and preserve the scheduler/request context around the failure.
- If it completes, the evidence shifts further toward a rare sampled-state/vLLM edge case rather than a deterministic PrimeRL scheduling bug.

Config:

- Throwaway ignored config: `outputs/mtp_rootcause_full_rl_long96.toml`.
- Output dir: `outputs/qwen35-2b-hendrycks-sanity-mtp-non-thinking-8xh200-rootcause-long96-filters`.
- Same model, filters, seed, GPU split, MTP settings, async/off-policy settings, and sampling settings as Isolation 7.
- `max_steps=96`.

Progress:

- Investigation branch: `codex/mtp-root-cause-isolation-20260514`.
- Run started at 2026-05-14 23:05 UTC.
- W&B run: `https://wandb.ai/primeintellect/mtp-qwen35-hendrycks/runs/061129638e2f40ae82a5d1bb20fe31c0`.
- Trainer completed step 8 at 23:24:50, broadcasted `step_9` weights, and started waiting for step 9 rollouts.
- Orchestrator step 9 began at 23:24:29. While 32/128 rollouts had completed, it got policy step 9 at 23:24:53, paused inference, updated weights, and resumed inference at 23:24:54.
- Inference failed immediately after `/update_weights` and `/resume` returned HTTP 200.

Failure details:

- Failing process: `EngineCore_DP0` worker PID `132013`.
- Top worker exception:

```text
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

- Stack location where the async CUDA error was reported:

```text
vllm/model_executor/models/qwen3_next.py:408 self.linear_attn(...)
vllm/model_executor/layers/mamba/gdn_linear_attn.py:517 forward
vllm/model_executor/layers/mamba/gdn_linear_attn.py:577 forward_cuda
torch.ops.vllm.gdn_attention_core(...)
vllm/model_executor/layers/mamba/gdn_linear_attn.py:1081 gdn_attention_core
vllm/model_executor/layers/mamba/gdn_linear_attn.py:956 _forward_core
self.chunk_gated_delta_rule(...)
vllm/model_executor/layers/mamba/gdn_linear_attn.py:173 forward_cuda
vllm/model_executor/layers/mamba/gdn_linear_attn.py:103 fi_chunk_gated_delta_rule
g=torch.exp(fi_g)
```

- Engine wrapper exception:

```text
RuntimeError: Worker failed with error 'CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
'
```

Scheduler dump at failure:

- `scheduled_new_reqs=[]`.
- `scheduled_cached_reqs` had 28 cached requests on DP0.
- `all_token_ids_lens` ranged from 2442 to 8191; five requests were near the 8192 model limit: 7983, 8090, 8134, 8180, 8191.
- `num_computed_tokens` was exactly one less than each corresponding `all_token_ids_lens`.
- `num_output_tokens` ranged from 2373 to 7370.
- `num_scheduled_tokens` was 2 for every cached request except one request at 1; total scheduled tokens was 55.
- `scheduled_spec_decode_tokens` contained only `[-1]` placeholders for the cached requests.
- `kv_cache_usage=0.02950531230967046`.

Notes:

- This is the same high-level failure surface as the original run: cached decode requests, preserved through an in-flight MTP weight update, followed by a GDN/Qwen3.5 CUDA device-side assert.
- This reproduction reports in the GDN prefill/chunk path (`chunk_gated_delta_rule`), while the original run reported in the recurrent decode convolution path (`causal_conv1d_update`). The CUDA error is explicitly asynchronous, so the stack frame may be the first sync point rather than the true failing kernel.
- The many `layerwise.py:225 Failed to load weights` warnings are not by themselves the root cause. The clean 32-step run produced 28,520 of the same warning and completed successfully; the failing long run produced 8,280 before crashing.

Next isolation:

- Added repo-level debug instrumentation around vLLM's GDN `_forward_core` path in `src/prime_rl/inference/patches.py`.
- The instrumentation is gated behind `PRIME_RL_GDN_DEBUG=1`; it wraps GDN core execution, validates obvious token/state index bounds, and appends the GDN metadata context if the wrapped call or a forced CUDA sync fails.
- Run the same workload with `CUDA_LAUNCH_BLOCKING=1` and the debug wrapper enabled so a future failure includes GDN metadata: token/state index ranges, query start locs, prefill/decode counts, spec masks, cache shapes, and chunk metadata.

### Isolation 9: CUDA_LAUNCH_BLOCKING plus GDN debug

Status: reproduced, but shifted the stack to embedding.

Config:

- Output dir: `outputs/qwen35-2b-hendrycks-sanity-mtp-non-thinking-8xh200-rootcause-gdn-debug`.
- W&B run: `https://wandb.ai/primeintellect/mtp-qwen35-hendrycks/runs/0d75ec1f6b05454594fb1c99e9164097`.
- Environment: `PRIME_RL_GDN_DEBUG=1 CUDA_LAUNCH_BLOCKING=1`.

Result:

- Trainer completed step 8 at 00:09:38 UTC and broadcasted `step_9`.
- Orchestrator had started step 9 at 00:09:10 UTC.
- At 00:09:42 UTC, with 8/128 rollouts complete, orchestrator got policy step 9, paused inference, wrote `NCCL_READY`, updated weights in 0.36s, and resumed.
- Inference died immediately afterward.

Key finding:

- With synchronous CUDA error reporting, the top failing stack was no longer GDN. It was:

```text
vllm/model_executor/models/qwen3_5.py:695 forward
vllm/model_executor/models/qwen3_next.py:506 forward
vllm/model_executor/models/qwen3_next.py:493 embed_input_ids
vllm/model_executor/layers/vocab_parallel_embedding.py:484 forward
vllm/model_executor/layers/vocab_parallel_embedding.py:78 embedding
torch.nn.functional.embedding(...)
torch.AcceleratorError: CUDA error: device-side assert triggered
```

- PyTorch also logged repeated vectorized gather bounds asserts:

```text
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel:
Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
```

Scheduler context:

- `scheduled_new_reqs=[]`.
- `scheduled_cached_reqs` contained 29 cached requests on DP0.
- `resumed_req_ids=set()`.
- `num_scheduled_tokens=2` for every cached request; total scheduled tokens was 58.
- `scheduled_spec_decode_tokens` was `[-1]` for every cached request.
- `kv_cache_usage=0.02253502064018409`.

Interpretation:

- The GDN stack from Isolation 8 was an asynchronous reporting site, not the root failing operation.
- The real invalid operation is an embedding gather over token id `-1`.
- The next run should guard Qwen3.5 embedding directly and print invalid token values before the CUDA kernel runs.

### Isolation 10: token-id guard before Qwen3.5 embedding

Status: reproduced and captured the bad IDs.

Config:

- Tracked config: `debug_configs/mtp_rootcause_full_rl_token_debug.toml`.
- Output dir: `outputs/qwen35-2b-hendrycks-sanity-mtp-non-thinking-8xh200-rootcause-token-debug`.
- W&B run: `https://wandb.ai/primeintellect/mtp-qwen35-hendrycks/runs/bedc53600754403e9f77bfe1949f0e59`.
- Environment: `PRIME_RL_TOKEN_DEBUG=1`.
- Instrumentation: `src/prime_rl/inference/patches.py` patches `Qwen3NextModel.embed_input_ids` and raises before embedding if any token is `< 0` or `>= vocab_size`.

Result:

- Trainer completed steps 0 through 7.
- Step throughputs after warmup were stable, around 40k-42k tokens/s:
  - step 1: 41242 tokens/s;
  - step 2: 39984 tokens/s;
  - step 3: 41424 tokens/s;
  - step 4: 41242 tokens/s;
  - step 5: 41356 tokens/s;
  - step 6: 41701 tokens/s;
  - step 7: 41602 tokens/s.
- Trainer completed step 7 at 00:29:48 UTC and broadcasted `step_8`.
- Orchestrator started step 8 at 00:29:26 UTC.
- At 00:29:52 UTC, with 32/128 rollouts complete, orchestrator got policy step 8, paused inference, created `step_8/NCCL_READY`, resumed, and logged `Updated weights to step 8 in 0.30s`.
- Inference failed in the same second.

Verbatim root exception:

```text
RuntimeError: PRIME_RL_TOKEN_DEBUG invalid input_ids before embedding:
  input_ids.shape=(56,)
  input_ids.dtype=torch.int32
  input_ids.device=cuda:3
  actual_vocab_size=248320
  org_vocab_size=248320
  padded_vocab_size=248320
  partition_vocab_size=248320
  min=-1
  max=35790
  invalid_count=28
  semantic_invalid_count=28
  sample_flat_positions=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55]
  sample_values=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  sample_unique_values=[-1]
```

The same worker then logged another invalid call:

```text
RuntimeError: PRIME_RL_TOKEN_DEBUG invalid input_ids before embedding:
  input_ids.shape=(56,)
  input_ids.dtype=torch.int32
  input_ids.device=cuda:3
  actual_vocab_size=248320
  org_vocab_size=248320
  padded_vocab_size=248320
  partition_vocab_size=248320
  min=-1
  max=-1
  invalid_count=56
  semantic_invalid_count=56
  sample_flat_positions=[0, 1, 2, ..., 55]
  sample_values=[-1, -1, -1, ..., -1]
  sample_unique_values=[-1]
```

Scheduler context at the first invalid embedding:

- DP rank: `EngineCore_DP3`.
- Worker PID: `148914`.
- `scheduled_new_reqs=[]`.
- `scheduled_cached_reqs` had 28 cached requests.
- `all_token_ids_lens` included long near-limit sequences: 8180, 7284, 7150, 7135, 7113, 7110, 7108, and shorter active sequences down to 1227.
- `num_computed_tokens` was exactly one less than each corresponding `all_token_ids_lens`.
- `num_output_tokens` ranged from 1118 to 8091.
- `num_scheduled_tokens=2` for every cached request.
- `total_num_scheduled_tokens=56`.
- `scheduled_spec_decode_tokens` was `[-1]` for every cached request.
- `resumed_req_ids=set()`.
- `kv_cache_usage=0.026730730188806895`.

Interpretation:

- This confirms the root value precisely: vLLM is passing MTP speculative placeholder token id `-1` into target-model embedding.
- The alternating invalid positions match the expected flattened layout for 28 cached decode requests with two scheduled tokens each: one real sampled token plus one speculative token per request. The speculative slot is `-1`.
- The all-`-1` call is consistent with a follow-on speculative/drafter path seeing the same poisoned scheduled batch after the first failure.
- The bug is not in the trainer, loss, or the Hendrycks reward path.
- The bug is in the inference-side interaction between preserved cached decode requests, MTP speculative decode bookkeeping, and PrimeRL's pause/update/resume cancellation/update cycle.

Recommended fix direction:

- Short-term validation workaround: disable MTP speculative decode across in-flight weight updates, or force a spec-decode state reset for cached requests while still preserving KV where possible.
- Safer PrimeRL-side mitigation: around `/pause` or `/update_weights`, clear only pending `request.spec_token_ids` / draft-token state for cached requests, not the KV cache. This preserves the PipelineRL value of keeping generated context while preventing stale or invalid speculative placeholders from being scheduled.
- Upstream vLLM fix candidate: the scheduler/model-runner should never schedule `-1` speculative placeholders as target-model `input_ids`. If invalid draft tokens are represented with `-1`, the target input preparation path must filter/trim those slots and reduce per-request scheduled token counts, or replace them only in metadata paths that explicitly mask them.
