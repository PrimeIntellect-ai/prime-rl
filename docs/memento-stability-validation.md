# Memento vLLM 0.21 Stability Validation

Date: 2026-05-21

This note records the Memento/vLLM validation gate before starting the Prime <> Still compactor integration. It also records the GitHub remote setup now that this checkout was cloned from `PrimeIntellect-ai/prime-rl` and needs to move onto the Baseten-owned `basetenlabs/compact-rl` repo.

## Verdict

The Memento vLLM 0.21 block-masking path is stable enough to start the compactor phase behind flags.

The validated surface is:

- Prime async RL loop with Memento disabled.
- Prime async RL loop with Memento enabled and deterministic boundary-triggered compaction.
- Prime async RL loop with Memento enabled in a realistic environment where compaction is not forced.
- Direct vLLM async/KV stress with repeated compactions, async barrier mode, and production-ish inference settings.
- A separate boundary async RL run allowing observed off-policy behavior.

This does not validate the Still compactor itself. The next phase still needs to prove Still sidecar loading, KV-to-compacted-KV runtime behavior, trace emission, trainer replay/backward, Still gradients, and joint Qwen+Still policy synchronization.

## Source Plans

The stability gate was checked against two existing plans:

- Memento migration plan: `/Users/thomasvilleneuve/.claude/plans/nifty-exploring-cascade.md`
- Prime <> Still training port plan: `/Users/thomasvilleneuve/Desktop/Code/rl-takehome/experiments/prime-still-training-port-plan.md`

The Memento plan's main risk was overlay drift from stock vLLM 0.21, especially scheduler/KV corruption around compaction, queue draining, block-table truncation, and async behavior. The Still port plan assumes that this Memento/vLLM layer can safely support rollout-time compaction and produce trace data for trainer-side replay.

## Test Matrix

| Job | Purpose | Result | Key Evidence |
| --- | --- | --- | --- |
| `w7d7xdw` | Baseline Prime async RL, Memento disabled | Passed | 100 steps, trainer step 99, stable broadcasts, finite metrics, zero Memento events |
| `qe97k53` | Boundary async RL, Memento enabled, compaction required | Passed | 100 steps, 404 successful compactions, KV copy and block truncation events present |
| `q05v97w` | Realistic async RL, Memento enabled, compaction not required | Passed | 100 steps, async scheduler/barrier active, stable broadcasts, no compaction expected or observed |
| `qvx61gw` | Direct vLLM async/KV stress, no trainer | Passed | Fixed-length non-empty outputs, 102 async stress requests, 109 successful compactions |
| `qkj4odw` | Boundary async RL with observed off-policy behavior | Passed | 100 steps, 550 successful compactions, max observed off-policy level 8, stable broadcasts |

An earlier realistic RL run, `qj7okgq`, completed training but the wrapper failed because zero-count `grep` checks interacted badly with `pipefail`. The wrapper was patched to make zero-count metrics explicit and safe, then `q05v97w` passed.

## Job Details

### Job 1: Baseline Async RL, Memento Disabled

Job id: `w7d7xdw`

Launch shape:

```bash
env BLOCK_MASKING_ASYNC_MODE=async_barrier EXPECT_MEMENTO_ENABLED=0 EXPECTED_FINAL_STEP=99 MIN_BROADCAST_STABLE=2 MEMENTO_RL_CONFIG=configs/ci/integration/memento_baseline_async_rl_100.toml MEMENTO_RL_OUTPUT_DIR=/b10/workspace/outputs/memento-rl-baseline-async100-r2 uvx truss train push memento_rl_test.py --job-name memento-vllm021-baseline-async100-r2b
```

Result:

- Passed 100 training steps.
- Orchestrator and trainer reached step 99.
- Trainer finished normally with finite metrics.
- Broadcast stability was observed at the required threshold.
- Memento event counts were all zero, as expected for the disabled path.
- Peak trainer memory was around 15 GiB.

### Job 2: Boundary Async RL, Memento Enabled, Compaction Required

Job id: `qe97k53`

Launch shape:

```bash
env BLOCK_MASKING_ASYNC_MODE=async_barrier EXPECT_MEMENTO_ENABLED=1 INSTALL_MEMENTO_BOUNDARY_ENV=1 REQUIRE_MEMENTO_COMPACTION=1 EXPECTED_FINAL_STEP=99 MIN_BROADCAST_STABLE=2 MIN_MEMENTO_COMPACTIONS=10 MIN_MEMENTO_KV_COPIES=10 MIN_MEMENTO_BLOCK_TRUNCATIONS=10 MEMENTO_RL_CONFIG=configs/ci/integration/memento_boundary_async_rl_100.toml MEMENTO_RL_OUTPUT_DIR=/b10/workspace/outputs/memento-rl-boundary-async100-r2 uvx truss train push memento_rl_test.py --job-name memento-vllm021-boundary-async100-r3
```

Result:

- Passed 100 training steps.
- `memento_compaction_count=404`
- `span_success_count=404`
- `kv_copy_count=202`
- `block_truncation_count=202`
- No failed spans or non-None compaction errors were observed.
- Trainer metrics stayed finite.
- Peak trainer memory was around 12.8 GiB.

This is the strongest async RL validation that the Memento KV update path is actually firing under trainer/backward and policy broadcast pressure.

### Job 3: Realistic Async RL, Memento Enabled, Compaction Not Required

Job id: `q05v97w`

Launch shape:

```bash
env BLOCK_MASKING_ASYNC_MODE=async_barrier EXPECT_MEMENTO_ENABLED=1 EXPECTED_FINAL_STEP=99 MIN_BROADCAST_STABLE=2 MEMENTO_RL_CONFIG=configs/ci/integration/memento_async_rl_100.toml MEMENTO_RL_OUTPUT_DIR=/b10/workspace/outputs/memento-rl-prod-async100-r3 uvx truss train push memento_rl_test.py --job-name memento-vllm021-prod-async100-r3
```

Result:

- Passed 100 training steps.
- Trainer finished normally.
- Async scheduler and async barrier paths were active.
- Policy updates and stable broadcasts were observed.
- `memento_compaction_count=0`
- `span_success_count=0`
- `kv_copy_count=0`
- `block_truncation_count=0`
- Max observed off-policy level was 0.

The zero compaction count is expected for this run because the realistic alphabet-sort environment does not deterministically emit the boundary-token history needed to trigger Memento compaction. This job validates that enabling Memento does not break a real Prime async RL loop, but it does not validate active compaction.

### Job 4: Direct vLLM Async/KV Stress

Job id: `qvx61gw`

Launch shape:

```bash
env BLOCK_MASKING_ASYNC_MODE=async_barrier MIN_MEMENTO_COMPACTIONS=10 MIN_MEMENTO_KV_COPIES=10 MIN_MEMENTO_BLOCK_TRUNCATIONS=10 MEMENTO_INFERENCE_EXTRA_ARGS="--no-enforce-eager --async-stress-repeats 6 --stress-compact-requests 16 --stress-long-tokens 384 --gpu-memory-utilization 0.85" uvx truss train push memento_inference_test.py --job-name memento-vllm021-infer-stress-r3
```

Result:

- Passed with `ALL TESTS PASSED`.
- Fixed-length generations were non-empty and had the expected token counts.
- Async client drain stress completed 6 waves and 102 total requests.
- `memento_compaction_count=109`
- `span_success_count=109`
- `kv_copy_count=16`
- `block_truncation_count=16`

This job isolates the vLLM overlay from trainer/orchestrator noise. It is the best current coverage for KV copy direction/layout, block-table truncation, async queue draining, repeated compaction, and non-eager inference behavior.

### Additional Off-Policy Boundary Run

Job id: `qkj4odw`

Launch shape:

```bash
env BLOCK_MASKING_ASYNC_MODE=async_barrier EXPECT_MEMENTO_ENABLED=1 INSTALL_MEMENTO_BOUNDARY_ENV=1 REQUIRE_MEMENTO_COMPACTION=1 EXPECTED_FINAL_STEP=99 MIN_BROADCAST_STABLE=2 MIN_MEMENTO_COMPACTIONS=10 MIN_MEMENTO_KV_COPIES=10 MIN_MEMENTO_BLOCK_TRUNCATIONS=10 MIN_MAX_OFF_POLICY_LEVEL=1 MEMENTO_RL_CONFIG=configs/ci/integration/memento_boundary_async_rl_100_offpolicy.toml MEMENTO_RL_OUTPUT_DIR=/b10/workspace/outputs/memento-rl-boundary-offpolicy100-a4-r1 uvx truss train push memento_rl_test.py --job-name memento-vllm021-boundary-offpolicy100-a4-r1
```

Result:

- Passed 100 training steps.
- `memento_compaction_count=550`
- `span_success_count=550`
- `kv_copy_count=198`
- `block_truncation_count=198`
- `max_observed_off_policy_level=8`
- `broadcast_stable_count=5`
- Trainer metrics remained finite.

This run shows Memento compaction surviving under a more asynchronous policy-update regime. It also exposed stale rollout cancellation warnings. Those are not a failure for this gate, but they should be monitored once Still is added because Still traces will need to be discarded with their owning rollout if the rollout is cancelled.

## Local Checks

Local verification included:

```bash
uv run --no-sync python -m py_compile memento_rl_test.py memento_inference_test.py
env PYTHONPATH=src:packages/prime-rl-configs/src uv run --no-sync pytest tests/unit/inference/test_block_masking.py -q
git diff --check
```

The block-masking unit tests passed: 5 tests passed.

The runtime wrappers were also checked for shell syntax by extracting and running `bash -n` on the embedded scripts.

## Confidence Bar

The aggressive stability plan required:

- Jobs 1, 2, and 3 pass 100 steps.
- Job 4 passes sustained async/KV stress.
- Boundary job shows repeated successful compaction.
- Realistic job shows repeated checkpoint/broadcast policy updates.
- No scheduler index errors, KV corruption, empty-output regression, wrong token counts, trainer NaNs, or memory drift.

That bar is met for the tested configuration.

It is reasonable to proceed to Prime <> Still integration, with the Still path behind explicit config flags and with the same hard-fail preflight/summary checks preserved.

## Remaining Hard Edges

The following are not yet proven:

- Actual Still compactor execution in the Prime vLLM loop.
- RoPE unrotation/rerotation for Still-generated compacted KV.
- Synthetic KV insertion from Still outputs.
- Trace emission from vLLM/Still and preservation into trainer samples.
- Trainer-side Still replay/backward.
- Still gradient correctness.
- Joint Qwen+Still checkpointing and weight broadcast as one policy bundle.
- Long-horizon production training beyond 100-step validation.
- Larger models and larger context windows.
- Multi-node, disaggregated serving, speculative decoding, structured outputs, or kv-connector combinations.
- DeepGEMM-enabled behavior.
- FlashInfer sampler behavior.

The RL jobs used conservative runtime settings, including eager inference and lower GPU memory utilization. The direct vLLM stress job covered `enforce_eager=false` with higher GPU memory utilization, but that is not the same as a full non-eager RL training run.

Off-policy settings also need care. `max_async_level` and the observed off-policy level are not the same practical bound in these runs. The off-policy boundary job observed level 8, so any Still training config should set and log both async and off-policy controls explicitly.

## Next Phase Gates

Before treating the Still integration as stable, the next phase should add gates for:

- Memento enabled and Still disabled: current validated path.
- Still loaded but compaction disabled: import/config/checkpoint safety.
- Still compaction enabled in deterministic boundary fixture: repeated Still compactions required.
- Still traces attached to rollouts and dropped when rollouts are cancelled.
- Trainer replay builds a valid autograd graph for Still.
- Still parameters receive non-zero finite gradients.
- Qwen parameters and Still parameters are checkpointed and broadcast with matching policy identity.
- Cancellation, dropped group, empty rollout, and failed compaction counts are summarized and optionally hard-failed.

## Baseten Fork And Remote Setup

Original remote state:

```text
origin https://github.com/PrimeIntellect-ai/prime-rl
```

Baseten fork:

```text
https://github.com/basetenlabs/compact-rl.git
```

Recommended setup before committing this work:

1. Preserve PrimeIntellect as `upstream`.
2. Make `basetenlabs/compact-rl` the `origin`.
3. Create a feature branch for the Memento port, for example `feat/memento-vllm021`.
4. Commit and push the current Memento port and this validation note to that feature branch.

Remote flow:

```bash
git remote rename origin upstream
git remote add origin https://github.com/basetenlabs/compact-rl.git
git switch -c feat/memento-vllm021
git push -u origin feat/memento-vllm021
```

If SSH is preferred:

```bash
git remote rename origin upstream
git remote add origin git@github.com:basetenlabs/compact-rl.git
git switch -c feat/memento-vllm021
git push -u origin feat/memento-vllm021
```

After setup, `origin` should point at Baseten's fork and `upstream` should point at `PrimeIntellect-ai/prime-rl`.

Do not push this dirty working tree directly to `main`. Keep the port on a named feature branch so follow-up Still integration can be reviewed independently from future upstream syncs.
