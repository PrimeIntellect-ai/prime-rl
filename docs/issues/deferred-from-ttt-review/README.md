# Deferred Prime-RL issues found during the TTT review

These notes capture general Prime-RL concerns discovered while reviewing the TTT feature
branch. They were intentionally separated from the TTT patch so each could be evaluated,
reproduced, and changed independently. Most remain proposals only; entries marked resolved
landed independently on `main` while this review was in progress.

The TTT branch originally addressed all of them. That made the branch much larger and changed
non-TTT behavior. The branch was reduced so normal Prime-RL behavior remains unchanged; only
behavior genuinely required by TTT remains, and it is TTT-gated.

## Issue index

| Issue | Concern | TTT branch disposition |
| --- | --- | --- |
| [SLURM namespace reservation](slurm-launch-namespace-reservation.md) | Concurrent submissions can share or overwrite one output/checkpoint namespace | Remove the general launcher change; keep distinct A0-A5 paths in experiment configs |
| [Token-batch payload accounting](train-sink-token-batching-payload-accounting.md) | Trace token counts can differ from actual trainer payload counts | Restore existing behavior for non-TTT runs; any exact accounting retained for TTT must be gated |
| [Empty/rejected cohort liveness](train-sink-empty-and-rejected-cohort-liveness.md) | A token-batched run can wait forever after cohorts produce no samples | Restore existing behavior for non-TTT runs; retain only the minimum TTT checkpoint-cleanup path |
| [Observation-window ownership](train-sink-observation-window-ownership.md) | A ready overflow batch can report a newly arrived partial group | Restore existing behavior for non-TTT runs; isolate any ownership bookkeeping needed by TTT GC |
| [Forked sampled-node deduplication](forked-trace-sampled-node-deduplication.md) | Shared sampled prefixes can contribute gradient more than once | Resolved independently on `main` by #2975 (`b453f40d8`); retained here as historical rationale |
| [vLLM `load_inplace` cleanup](vllm-load-inplace-cleanup-on-failure.md) | A failed adapter reload can leave mutable request state set | Remove from the TTT branch and consider independently |
| [LoRA inference capacity validation](lora-inference-capacity-validation.md) | `max_cpu_loras < max_loras` fails late in serving | Remove the global validator; the TTT experiment already supplies valid values |
| [Fused LM-head unused gradients](fused-lm-head-unused-gradient-allocation.md) | Backward allocates gradients for inputs that do not require them | Remove from the TTT branch; evaluate as an isolated performance change |
| [Nemotron-H adapter export mapping](nemotron-h-adapter-export-mapping.md) | Some exported adapter names do not match Hugging Face Nemotron-H names | Remove from the GLM-based TTT branch; address with a dedicated Nemotron test |
| [Echo sample/branch alignment](echo-sample-branch-alignment.md) | Fork/version expansion can misalign Echo samples and branches | Preserve existing Echo behavior; either reject Echo+TTT or implement separately |
| [Eval renderer-client parity](eval-renderer-client-parity.md) | Some experiments need eval through the renderer rather than chat relay | Remove the generic option from TTT; reconsider as a standalone feature |
| [Final NCCL dispatch gate](final-nccl-dispatch-gate.md) | The orchestrator could wait for a final policy version the trainer intentionally never broadcasts | Resolved independently on `main` by #2990 (`f34efdbf4`) |

## Triage guidance

Before implementing an issue:

1. Reproduce it against a clean, current `main` worktree.
2. Decide whether the behavior is a correctness bug, a performance opportunity, or an
   unsupported configuration that should instead be documented.
3. Keep the patch in the owning subsystem and add tests to that subsystem's normal suite.
4. Check non-target behavior explicitly; several proposed fixes alter batching or launch
   lifecycle semantics even when their motivating example is narrow.
5. Avoid coupling the issue to TTT unless the failure truly requires TTT state.

The notes describe the implementation considered on the TTT branch, not a decision that the
same implementation should be adopted on `main`.
