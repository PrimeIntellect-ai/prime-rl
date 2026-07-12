# Issue: the final NCCL batch could wait for a policy version that never arrives

## Status

Resolved independently on Prime-RL `main` by PR #2990, commit `f34efdbf4`
(`fix(orchestrator): allow final nccl batch through dispatch gate`). No TTT-branch fix is
needed after merging current `main`.

## Failure mode

The NCCL trainer intentionally skips its final inference-weight broadcast because the
inference NCCL group is torn down before that broadcast could be consumed. Consequently,
`policy.version` stops at the penultimate published policy version.

The orchestrator's lag gate previously treated every missing version alike. Near the end of a
finite run it could pause rollout dispatch while waiting for the final, intentionally omitted
broadcast. If the buffered rollouts could not fill the last training batch, neither side could
make progress and the run deadlocked.

## Resolution on `main`

The dispatch gate now mirrors the trainer's NCCL final-broadcast rule. While building the
final batch of a bounded NCCL run, it permits dispatch despite the apparent version lead. That
batch is expected to sample from the penultimate published policy.

## Why it is recorded here

The TTT review encountered the same end-of-run ownership boundary while examining checkpoint
consumption. It is a general orchestrator/NCCL lifecycle issue, not part of test-time training.
Landing it independently avoids coupling ordinary training behavior to the TTT feature.

## Relevant code

- `src/prime_rl/orchestrator/orchestrator.py::update_dispatch_gate`
- `src/prime_rl/trainer/rl/train.py::nccl_broadcast_unused`
