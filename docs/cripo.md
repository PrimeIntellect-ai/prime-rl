# CriPO-S: criterion-level credit assignment

`cripo_s` implements the suppressed-criteria branch of
[CriPO](https://arxiv.org/abs/2607.18082). It preserves useful rubric behaviors
that would otherwise receive negative credit from an aggregate reward.
It uses prime-rl's GRPO baseline, including its optional length penalty and
absence of standard-deviation normalization.

```toml
[orchestrator.algo]
type = "cripo_s"
criteria_key = "criteria"
max_criteria = 3
flip_threshold = 0.1
flip_advantage = 0.1
flip_zero_advantage = false
```

## Rubric contract

The environment supplies explicit text keyed by reward name:

```python
import verifiers.v1 as vf

trace.info["criteria"] = {
    "explanation": "Explain why the proposed approach solves the problem.",
    "limitations": "Describe the limitations of the proposed approach.",
}
trace.rewards["explanation"] = vf.Reward(score=1.0, weight=2.0)
trace.rewards["limitations"] = vf.Reward(score=0.0, weight=1.0)
```

If the info entry is absent, the algorithm reads the same key from the typed
`trace.task.data`, allowing prompt-specific rubrics. Every eligible trace in
a cohort must provide the same criterion texts and weights. Each listed
criterion needs an unweighted binary `Reward.score` and a finite positive
`Reward.weight`. Missing scores, inconsistent rubrics, and invalid values
raise errors. Names alone are not inferred to be criterion instructions.
Rewards omitted from the text mapping still contribute to the GRPO aggregate
but do not participate in criterion selection. Thus continuous rewards or
penalties can coexist with the binary rubric.

The built-in `RubricJudge` records per-criterion metrics and returns one
aggregate reward. An environment using it must expose the named rewards and
criterion texts required here; enabling `cripo_s` alone does not convert
those metrics. When doing so, avoid counting both the aggregate judge reward
and its individual components in the total reward.

## Scoring and token credit

The algorithm scores the finalized cohort before admission. Errored and
non-trainable traces are excluded. For each criterion it sums the GRPO
advantages of all satisfying traces; a negative sum indicates suppression.
A zero sum also indicates suppression when fewer than half the cohort
satisfies the criterion.

For each negative-advantage trace, the highest-weight suppressed criteria
that it satisfies are combined in a removal instruction. The counterfactual
teacher reads this instruction and the original response, followed by the
verbatim branch tokens. Separate unconditioned and conditioned prefills use
the live policy at temperature 1.0. Selection compares these fresh scores;
the stored sampling logprobs remain the trainer's importance-ratio inputs.

A sampled token is selected only if its probability drops under the removal
instruction and is below `flip_threshold` times the teacher's maximum
next-token probability. Its advantage is **replaced** with `flip_advantage`.
Other token advantages retain their GRPO values. Shared sampled graph nodes
receive credit through the first trainable branch only; prompts and tool
observations remain masked.

For example, a trace with aggregate advantage `-1.0` may have token
advantages `[-1.0, 0.1, -1.0]` after localization. Adding `0.1` to the selected
token would leave its credit negative and would not implement the flip.

The default updates negative-advantage traces, following the paper's
algorithm. Setting `flip_zero_advantage = true` additionally rescues
zero-advantage traces for suppressed criteria. This is an explicit extension
and can create a training signal in otherwise all-zero groups.

## Operational scope

- Requires a tokens-in inference endpoint supporting prompt logprobs. The
  client extracts the target by token ID and validates all positions; it
  also returns the highest logprob from the top-1-plus-target response.
- Adds two prefills per eligible branch, independent of criterion count.
  Branches are processed sequentially; each student/teacher pair runs
  concurrently. Groups without eligible candidates make no scoring requests.
- The counterfactual prompt includes a copy of the response. Reserve context
  space for that copy and the criterion instruction. Endpoint errors propagate.
- Supports text branches with one trainable trace per episode. Multimodal
  counterfactual scoring and multi-agent cohort definitions are unsupported.
- Both prefills use the live inference pool. They are not an atomic snapshot
  across a weight update; tight version synchronization requires deployment
  support. Fresh scoring avoids comparing a counterfactual against aged or
  temperature-scaled rollout scores, but does not remove asynchronous drift.
- Uses the existing advantage stream and RL loss. CriPO's unexplored-criteria
  behavior injection, localized forward KL, and beta warmup are not implemented
  by `cripo_s`. The scalar `ref_kl` stream cannot represent that objective.

Unit tests cover the credit-assignment rules and transport annotations. They
do not establish training quality or reproduce the paper's benchmark results.
