# Issue: fused LM-head backward allocates gradients that are not required

## Status

Deferred performance issue. It is not needed for the GLM TTT feature's functional contract.

## Summary

`_SequenceChunkedLogProbEntropyFn.backward` allocates both `grad_hidden` and `grad_weight`
unconditionally. Autograd may require only one of them, for example when the output-head weight
is frozen and only upstream activations need gradients.

## Potential cost

The output-head weight can be very large. Allocating and accumulating an unused gradient adds
memory pressure and matrix multiplications even though autograd will discard that result. This
is a performance/memory concern, not a demonstrated numerical correctness bug.

## Change considered on the TTT branch

The implementation read `ctx.needs_input_grad[:2]`, allocated each gradient conditionally, and
skipped its accumulation when not required. The custom autograd function returned `None` for an
unneeded gradient.

## Why it is deferred

The function is part of Prime-RL's general trainer path. A custom-autograd optimization should
be benchmarked and gradchecked independently across dtype, chunking, frozen/unfrozen weights,
and distributed model configurations. The TTT experiment can proceed without changing it.

## Suggested tests

- `torch.autograd.gradcheck` or a high-precision reference for all four requires-grad
  combinations.
- Numerical parity with the current backward for hidden-only, weight-only, and both gradients.
- Peak-memory and runtime benchmark at a representative vocabulary size.
- BF16/FP32 accumulation and empty-mask edge cases.

## Relevant code

- `src/prime_rl/trainer/models/layers/lm_head.py`
- `tests/unit/train/rl/test_fused_lm_head.py`
