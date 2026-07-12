# Issue: failed vLLM adapter reload can leave `load_inplace` set

## Status

Deferred from the TTT review. The proposed endpoint change is not applied on `main` here.

## Summary

Prime-RL's vLLM admin endpoint sets `LoadLoRAAdapterRequest.load_inplace = True` before calling
the model handler. After success it resets the stored request's flag. If the handler raises,
the reset code is skipped.

## Why it may matter

The stored request object participates in later adapter operations. Leaving a process-wide
adapter record marked for in-place loading after a failed request can change later reload or
unload behavior and make recovery depend on the exact failure point inside vLLM.

## Change considered on the TTT branch

The handler call was wrapped in `try/finally`, with the stored request lookup and flag reset in
the `finally` block. A focused test injected a handler exception and asserted that the flag was
cleared.

## Why it is deferred

This endpoint is used by ordinary LoRA serving and elastic adapter loading, not only TTT. The
change is small, but its correctness depends on vLLM's ownership and mutation rules for the
incoming versus stored request objects. It should be validated against the supported vLLM
version and landed with the inference subsystem's tests.

## Questions

- Can the handler replace or remove the stored request before raising?
- Should the incoming request flag also be reset?
- Is `load_inplace` meant to persist for any recovery path?
- Does a failed in-place load require unloading or reconciling adapter tensors as well?

## Suggested tests

- Success resets the stored flag.
- Failure before storing, after storing, and after replacing a request all have defined results.
- A second load after failure behaves like a clean request.
- Existing elastic and manual LoRA load paths remain unchanged.

## Relevant code

- `src/prime_rl/inference/vllm/server.py`
- adapter clients in `src/prime_rl/utils/client.py`
- elastic adapter management in `src/prime_rl/utils/elastic.py`
