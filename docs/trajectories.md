# Trajectories

Verifiers [v0.1.8](https://github.com/PrimeIntellect-ai/verifiers/releases/tag/v0.1.8) introduced trajectory-based rollouts, where each LLM request/response pair in a multi-turn interaction is recorded as an independent step. For details on the design decision, check the detailed [design document](https://github.com/PrimeIntellect-ai/verifiers/blob/main/notes/TRAJECTORIES.md) in the verifiers repository.

## Best-Effort Interleaved Rollouts

PRIME-RL uses a best-effort interleaving strategy that automatically merges consecutive trajectory steps when possible, and starts a new training sample when the extension property breaks.

### The Extension Property

A sequence of trajectory steps has the **extension property** when each successive step's prompt contains all previous prompts and completions as a prefix. When this holds:
- Multiple steps can be merged into a single training sample
- Compute scales as O(T) for a trajectory of length T

When extension breaks (e.g., due to context compaction or thinking being stripped):
- A new training sample is started from that step
- Compute scales as O(T²) in the worst case (every step breaks extension)

### How It Works

```
5-step trajectory where extension breaks at step 4:

Steps 1-3: extension holds → merged into Sample 1
Step 4: extension breaks (e.g., thinking stripped from history)
Steps 4-5: extension holds → merged into Sample 2

Result: 2 training samples instead of 5
```

This approach gives you the best of both worlds:
- When extension holds: O(T) compute, single merged sample
- When extension breaks: graceful fallback, no corrupted data
- Mixed scenarios: optimal merging where possible

### The Exact Prefix Invariant

Interleaving enforces a strict invariant:

> The prompt at turn $t$ must be the exact concatenation of prior messages exactly as the LLM originally generated them

We call this the "exact prefix" invariant. For example, at turn 2, the LLM should see U1,A1,U2 as the prompt, where U1 exactly matches the user message in turn 1 and A1 exactly matches the produced assistant message in turn 1. Any violation to this invariant will result in downstream problems when computing the importance sampling ratio during training.

For example, assume that at turn 2 the prompt is U1,A1',U2 where A1' varies from A1. In this scenario it is not clear whether to add A1 or A1' to the interleaved rollout:
- If we add A1', the logprobs from turn 1 might be off because the inference LLM produced A1 but the trainer LLM is computing logprobs for A1'
- If we add A1, the logprobs from turn 2 might be off because the inference LLM is attending to A1' but the trainer LLM is attending to A1

When the invariant is violated (extension breaks), PRIME-RL automatically starts a new training sample rather than producing corrupted data.

### Arbitrary Chat Templates

There exist chat templates which add, modify, or remove tokens across turns. One good example is the chat template of the Qwen3-series of models, which strips thinking across user turns.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

messages = [
    {"role": "user", "content": "U1"},
    {"role": "assistant", "content": "<think>R1</think>A1"},
    {"role": "user", "content": "U2"},
]

print(tokenizer.apply_chat_template(messages[:1], tokenize=False))
# <|im_start|>user
# U1<|im_end|>

print(tokenizer.apply_chat_template(messages, tokenize=False))
# <|im_start|>user
# U1<|im_end|>
# <|im_start|>assistant
# A1<|im_end|>
# <|im_start|>user
# U2<|im_end|>
```

The chat template automatically strips away past thinking sections across user turns, which is often referred to as "interleaved thinking". Many chat templates, such as GLM or MiniMax, implement this approach.

With best-effort interleaving, PRIME-RL handles this gracefully: when the thinking is stripped and the prefix no longer matches, a new training sample is started automatically.

### Discontinuous Trajectories by Design

Some multi-turn environments are intentionally discontinuous. For example, in a sub-agent calling scenario:

1. Main agent receives a task and decides to delegate to a sub-agent
2. Sub-agent runs independently (possibly multiple turns with its own context)
3. Control returns to main agent with only the sub-agent's final result

The main agent's trajectory is discontinuous because the sub-agent's internal conversation isn't part of its context. When the main agent resumes, its prompt doesn't extend the previous turn - it contains a summarized result instead.

Best-effort interleaving handles this naturally: each agent's contiguous turns get merged, but the handoff between agents starts a new sample.

## Deprecated: Branching Mode

The `--trajectory-strategy branching` option is deprecated. The best-effort interleaving strategy now handles all cases automatically, falling back to separate samples (equivalent to branching) when the extension property breaks.
