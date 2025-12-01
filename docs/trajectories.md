# Trajectories

Verifiers [v0.1.8](https://github.com/PrimeIntellect-ai/verifiers/releases/tag/v0.1.8) introduced trajectory-based rollouts, where each LLM request/response pair in a multi-turn interaction is recorded as an independent step. For details on the design decision, check the detailed [design document](https://github.com/PrimeIntellect-ai/verifiers/blob/main/notes/TRAJECTORIES.md) in the verifiers repository.

This allows two different training modes:
- **Interleaved Rollouts**: A multi-turn interaction is concatenated into a single training example (default, enabled via `--trajectory-strategy interleaved` on the orchestrator)
- **Branching Rollouts**: A multi-turn interaction is split into separate training examples (enabled via `--trajectory-strategy branching` on the orchestrator)

## Interleaved Rollouts

Prior to verifiers v0.1.8, interleaved rollout was the only supported mode. Here, any multi-turn interaction is concatenated into a single training example. For example, a 3-turn conversation with alternating user and assistant messages will be concatenated into the following training example:

```txt
(U1,A1,U2,A2,U3,A3)
```

This approach enforces a strict invariant:

> The prompt at turn $t$ must be the exact concatenation of prior messages exactly as the LLM originally generated them

We call this the "exact prefix" invariant. For example, at turn 2, the LLM should see U1,A1,U2 as the prompt, where U1 exactly matches the user message in turn 1 and A1 exactly matches the produced assistant message in turn 1. Any violation to this invariant will result in downstream problems when computing the importance sampling ratio during training. For example, assume that at turn 2 the prompt is U1,A1',U2 where A1' varies from A1. In this scenario it is not clear whether to add A1 or A1' to the interleaved rollout:
- If we add A1', the logprobs from turn 1 might be off because the inference LLM produced A1 but the trainer LLM is computing logprobs for A1'
- If we add A1, the logprobs from turn 2 might be off because the inference LLM is attending to A1' but the trainer LLM is attending to A1.

While it would seem that this invariant is easy to enforce, there are two surprisingly common violations that can occur:
- **Retokenization**: Modern tokenizers are not symmetric, and so retokenizing A1 to prepare the prompt at turn 2 may produce A1' != A1
- **Arbitrary Chat Templates**: The chat template may add or remove tokens across turns

### Retokenization

There are two main reasons why retokenization is a problem:
- `verifiers` uses an OAI client for collecting a trajectory, which only supports text-in prompts, which implies that at any turn $t>1$, we have to retokenize the assistant messages from turns $1,\dots,t-1$.
- Modern tokenizers are not symmetric, and so retokenizing the assistant messages from turns $1,\dots,t-1$ may produce different tokens than the original decoded tokens by the LLM.

For example, it may be that the LLM, during decoding, produced the number `11` as two separate tokens, corresponding to `1` and `1`, but when retokenizing `11` becomes a single token. We have violated the exact prefix invariant: It is unclear whether to add `1` and `1` or `11` to the interleaved rollout:
- If we choose to add `1` and `1`, the logprobs from turn 2 might be off because the attention of the trainer and inference LLM is different.
- If we choose to add `11`, the logprobs from turn 1 might be off because `11` has a different likelihood than `1` and `1`.

In moderately complex multi-turn environments (e.g. `wordle` or `wiki-search`) we have found that our Icepop-style double-sided masking is able to mitigate such discrepancies, but it is unclear whether remains robust for large-scale agentic training in environments that are many hundreds of turns long. A simple solution to this problem is given by allowing for token-in requests, but this is neither standard OAI nor vLLM spec, which is why it is not easily supported yet.

### Arbitrary Chat Templates

There exist chat templates which add, modify, or remove tokens across turns. One good example, is the chat template of the Qwen3-series of models, which strips thinking across user turns.

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

The chat template automatically strips away past thinking section across user turn, which is often referred to as "interleaved thinking". Many chat templates, such as GLM or MiniMax, implement this approach, which makes training in multi-user turn environment (e.g. `alphabet-sort`) impossible when interleaving rollouts: Should we include the thinking section in the interleaved rollout or not? No matter what we do, we violate the exact prefix invariant, and will run into large logprob discrepancies between the trainer and inference.

## Branching Rollouts

When branching rollouts, a $n$-length multi-turn interaction will be split into $n$ separate training examples. For example, a 3-turn conversation with alternating user and assistant messages will be split into the following training examples:

```txt
1. U1,A1
2. U1,A1,U2,A2
3. U1,A1,U2,A2,U3,A3
```

This approach alleviates both issues that interleaved rollouts suffer from: it naturally handles any chat template, and avoids the retokenization issues all together. This is, because we *turn a single multi-turn interaction into many single-turn interactions*. In a sense this method is a safer default that works out of the box, but it comes at the (significant) cost of increased training time, because the number of training tokens now increases linearly with the number of turns.

## Verifiers vs. PRIME-RL

The verifiers trainer is using branching rollouts by default, while PRIME-RL supports two training modes, with interleaved rollouts being the default. It is an example of different design philosophies between the two projects. Verifiers is designed to be a general-purpose trainer that "just works" out of the box, while PRIME-RL is designed to be a high-performance production-scale trainer.