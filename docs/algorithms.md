# Algorithms

This page covers the math and the configurable algorithmic components: how off-policy training works, the default loss and advantage functions, how to plug in your own, the filters applied between rollout and training, and how multi-turn rollouts get merged into training samples.

## Table of Contents

- [Async / Off-Policy Training](#async--off-policy-training)
- [Loss](#loss)
  - [Default Loss](#default-loss)
  - [Custom Loss](#custom-loss)
- [Algorithms](#algorithms)
  - [Built-Ins](#built-ins)
  - [Environment-Owned Algorithms](#environment-owned-algorithms)
- [Filters](#filters)
- [Difficulty Pools](#difficulty-pools)
- [Online Difficulty Filtering](#online-difficulty-filtering)
- [Multi-Turn Trajectories](#multi-turn-trajectories)
  - [Extension Property](#extension-property)
  - [Best-Effort Interleaving](#best-effort-interleaving)
  - [Renderers](#renderers)
  - [Discontinuous Trajectories](#discontinuous-trajectories)

## Async / Off-Policy Training

`prime-rl` is asynchronous by default. The trainer and inference always run one step overlapped: while the trainer is producing $\pi_n$ from rollouts at step $n$, inference is already generating the rollouts for step $n+1$ using $\pi_{n-1}$. With matched trainer and inference step times this produces fully-overlapped pipeline parallelism — neither side ever idles.

![Async pipeline: trainer step n produces $\theta_n$, inference at step n samples with $\theta_{n-1}$](assets/async-pipeline.png)

At step $n = 1, 2, 3, \dots$:

- **Trainer** produces policy $\pi_n$ with weights $\theta_n$ from rollouts $(x_n, y_n)$.
- **Inference** produces rollouts $(x_n, y_n)$ from policy $\pi_{\max(0,\,n-1)}$.

Step indices are 0-indexed so the gap holds at startup — inference is exactly one step behind the trainer.

## Loss

### Default Loss

The default RL loss is a DPPO policy-gradient term combined with a KL regularizer similar to Kimi-K2.5. For each prompt $x_j$ we sample a group of $G$ rollouts $\{y_i\}_{i=1}^G$, score them to get $s_i$, then optimize:

$$
\mathcal{L}(\theta) = -\,\mathcal{J}_{\text{PG}}(\theta) \;+\; \tau_{KL}\,\mathcal{L}_{KL}(\theta)
$$

where the policy-gradient term is

$$
\mathcal{J}_{\text{PG}}(\theta)
= \frac{1}{\sum_{j,i} |y_i^{(j)}|}
\sum_{j,i,t}
\min\!\left(\frac{\pi(y_{i,t}^{(j)}\mid x_j, y_{i,<t}^{(j)})}{\mu(y_{i,t}^{(j)}\mid x_j, y_{i,<t}^{(j)})}, \delta\right) \hat{A}^{(j)}_{i,t}
$$

and the KL regularizer penalizes drift between trainer and inference policies via the squared log importance ratio:

$$
\mathcal{L}_{KL}(\theta) = \frac{1}{\sum_{j,i} |y_i^{(j)}|}
\sum_{j,i,t} \log^2\!\left(\frac{\pi(y_{i,t}^{(j)}\mid x_j, y_{i,<t}^{(j)})}{\mu(y_{i,t}^{(j)}\mid x_j, y_{i,<t}^{(j)})}\right).
$$

$\mu$ is the policy that generated the rollout (inference), $\pi$ is the current policy (trainer), $\hat{A}_{i,t}$ is the token-level advantage, $\delta$ is the importance-sampling clipping ratio, and $\tau_{KL}$ is the KL temperature. The `min` clamps the importance ratio from above so a stale rollout assigning very low probability to a high-reward token doesn't produce a runaway gradient.

The knobs (under `[trainer.loss]` with `type = "default"`):

| Knob | Default | What it does |
|---|---|---|
| `dppo_mask_low` / `dppo_mask_high` | 0.2 / 0.2 | Lower / upper thresholds for DPPO-style token-level masking. |
| `adv_tau` | 1.0 | Temperature on the advantage term. Set to 0 for pure distillation (no RL signal). |
| `kl_tau` | 1e-3 | Temperature on the KL regularizer. Set to 0 to disable. |

The orchestrator writes per-token advantage channels. Each channel names the loss it feeds:

- `rl` → DPPO + KL with the channel's advantage values.
- `ce` → token-level NLL, using the channel values as per-token weights.

Select built-in or environment-owned algorithms with `[[orchestrator.algorithms]]` or per-env `[[orchestrator.train.env.algorithms]]`. The default is `id = "grpo"`.

### Custom Loss

The loss is computed **per sequence**: you write a function that takes one sequence's tensors and returns a scalar loss. The trainer iterates and aggregates.

```python
# my_module.py
import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    surr1 = ratio * inputs.advantages
    surr2 = clipped * inputs.advantages
    loss = -torch.min(surr1, surr2)[inputs.loss_mask].sum()
    return LossOutputs(
        loss=loss,
        metrics={
            "clip_frac": (ratio != clipped)[inputs.loss_mask].float().mean(),
        },
    )
```

Wire it up:

```toml
[trainer.loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"
kwargs = { clip_eps = 0.2 }
```

The dataclasses:

```python
@dataclass
class LossInputs:
    trainer_logprobs: Float[Tensor, "seq"]      # current policy
    inference_logprobs: Float[Tensor, "seq"]    # rollout-time policy
    advantages: Float[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]

@dataclass
class LossOutputs:
    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]
```

Anything you put in `metrics` is averaged across sequences and logged with the other trainer metrics.

## Algorithms

Algorithms run after scoring and before filtering. Each algorithm receives a group of `vf.Trace` objects and writes `branch.advantages` plus `branch.mask` in place for each trace branch it wants to train. `branch.advantages` and `branch.mask` must both align to `branch.token_ids`.

The trainer currently has two built-in loss channels:

- `rl` applies the configured RL stability loss (`[trainer.loss]`) using the algorithm's values as advantages.
- `ce` applies weighted token-level NLL using the algorithm's values as weights.

Configure defaults at the orchestrator level:

```toml
[[orchestrator.algorithms]]
id = "grpo"
```

Or override per environment:

```toml
[[orchestrator.train.env]]
id = "math-env"

[[orchestrator.train.env.algorithms]]
id = "rl"
```

### Built-Ins

| Algorithm | Loss | Signal |
|---|---|---|
| `grpo` | `rl` | Reward minus the group's mean reward, assigned to sampled tokens. |
| `max_rl` | `rl` | Reward minus group mean, divided by group mean when positive. |
| `rl` | `rl` | Raw scalar reward, assigned to sampled tokens. |
| `sft` | `ce` | Weight `1.0` on sampled tokens. |
| `echo` | `ce` | Weight `0.1` on tool-output context tokens. |
| `opd` | `rl` | Reference logprob minus actor logprob, assigned to sampled tokens. |
| `opsd` | `rl` | Reference-conditioned demonstration logprob minus actor logprob, assigned to sampled tokens. |

`opd` and `opsd` need a token-capable model endpoint. The endpoint is just a named model under `[orchestrator.models]`; `reference` is a common key, not a special role:

```toml
[[orchestrator.algorithms]]
id = "opd"
model = "reference"

[orchestrator.models.reference]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
tokens = true

[orchestrator.models.reference.client]
base_url = ["http://localhost:8001/v1"]
```

### Environment-Owned Algorithms

If an algorithm id is not one of the prime-rl built-ins, prime-rl sends the traces and configured model runtimes to the env server. The env package owns loading the algorithm and its dependencies, so the orchestrator process does not need to import env-specific algorithm modules.

An environment-owned algorithm uses the verifiers interface:

```python
# my_env/algorithms/normalized_reward.py
import verifiers.v1 as vf


class NormalizedReward(vf.Algorithm[vf.AlgorithmConfig]):
    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        rewards = [trace.reward for trace in traces]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        scale = variance**0.5 or 1.0
        for trace, reward in zip(traces, rewards, strict=True):
            value = (reward - mean) / scale
            for branch in trace.branches:
                branch.mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [value if keep else 0.0 for keep in branch.mask]
        return traces


__all__ = ["NormalizedReward"]
```

```toml
[[orchestrator.train.env.algorithms]]
id = "my_env.algorithms.normalized_reward"
```

## Filters

Filters drop rollouts between scoring and training. Built-ins (composable):

| Filter | Effect |
|---|---|
| `gibberish` | Drops rollouts whose mean log-prob fall below a threshold — usually a sign of degenerate output. |
| `repetition` | Drops rollouts with high n-gram repetition. |
| `zero_advantage` | Drops rollouts whose advantage is zero, so the trainer doesn't waste tokens on them. |

The default `[orchestrator]` config already includes all three filters with their defaults. To override, set `filters` explicitly — the list replaces the defaults wholesale:

```toml
[[orchestrator.filters]]
type = "zero_advantage"

[[orchestrator.filters]]
type = "repetition"
threshold = 0.4
```

Filtered rollouts still appear in W&B distributions, just not in the trainer batch — useful for spotting whether filtering is doing its job.

## Difficulty Pools

Difficulty pools gradually retire problems the model has solved or never solves. After each rollout, the average reward across a problem's group is compared to two thresholds:

- `buffer.easy_threshold` — at or above this, the problem moves into the `easy` pool and is no longer sampled.
- `buffer.hard_threshold` — at or below this, the problem moves into the `hard` pool and is no longer sampled.
- Otherwise the problem stays in `normal` and remains in the sampling rotation.

Pool assignments persist across checkpoints (`easy_examples.jsonl` / `hard_examples.jsonl` under each step's orchestrator checkpoint). When you resume — or want to broaden the curriculum mid-run — `buffer.easy_fraction` / `buffer.hard_fraction` randomly lift that fraction of pooled problems back into `normal` so they re-enter sampling.

```toml
[orchestrator.buffer]
easy_threshold = 0.95
hard_threshold = 0.05
easy_fraction = 0.0   # default; bump on resume to bring some easy problems back
hard_fraction = 0.0   # default; bump on resume to bring some hard problems back
```

Watch `pool/{env}/{easy,normal,hard}` (current pool ratios) and `evicted_examples/{env}/{easy,hard}` (per-step eviction rate).

## Online Difficulty Filtering

Online difficulty filtering (ODF) drops collapsed-advantage groups on the way *into* the buffer. Set `buffer.online_difficulty_filtering = true` (default `false`) to enable:

- Average reward across the group is **0.0** (every rollout failed) → drop the group, count under `filtered_rollouts/{env}/hard`.
- Average reward **1.0** (every rollout succeeded) → drop, count under `filtered_rollouts/{env}/easy`.
- Otherwise → into the buffer.

These are exactly the groups whose within-group advantage collapses to zero — DR-GRPO produces no gradient signal for them, so the trainer would burn step time on tokens it can't learn from.

```toml
[orchestrator.buffer]
online_difficulty_filtering = true
```

**Tradeoff: trainer stability vs. inference speed.** With ODF on, every rollout that reaches the trainer carries non-zero advantage — each trainer step's effective batch is predictable and the gradient signal is denser. The cost is paid on the inference side: rollouts get produced and then thrown away, so the orchestrator has to oversample to keep the trainer fed. If the orchestrator is your bottleneck (`time/wait_for_batch` high on the trainer), ODF can starve the loop. Bump `orchestrator.oversampling_factor` so inference produces enough groups per step to absorb the drops.

ODF is orthogonal to the [pools](#difficulty-pools): ODF reacts to the *current* group's reward distribution, the pools track the *running* per-problem average. Many configs use both — ODF for per-step density, pools for long-horizon curriculum cleanup.

## Multi-Turn Trajectories

Multi-turn rollouts (tool use, browser environments, long conversations) are recorded as trace branches with token attribution for each LLM turn. `prime-rl` converts those branches into training samples with best-effort interleaving, using [renderers](#renderers) to preserve exact token identity across turns.

### Extension Property

A sequence of trajectory steps has the **extension property** when each successive step's prompt contains all previous prompts and completions as an exact prefix. The trainer relies on this property — when it holds:

- Multiple steps merge into one training sample.
- Compute scales as $O(T)$ in the trajectory length.

When it breaks (chat template strips past thinking, environment compacts context, an agent hands off to a sub-agent, etc.), the trainer starts a new training sample from that step:

- Graceful fallback to multiple samples — no corrupted data.
- Worst case (every step breaks extension) is $O(T^2)$.

### Best-Effort Interleaving

Concretely:

```
5-step trajectory where extension breaks at step 4:

steps 1–3: extension holds   → merged into Sample 1
step 4:    extension breaks  (e.g. thinking stripped from history)
steps 4–5: extension holds   → merged into Sample 2

result: 2 training samples instead of 5
```

The orchestrator enforces an **exact prefix invariant**: the prompt at turn $t$ must be the exact concatenation of prior messages exactly as the LLM originally generated them. If turn 2's prompt is `U1, A1', U2` while `A1' ≠ A1`, the orchestrator can't safely merge — either choice produces logprob drift between trainer and inference. Starting a fresh sample is the only correct behavior, so that's what happens.

### Renderers

Best-effort interleaving works because the renderer guarantees the exact-prefix invariant *by construction* — it never re-renders prior turns, so it can't lose tokens to chat-template normalization, BPE retokenization drift, or thinking stripping. A renderer turns a model's chat template into a Python object that can:

- `render_ids(messages)` — tokenize messages to ids the inference engine accepts.
- `parse_response(completion_ids)` — recover structured `(content, reasoning_content, tool_calls)` from sampled ids.
- `bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages)` — extend the previous turn's tokens verbatim with the new environment turn, instead of re-rendering history.

When `bridge_to_next_turn` succeeds, the trainer sees the exact token stream the sampler produced; when it can't be proven safe (e.g. the renderer is `DefaultRenderer` and the template's stop sequence is unknown), it returns `None` and the orchestrator falls back to a full re-render — which triggers the new-sample fallback above.

A common source of breakage in the absence of a hand-coded renderer is models like Qwen3 whose chat templates strip past `<think>` blocks across user turns:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "U1"},
    {"role": "assistant", "content": "<think>R1</think>A1"},
    {"role": "user", "content": "U2"},
]
tok.apply_chat_template(messages[:1], tokenize=False)
# <|im_start|>user
# U1<|im_end|>

tok.apply_chat_template(messages, tokenize=False)
# <|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nU2<|im_end|>
# (the <think>R1</think> from turn 2 is gone)
```

Hand-coded renderers ship for `qwen3`, `qwen3-vl`, `qwen3.5`, `glm5`, `glm4.5`, `minimax-m2`, `deepseek-v3`, `kimi-k2`, `kimi-k2.5`, `nemotron-3`, `gpt-oss`; anything else falls back to `DefaultRenderer` (a generic `apply_chat_template` wrapper). Pick one via:

```toml
[orchestrator.renderer]
name = "auto"   # detect from tokenizer; pass an explicit name for fine-tunes
```

For the full design rationale (failure modes ruled out, empirical token-identity comparison against `apply_chat_template`, when to write a hand-coded renderer), see [the renderers writeup on the Prime Intellect blog](https://www.primeintellect.ai/blog/renderers) — the canonical reference.

### Discontinuous Trajectories

Some envs are discontinuous by design — e.g. a main agent delegating to a sub-agent and getting back only a summarized result, not the sub-agent's whole conversation. Best-effort interleaving handles this naturally: each agent's contiguous turns merge, the handoff starts a new sample. The trainer never sees fabricated extension where there is none.
