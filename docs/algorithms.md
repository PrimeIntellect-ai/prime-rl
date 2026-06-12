# Algorithms

This page covers the math and the configurable algorithmic components: the algorithm abstraction and its algorithms, how off-policy training works, the loss components and advantage functions, how to plug in your own, the filters applied between rollout and training, and how multi-turn rollouts get merged into training samples.

## Table of Contents

- [The Algorithm Abstraction](#the-algorithm-abstraction)
  - [Model References](#model-references)
  - [The Algorithms](#the-algorithms)
  - [Customizing Components](#customizing-components)
  - [Per-Env Algorithms](#per-env-algorithms)
  - [The Algorithm Classes](#the-algorithm-classes)
- [Async / Off-Policy Training](#async--off-policy-training)
- [Loss](#loss)
  - [Loss Components](#loss-components)
  - [Default RL Loss](#default-rl-loss)
  - [Custom Loss](#custom-loss)
- [Advantage](#advantage)
  - [Default Advantage](#default-advantage)
  - [Custom Advantage](#custom-advantage)
  - [Reference Scoring](#reference-scoring)
- [Filters](#filters)
- [Multi-Turn Trajectories](#multi-turn-trajectories)
  - [Extension Property](#extension-property)
  - [Best-Effort Interleaving](#best-effort-interleaving)
  - [Renderers](#renderers)
  - [Discontinuous Trajectories](#discontinuous-trajectories)

## The Algorithm Abstraction

A training algorithm in `prime-rl` is a bundle of two components, configured under `[orchestrator.algo]`:

1. **Sampling** (`algo.sampling`) — how train rollouts are produced: which model generates them. `source` is a [model reference](#model-references): `"policy"` (the live policy, the default) or an inline frozen hosted model. Group sizing stays on the env config (`group_size`).
2. **Advantage** (`algo.advantage`) — the per-token training signal: credit assignment and loss routing, fused. One mapping from a finalized rollout to per-token *(loss component, weight)* pairs — the credit a token gets and the loss that consumes it are two coordinates of the same output. Group-relative strategies compute scalars on the orchestrator and ship numbers; reference-KL strategies query a reference model at batch-ship time (bounded concurrency) and ship its prefill logprobs for the trainer to evaluate against the live policy. The strategy determines which loss component consumes the action tokens (`rl` / `ce` / `ref_kl`) and what happens to env-provided observation tokens in multi-turn rollouts (masked out by default; `echo` trains on them with weighted CE).

The trainer is algorithm-blind: the loss is a sum of three components (rl, ce, ref_kl), each normalized by its own global token count; per-token component weight streams ship on the wire (`rl_weights` / `ce_weights` / `ref_kl_weights` on each training sample) and the trainer just executes them. Adding an algorithm never touches the dispatcher, packer, or trainer hot path.

### Model References

`prime-rl` hosts exactly one model: the trainable policy (`[orchestrator.model]`). Every other model an algorithm uses is an external OpenAI-compatible endpoint, declared *inline on the component that uses it*. A model reference is either the string `"policy"` (the live policy) or a frozen hosted model (`name` + `base_url`):

```toml
[orchestrator.algo.advantage]
type = "opd"

[orchestrator.algo.teacher]   # alias for `model`; folds into advantage.model
name = "Qwen/Qwen3-32B"
base_url = ["http://localhost:8001/v1"]
```

Model *roles* are labels the algorithm itself declares over these references — the distillation algorithms declare their reference's role as `teacher`, so `[orchestrator.algo.teacher]` parses as an alias for the `model` shorthand and validation errors speak the same language ("advantage 'opd' needs a teacher"). No role exists outside the algorithm that declares it: the dispatcher, sink, and trainer branch on liveness alone, never on what an algorithm calls a model.

`algo.model` (alias: `algo.teacher`) is shorthand for the slot the advantage type declares for its reference — `advantage.model` for `opd` / `opsd`, `sampling.source` for `sft` (its teacher is the sampling source). A slot you didn't set takes the shorthand; an explicit reference that already equals it is accepted, a disagreeing one is an error. Set the component fields directly for multi-model setups.

Liveness is a property of the reference, not of any role: rollouts sampled from `"policy"` get version-salted prefix caches, carry sampling logprobs for importance ratios, and age off-policy as weights update; rollouts and scores from frozen models get a stable prefix cache and never go stale. Frozen models are externally hosted (`base_url` is required) — `prime-rl` never launches or updates them, and each env's algorithm builds its own client pool to the endpoints it declares.

### The Algorithms

The advantage `type` names the algorithm, and each type's class defaults are its vetted setting — picking a type with no other keys IS the algorithm:

```toml
[orchestrator.algo.advantage]
type = "grpo"  # the default
```

| `type` | Sampling | Loss | What it is |
|---|---|---|---|
| `grpo` | policy | `rl` on actions | Standard group-relative RL. |
| `max_rl` | policy | `rl` on actions | MaxRL ([arXiv:2602.02710](https://arxiv.org/abs/2602.02710)): GRPO's centered reward normalized by the group **mean** instead of the standard deviation — the gradient is unbiased for the order-`group_size` truncation of the maximum-likelihood objective, upweighting hard examples like `1/p`. |
| `opd` | policy | `ref_kl` on actions | On-policy distillation ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)): the policy samples, per-token reverse KL against a reference model as the gradient signal. Needs a `teacher`. |
| `sft` | *(the teacher)* | `ce` on actions | Hard distillation: a frozen model generates rollouts, the policy trains with CE on its tokens. Needs a `teacher` (folds into `sampling.source`). |
| `opsd` | policy | `ref_kl` on actions | SDFT ([arXiv:2601.19897](https://arxiv.org/abs/2601.19897)): the model is its own reference, conditioned on an expert demonstration. Defaults to the live policy (the paper's setting, no extra deployment); set an inline `model` to score under a frozen copy instead. |
| `echo` | policy | `rl` on actions + weighted `ce` on observations | ECHO: standard GRPO plus a cross-entropy loss on env-provided tokens already present in the rollout, selected by message role (needs the renderer's role attribution). Defaults to tool-response bodies at `alpha = 0.1` (ECHO's λ); set `roles` to train other roles, each at its own weight. |
| `reward` | policy | `rl` on actions | REINFORCE-style: advantage = raw reward, no group baseline. |
| `custom` | policy | `rl` on actions | Your own advantage function (`import_path`), scalar per rollout, optionally per-token — see [Custom Advantage](#custom-advantage). |

### Customizing Components

Every key beyond `type` is visibly your own assembly — there is no preset layer to diverge from. The vetted setting is the class defaults; what you set is what runs:

```toml
# echo on tool AND user feedback tokens, each at its own weight.
# Setting any role replaces the whole table.
[orchestrator.algo.advantage]
type = "echo"

[orchestrator.algo.advantage.roles.tool]
alpha = 0.25

[orchestrator.algo.advantage.roles.user]
alpha = 0.05

# or a custom advantage strategy:
# [orchestrator.algo.advantage]
# type = "custom"
# import_path = "my_module.normalized_advantage"
```

Echo also takes an optional user-supplied token filter that narrows the role selection per rollout — e.g. dropping warning lines from tool output, or tokens the sampler found unlikely:

```toml
[orchestrator.algo.advantage.filter]
import_path = "my_module.drop_warnings"
kwargs = { patterns = ["WARNING"] }
```

```python
# my_module.py — sees the raw rollout (message text, sampling logprobs);
# returns one keep-mask per trajectory step, spanning that step's
# prompt_ids + completion_ids. False = never echo-trained.
def drop_warnings(rollout, *, patterns: list[str]) -> list[list[bool]]: ...
```

Component compatibility is validated at config time: frozen-model sampling can only feed the `ce` loss component — the `rl` and `ref_kl` components need the live policy's own sampling logprobs for importance ratios — `opd` pointed at `"policy"` is rejected as degenerate (zero KL), `sft` without a frozen source is rejected (CE on the policy's own tokens is not a distillation target), and group-relative advantage with `group_size = 1` warns that every advantage collapses to zero.

### Per-Env Algorithms

Both components resolve per environment. Each env inherits `[orchestrator.algo]` unless it sets its own, so a single run can mix algorithms across envs — e.g. GRPO on math, ECHO on a terminal env:

```toml
[orchestrator.algo.advantage]
type = "grpo"

[[orchestrator.train.env]]
id = "math-env"     # inherits grpo

[[orchestrator.train.env]]
id = "terminal-env"
advantage = { type = "echo" }   # shorthand: the env assembles its own algorithm
```

### The Algorithm Classes

At runtime, each env's resolved config builds two objects: a `Sampler` (`prime_rl.orchestrator.sampler`) from the `sampling` component — the pool rollouts are generated from, and the home of future sampling strategies like replay buffers or branching — and one of the named algorithm classes in `prime_rl.orchestrator.algo` from the `advantage` component. Algorithm dispatch is keyed on `advantage.type` — it names the algorithm, and each config class's defaults are its vetted parameterization:

| `advantage.type` | Class | `assign` (group time) | `score` (ship time) |
|---|---|---|---|
| `grpo` | `GRPOAlgorithm` | group-norm credit (optional length penalty) | — |
| `echo` | `EchoAlgorithm` | group-norm credit, plus weighted ce on observation tokens | — |
| `max_rl` | `MaxRLAlgorithm` | mean-normalized group credit | — |
| `opd` | `OPDAlgorithm` | — | own-context prefill under the teacher |
| `opsd` | `OPSDAlgorithm` | — | demo-conditioned prefill under the teacher |
| `sft` | `SFTDistillAlgorithm` | group-norm credit (feeds filters) | — |
| `reward` | `RewardAlgorithm` | raw reward | — |
| `custom` | `CustomAlgorithm` | your function | — |

Each class owns its methods outright — reading one top to bottom reads the algorithm. The two execution points of the training signal:

- `assign(rollouts)` — at group finalization, cheap and synchronous; sets rollout-level scalar (and optionally per-token) advantages.
- `async score(rollouts)` — at batch-ship time; attaches per-token reference data by querying `self.reference_pool` (connected in `setup()` from the algorithm's declared model reference — the live policy pool when the reference is `"policy"`).

Class-level declarations state what the algorithm needs: which loss component its action tokens feed (`action_loss_type`) and what it calls its reference model (`model_role`, e.g. `"teacher"`). Every class is constructed with the policy pool and the policy's renderer — text → token ids always goes through the renderer, the same path the policy's own prompts take (`opsd` requires one, validated at config time). The pipeline only ever calls the base-class hooks — writing your own algorithm is subclassing `Algorithm` and overriding `assign` and/or `score`. For pure scalar credit, no subclass is needed: `advantage.type = "custom"` imports a plain advantage function (see [Custom Advantage](#custom-advantage)); custom reference scoring means forking one of the named classes. Shared math (group normalization, prefill alignment) lives as plain functions in `prime_rl.orchestrator.algo.advantage`.

## Async / Off-Policy Training

`prime-rl` is asynchronous by default. The trainer and inference always run one step overlapped: while the trainer is producing $\pi_n$ from rollouts at step $n$, inference is already generating the rollouts for step $n+1$ using $\pi_{n-1}$. With matched trainer and inference step times this produces fully-overlapped pipeline parallelism — neither side ever idles.

![Async pipeline: trainer step n produces $\theta_n$, inference at step n samples with $\theta_{n-1}$](assets/async-pipeline.png)

At step $n = 1, 2, 3, \dots$:

- **Trainer** produces policy $\pi_n$ with weights $\theta_n$ from rollouts $(x_n, y_n)$.
- **Inference** produces rollouts $(x_n, y_n)$ from policy $\pi_{\max(0,\,n-1)}$.

Step indices are 0-indexed so the gap holds at startup — inference is exactly one step behind the trainer.

## Loss

### Loss Components

The training loss is a **sum of three components**, each with its own per-token weight stream and its own normalization:

$$
\mathcal{L} = \frac{\sum \mathcal{L}_{rl}}{N_{rl}} + \frac{\sum \mathcal{L}_{ce}}{N_{ce}} + \frac{\sum \mathcal{L}_{ref\_kl}}{N_{ref\_kl}}
$$

- `rl` — the configured RL loss (`[trainer.loss]`): DPPO + KL by default, or a [custom loss](#custom-loss). Fed by the scalar advantage strategies (`grpo`, `max_rl`, `reward`, `custom`, and `echo`'s action tokens).
- `ce` — masked NLL. Used for frozen-model tokens (`sft`) and env-observation tokens (`echo`).
- `ref_kl` — the per-token reverse KL to a reference model ($\log \pi_{\text{ref}} - \log \pi$) as the policy-gradient signal, importance-ratio corrected with a one-sided trust region (`opd`, `opsd`). Requires `ref_logprobs` from a [reference scoring](#reference-scoring); the scoring model must be a vLLM server (it's the only one that exposes `prompt_logprobs`).

The orchestrator stamps each sample's component membership as per-token weight streams (`rl_weights` / `ce_weights` / `ref_kl_weights` on the wire): a weight scales that component's per-token loss, `0.0` leaves the token out of the component entirely (mask *and* denominator), and components may overlap on the same token — their gradients sum. Each $N$ is the global (all-reduced) count of that component's member tokens, so the components don't dilute each other: adding echo observation tokens never changes the rl term's effective per-token learning rate, and an sft env packed next to a GRPO env doesn't soften its gradient. Tokens of different components pack freely into the same micro batch, and a plain GRPO run ships no streams at all (absent streams mean rl weight 1.0 on every trainable token — the unchanged hot path).

### Default RL Loss

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
| `adv_tau` | 1.0 | Temperature on the advantage term. Set to 0 to drop the policy-gradient term, leaving only the KL regularizer. |
| `kl_tau` | 1e-3 | Temperature on the KL regularizer. Set to 0 to disable. |

Set `[trainer.loss] type = "default"` and configure via the knobs above. The `ce` and `ref_kl` components are fixed and unaffected by `[trainer.loss]`.

### Custom Loss

`[trainer.loss] type = "custom"` replaces the `rl` component. The loss is computed **per sequence**: you write a function that takes one sequence's tensors and returns a scalar loss. The trainer iterates and aggregates. `inputs.loss_mask` selects exactly the rl member tokens (for a plain GRPO run, all trainable tokens).

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
    ref_logprobs: Float[Tensor, "seq"] | None   # set by reference-scoring algorithms
    advantages: Float[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]              # this component's member tokens
    loss_weights: Float[Tensor, "seq"] | None   # the component's weight stream (None = 1.0)

@dataclass
class LossOutputs:
    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]
```

Anything you put in `metrics` is averaged across sequences and logged with the other trainer metrics.

## Advantage

The advantage strategy is the `advantage` component of the [algorithm](#the-algorithm-abstraction) — every training signal is an advantage, varying in granularity (group-scalar vs. per-token) and evaluation site (orchestrator vs. trainer). `[orchestrator.advantage]` (and per-env `advantage = {...}`) is shorthand for `algo.advantage`. Types:

| Type | Component | Effect |
|---|---|---|
| `grpo` | `rl` | Group-norm: reward minus per-group baseline, optional length penalty. |
| `max_rl` | `rl` | Mean-normalized group credit (maximum-likelihood RL). |
| `echo` | `rl` + `ce` | Group-norm on action tokens, plus weighted CE on env-provided tokens selected by message role (each role's `alpha` is its ECHO λ), optionally narrowed by a user filter. |
| `reward` | `rl` | Advantage = raw reward, no baseline. |
| `opd` | `ref_kl` | On-policy distillation: per-token reverse KL to a reference model (`model`, an inline frozen hosted model), evaluated in the trainer from shipped reference logprobs. No scalars — rollouts keep `advantage = None` (advantage-based filters never fire) and ship a neutral 0.0; `group_size` only fans out sampling. |
| `opsd` | `ref_kl` | SDFT: per-token reverse KL to a demo-conditioned reference. No scalars — rollouts keep `advantage = None` (advantage-based filters never fire) and ship a neutral 0.0. |
| `sft` | `ce` | Cross-entropy on the sampled tokens. The loss ignores scalars, but group-relative scalars are still assigned so reward-based filtering keeps working. |
| `custom` | `rl` | Your function (below); scalar per rollout, optionally per-token. |

### Default Advantage

The default advantage is per-group reward minus per-group baseline (DR-GRPO without std normalization). For each prompt's group of `group_size` rollouts, every token in rollout $i$ receives advantage $s_i - \bar{s}$ where $\bar{s}$ is the group mean.

This is intentionally simple — it does the right thing for most envs. Switch to a [custom advantage](#custom-advantage) when you need group-aware shaping that depends on trajectory metadata (sub-agent rollouts, relative-rank shaping, …).

Two built-in **length penalties** (`length_penalty` on the `grpo`-family strategies) can be layered on top to discourage rambling: `tokens` penalizes long completions by weighted token cost, `turns` penalizes long multi-turn rollouts by turn count.

```toml
[orchestrator.advantage]
type = "grpo"

[orchestrator.advantage.length_penalty]
type = "tokens"
```


### Custom Advantage

Advantages are computed **per group**. You write a function that takes one group of rollouts and returns one advantage scalar per rollout. The orchestrator handles groups of varying size automatically — partial-group training kicks in when some rollouts in a group errored.

```python
# my_module.py
import statistics
from prime_rl.orchestrator.algo import AdvantageInputs, AdvantageOutputs

def normalized_advantage(inputs: AdvantageInputs, eps: float = 1e-8) -> AdvantageOutputs:
    rewards = [r["reward"] for r in inputs.rollouts]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return AdvantageOutputs(advantages=[(r - mean) / (std + eps) for r in rewards])
```

```toml
[orchestrator.advantage]
type = "custom"
import_path = "my_module.normalized_advantage"
kwargs = { eps = 1e-8 }
```

`AdvantageInputs.rollouts` is a list of `verifiers.RolloutOutput`, so you have access to the full rollout (turns, tool calls, custom metadata) — not just the reward. Use this for anything reward-shaping-like that needs trajectory context.

#### Per-token advantages

A custom function can also emit **per-token advantages** (process rewards, step-level credit assignment) via `AdvantageOutputs.token_advantages` — one optional list per rollout, aligned to that rollout's completion tokens. `None` entries (or omitting the field) broadcast the scalar over the sequence; prompt positions are padded internally and never trained.

```python
def step_weighted_advantage(inputs: AdvantageInputs) -> AdvantageOutputs:
    rewards = [r["reward"] for r in inputs.rollouts]
    baseline = statistics.fmean(rewards)
    scalars = [r - baseline for r in rewards]
    token_advantages = [
        [scalar * w for w in my_token_weights(rollout)]  # one float per completion token
        for scalar, rollout in zip(scalars, inputs.rollouts)
    ]
    return AdvantageOutputs(advantages=scalars, token_advantages=token_advantages)
```

The scalar `advantages` are still required — advantage-based filters and metrics read them. Each list must match the rollout's completion token count exactly (for multi-turn envs that's the merged completion, including interleaved observation tokens), and the rollout must map to a single training sample — both are validated loudly at group finalization. Signals that depend on the live policy's weights (like OPD's reverse KL) cannot be precomputed here; those are reference-scoring algorithms, evaluated in the trainer.

### Reference Scoring

`OPDAlgorithm` / `OPSDAlgorithm` have an async ship-time half (`score`): at batch-ship time they query their teacher (`model`, a [model reference](#model-references)) with bounded concurrency (`max_concurrent`, default 32) and attach per-token reference logprobs to each sample:

- `opd` — score each sample's own context under the reference model via prefill; fills `ref_logprobs` for the `ref_kl` loss component (on-policy distillation). `model = "policy"` is rejected (the KL would be identically zero).
- `opsd` — SDFT: rebuild the prompt with an expert demonstration woven into the last user message (`template`, with `{question}` / `{demonstration}` placeholders), score the policy's completion under that demo-conditioned context. `model = "policy"` scores under the live policy itself — the SDFT setting, no extra deployment. The demonstration is read from the example's `info[demo_key]`, falling back to a top-level rollout field of the same name (e.g. `answer`); single-step trajectories only.

```toml
[orchestrator.algo.advantage]
type = "opsd"
model = "policy"
demo_key = "demonstration"
max_concurrent = 64
```

Only batch survivors get scored — rollouts that are filtered or cancelled never cost reference compute. The time shows up as `time/scoring` in the step timing.

## Filters

Filters drop rollouts between scoring and training. Built-ins (composable):

| Filter | Effect |
|---|---|
| `gibberish` | Drops rollouts whose mean log-prob fall below a threshold — usually a sign of degenerate output. |
| `repetition` | Drops rollouts with high n-gram repetition. |
| `zero_advantage` | Drops rollouts whose advantage is zero, so the trainer doesn't waste tokens on them. |

The default `[orchestrator]` config registers all three in both filter slots: `post_batch_filters` enforce by default (flagged rollouts are recorded but not shipped to the trainer), while `pre_batch_filters` run in monitor mode (`enforce = false`); flip `enforce = true` there to drop matching rollouts before they consume a slot in the batch. Setting a slot replaces its defaults wholesale:

```toml
[[orchestrator.post_batch_filters]]
type = "zero_advantage"

[[orchestrator.post_batch_filters]]
type = "repetition"
threshold = 0.4
```

Filtered rollouts still appear in W&B distributions, just not in the trainer batch — useful for spotting whether filtering is doing its job.

## Multi-Turn Trajectories

Multi-turn rollouts (tool use, browser environments, long conversations) used to be stitched into a single fake "single-turn" sample, which silently corrupted the importance ratio when chat templates didn't roundtrip. Since [`verifiers` v0.1.8](https://github.com/PrimeIntellect-ai/verifiers/releases/tag/v0.1.8), `prime-rl` records each LLM request/response as an independent **trajectory step** and merges them at training time using best-effort interleaving — with [renderers](#renderers) as the mechanism that keeps the merge safe by construction.

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
