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
  - [Authoring an Algorithm](#authoring-an-algorithm)
  - [Reference Scoring](#reference-scoring)
- [Filters](#filters)
- [Multi-Turn Trajectories](#multi-turn-trajectories)
  - [Extension Property](#extension-property)
  - [Best-Effort Interleaving](#best-effort-interleaving)
  - [Renderers](#renderers)
  - [Discontinuous Trajectories](#discontinuous-trajectories)

## The Algorithm Abstraction

A training algorithm in `prime-rl` is configured under `[orchestrator.algo]`, where **`type` names the algorithm** (`grpo`, `opd`, `sft`, …) and the class defaults are its vetted setting. It has two parts:

1. **Sampling** (`algo.sampling`) — how train rollouts are produced: which model generates them. `source` is a [model reference](#model-references): `"policy"` (the live policy, the default) or an inline frozen hosted model. Group sizing stays on the env config (`group_size`).
2. **The per-token training signal** — credit assignment and loss routing, fused; the algorithm's own parameters sit directly on `algo`. One mapping from a finalized rollout to per-token *(loss component, weight)* pairs — the credit a token gets and the loss that consumes it are two coordinates of the same output. Group-relative algorithms compute credit on the orchestrator and ship per-token advantage streams; reference-KL algorithms query a reference model at batch-ship time (bounded concurrency) and ship its prefill logprobs for the trainer to evaluate against the live policy. The `type` determines which loss component consumes the action tokens (`rl` / `ce` / `ref_kl`) and what happens to env-provided observation tokens in multi-turn rollouts (masked out by default; `echo` trains on them with weighted CE).

The trainer is algorithm-blind: the loss is a sum of three components (rl, ce, ref_kl), each normalized by its own global token count; per-token streams ship on the wire (the `rl_weights` / `ce_weights` / `ref_kl_weights` component weights plus the `advantages` stream on each training sample) and the trainer just executes them. Adding an algorithm never touches the dispatcher, packer, or trainer hot path.

### Model References

`prime-rl` hosts exactly one model: the trainable policy (`[orchestrator.model]`). Every other model an algorithm uses is an external OpenAI-compatible endpoint, declared *inline on the component that uses it*. A model reference is either the string `"policy"` (the live policy) or a frozen hosted model (`name` + `base_url`):

```toml
[orchestrator.algo]
type = "opd"

[orchestrator.algo.teacher]   # opd's teacher: the frozen model it scores against
name = "Qwen/Qwen3-32B"
base_url = ["http://localhost:8001/v1"]
```

Model *roles* are algorithm-local vocabulary — each algorithm names its reference on the field where the model is actually used, and there is no shared `teacher` slot. `opd` declares a `teacher` field (the frozen model whose reverse KL the policy distills toward); `sft`'s teacher *is* its `sampling.source` (the frozen model it imitates); `opsd` self-distills against the live policy and names no model at all. No role exists outside the algorithm that declares it: the dispatcher, sink, and trainer branch on liveness alone, never on what an algorithm calls a model.

So for `opd` set `[orchestrator.algo.teacher]`; for `sft` set `[orchestrator.algo.sampling.source]`; `opsd` needs neither. `opd`'s teacher must be a frozen endpoint — it is typed `FrozenModelConfig`, so `"policy"` isn't representable (the KL would be identically zero); `opsd`'s teacher *is* the live policy by definition (self-distillation conditioned on a demonstration), so it exposes no reference to configure.

Liveness is a property of the reference, not of any role: rollouts sampled from `"policy"` get version-salted prefix caches, carry sampling logprobs for importance ratios, and age off-policy as weights update; rollouts and scores from frozen models get a stable prefix cache and never go stale. Frozen models are externally hosted (`base_url` is required) — `prime-rl` never launches or updates them, and each env's algorithm builds its own client pool to the endpoints it declares.

### The Algorithms

The `algo.type` names the algorithm, and each type's class defaults are its vetted setting — picking a type with no other keys IS the algorithm:

```toml
[orchestrator.algo]
type = "grpo"  # the default
```

| `type` | Sampling | Loss | What it is |
|---|---|---|---|
| `grpo` | policy | `rl` on actions | Standard group-relative RL. |
| `max_rl` | policy | `rl` on actions | MaxRL ([arXiv:2602.02710](https://arxiv.org/abs/2602.02710)): GRPO's centered reward normalized by the group **mean** instead of the standard deviation — the gradient is unbiased for the order-`group_size` truncation of the maximum-likelihood objective, upweighting hard examples like `1/p`. |
| `opd` | policy | `ref_kl` on actions | On-policy distillation ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)): the policy samples, per-token reverse KL against a reference model as the gradient signal. Needs a `teacher`. |
| `sft` | *(the teacher)* | `ce` on actions | Hard distillation: a frozen model generates rollouts, the policy trains with CE on its tokens. Needs a frozen `sampling.source` (the teacher it samples from). |
| `opsd` | policy | `ref_kl` on actions | SDFT ([arXiv:2601.19897](https://arxiv.org/abs/2601.19897)): the model is its own reference, conditioned on an expert demonstration. The teacher *is* the live policy (the paper's setting, no extra deployment) — no model to configure. |
| `echo` | policy | `rl` on actions + weighted `ce` on observations | ECHO: standard GRPO plus a cross-entropy loss on env-provided tokens already present in the rollout, selected by message role (needs the renderer's role attribution). Defaults to tool-response bodies at `alpha = 0.1` (ECHO's λ); set `roles` to train other roles, each at its own weight. |

### Customizing Components

Every key beyond `type` is visibly your own assembly — there is no preset layer to diverge from. The vetted setting is the class defaults; what you set is what runs:

```toml
# echo on tool AND user feedback tokens, each at its own weight.
# Setting any role replaces the whole table.
[orchestrator.algo]
type = "echo"

[orchestrator.algo.roles.tool]
alpha = 0.25

[orchestrator.algo.roles.user]
alpha = 0.05
```

A new algorithm is a named class in code, not a config that points at an import path — see [Authoring an Algorithm](#authoring-an-algorithm).

Echo also takes an optional user-supplied token filter that narrows the role selection per rollout — e.g. dropping warning lines from tool output, or tokens the sampler found unlikely:

```toml
[orchestrator.algo.filter]
import_path = "my_module.drop_warnings"
kwargs = { patterns = ["WARNING"] }
```

```python
# my_module.py — sees the raw rollout (message text, sampling logprobs);
# returns one keep-mask per trainable branch, spanning that branch's
# token_ids. False = never echo-trained.
def drop_warnings(rollout, *, patterns: list[str]) -> list[list[bool]]: ...
```

Component compatibility is validated at config time: frozen-model sampling can only feed the `ce` loss component — the `rl` and `ref_kl` components need the live policy's own sampling logprobs for importance ratios — `opd` pointed at `"policy"` is rejected as degenerate (zero KL), `sft` without a frozen source is rejected (CE on the policy's own tokens is not a distillation target). A group-relative algorithm with `group_size = 1` produces all-zero advantages; the resulting empty batch is caught at runtime (the orchestrator warns and aborts after repeated zero-trainable batches), not at config time.

### Per-Env Algorithms

Both components resolve per environment. Each env inherits `[orchestrator.algo]` unless it sets its own, so a single run can mix algorithms across envs — e.g. GRPO on math, ECHO on a terminal env:

```toml
[orchestrator.algo]
type = "grpo"

[[orchestrator.train.env]]
id = "math-env"     # inherits grpo

[[orchestrator.train.env]]
id = "terminal-env"
algo = { type = "echo" }   # this env runs its own algorithm
```

### The Algorithm Classes

At runtime, each env's resolved config builds two objects: a `Sampler` (`prime_rl.orchestrator.sampler`) from the `sampling` component — the pool rollouts are generated from, and the home of future sampling strategies like replay buffers or branching — and one of the named algorithm classes in `prime_rl.orchestrator.algo` (one module per algorithm: `algo/grpo.py`, `algo/opd.py`, …) from the algorithm config. Algorithm dispatch is keyed on `algo.type` — it names the algorithm, and each config class's defaults are its vetted parameterization:

| `algo.type` | Class | hook(s) — stage |
|---|---|---|
| `grpo` | `GRPOAlgorithm` | `score_group`: group-norm credit (optional length penalty) |
| `echo` | `EchoAlgorithm` | `score_rollout`: weighted ce on observation tokens; `score_group`: group-norm credit (inherited) |
| `max_rl` | `MaxRLAlgorithm` | `score_group`: mean-normalized group credit |
| `opd` | `OPDAlgorithm` | `score_rollout`: own-context prefill under the teacher |
| `opsd` | `OPSDAlgorithm` | `score_rollout`: demo-conditioned prefill under the live policy |
| `sft` | `SFTDistillAlgorithm` | `score_group`: group-norm credit (feeds filters) |

Each class owns its hooks outright — reading one top to bottom reads the algorithm, and everything on the class is an override point. The two hooks are one scope-and-timing ladder — the wider scope is unlocked by a later barrier, so the two axes coincide. Each is handed the `Rollout` directly — the env's typed trace (`reward`, `nodes`, `num_turns`, ...) with `samples` attached, plus `assign_advantages` to write credit:

- `async score_rollout(rollout)` — one rollout, **on arrival** (as it's tokenized, before its group is complete): rollout-local credit (`rollout.assign_advantages(...)`, scalar broadcast or per-token), observation ce weights, **or** model I/O — query a reference pool (e.g. `self.teacher_pool`, connected in `setup()` via `self.connect(...)`, or the live `self.policy_pool` for opsd) and attach per-token results (e.g. teacher logprobs) with bounded concurrency. No siblings. `echo` weights observation tokens here, identifying env-provided observation nodes by their non-sampled status and source step role attribution, applying the optional user filter, and writing the `ce_weights` stream. Model I/O runs *before* the pre-batch filters, so it pays compute on rollouts that may then be filtered out.
- `score_group(group)` — the cohort, **before filtering** (filters read the streams), synchronous: group-relative credit (GRPO/MaxRL baselines). `group` is a list of `Rollout`.

The pipeline drives the hooks through two non-virtual methods it never looks inside: `algorithm.finalize_rollout(rollout)` per arrival (rollout-local scoring + reference I/O) and `algorithm.finalize_group(rollouts)` per group (scoring + wire stamping; after this the records are frozen — groups die at stamping). Sample construction (interleaving) is pure pipeline — observation-token provenance is available through structural attribution (`node.sampled`, `node.is_content`) for any algorithm that trains on env-provided tokens.

Class-level declarations state what the algorithm needs: which loss component its action tokens feed (`action_loss_type`). Every class is constructed with its algorithm config plus the one host-owned resource it can't rebuild — the live policy pool (`self.policy_pool`). Everything else an algorithm needs it builds from its own config in `setup()`: `opd` connects its frozen `teacher`; `opsd` builds the renderer for its demonstration hint (tokenizer is always the live policy's — self-distillation has no separate model). The pipeline only ever calls the two `finalize_*` methods — writing your own algorithm is subclassing `Algorithm` and overriding the hooks its signal needs (see [Authoring an Algorithm](#authoring-an-algorithm)). Shared math (efficiency shaping, prefill alignment) lives as plain functions in `prime_rl.orchestrator.algo.advantage`.

## Async / Off-Policy Training

`prime-rl` is asynchronous by default. The trainer and inference always run one step overlapped: while the trainer is producing $\pi_n$ from rollouts at step $n$, inference is already generating the rollouts for step $n+1$ using $\pi_{n-1}$. With matched trainer and inference step times this produces fully-overlapped pipeline parallelism — neither side ever idles.

![Async pipeline: trainer step n produces $\theta_n$, inference at step n samples with $\theta_{n-1}$](assets/async-pipeline.png)

At step $n = 1, 2, 3, \dots$:

- **Trainer** produces policy $\pi_n$ with weights $\theta_n$ from rollouts $(x_n, y_n)$.
- **Inference** produces rollouts $(x_n, y_n)$ from policy $\pi_{\max(0,\,n-1)}$.

Step indices are 1-indexed; policy versions are 0-indexed, with $\pi_0$ the base model. At step 1 inference samples from $\pi_0$.

## Loss

### Loss Components

The training loss is a **sum of three components**, each with its own per-token weight stream and its own normalization:

$$
\mathcal{L} = \frac{\sum \mathcal{L}_{rl}}{N_{rl}} + \frac{\sum \mathcal{L}_{ce}}{N_{ce}} + \frac{\sum \mathcal{L}_{ref\_kl}}{N_{ref\_kl}}
$$

- `rl` — the configured RL loss (`[trainer.loss]`): DPPO + KL by default, or a [custom loss](#custom-loss). Fed by the group-relative algorithms (`grpo`, `max_rl`, and `echo`'s action tokens).
- `ce` — masked NLL. Used for frozen-model tokens (`sft`) and env-observation tokens (`echo`).
- `ref_kl` — the per-token reverse KL to a reference model ($\log \pi_{\text{ref}} - \log \pi$) as the policy-gradient signal, importance-ratio corrected with a one-sided trust region (`opd`, `opsd`). Requires `ref_logprobs` from a [reference scoring](#reference-scoring); the scoring model must be a vLLM server (it's the only one that exposes `prompt_logprobs`).

The orchestrator stamps each sample's component membership as per-token weight streams (`rl_weights` / `ce_weights` / `ref_kl_weights` on the wire): a weight scales that component's per-token loss, `0.0` leaves the token out of the component entirely (mask *and* denominator), and components may overlap on the same token — their gradients sum. Each $N$ is the global (all-reduced) count of that component's member tokens, so the components don't dilute each other: adding echo observation tokens never changes the rl term's effective per-token learning rate, and an sft env packed next to a GRPO env doesn't soften its gradient. Tokens of different components pack freely into the same micro batch, and a plain GRPO run ships no weight streams at all (absent streams mean rl weight 1.0 on every trainable token — the unchanged hot path). Advantages always ship per token (`advantages` on the wire), assigned as per-token streams from the start — uniform group credit is broadcast over completion tokens at assignment; algorithms with no rl credit (opd, opsd) ship none.

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

The per-token training signal is set by `algo.type` and the [algorithm](#the-algorithm-abstraction)'s parameters — every signal is a per-token advantage stream, varying in evaluation site (orchestrator vs. trainer). The `algo.type` values:

| Type | Component | Effect |
|---|---|---|
| `grpo` | `rl` | Group-norm: reward minus per-group baseline, optional length penalty. |
| `max_rl` | `rl` | Mean-normalized group credit (maximum-likelihood RL). |
| `echo` | `rl` + `ce` | Group-norm on action tokens, plus weighted CE on env-provided tokens selected by message role (each role's `alpha` is its ECHO λ), optionally narrowed by a user filter. |
| `opd` | `ref_kl` | On-policy distillation: per-token reverse KL to a reference model (`model`, an inline frozen hosted model), evaluated in the trainer from shipped reference logprobs. No credit — rollouts keep `advantages = None` (advantage-based filters never fire) and ship no advantage stream; `group_size` only fans out sampling. |
| `opsd` | `ref_kl` | SDFT: per-token reverse KL to a demo-conditioned reference. No credit — rollouts keep `advantages = None` (advantage-based filters never fire) and ship no advantage stream. |
| `sft` | `ce` | Cross-entropy on the sampled tokens. Assigns no advantage — trains on every sampled token. |

### Default Advantage

The default advantage is per-group reward minus per-group baseline (DR-GRPO without std normalization). For each prompt's group of `group_size` rollouts, every token in rollout $i$ receives advantage $s_i - \bar{s}$ where $\bar{s}$ is the group mean.

This is intentionally simple — it does the right thing for most envs. Write a named algorithm class when you need group-aware shaping that depends on trajectory metadata (sub-agent rollouts, relative-rank shaping, …) — see [Authoring an Algorithm](#authoring-an-algorithm).

A **length penalty** (`length_penalty` on the `grpo`-family algorithms) can be layered on top to discourage rambling. The `linear` penalty subtracts a single `pass_rate`-scaled penalty from each reward before the GRPO baseline, combining output tokens (`num_output_tokens_weight`), input / context tokens (`num_input_tokens_weight`), and turns (`num_turns_weight`) — each normalized by the group's own max for that quantity, with `num_input_tokens_weight` and `num_turns_weight` defaulting to `0.1`.

```toml
[orchestrator.algo]
type = "grpo"

[orchestrator.algo.length_penalty]
type = "linear"
```


### Authoring an Algorithm

There is no config hook that points at user code — a new credit-assignment scheme is a new named algorithm in the repo. Subclass `Algorithm`, assign credit in the scoring hook whose timing fits your signal, and register the class. The hook receives the group's `Rollout`s (each the env's typed `verifiers.Trace` — turns, tool calls, metadata in `info` — with `samples` attached) and writes credit via `assign_advantages`:

```python
# src/prime_rl/orchestrator/algo/my_algo.py
import torch

from prime_rl.orchestrator.algo.base import Algorithm


class MyAlgorithm(Algorithm):
    async def score_group(self, group):
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        advantages = ...  # one value per rollout
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
```

Add a typed `MyAlgoConfig` to `prime_rl.configs.algorithm` and its discriminated union, then register `"my_algo": MyAlgorithm` in `ALGORITHM_CLASSES`. Pick the hook by *when* your signal is ready: `score_rollout` for per-arrival credit or credit that needs a model call (it's `async`), `score_group` for group-relative credit (GRPO/MaxRL). `assign_advantages` takes a scalar (broadcast over the rollout's trainable tokens — the common case) or a full-length per-token list aligned to the concatenated sample token_ids (process rewards, step-level credit; `0.0` off-mask).

Each per-token list must match the rollout's completion-token count exactly — validated loudly when the view writes it. Advantage-based filters and metrics derive from the streams (the zero-advantage filter checks for all-zero streams; logged distributions use per-rollout means). Signals that depend on the live policy's weights (like OPD's reverse KL) cannot be precomputed here; those are reference-scoring algorithms, evaluated in the trainer.

### Reference Scoring

`OPDAlgorithm` / `OPSDAlgorithm` do their model I/O in `score_rollout`: as each rollout arrives they query a reference (the sample's own context for `opd`, the demo-conditioned context for `opsd`) and attach per-token reference logprobs to each sample. Rollouts are consumed serially by the orchestrator's main loop and each carries only a handful of samples, so the in-flight request count is naturally bounded — no explicit concurrency cap:

- `opd` — score each sample's own context under the `teacher` (a frozen [model reference](#model-references)) via prefill; fills `ref_logprobs` for the `ref_kl` loss component (on-policy distillation). The `teacher` is typed `FrozenModelConfig`, so `"policy"` isn't representable (the KL would be identically zero).
- `opsd` — SDFT: prepend an expert demonstration as a leading system message (`template`, with a `{demonstration}` placeholder) and score the sample under that demo-conditioned context. The sample is scored verbatim (`hint_block + token_ids`, slicing the hint's logprobs back off), so the join is BPE-clean and it's robust to tool/multimodal prompts and any number of turns. The scoring reference *is* the live policy — self-distillation names no teacher. opsd builds its own renderer to tokenize the hint block: the tokenizer is always the live policy's (not configurable — there is no separate model), and only the `renderer` family is settable (defaults to `"auto"`, resolved from the policy tokenizer; set it to match a non-auto policy renderer). The demonstration is read from the example's `info[demo_key]`, falling back to a top-level rollout field of the same name (e.g. `answer`).

```toml
[orchestrator.algo]
type = "opsd"
demo_key = "demonstration"
```

Scoring runs at arrival, *before* the pre-batch filters, so a rollout that is later filtered still cost its reference compute — accepted for the simpler one-rollout-at-a-time shape (advantage-based filters never fire for opd/opsd anyway, since neither assigns an advantage).

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
