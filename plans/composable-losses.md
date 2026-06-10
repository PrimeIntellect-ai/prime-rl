# Composable losses — the design

> **Single source of truth** for the composable-loss / advantage-function work on
> `sebastian/losses-2026-06-04` (draft PR #2715). This document consolidates and **replaces** the
> earlier `plans/loss-design.md`, `plans/losses.md`, and `plans/pedagogical-rl.md`.
>
> It is meant to be read top-to-bottom by someone new to the PR: **§1–2** are the idea and the mental
> model, **§3** is the config surface by example (presets → custom → multi-term → per-env →
> references), **§4–6** are the component axes / presets / references in detail, **§7–8** are the
> internals and edge cases, **§9** the applications, **§10** the comparison to
> `feat/algorithm-abstraction`, and **§11–12** the status (done / TODO) and open questions.
>
> Markers: **✅ built** · **🟡 designed (near-term)** · **⚪ backlog**. The default (no config) is
> **byte-for-byte today's DPPO+KL** — that invariant is the regression guard for every chunk.

---

## 1. What we're building (the thesis)

Today's RL loss bakes three things into one function: the **objective** (DPPO+KL), the **token
selection** (which tokens train), and the **per-token weighting** (the GRPO advantage). This PR pulls
them apart so a training run becomes a **list of composable loss terms summed over one shared
forward**.

The shape is the point — we win on *shape*, not feature count:

- **N-term composition.** Any number of terms, each with its **own λ** and its **own normalization**,
  so terms neither dilute nor mask each other.
- **Per-token everything.** Selection and weighting are per-token floats, not coarse action/observation
  routing.
- **One uniform custom surface.** Every component of a term — advantage, core, reduce, hooks — is *the
  same kind of thing*: a **pointer** (a built-in preset name *or* a dotted import path) **plus kwargs**.
  A custom *loss* is declared exactly like a custom *advantage*. No "custom loss is configured
  differently from custom advantage."
- **Arbitrary filtering.** `0`-advantage is a true mask (no KL leak); trainer-side gates live in hooks.
- **Trainer-blind references.** Another model's logprobs are prefilled orchestrator-side and shipped
  per-token; the trainer just consumes them.

The default `losses` reproduces today's DPPO+KL exactly.

## 2. The mental model

A **term** is:

```
LossTerm = { name, advantage, loss (core), lambda_weight, reduce, hooks }
```

and the run is `losses: list[LossTerm]`. Per-env `enabled_losses` selects which terms apply;
per-env `loss_overrides` tweaks them (advantage axis only — see §3.5).

### The five-stage per-term pipeline

```
advantage_fn   →   core            →   hooks               →   reduce        →   λ·sum
(orchestrator)     (trainer)           (trainer)               (per term)        (combine)
per-token float    per-token loss      per-token→per-token     → scalar          total = Σ λ_t·s_t
0 = masked         NOT reduced         sees the live forward
```

1. **advantage_fn** (orchestrator) — one float per token; `0` = masked. Group-aware (sees the whole
   GRPO group); also emits constant / dataset-driven / reference-derived signals. It's the adapter
   between heterogeneous rollout data and uniform loss math — **and it is the filtering mechanism, for
   free and by default**: `0` is a true mask, so a term whose advantage is all-zero for a sample drops
   out automatically, and a sample that *no* term applies to is dropped from the batch (emergent
   zero-advantage dismissal, §8). Filtering is not a separate stage — you filter by zeroing the
   advantage.
2. **core** (trainer) — `core(LossInputs) -> LossOutputs`; per-token loss, **not** reduced. One
   parameterizable policy-gradient core underlies `dppo_kl` and `ce` (§4.2).
3. **hooks** (trainer) — `hook(per_token_loss, inputs) -> per_token_loss`, chainable; for transforms
   that need the live forward (current-policy prob/entropy gating, smoothing, penalties).
4. **reduce / normalize** (per term) — per-token → scalar. **This is where normalization lives** — its
   own axis, **not** the core and **not** a hook. Default `mean` (global per-token mean = today's
   `loss_scale`); customize with a `custom` reduce (§4.3). Per-term, so terms normalize independently
   and don't dilute each other.
5. **λ·sum** — `total = Σ_terms lambda_weight_t · scalar_t`.

### The execution seam (split execution, unified surface)

A term spans both processes along a forced seam: the **orchestrator** owns the rendered rollout +
rewards + group + reference logprobs (so the advantage is computed there); the **trainer** owns the
forward (so core + hooks + reduce run there). What is *not* forced — and what we deliberately do **not**
do — is leak that seam into the config. The user **co-locates all of a term's components**; the wire
carries per-token advantages + per-term routing; each process reads its slice.

### Primary vs overlay

Exactly one term may be the **primary** — the RL objective, identified by a **`grpo` advantage** (with
a `dppo_kl`/`custom` core). It is dispatched per sample by `training_mode`. Every other term is an
additive **overlay** (a `ce`/`custom` core with an `echo`/`sft`/`custom` advantage). This is a
resolved-internal distinction; the user just writes terms.

### Why one shared forward, one summed backward

Every term differentiates the **same** per-token `trainer_logprobs` from the single forward.
`∇(rl + echo) = ∇rl + ∇echo`, so summing the per-term scalars and calling `backward()` **once** is
correct and cheapest. Separate backward passes would double the backward cost for an identical
gradient (the shared forward can't be freed between them without `retain_graph`, which frees nothing).
Gradient surgery / PCGrad is the only reason to split — out of scope.

A core is **not** a two-stage `loss ∘ post-loss` pipeline. The DPPO+KL core is a function of the
importance ratio `exp(trainer_lp − inference_lp)` plus a squared-KL term, owning its advantage
multiplication and trust-region mask internally — it is *not* "cross-entropy × advantage." Masked-NLL
(`ce`) *is* cleanly `weight · (−logprob)` masked; RL is not, and we don't force it to be. Cross-cutting
transforms that *are* post-processors of the per-token loss go in **hooks** (§4.4).

---

## 3. Config by example

One shared `losses` list defines the terms; per-env config selects from it by name. The common
surface stays tiny; the machinery only appears when you deliberately go custom. (`losses` is set once
at the `RLConfig` level and propagated to both the trainer and orchestrator sub-configs.)

### 3.1 Default — write nothing  ✅

```toml
# (no [[losses]] at all)  ==  today's DPPO+KL, byte-for-byte
```

The default `losses` is a single `rl` term: `{ loss = dppo_kl, advantage = grpo, reduce = mean, λ = 1 }`.

### 3.2 One-line presets  ✅

Plain RL, written explicitly:

```toml
[[losses]]
type = "rl"                      # single-term preset: dppo_kl core + grpo advantage
```

Echo — **one block gives you both** the RL objective *and* the cross-entropy supervision, because the
`echo` recipe expands to two terms (`rl ⊕ ce-on-roles`):

```toml
[[losses]]
type  = "echo"                   # COMPOUND RECIPE -> the `rl` primary + a `ce`-on-roles overlay
roles = ["user", "tool"]         # what to echo (routes to the ce overlay's advantage)
```

`echo` is not a primitive — it is the policy-gradient objective on the completion **plus**
cross-entropy supervision on the chosen role tokens. Because the `echo` block **already emits the `rl`
primary, you do not also write a separate `rl` term** — doing so is a duplicate-name error (§5).

### 3.3 Tuning a preset

Override an axis; the preset's other defaults stay (so `loss = { kl_tau = 5e-4 }` keeps
`dppo_mask_low/high = 0.2`) — the same semantics as a full term.

Plain RL, retuned:

```toml
[[losses]]
type = "rl"
loss = { kl_tau = 5e-4 }
```

Echo, **fully tuned through the one `echo` block** — its own knobs route to the ce overlay, and a
nested `rl = { ... }` tunes the bundled policy-gradient half. You never add a separate `rl` term:

```toml
[[losses]]
type          = "echo"
roles         = ["assistant"]                  # -> the ce overlay's echo advantage
lambda_weight = 0.5                            # -> the ce overlay's λ (the echo magnitude)
rl            = { loss = { kl_tau = 5e-4 } }   # -> tunes the bundled rl primary
```

### 3.4 Multiple terms, written out in full

Presets are sugar; the canonical form names each axis explicitly. Terms may overlap — gradients sum.

```toml
[[losses]]                                     # the primary (== {type = "rl"})
name      = "rl"
loss      = { type = "dppo_kl", kl_tau = 1e-3 }
advantage = { type = "grpo", tau = 1.0 }

[[losses]]                                     # behavior-cloning regularizer: constant-weight NLL on the sampled tokens
name          = "bc"
loss          = { type = "ce" }
advantage     = { type = "sft", alpha = 1.0 }
lambda_weight = 0.1

[[losses]]                                     # a custom per-token advantage (the uniform pointer surface)
name          = "shaped"
loss          = { type = "ce" }
advantage     = { import_path = "pkg.my_advantage", kwargs = { scale = 0.5 } }
lambda_weight = 0.05
```

(The same three-term shape with two `echo` overlays — different roles, one advantage-weighted — is the
echo recipe's territory; see §5.)

### 3.5 Per-env selection + overrides  ✅

```toml
[[orchestrator.train.env]]
id = "math"                                    # default enabled_losses = all terms

[[orchestrator.train.env]]
id             = "tool-traces"
enabled_losses = ["rl", "echo"]                # only these two apply here
loss_overrides = { echo = { advantage = { tau = 0.01 } } }   # per-env tweak
```

`loss_overrides` may override **only the `advantage` axis**, **only on overlay terms that are
enabled** in that env. Rationale: the advantage_fn is resolved per env (orchestrator-side), so it is
the only per-env-cheap axis; `loss` (the core), `reduce`, and `hooks` are global (trainer-side), so a
per-env override there would validate but be silently ignored — we reject it rather than mislead. An
override on a disabled term is a silent no-op, so that's rejected too.

### 3.6 Fully custom — the uniform pointer surface  ✅

Every axis takes a built-in `type` **or** a dotted `import_path` + `kwargs`, with identical shape:

```toml
[[losses]]
name      = "myterm"
loss      = { import_path = "pkg.my_core",   kwargs = { temperature = 0.7 } }   # trainer-side
advantage = { import_path = "pkg.my_adv",    kwargs = { k = 3 } }               # orchestrator-side
reduce    = { import_path = "pkg.my_reduce" }                                   # trainer-side
hooks     = [ { type = "min_prob_filter", min_logprob = -8.0 },
              { import_path = "pkg.my_hook" } ]                                  # trainer-side, in order
lambda_weight = 0.5
```

Signatures (§4): `advantage_fn(group: list[RenderHints], **kwargs) -> list[list[float]]` ·
`core(inputs: LossInputs, **kwargs) -> LossOutputs` · `reduce(inputs: ReduceInputs, **kwargs) -> Tensor`
· `hook(per_token_loss: Tensor, inputs: LossInputs, **kwargs) -> Tensor`. Custom pointers are
caller-beware: the config is structurally validated and the target imported at setup, but the
*semantics* of a user fn are the user's responsibility.

### 3.7 References — scoring against another model  🟡 *(designed; see §6)*

A term can consume another (frozen) model's per-token logprobs (an OPD distillation target, a
pedagogical frozen-student's surprise). The reference is prefilled orchestrator-side and shipped
per-token; a `ref_kl` advantage (or core/hook) reads it:

```toml
[[losses]]
type = "opd"                                   # 🟡 future preset == { ce/dppo_kl core, ref_kl advantage }
advantage = { type = "ref_kl", scorer = "reference" }   # points at a named scorer (§6)
```

`top_k` is a property of the scorer (how many `(token_id, logprob)` per position it returns);
`top_k = 1` is the sampled-token estimate.

### 3.8 Filtering

```toml
# trainer-side gate (needs the live forward): zero a term's loss on tokens the current policy is
# already confident about. Hooks attach to any term — here, the rl primary in full form:
[[losses]]
name      = "rl"
loss      = { type = "dppo_kl" }
advantage = { type = "grpo" }
hooks     = [ { type = "min_prob_filter", min_logprob = -2.0 } ]
```

- **`0`-advantage is a true mask** (§8): set a token's advantage to `0` and it leaves both the
  numerator and the denominator — no gradient, no KL. Anything computable from the rollout (role, tool,
  sampling-logprob threshold, custom predicate) filters by zeroing the advantage_fn output.
- **Trainer-side filters are hooks** (`min_prob_filter` ✅, or custom) — for anything needing the live
  forward.
- **Per-rollout filters** (`zero_advantage`, gibberish, repetition) stay as orchestrator post-batch
  filters.

---

## 4. The component axes (reference)

### 4.1 advantage_fn — orchestrator-side  ✅

`fn(group: list[RenderHints], **kwargs) -> list[list[float]]` — one list per unit, one float per
token, aligned to `prompt_ids + completion_ids`; `0` masks. Resolved per env by
`orchestrator/advantage.py::resolve_advantage_fn` and run per group in `train_sink`.

| `type`   | meaning |
|----------|---------|
| `grpo`   | the per-rollout reward-baseline advantage × `tau`, broadcast over sampled tokens, `0` elsewhere — **the primary / RL objective** |
| `echo`   | selection mask: `1.0` on role-matched context tokens (optionally narrowed by `tool_names`), `0` elsewhere; `× advantage·tau` when `by_advantage`. **No `alpha`** — magnitude is the term's λ |
| `sft`    | constant `alpha` on sampled tokens, `0` elsewhere (masked NLL) |
| `custom` | a dotted `import_path` + `kwargs` |

`RenderHints` (read-only, per-token-aligned) is the advantage_fn's data contract:

```python
@dataclass
class RenderHints:
    token_id: list[int]
    role: list[str | None]              # per-token role (None = unattributed)
    tool_name: list[str | None]         # set when role == "tool"
    is_sampled: list[bool]              # the sampled completion tokens (today's loss_mask)
    inference_logprob: list[float]      # sampling-policy logprob; 0.0 on prompt tokens
    reward: float | None                # per-rollout scalar
    advantage: float | None             # per-rollout scalar from Layer 1 (pre-tau)
    rollout: vf.RolloutOutput | None    # raw renderer output
    # 🟡 to add (§6): reference_logprob / reference_topk_* for reference-reading advantage_fns
```

Group-awareness (the whole list) is what lets `grpo` center group-relative; constant/dataset signals
ignore it. Orchestrator-side signals (sampled entropy, reward) belong here; trainer-side signals
(current-policy entropy, the live loss) belong in a **hook**.

### 4.2 core — trainer-side  ✅

`core(inputs: LossInputs) -> LossOutputs`; returns a **per-token** loss (not reduced).

```python
@dataclass
class LossInputs:
    trainer_logprobs:   Tensor          # current policy, from the forward
    inference_logprobs: Tensor          # sampling policy (the rollout)
    teacher_logprobs:   Tensor | None   # a second model's logprobs (today: the reference scorer; misnamed — §6)
    advantages:         Tensor          # the term's resolved per-token advantage (0 = masked)
    loss_mask:          Tensor          # eligible tokens

@dataclass
class LossOutputs:
    loss:           Tensor              # scalar (byte-for-byte today's when no hooks — §7)
    per_token_loss: Tensor | None       # the pre-sum tensor the hook chain runs on
    metrics:        dict
```

| `type`     | meaning |
|------------|---------|
| `dppo_kl`  | DPPO+KL policy-gradient core (the RL objective): trains filtered tokens with the per-token advantage as weight; masks trust-region violators; squared-KL regularizer (`kl_tau`) |
| `ce`       | cross-entropy / weighted NLL: `−Σ weightₜ · logprobₜ` over filtered tokens (echo / SFT) |
| `custom`   | dotted `import_path` + `kwargs` |

**One parameterizable core.** `dppo_kl` and `ce` are the same policy-gradient core with different
flags: `use_importance_ratio=False` ⇒ `−(advantage · trainer_lp)` = weighted NLL (`ce`);
`True` + clip + `kl_weight>0` ⇒ DPPO+KL (`rl`). The DPPO trust-region clip and the KL term stay
**inside** the core (intrinsic, ratio-dependent) — never in hooks. KL is a core arg, so a `0`-advantage
token is masked ⇒ no KL on it.

**Validity:** `dppo_kl` needs real `inference_logprobs` and GRPO advantages → only sampled completion
tokens (context tokens have `inference_logprob = 0`). `ce` works on **any** token — which is exactly
why echo uses it on context tokens. The grpo↔dppo_kl / overlay↔ce pairing is validated at config time.

### 4.3 reduce — per term, trainer-side  ✅

`reduce(inputs: ReduceInputs, **kwargs) -> Tensor`. **Normalization is this axis.** To change how a
term is normalized you write a `custom` reduce — **not** a custom loss and **not** a hook (the core
returns *unreduced* per-token loss by design, and hooks are per-token→per-token; reduce is the only
per-token→scalar step, which is also why masking hooks compose). `mean` (the default) = global
per-token mean over the term's eligible tokens (all-reduced count, `max(·,1)`) — bit-compatible with
today's `loss_scale` and unbiased (masked tokens leave both numerator and denominator). `custom` = a
dotted path (e.g. the pedagogical `1/Σ wₜ` normalization, §9). Reduce is **per-term, not per-env** —
per-env λ already gives per-env weighting; normalizing slices of one term differently is incoherent.

### 4.4 hooks — trainer-side, post-core  ✅

`hook(per_token_loss: Tensor, inputs: LossInputs) -> per_token_loss`, **chainable**, **no scalar
return** (reduction is the separate stage 4, so masking hooks compose). For signals only available
after the forward.

| `type`            | meaning |
|-------------------|---------|
| `min_prob_filter` | zero the per-token loss where the current-policy logprob `< min_logprob` (a trainer-side filter) |
| `surprisal_gate`  | weight the per-token loss by `σ(kappa·(trainer_logprob − gamma))` — the pedagogical student-assimilation gate over the trained policy's logprob (§9) |
| `custom`          | dotted `import_path` + `kwargs` |

Principle: intrinsic objective math (DPPO clip + KL) stays inside the core; hooks are cross-cutting
transforms layered on top.

**Hooks are per-sample, not group-level (decided).** A hook sees one sample's live forward, not the
GRPO group. The group is an orchestrator concept and isn't preserved through the trainer's micro-batch
packing, so group-level trainer hooks would mean shipping group IDs and regrouping in the loss —
duplicating what the advantage_fn already does naturally. The clean split: **needs-the-group →
orchestrator advantage; needs-the-live-forward → trainer hook.** (A transform needing *both* — exotic —
is the only thing this rules out; nothing on the roadmap needs it.)

### 4.5 λ (`lambda_weight`)

Per-term coefficient applied pre-reduce (default `1.0`). It owns a term's **magnitude** (e.g. the echo
strength), keeping the coefficient independent of the reward-derived advantage. Per-env λ is the
intended per-env weighting knob.

---

## 5. Presets & recipes  ✅

Resolution cascades by **config position** and is deep-merged at any layer:

- **Component presets** sit in an axis slot: `advantage = "grpo"`, `loss = "dppo_kl"`,
  `reduce = "mean"` → an in-repo path + default kwargs.
- **Full presets** sit in a term's `type` and expand **one** term: `{type = "rl"}` →
  `{ loss = dppo_kl, advantage = grpo, reduce = mean, λ = 1 }`.
- **Compound recipes** sit in a term's `type` and expand to a **list** of terms. This is the headline
  demonstration that the framework composes.

### `echo` = `rl ⊕ ce-on-roles`

`{type = "echo", ...}` fans out (at the `losses` list level) into **two** terms — the standard `rl`
primary and a `ce` overlay whose advantage selects the echoed role tokens:

- echo's advantage knobs — `roles` / `tool_names` / `by_advantage` / `tau` — route to the **ce
  overlay's** echo advantage;
- `lambda_weight` / `reduce` / `hooks` set the **ce overlay term** (λ owns the echo magnitude);
- an `rl = { ... }` block deep-merges into the **rl sub-term**, so the policy-gradient half is tuned
  *through* echo, never as a separate, name-colliding sibling.

**Magnitude is λ, not a second alpha.** `EchoAdvantageConfig.alpha` was dropped — the advantage is a
pure selection mask; non-uniform per-token weights stay reachable via a `custom` advantage.

**Names stay unique — duplicate names raise.** A name is the trainer's core-registry key and what
`enabled_losses`/`loss_overrides` reference, so a clash must fail, not silently merge. Recipe expansion
tags each emitted term with its source preset, so a collision with a recipe's sub-term gives a pointed
error, e.g. *"Duplicate loss term name 'rl': emitted by the 'echo' compound preset and also defined
elsewhere — tune echo's rl through the echo config instead."* `rl`/`sft`/`opd` are reserved names
(trainer dispatch keys); at most one primary term is allowed.

---

## 6. References — second-model logprobs

Some terms need **another model's per-token logprobs over the trajectory, computed before the loss**:
the OPD distillation target, a pedagogical frozen-student's surprise. This is orchestrator-side
(prefill) and **trainer-blind** (ship per-token, the trainer just consumes).

### What exists today  ✅

- A single global `orchestrator.reference` (a `ClientConfig` + model name + a `logprobs.top_k` field)
  and a **managed `reference_inference` pool** (prime-rl hosts it). Distinct from `orchestrator.teacher`,
  the SFT *generator* (it samples rollouts) — a different thing entirely.
- `compute_reference_logprobs` prefills the full `prompt_ids + completion_ids` against the reference
  (currently `prompt_logprobs = 1`, **top-1**) and ships a flat per-token `reference_logprobs` on the
  wire. The **OPD core** consumes it (`reference_kl = reference_logprobs − trainer_logprobs`, used as
  the per-token signal), so the reference-consuming *core* seam (`LossInputs`, trainer-side) works.
- The wire / `MicroBatch` / `LossInputs` field was renamed **`teacher_logprobs` → `reference_logprobs`**
  (it carries the reference scorer's output, never the SFT generator's — the old name was a live
  foot-gun). **Done.**
- `logprobs.top_k` is still **inert** — making it live is coupled to a consumer (see below).

### The end-game (proposed): a named-scorer registry  🟡

The cleanest general form falls straight out of the framework's own thesis (every component is a
pointer; references are trainer-blind; roles are component-local labels). Three pieces:

1. **Named scorers.** A scorer is a pointer: a name → `{ how-to-reach-it, logprobs.top_k }`.
   `"reference"`, `"frozen_student"`, `"policy@iter-1000"` are just *keys*. teacher / reference /
   frozen-student collapse into one concept — "a model that can score a trajectory."
2. **A generic per-token map on the wire**, parallel to `term_advantages`:
   `scored_logprobs: dict[str, PerTokenTopK]`. The orchestrator prefills each configured scorer **once**
   and ships its top-k per token under its name.
3. **Consumers point at a scorer by name** — `advantage = { type = "ref_kl", scorer = "reference" }`, or
   a pedagogical `G_spike` advantage reads `scorer = "frozen_student"`. The framework guarantees a
   referenced scorer is computed, shipped, and surfaced into `RenderHints` (advantages) + `LossInputs`
   (cores/hooks). (The `surprisal_gate` hook is *not* a scorer consumer — it reads the live trained
   policy, `trainer_logprobs`.)

Why this is the general form — and why it dissolves the global-vs-inline fork: **chunk-1's global field
and the "inline-on-component" idea are both special cases of the registry** (one hosted entry; one
external entry at its only use-site). "External-only / never host" is a **policy knob on a scorer** (a
scorer resolves to an external `base_url`, a managed pool, or — later — a policy snapshot), not a
property of the architecture. One prefill, N consumers.

### Done ✅

The rename; the flat `reference_logprobs` feed → `LossInputs` → the OPD core (the reference-consuming
*core* seam works for `top_k = 1`, the sampled-token estimate). The `surprisal_gate` hook (§4.4) lands
the pedagogical Phase-2 gate (it reads the live trained policy, `trainer_logprobs`, so needs no
reference feed).

### The reorder — a lazy reference handle (decided)

To let orchestrator-side advantage logic *use* the reference, it must be available **before** GRPO
centering — the "reorder." We do it **paper-faithfully** (`center(R · G_spike)`: the reference feeds the
reward that centering sees), not the cheaper post-hook approximation (`center(R) · G_spike`, which was
considered and rejected — it filters on `R`-ties and centers the unshaped reward).

**Mechanism — pull-based, zero new config.** A single memoized `ReferenceHandle` per group is the *only*
prefill primitive. `await reference(sample)` triggers one batched prefill against the reference model
and caches the result **on the sample**; not calling it costs nothing. No scheduler, nothing to declare
— **the call is the opt-in, and prefill timing follows whoever pulls first:**

- an **advantage *strategy*** (the per-rollout scalar — where centering lives) pulls it → reference is
  there *before* centering → reward shaping like `G_spike`;
- a **per-term advantage_fn** pulls it → there *after* centering → per-token signals like a `ref_kl`
  surprise weight;
- a **trainer-side core** can't pull mid-loss, so it **declares** it needs the reference shipped (OPD
  does); the orchestrator pulls the handle at **post-filter ship-prep** and stamps `reference_logprobs`
  on the sample. This makes **OPD functionally identical** (same prefill call, same values, same
  post-filter timing, same `opd_loss_fn`) — but now OPD is "just a trainer-side reference consumer
  through the shared handle," and the bespoke finalize prefill goes away. Caching-on-sample means an
  orchestrator-side puller and OPD share one prefill (no double work).

So one primitive, two access modes: **orchestrator-side consumers pull (lazy); trainer-side consumers
declare and the orchestrator pulls-and-ships.**

**Footprint (deliberately small).** The handle rides on `RenderHints`, so **no advantage_fn signature
changes** — a reference-using fn is just `async` and does `await hints.reference_logprobs()`; built-in
sync fns (`grpo` / `echo` / `sft`, the GRPO strategy) never touch it and are unchanged. The
orchestrator-side advantage step becomes `async`, but it already runs inside an async context
(`process_group` ← async `add`), so the change is localized (`process_group` + `assign_advantages` + an
`isawaitable` await) with **zero user-facing config**. Generalizes for free to the multi-scorer registry
(each scorer is its own lazy handle).

**Why this is generally more powerful** (not just "enables pedagogical") — all orchestrator-only,
impossible trainer-side:

- **pre-centering reward shaping** (`r' = f(R, reference)`);
- **group-relative reference advantages** (an advantage_fn combining the group *and* the reference);
- **reference-based filtering *before* the forward** — a reference-derived advantage can go to `0` →
  emergent dismissal → no forward, so the "filter-before-forward saves compute" property extends to
  reference signals.

(Hooks deliberately stay **per-sample**, not group-level — §4.4: needs-the-group → orchestrator
advantage; needs-the-live-forward → trainer hook. Pedagogical splits cleanly: `G_spike` is
orchestrator-side, `surprisal_gate` is the live-forward hook.)

### Still deferred 🟡

- **Top-k distillation core** — `logprobs.top_k > 1` only pays off with a core that gathers the
  trainer's logprobs at the reference's top-k ids (a forward-pass gather); feed + gather-core ship
  together. `top_k = 1` (the sampled-token estimate) already works.
- **The named-scorer registry config surface** — the open team item (§12); the wire stays the renamed
  flat `reference_logprobs` (one scorer) until a second scorer lands.

---

## 7. Internals — how it executes

- **The wire** (`transport/types.py`, msgspec `array_like` + `omit_defaults`). `TrainingSample` carries
  `term_advantages: dict[str, list[float]] | None` — the **primary keyed by `training_mode`, overlays
  keyed by name** — plus `roles` / `tool_names` (per-token attribution) and the reference logprob
  field(s) (§6). Default path ships nothing extra.
- **Primary/overlay split stays internal.** `MicroBatch` keeps `overlay_masks` / `overlay_weights`;
  `prepare_sample` re-splits `term_advantages` by `training_mode` (the primary key) vs overlay names.
- **`compute_loss`** iterates terms, runs each core over its `(advantages, loss_mask)`, applies that
  term's hook chain, reduces (its own scale), multiplies by λ, and **sums**; aggregates per-term
  metrics; **one `backward()`**.
- **Normalization.** One scale per term = the global (dp_cp) all-reduced count of that term's eligible
  tokens (`max(·,1)`), so each term is a true global per-token mean and terms don't dilute each other.
  With `losses = [rl]`, the rl scale equals today's `loss_scale` ⇒ identical result. The FSDP
  per-rank `fsdp_gradient_divide_factor` undo is unchanged.
- **Bit-identity strategy for hooks.** Rather than invert the core↔reduce boundary, cores
  *additionally* return `per_token_loss` while the scalar `loss` stays byte-for-byte unchanged. The
  no-hook path consumes the scalar (bit-identical — golden tests pass untouched); the hook chain runs
  on `per_token_loss` and is summed only when hooks are present.
- **Validation timing.** Structural validation (types, unique/reserved names, single primary,
  `enabled_losses ⊆ defined`, supported axis combos) is **always**, at config parse. A component's
  existence/import is verified **in the process that runs it, at setup** — never an eager cross-process
  import at parse time (a trainer-side core isn't importable in the orchestrator that validates the
  shared config). Per-env overrides are validated eagerly via `apply_term_override` (reconstruct the
  `LossTerm`, so a bad value is caught at config time, not deferred to resolve).
- **adv_tau baked orchestrator-side.** The advantage's `tau` folds into the shipped per-token value (a
  scalar multiply), so per-env `tau` is free and the core consumes the already-scaled advantage.

---

## 8. Edge cases & settled decisions

- **`0` = mask** (no KL leak). A `0`-advantage token leaves both numerator and denominator and gets no
  KL — matching today's zero-advantage handling. (`None` is reserved if KL-on-zero is ever wanted.)
  This is strictly better than `feat/algorithm-abstraction`, where `advantage = 0` zeroes the PG term
  but the KL term still pulls the token.
- **Emergent zero-advantage dismissal** ✅. The default post-batch `zero_advantage` filter is
  monitor-only; `train_sink.process_batch` masks the primary on any zero-advantage rollout and ships a
  sample iff some term still applies (`_sample_has_trainable_tokens`). A tied / size-1 GRPO group →
  all-zero adv → primary masked → dropped if no overlay applies; an RL+constant-echo rollout keeps its
  echo gradient. No special filter.
- **Empty-batch safety.** The trainer already guards an all-zero batch (`loss_scale = max(·,1)`), and
  empty batches were already reachable via the old enforcing filter — so dropping no-op samples adds no
  new crash hazard; worst case is a harmless zero-gradient step. The planned env-sampler backfills to
  make that rare (not required for correctness now).
- **Duplicate / reserved names.** Duplicate term names raise (provenance-aware — §5). `rl` is reserved
  for the primary; `sft` / `opd` are reserved (trainer dispatch keys). At most one primary term.
- **Default is byte-for-byte today's DPPO+KL** — the regression guard for every chunk.
- **`training_mode`** (`rl`/`sft`/`opd`) is a transitional bridge: it routes by data source/path and
  dispatches the primary core. Kept until the env-sampler refactor folds the data axis; the design
  doesn't depend on it long-term. The `sft`/`opd` data paths are unchanged here.
- **Per-env = per-sample resolved DATA, not per-call config.** The orchestrator resolves `global ⊕
  per-env` per sample and ships the result; the trainer runs the core **once, vectorized**, over the
  mixed batch. Structural variation (a genuinely different core) = a separate term selected via
  `enabled_losses`, not a per-env override — config repeats only when the *math* differs.

---

## 9. Applications (each is a composition, not a feature)

- **RL** — the default single `rl` term.
- **Echo** — the `echo` recipe (`rl ⊕ ce-on-roles`), §5.
- **SFT** — a `ce` core with an `sft` advantage (constant on sampled tokens); strength via λ. (The
  `sft` *training_mode* data path is separate.)
- **OPD** — today a hardcoded `opd_loss_fn`; the target is a composable term (`ce`/`dppo_kl` core +
  `ref_kl` advantage reading a scorer, §6).
- **Pedagogical RL** 🟡 (`https://noahziems.com/pedagogical-rl`) — a self-teacher is trained to produce
  trajectories that are correct **and** easy for a frozen student to imitate; the student then
  assimilates them. Two **separate runs** (freeze one, train the other); the teacher↔student
  alternation/curriculum lives *above* the loss layer (multi-run orchestration) and is **out of scope**:

  | Run | Trains | Frozen | What it needs |
  |---|---|---|---|
  | **Phase 1** (teacher) | the teacher | the student | RL on `r_ped = R · G_spike^θS`; `G_spike` is an **advantage *strategy*** that pulls the lazy reference handle (§6) to shape the reward *before* GRPO centering — orchestrator-side, since centering is group-relative (so not a post-loss hook) |
  | **Phase 2** (student) | the student | the teacher | OPD imitation + a **surprisal gate** `wₜ = σ(κ(logπθS − γ))` and a `1/Σwₜ` **custom reduce** |

  The Phase-2 gate depends on the **student being trained** → live `trainer_logprobs` → a **hook** (the
  one genuine live-hook use case). **Snapshot escape hatch:** if a per-iteration snapshot is acceptable,
  the orchestrator scores the current student and ships `wₜ` as a per-token advantage → no hook needed.
  Whether to build the live gate vs the snapshot path is an open decision (§12).

---

## 10. Relationship to `feat/algorithm-abstraction`

This branch anchors against the competing `feat/algorithm-abstraction`. We **borrow its genuinely
better ideas** rather than reinvent them, and keep what our shape uniquely gives:

| Borrow from them | Keep / from us |
|---|---|
| Algorithm-local inline **external references** (→ generalized to the scorer registry, §6) | **N-term composition**: λ-summed, independently normalized |
| **Trainer-blind** per-token wire discipline | **Per-token custom advantages** (`fn(group) -> list[list[float]]`) |
| **Bundle-preset** ergonomics | **Arbitrary echo**: any role + `tool_names` + custom filters |
| `ref_kl` / `sft_distill` / `self_distill` as *future presets* (expressible as terms) | **Emergent `0`=mask filtering** (no KL leak) + per-term λ/reduce |
| | **Uniform component-pointer surface** (custom loss == custom advantage) |

We deliberately do **not** chase preset parity — their `grpo`/`opd`/`sft_distill`/`self_distill`/`echo`
are all expressible as terms here and can ship as presets later (⚪ backlog).

---

## 11. Status — done / TODO

### ✅ Built (each chunk CI-green; GRPO bit-identical where noted)

- Composable per-term `losses` (core + advantage_fn); per-env `enabled_losses` + `loss_overrides`.
- The advantage axis (`grpo` / `echo` / `sft` / `custom`); one parameterizable pg core under
  `dppo_kl` / `ce`.
- Per-term **λ** (`lambda_weight`) + pluggable **reduce** (`mean` / `custom`).
- **Emergent zero-advantage dismissal** (`0` = mask; no-gradient samples dropped).
- **Reference scorer split** in config: `orchestrator.teacher` (SFT generator) vs
  `orchestrator.reference` (scorer) + a `logprobs.top_k` field (`top_k>1` not yet consumed — §6).
- **Reference feed rename** `teacher_logprobs → reference_logprobs` end-to-end; the OPD core consumes
  the flat (`top_k = 1`) reference feed via `LossInputs`.
- **Hooks** end-to-end (seam → `LossTerm.hooks` config → built-ins `min_prob_filter` + `surprisal_gate`).
- **Compound recipes + `echo`** (`rl ⊕ ce-on-roles`; knob routing; λ owns magnitude — `alpha` dropped;
  provenance-aware duplicate errors).

### 🟡 TODO (rough order)

1. **The reorder — lazy reference handle + OPD unification** (§6, *designed/decided*) — build the
   `ReferenceHandle` (memoized, cache-on-sample); route OPD's prefill through it (byte-identical, no more
   bespoke finalize path); let orchestrator-side advantage strategies / advantage_fns pull it
   (pre-/post-centering by who pulls); add the pedagogical Phase-1 `G_spike` advantage strategy. **← next.**
2. **Top-k distillation core** (§6) — `reference_topk_*` on the wire + a core that gathers the trainer's
   logprobs at the reference's top-k ids (coupled — ship together). `top_k = 1` already works.
3. **Filter niceties** (§3.8) — an orchestrator-side sampling-prob filter (the live-prob one is done via
   `min_prob_filter`); wants the filter/advantage-composition surface, so it couples with #5.
4. **Pedagogical RL** (§9) — `surprisal_gate` is built (Phase-2 live gate); `G_spike` lands with #1;
   remaining is the curriculum (backlog).
5. **Surface polish** toward the full pointer / cascading-preset model **and the scorer-registry config
   surface** (§6) — proposed here, **deferred in code until the team aligns** (typed configs + the
   `custom` escape hatch are ~equivalent today; global-vs-registry-vs-inline is the open call).

### ⚪ Backlog (not now)

- Preset parity with the algorithm-abstraction zoo (`opd` / `sft_distill` / `self_distill` as presets).
- The teacher↔student curriculum orchestration (above the loss layer).
- `training_mode` removal (folds into the env-sampler refactor — the data axis).

---

## 12. Open questions (to settle with the team)

- **Scorer config surface (§6).** Global field (today) vs the named-scorer **registry** (proposed
  end-game) vs pure inline-on-component. The lazy `ReferenceHandle` *mechanism* is decided and
  generalizes (each scorer is its own handle); only the *config surface* is open. Lean: build on the
  handle now, keep the flat `reference_logprobs` wire + the global `orchestrator.reference` config until
  a second scorer actually lands, and settle the surface then.
- **Pedagogical Phase-2 gate (§9):** the live-hook path is built (`surprisal_gate`); the per-iteration
  snapshot path (→ custom advantage + custom reduce) stays an alternative. Open: which to use in
  practice, and whether the exact `1/Σ wₜ` normalization warrants a built-in custom reduce.
- **Pointer-surface refactor now, or propose-and-keep typed configs?** Lean: propose here, don't
  refactor pre-discussion (it's the exact thing to align on).
- **Preset namespacing** if component / full / recipe names ever collide beyond what config position
  disambiguates.
- **OPD top-k distillation form:** forward-KL over the reference's top-k vs the current sampled-token
  NLL (couples with the trainer-gather note in §6).
