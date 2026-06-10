# Composable loss framework — converged design

Status: **design + partially implemented** on `sebastian/losses-2026-06-04`, anchoring the discussion
against `feat/algorithm-abstraction`. Builds on this branch (composable per-term losses) and borrows
the genuinely-better ideas from that branch. This is the authoritative shape; `plans/losses.md` is the
journey + per-stage detail, and `plans/pedagogical-rl.md` is one application. **§10 lists what's built;
§12 lists what's still TODO.**

## Thesis — win on shape, not feature count

A training run is a **list of composable loss terms**, summed over one shared forward. **Every
component of a term — advantage, loss core, filters, reduce, hooks — is the same kind of thing: a
pointer (a preset name *or* a dotted path) plus kwargs.** A term spans the orchestrator→trainer seam,
but the user writes one uniform surface; each process picks up the components it runs.

That shape gives the things `feat/algorithm-abstraction` structurally can't express — N-term
composition with per-term λ and independent normalization, per-token everything, arbitrary token
filtering, and a *uniform* custom surface (no "custom loss is configured differently from custom
advantage"). We deliberately **borrow that branch's better ideas** (algorithm-local external model
references, a trainer-blind per-token wire, and bundle-preset ergonomics) rather than reinvent them.
We do **not** chase preset parity — its `grpo`/`opd`/`sft_distill`/`self_distill`/`echo` are all
expressible as terms here and can ship as presets later; that's backlog, not the point.

## 1. The core model — a list of terms

`losses` is a list; per-env `enabled_losses` selects which apply; per-env `loss_overrides` tweaks them.
Terms sum into one backward; each term has its **own λ** (pre-reduce) and its **own normalization**
(`reduce`), so terms neither dilute nor mask each other. Default `losses` reproduces today's DPPO+KL
**bit-for-bit**. (All of this already exists on this branch.)

## 2. The component-pointer model (the key unification)

Every component is the same shape — a built-in preset name that resolves to an in-repo function path,
or any dotted `import_path`, plus kwargs:

```toml
advantage = { type = "grpo", tau = 0.5 }                    # built-in preset
advantage = { import_path = "pkg.my_adv", kwargs = { .. } } # custom — identical shape
loss      = { type = "dppo_kl", kl_tau = 1e-3 }
reduce    = { type = "mean" }
hooks     = [ { type = "entropy_gate", threshold = 2.0 }, { import_path = "pkg.my_hook" } ]
filters   = [ { type = "role", roles = ["assistant"] }, { import_path = "pkg.low_prob_filter" } ]
```

This is what kills the `feat/algorithm-abstraction` asymmetry where a custom *advantage* is an
algorithm-local pointer but a custom *loss* is a separate global `[trainer.loss]` that can only
replace the `rl` type. Here a custom loss is declared exactly like a custom advantage, per term.

**Validation:** a component whose target declares a kwargs schema gets full pydantic validation
(every built-in does; a custom path may); otherwise only existence is checked. Existence/import is
verified **in the process that runs the component, at setup** — never eager cross-process import at
parse time (a trainer-side core isn't importable in the orchestrator that validates the shared config).

## 3. The per-term pipeline (five stages, each a pointer)

```
advantage_fn  →  core           →  hooks            →  reduce        →  λ·sum
(orchestrator)   (trainer)         (trainer)           (per term)       (combine)
per-token float  per-token loss    per-token→per-token  →scalar
0 = masked       NOT reduced       sees the live fwd
```

- **advantage_fn** (orchestrator): one float per token, `0` = masked. Group-aware. Sees rewards +
  attribution + (when configured) a reference's shipped logprobs.
- **core** (trainer): `core(LossInputs) -> per-token loss tensor` (not reduced). One parameterizable
  policy-gradient core underlies `dppo_kl`/`ce`; `custom` is a path.
- **hooks** (trainer): `hook(loss_per_token, ctx) -> loss_per_token`, chainable — §7.
- **reduce** (per term): per-token → scalar (`mean` = global per-token mean = today; or custom).
- **λ·sum**: `total = Σ_terms λ_t · scalar_t`.

## 4. Presets: component, full, and compound recipes

Resolution cascades by **config position**, deep-merged at any layer, with **dumpable resolved configs**
so a user sees what a name expanded to:

- **Component presets** — `advantage="grpo"`, `loss="dppo_kl"`, `reduce="mean"` → an in-repo path + default kwargs (sit in the `advantage`/`loss`/`reduce` slot).
- **Full presets (one term)** — `rl` → `{ loss="dppo_kl", advantage="grpo", reduce="mean", λ=1.0 }` (sit in the term's `type`).
- **Compound recipes (a *list* of terms)** — a preset that expands to several terms. This is the
  headline demonstration of the framework: **`echo` is not a primitive, it's `rl ⊕ ce-on-roles`** — the
  policy-gradient objective on the completion plus cross-entropy supervision on chosen role tokens,
  stitched from two primitives with the right mask + advantage.

### `echo` as a recipe

`{ type = "echo", ... }` expands to **two terms** — the standard `rl` term and a `ce` overlay whose
advantage selects the echoed role tokens:

```toml
# common case — one line; no separate rl term to write or collide with:
[[losses]]
type  = "echo"
roles = ["user", "tool"]            # -> the echo (ce) sub-term's advantage (a 0/1 selection mask)
# tool_names / by_advantage / tau also route to the echo sub-term;
# lambda_weight sets the echo strength (the ce sub-term's λ)

# tune the policy-gradient half *through* echo (no separate term):
[[losses]]
type  = "echo"
roles = ["assistant"]
rl    = { loss = { kl_tau = 5e-4 } }   # deep-merges into the rl sub-term
```

The recipe routes its own knobs (`roles`/`tool_names`/`by_advantage`/`tau`/`lambda_weight`) to the
**ce** sub-term, and any named sub-term block (`rl = {...}`) to that sub-term — so you never need a
separate `rl` term, and the common case is one line.

**Magnitude is λ, not a second alpha.** The echo strength is the ce term's `lambda_weight`; its
advantage is a pure selection mask (`1.0` on matched tokens, `× advantage·tau` when `by_advantage`).
`EchoAdvantageConfig.alpha` is dropped — λ owns magnitude, the advantage owns selection/shape,
per-token weights stay reachable via the advantage.

**Names stay unique — duplicate names raise.** A name is the trainer's core-registry key and what
`enabled_losses`/`loss_overrides` reference, so a clash must fail, not silently merge. Recipe expansion
**tags each emitted term with its source preset**, so a collision with a recipe's sub-term gives a
pointed error, e.g. *"Duplicate loss term name 'rl': emitted by the 'echo' compound preset and also
defined explicitly — tune echo's rl via the echo config instead."*

## 5. References — borrowed from `feat/algorithm-abstraction`, improved

Adopt their model and **drop our chunk-1 global `orchestrator.reference` + managed pool**:

- A reference is an **inline `FrozenModelConfig`** (`ClientConfig` + `name`), declared **on the
  component that scores against it** (e.g. an `opd`/`ref_kl` advantage's `model`), not a global field.
- **External-only**: `base_url` required; prime-rl never hosts, launches, or weight-syncs frozen
  models — only the trainable policy. No prefix-cache salting, no off-policy aging.
- **Trainer-blind**: the orchestrator prefills the reference and ships per-token signal on the wire;
  the trainer just consumes it. "Roles" (teacher / reference / student) are **component-local labels**,
  not global vocabulary.
- **Plus top-k** (the thing their branch lacks): `model.logprobs.top_k` ships the reference's top-k
  (ids, logprobs) per token for richer distillation; `top_k=1` is the sampled-token estimate (≈ their
  current `ref_kl`). Additive — it does not conflict with the trainer-blind design.

```toml
[[losses]]
type = "opd"
advantage = { type = "ref_kl",
              model = { name = "Qwen/Qwen3-32B", base_url = ["http://host:8001/v1"],
                        logprobs = { top_k = 8 } } }
```

## 6. Filtering — emergent + arbitrary

- **`0`-advantage is a true mask** (emergent dismissal, already built): a token with `0` leaves both
  numerator and denominator; a rollout with no nonzero term is dropped and the sampler backfills. No
  KL leak (unlike `feat/algorithm-abstraction`, where `advantage=0` zeroes PG but the KL term still
  pulls the token — it can only filter via coarse action/observation routing or whole-rollout filters).
- **Orchestrator-side filters** live in the advantage_fn / `filters` pointers — anything computable
  from the rollout (role, tool, sampling-logprob threshold, custom predicate).
- **Trainer-side filters** live in **hooks** — anything needing the live forward (current-policy
  prob threshold, entropy gate). This is why §7 is required for "drop tokens the policy is already
  confident about."
- Per-rollout filters (`zero_advantage`, gibberish, repetition) stay as-is.

## 7. Hooks — stage 3 ✅ built

The capability neither branch had; now implemented on this branch (seam → config → first built-in).

- **Contract:** `hook(per_token_loss, inputs) -> per_token_loss`, chainable, **no scalar return**
  (reduction is the separate stage 4, so masking hooks compose). `inputs` is the `LossInputs` — the
  live `trainer_logprobs` / `inference_logprobs` / `advantages` / `loss_mask`.
- **Use cases:** live-policy-prob / entropy-gated masking (§6), the pedagogical surprisal gate
  (`plans/pedagogical-rl.md`), smoothing, penalties.
- **Principle:** intrinsic objective math (DPPO clip + KL) stays **inside the core**; hooks are
  cross-cutting transforms layered on top.
- **How it stayed bit-identical:** rather than invert the core↔reduce boundary, cores *additionally*
  return `per_token_loss` (the pre-sum tensor) while the scalar `loss` is byte-for-byte unchanged. The
  no-hook path consumes the scalar (bit-identical — golden tests pass untouched); the hook chain runs
  on `per_token_loss` and is summed only when hooks are present.
- **Built in:** `min_prob_filter` (zero the loss where the current-policy logprob < threshold — a
  trainer-side filter that needs the live forward); `HookConfig` is a discriminated union
  (`custom` + built-in presets).

## 8. The execution seam (uniform surface, split execution)

Advantage is computed orchestrator-side (it needs the group, rewards, and reference logprobs);
core+hooks+reduce run trainer-side (they need the forward). That split is forced. What is **not**
forced — and what `feat/algorithm-abstraction` got wrong — is leaking the seam into the config
(advantage configured under `orchestrator.*`, loss under `trainer.*`). Here the term **co-locates all
components**; the wire carries per-token advantages + per-term routing; each process reads its slice.
The trainer stays a core-executor, exactly as in their design — we just don't fragment the surface.

## 9. Borrow / keep (vs `feat/algorithm-abstraction`)

| Borrow from them | Keep / from us |
|---|---|
| Algorithm-local inline **external references** (drop our global field + managed pool) | **N-term composition**: λ-summed, independently normalized |
| **Trainer-blind** per-token wire discipline | **Per-token custom advantages** (`fn(group)->list[list[float]]`) |
| **Full-preset bundles** as top-layer ergonomics | **Arbitrary echo**: any role + `tool_names` + custom filters |
| `ref_kl`/`sft_distill`/`self_distill` as *future presets* (expressible as terms) | **Emergent `0`=mask filtering** (no KL leak) + per-term λ/reduce |
| | **Uniform component-pointer surface** (custom loss == custom advantage) |

## 10. Status — what's built

On `sebastian/losses-2026-06-04`, each chunk CI-green (GRPO bit-identical where noted):

- Composable per-term `losses` (loss core + advantage_fn), per-env `enabled_losses` + `loss_overrides`.
- The advantage axis (`grpo`/`echo`/`sft`/`custom`); one parameterizable pg core under `dppo_kl`/`ce`.
- Per-term **λ** (`lambda_weight`) + pluggable **reduce** (`mean`/`custom`).
- **Emergent zero-advantage dismissal** (the old ①/⑤): `0`=mask; no-gradient samples dropped.
- **Reference scorer split** (chunk 1): `orchestrator.teacher` (sft generator) vs `orchestrator.reference`
  (scorer) + a `logprobs.top_k` config field — *not yet consumed* (see §12).
- **Hooks** end-to-end (§7): seam → config (`LossTerm.hooks`) → first built-in (`min_prob_filter`).
- This design doc + `plans/pedagogical-rl.md`.

## 11. Open questions (to discuss)

- **Pointer-surface refactor now, or propose-and-keep typed configs?** Lean: propose here, don't
  refactor pre-discussion (it's the exact thing to align on).
- **Pedagogical gate: per-iteration snapshot vs live hook** (`plans/pedagogical-rl.md`).
- **Preset namespacing** if component/full names collide beyond what config position disambiguates.
- How far to push the wire toward fully type-driven routing (we already unified to `term_advantages`).

## 12. Still TODO (rough order)

1. **Compound recipes + `echo`** (§4) — list-level, provenance-tagged preset expansion; `echo` =
   `rl ⊕ ce-on-roles` with knob routing; drop `EchoAdvantageConfig.alpha` (λ owns magnitude); generic
   duplicate-name error with the compound-aware message. **← next chunk.**
2. **Reference logprobs feed + top-k** (§5) — reconcile chunk-1's reference with the algorithm-local
   inline external model; have the prefill compute top-k, ship it on the wire, expose it to
   `RenderHints` (advantage_fns) + `LossInputs` (cores/hooks); a `ref_kl` advantage/core consumes it.
3. **Filter niceties** (§6) — the orchestrator-side sampling-prob filter as a built-in (the live-prob
   one is done via `min_prob_filter`).
4. **Surface polish** toward the full pointer/cascading-preset model — proposed here, *deferred in
   code* until the team aligns (typed configs + the `custom` escape hatch are ~equivalent today).
5. **Pedagogical RL** (`plans/pedagogical-rl.md`) — the application that motivated hooks + the
   reference feed; needs #2 and, for the live gate, a `surprisal_gate` hook.

**Backlog / not now:** preset parity with the algorithm-abstraction zoo (`opd`/`sft_distill`/
`self_distill` as our presets — easily expressible, not a priority); the teacher↔student curriculum
orchestration (above the loss layer).
