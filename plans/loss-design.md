# Composable loss framework — converged design

Status: **design**, anchoring the discussion against `feat/algorithm-abstraction`. Builds on this
branch (composable per-term losses) and borrows the genuinely-better ideas from that branch. This is
the authoritative shape; `plans/losses.md` is the journey + the per-stage detail, and
`plans/pedagogical-rl.md` is one application of it.

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

## 4. Cascading presets (two layers)

- **Component presets**: `advantage="grpo"`, `loss="dppo_kl"`, `reduce="mean"` → resolve to in-repo
  paths + default kwargs.
- **Full presets**: `rl`/`echo`/`opd`/… → a *bundle* of component-preset choices → which resolve to
  paths. `rl` → `{ loss="dppo_kl", advantage="grpo", reduce="mean", λ=1.0 }`.

Resolution cascades full → components → paths, with user overrides **deep-merged at any layer**.
Guardrails: disambiguate layers **by config position** (a full-preset `"rl"` sits in the term's
`type`; a component preset sits in the `advantage`/`loss` slot — same string, unambiguous by slot);
**exactly two layers**; resolved configs are **dumpable** so a user sees what `"rl"` expanded to.

```toml
[[losses]]
type = "rl"                       # full preset

[[losses]]
name      = "echo"                # explicit composition
loss      = { type = "ce" }
advantage = { type = "echo", roles = ["user", "tool"], tool_names = ["python"] }
lambda    = 0.3
```

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

## 7. Hooks — the new build (stage 3)

The one capability neither branch has today, and the highest-leverage addition.

- **Contract:** `hook(loss_per_token, ctx) -> loss_per_token`, chainable, **no scalar return**
  (reduction is the separate stage 4, so masking hooks compose). `ctx` exposes all trainer-side data:
  the live `trainer_logprobs`, the importance ratio, current-policy entropy, the per-token loss so far.
- **Use cases:** live-policy-prob / entropy-gated masking (§6), the pedagogical surprisal gate
  (`plans/pedagogical-rl.md`), label smoothing, per-token penalties.
- **Principle:** intrinsic objective math (DPPO trust-region clip + KL) stays **inside the core**;
  hooks are cross-cutting transforms layered on top.
- **Cost:** requires inverting the current core↔reduce boundary — cores must return a **per-token
  tensor** (today they return a per-sample scalar) and the masking/sum move into `reduce`. That's the
  real refactor in this PR.

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

## 10. What this PR ships (build order)

Most of the composability is already on this branch (terms, λ/reduce, per-token advantages, role/tool
echo, emergent dismissal). The deltas:

1. **This doc** (the shape, for discussion).
2. **Hooks** (§7) — the headline capability; includes the core→per-token + separate-reduce refactor.
3. **References** (§5) — adopt the inline external model, drop chunk-1's global field/pool, add `top_k`.
4. **Filter niceties** — custom token-filter pointers + the sampling-prob / live-prob filters as
   built-ins (the latter via a hook).
5. **Surface polish** toward the full component-pointer + cascading-preset model — **proposed here,
   deferred in code** until the team aligns on the shape (today's typed configs + `custom` escape
   hatch are ~equivalent in capability).

Not in scope: preset parity with their algorithm zoo; the teacher↔student curriculum orchestration.

## 11. Open questions (to discuss)

- **Pointer-surface refactor now, or propose-and-keep typed configs?** Lean: propose here, don't
  refactor pre-discussion (it's the exact thing to align on).
- **Pedagogical gate: per-iteration snapshot vs live hook** (`plans/pedagogical-rl.md`).
- **Preset namespacing** if component/full names collide beyond what config position disambiguates.
- How far to push the wire toward fully type-driven routing (we already unified to `term_advantages`).
