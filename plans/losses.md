# Composable per-term losses — implementation plan

Branch: `sebastian/losses-2026-06-04` (from `main`).

Today the RL loss bakes the objective, the token selection, and the per-token weighting into
one function. This plan pulls them apart and lets the trainer apply a **list of loss terms**
over a single shared forward.

> **Echo** — per-role, per-token cross-entropy supervision over *context* tokens (system / user /
> tool / assistant turns), as opposed to the usual policy-gradient signal on the sampled
> completion — is **one preset** in this framework, not a bespoke feature. It expands to
> `{loss: sft, filters: [role], weight: constant}`.

## 1. Goal

Decouple three concerns that are entangled in today's RL loss:

1. **the objective** ("loss" / core) — DPPO+KL for RL, masked-NLL for SFT/echo, or custom;
2. **token selection** ("filters") — which tokens a term trains on;
3. **per-token weight** — the GRPO advantage for RL, a constant `alpha` for echo, or custom.

A term is `{name, loss, filters, weight}`. The trainer applies a **list** of terms over one
shared forward and sums them into a single backward.

**Hard requirement: the default is byte-for-byte today's training.** With no configuration,
`losses = ["rl"]` must reproduce the current DPPO+KL loss exactly — same masks, same
normalization, same gradient.

This is also preparatory for the planned env-sampler refactor: a sample will carry which loss
term(s) to apply, the same way it carries `training_mode` today.

## 2. Why this is sound (and two things it deliberately is *not*)

- **One shared forward, one summed backward.** Every term differentiates the *same* per-token
  `trainer_logprobs` from the single forward. `∇(rl + echo) = ∇rl + ∇echo`, so summing the
  per-term scalars and calling `backward()` once is correct and cheapest. We do **not** use
  separate backward passes: the forward (the expensive, shared part) cannot be freed between
  them without `retain_graph=True` (which frees nothing), so separate backwards would double
  the backward cost for an identical gradient. (Gradient surgery / PCGrad would be the only
  reason to split — out of scope.)
- **The term abstraction is not a two-stage `loss ∘ post-loss` pipeline.** The DPPO+KL core is
  a function of the *importance ratio* `exp(trainer_lp − inference_lp)` plus a squared-KL term,
  and it owns its advantage multiplication and its trust-region mask internally — it is not
  "cross-entropy times an advantage." So a core is "a function that receives `(logprobs, mask,
  weight, scale)` and decides how to combine them," not a post-processor of per-token losses.
  Masked-NLL (sft/echo) *is* cleanly `weight · (−logprob)` masked; RL is not, and we don't
  force it to be.

### Validity constraints (these gate which combinations make sense)

- `loss=rl` needs **sampled completion tokens with real `inference_logprobs`** and GRPO
  advantages. Context tokens have `inference_logprobs = 0.0`, so RL on them is meaningless.
  ⇒ the `rl` preset's filter is `completion`, and this pairing is validated.
- `loss=sft` (masked NLL) works on **any** token — which is exactly why echo can use it on
  context tokens. This is the permissive core.
- (`opd` needs `teacher_logprobs`; it stays a separate per-sample path for now — see §8.)

## 3. The execution seam (what runs where)

A term spans both processes, along a natural seam: the **orchestrator** owns the rendered
rollout, the **trainer** owns the forward.

| Slot      | Runs on      | Input it needs                               | Output |
|-----------|--------------|----------------------------------------------|--------|
| `filters` | orchestrator | rollout + `prompt_attribution` (roles/tools) | per-token bool mask (AND-composed) |
| `weight`  | orchestrator | rollout (rewards → GRPO advantage; roles)    | per-token float coefficient |
| `loss`    | trainer      | `trainer_logprobs` from the forward          | per-token objective → scalar |

- `prompt_attribution` (`message_roles`, `message_indices`, `is_content`, `message_tool_names`)
  is emitted by the **verifiers renderer** (the pinned `deps/verifiers`) on each trajectory
  step's `tokens` dict. The trainer never sees it — hence filters/weights are orchestrator-side.
- GRPO advantage is group-relative (needs the whole rollout group) → intrinsically
  orchestrator-side. It is already computed there and shipped per-token.
- The two sides are tied by the term **name**. The orchestrator ships each enabled term's
  per-token `(mask, weight)`; the trainer looks the core up by name and applies it.

The **DPPO trust-region mask** (`probs_diff` thresholds) is *internal* to the `rl` core — it
zeroes the per-token loss for violators and is **not** a user-facing filter.

## 4. Pointers + presets (resolution model)

Every slot value is **either a built-in key or a dotted import path**, plus a `kwargs` dict,
resolved with the existing convention: `import_object(path)` then
`functools.partial(fn, **kwargs)` (see `orchestrator/advantage.py` for custom advantages and
`trainer/rl/loss.py`'s `setup_loss_fns` for custom losses).

Built-in registry (resolve by key to in-repo functions):

- **cores**: `rl` → the current DPPO+KL loss fn, `sft` → masked NLL.
- **filters**: `completion` (the sampled-completion / `loss_mask` tokens),
  `role` (kwargs: `roles: list[str]`, `tools: set[str] | None`).
- **weights**: `grpo` (the shipped per-token advantage, `× adv_tau`),
  `constant` (kwargs: `value: float` — echo's `alpha`).

**Presets** are sugar that expand to a full validated triple:

```
"rl"   → { name: "rl",
           loss:    { type: "rl", kl_tau: 1e-3, dppo_mask_low: 0.2, dppo_mask_high: 0.2 },
           filters: [ { type: "completion" } ],
           weight:  { type: "grpo", adv_tau: 1.0 } }

"echo" → { name: "echo",
           loss:    { type: "sft" },
           filters: [ { type: "role", roles: [...], tools: [...] } ],   # + optional user filter, AND-composed
           weight:  { type: "constant", value: <alpha> } }

"sft"  → { name: "sft",
           loss:    { type: "sft" },
           filters: [ { type: "completion" } ],
           weight:  { type: "constant", value: 1.0 } }
```

Validation has two layers:

- **Structural** (always): types, required fields, unique term names, `enabled_losses ⊆`
  defined term names, filters non-empty.
- **Semantic** (presets only): e.g. the `rl` preset's filter must be `completion`-like; reject
  obviously broken pairings. Custom pointers are **caller-beware** — every option is exposed and
  the config is structurally validated, but the semantics of user functions are the user's
  responsibility.

## 5. Config surface

One shared `losses` list defines the terms; per-env config selects from it by name. The common
surface stays tiny — the three-slot machinery only appears when deliberately going custom.

Default — nothing to write (`losses` defaults to the `rl` preset):

```python
losses = ["rl"]          # == today's training, exactly
```

Echo via preset:

```python
losses = [
  "rl",
  { preset: "echo", roles: ["system", "tool"], tools: ["calculator"], alpha: 0.005 },
]
```

Per-env: select by name, with optional kwarg overrides (see §7):

```python
environments:
  - { id: "math" }                                   # uses default enabled_losses = ["rl"]
  - { id: "tool-traces", enabled_losses: ["rl", "echo"],
      loss_overrides: { echo: { alpha: 0.01 } } }
```

Fully custom (escape hatch, structural validation only):

```python
losses = [
  "rl",
  { name: "myterm",
    loss:    { type: "custom", import_path: "pkg.core",   kwargs: {...} },   # trainer-side
    filters: [ { type: "role", roles: ["assistant"] },
               { import_path: "pkg.filter", kwargs: {...} } ],               # orchestrator, AND
    weight:  { type: "custom", import_path: "pkg.weight", kwargs: {...} } }, # orchestrator
]
```

## 6. Overlap semantics

- Terms may overlap; **gradients sum**. There is no automatic cross-term exclusion.
- Training echo *instead of* RL on the completion is expressed by **omitting `rl`**:
  `enabled_losses = ["echo"]`. Deactivating RL is trivial.
- A token trained by two terms receives both gradients and counts toward both terms'
  denominators (§9). This is intended ("do both and add the losses").

## 7. Per-env kwarg overrides

Global term defaults, overridable per-env. Merge is per-slot shallow: the term's global
`kwargs` provide defaults, the per-env override patches individual keys. **The cost depends on
which side the slot runs on:**

- **Orchestrator-side kwargs (filters, weight — incl. `alpha`, and `adv_tau`): free.**
  The orchestrator resolves per-env config and bakes the *result* into the shipped per-token
  `(mask, weight)`. `adv_tau` folds into the advantage value (a scalar multiply) before
  shipping — **decided: bake it in orchestrator-side**, so per-env `adv_tau` is free and the
  `rl` core just consumes the pre-scaled advantage.
- **Trainer-side core kwargs that touch trainer-only quantities (`kl_tau`,
  `dppo_mask_low/high`): require a per-sample field.** They scale quantities that only exist
  after the forward (`(trainer_lp − inference_lp)²`, `exp(trainer_lp) − exp(inference_lp)`), so
  they cannot be baked orchestrator-side. **Decided: ship resolved core kwargs per sample** —
  the orchestrator resolves `global ⊕ per-env` and stamps them onto the sample; the trainer
  reads them per sample (cheap via msgspec `omit_defaults`, nothing ships when unset). This is
  the env-sampler-aligned shape.

Rule of thumb: *a core kwarg is per-env-free if it folds into a shipped per-token quantity
(`adv_tau` → advantage); otherwise it ships per sample.*

## 8. Relationship to `training_mode` (sft/opd paths)

`training_mode` (`sft`/`opd`/`rl`, stamped per sample by the orchestrator) is a **separate
axis** — it routes by data source/path. We **keep it as-is**. The `losses` list governs
composition *within the `rl` path*; the `opd` and `sft` per-sample paths stay their own
functions for now.

Note the naming: the `sft` *core* in a loss term (masked NLL — the objective form) is **not**
the same thing as a sample whose `training_mode="sft"` (a data path). The framework adds the
`sft` core; the `sft` path's data-sourcing is unchanged.

## 9. Normalization

Generalize the single global `loss_scale` to **one scale per term** = the global (dp_cp) count
of that term's mask tokens (all-reduced, `max(·, 1)`), so each term is a true per-token mean
over the global batch and terms do not dilute each other. The FSDP per-rank
`fsdp_gradient_divide_factor` undo after the micro-batch loop is unchanged.

(Today: `compute_loss(..., loss_scale)` then `scaled_loss = total_loss / loss_scale`. New: each
term divides by its own scale inside `compute_loss`, then terms sum. With `losses = ["rl"]`, the
rl scale equals today's `loss_scale` ⇒ identical result.)

## 10. Wire format (`transport/types.py`)

- The `rl` term reuses existing fields (`loss_mask`, `advantage`, `inference_logprobs`) — the
  default path adds **no** new per-token data.
- Each **additional** enabled term ships per-token `(mask, weight)`, carried as a
  `term_weights: dict[str, list[float | None]]` keyed by term name (`None` = ineligible). The
  boolean mask is `weight is not None`. Add this to `TrainingSample` and `MicroBatch`.
- Optional `term_loss_kwargs: dict[str, dict]` for per-sample resolved core kwargs (§7),
  omitted when equal to the global default.
- Optionally ship `enabled_losses` (the names) for clarity.

All fields use msgspec `omit_defaults`, so the default path adds nothing to the wire.

## 11. File-by-file changes

### Config (`packages/prime-rl-configs/src/prime_rl/configs/`)

- **`trainer.py`** — add the term registry: `LossTermConfig` (`name`, `loss`, `filters`,
  `weight`) with discriminated sub-configs per slot (`RLCore`/`SFTCore`/`CustomCore`;
  `CompletionFilter`/`RoleFilter`/`CustomFilter`; `GRPOWeight`/`ConstantWeight`/`CustomWeight`).
  Add `losses: list[LossTermConfig | str]` (default `["rl"]`; the `str` form is a preset name).
  The existing `DefaultLossConfig`/`CustomLossConfig` already carry the `rl`/`custom` core
  fields (`kl_tau`, `dppo_mask_*`, `adv_tau`, `import_path`, `kwargs`) — reuse them as the `rl`
  and `custom` core configs. Preset-expansion + validators live here.
- **`orchestrator.py`** — add per-env `enabled_losses: list[str]` + `loss_overrides: dict[str, dict]`.
- **`rl.py` (`RLConfig`)** — host the single shared `losses` definition (see §12) and distribute
  it to both the trainer and orchestrator sub-configs.

### Transport (`src/prime_rl/transport/types.py`)

- Add `term_weights` (+ optional `term_loss_kwargs`, `enabled_losses`) to `TrainingSample`, and
  the per-token `term_weights` to `MicroBatch`; all `omit_defaults`.

### Orchestrator (`src/prime_rl/orchestrator/`)

- **`losses.py`** (new module) — bind each enabled term's filters + weight per env
  (`import_object` + `functools.partial`), run them per rollout to produce per-token
  `(mask, weight)`, AND-compose filters, and **validate the filter return shape**: a filter must
  return `list[list[bool]]` with the outer length equal to the number of trajectory steps, each
  inner length equal to that step's `prompt_ids + completion_ids`, and every element a plain
  `bool`. Built-in `completion`/`role` filters and `grpo`/`constant` weights live here.
- **`envs.py`** — bind each enabled term's filters/weights for the env (from `enabled_losses`
  + `loss_overrides`).
- **`train_sink.py`** — run the per-term builder per rollout and stamp `term_weights`
  (+ resolved `term_loss_kwargs`) onto each `TrainingSample`.
- **`trajectories.py`** — extend per-term weights across multi-step trajectories (alongside the
  existing per-token field extension).
- **`advantage.py`** — unchanged mechanism; the `grpo` weight reads its output, with `adv_tau`
  baked in at stamp time.

### Trainer (`src/prime_rl/trainer/`)

- **`rl/loss.py`** — `default_loss_fn` stays **exactly as it is today** (DPPO+KL over its
  `loss_mask`, unaware of terms). Add an `sft`/echo core = masked NLL
  `−(weight · logprob)[mask].sum() / scale`. Add a built-in **core registry**
  `{"rl": default_loss_fn, "sft": ...}` plus custom import. Rework `compute_loss` to iterate
  enabled terms, apply each core over its `(mask, weight, scale, kwargs)`, **sum**, and
  aggregate per-term metrics. Keep `LossInputs`/`LossOutputs`.
- **`batch.py`** — `prepare_sample`: build per-term per-token `(mask, weight)` from the shipped
  `term_weights`. Propagate through `packed_samples_into_micro_bs`, `pad_micro_batch`,
  `_make_dummy_batch` (parallel lists).
- **`rl/packer.py`** — add per-term weight length assertions (parallel to the existing
  per-token field checks).
- **`rl/train.py`** — compute **per-term scales** (generalize the single all-reduced
  `loss_scale`); move per-term `term_weights` to CUDA + `.split(response_lengths)`; pass to
  `compute_loss`; keep the **single** `loss.backward()` and the FSDP-undo loop unchanged.
- **`rl/token_export.py`** — export per-term masks/weights.

## 12. Config placement (decided)

The core runs trainer-side; filters/weights run orchestrator-side. To keep a single source of
truth and make the user-facing surface easy, **define the `losses` list once at the `RLConfig`
level** and distribute it to both processes; each reads the slots it executes, and per-env
config selects terms by name. Implementation check: confirm how `RLConfig` hands sub-config to
the trainer and orchestrator processes, and thread the shared `losses` section through both.

## 13. Testing (conservative)

- **Golden: `losses=["rl"]` is bit-identical to today's loss** on a fixed input — the contract
  for "default unchanged." (Primary regression guard.)
- Term composition sums correctly; overlapping masks double-train; per-term scales correct.
- Filter AND-composition; `completion`/`role` built-ins select the right tokens.
- Only pure-logic units (loss cores, mask/weight builders, preset expansion, validators). No new
  framework-glue tests.

## 14. Suggested PR phases (keep the default green throughout)

1. **Term abstraction + core registry + `compute_loss` over a list**, with `losses=["rl"]` only.
   No behavior change; the golden test guards it.
2. **`sft` core + `echo` preset** (role filter + constant weight) — reaches the echo objective
   as a preset (orchestrator `losses.py`, wire format, `batch.py`).
3. **Per-env `enabled_losses` + overrides**, including the per-sample resolved core kwargs.
4. **Custom pointers + validation polish.**

Open as a draft PR.

## 15. Settled decisions

- **Single shared `losses` list at `RLConfig` level** (§12); per-env selects by name. *Impl
  check remaining: how `RLConfig` distributes sub-config to the two processes.*
- **Ship per-sample resolved core kwargs** for trainer-side knobs (`kl_tau`, DPPO thresholds);
  do it in phase 3 (§7).
- **`adv_tau` is baked into the advantage orchestrator-side** (§7), so it's per-env-free.
