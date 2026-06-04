# Composable per-term losses — implementation plan

Branch: `sebastian/losses-2026-06-04` (from `main` @ `63af82343`).
Supersedes the bespoke per-role echo work on `feat/per-role-echo` (#2677): echo becomes
a *preset*, not a feature.

## 1. Goal

Decouple three things that are currently entangled in the RL loss:

1. **the objective** ("loss" / core) — DPPO+KL for RL, masked-NLL for SFT/echo, custom;
2. **token selection** ("filters") — which tokens a term trains on;
3. **per-token weight** — GRPO advantage for RL, a constant `alpha` for echo, custom.

Then let the trainer apply a **list** of such terms over one shared forward, summing them
into a single backward. Echo stops being special: it is `{loss: sft, filters: [role], weight: constant}`.

**Hard requirement: the default is byte-for-byte `main`.** With no configuration, `losses = ["rl"]`
must reproduce today's DPPO+KL training exactly (same masks, same normalization, same gradient).

This is also preparatory for the planned env-sampler refactor: a sample carries which loss
term(s) to apply, exactly as it carries `training_mode` today.

## 2. Why this is sound (and the two corrections that shaped it)

- **One shared forward, one summed backward.** Every term differentiates the *same* per-token
  `trainer_logprobs` produced by the single forward. `∇(rl + echo) = ∇rl + ∇echo`, so summing
  the per-term scalars and calling `backward()` once is correct and cheapest. Separate
  backward passes are **not** used: the forward (the expensive, shared part) cannot be freed
  between them without `retain_graph=True` (which frees nothing), so they would double the
  backward cost for an identical gradient. The only thing that would justify separate backwards
  is gradient surgery (PCGrad) — out of scope.
- **RL is not "cross-entropy + an advantage step."** The DPPO+KL core is a function of the
  *importance ratio* `exp(trainer_lp − inference_lp)` plus a squared-KL term, and it owns its
  advantage multiplication and its trust-region mask internally. So the term abstraction is
  **not** a two-stage `loss ∘ post-loss` pipeline; it is "a core fn that receives `(logprobs,
  mask, weight, scale)` and decides how to combine them." Echo/SFT (masked NLL) *is* cleanly
  `weight · (−logprob)` masked, but RL is not, and we do not force it to be.

### Validity constraints (these gate which combos make sense)

- `core=rl` needs **sampled completion tokens with real `inference_logprobs`** and GRPO
  advantages. Prompt/context tokens have `inference_logprobs = 0.0` → RL on them is garbage.
  ⇒ the `rl` preset's filter is `completion`, and this pairing is validated.
- `core=opd` needs `teacher_logprobs` (a configured teacher).
- `core=sft` (masked NLL) works on **any** token — which is exactly why echo can use it on
  context tokens. This is the permissive core.

## 3. The execution seam (what runs where)

A term spans both processes, along the seam echo already uses today:

| Slot      | Runs on      | Input it needs                              | Output |
|-----------|--------------|---------------------------------------------|--------|
| `filters` | orchestrator | rollout + `prompt_attribution` (roles/tools)| per-token bool mask (AND-composed) |
| `weight`  | orchestrator | rollout (rewards → GRPO advantage; roles)   | per-token float coefficient |
| `loss`    | trainer      | `trainer_logprobs` from the forward         | per-token objective → scalar |

- `prompt_attribution` (`message_roles`, `message_indices`, `is_content`, `message_tool_names`)
  is emitted by the **verifiers renderer** (`deps/verifiers` @ `05c66c235`, same pin as main) on
  each trajectory step's `tokens` dict. The trainer never sees it — hence filters/weights are
  orchestrator-side.
- GRPO advantage is group-relative (needs the whole rollout group) → intrinsically
  orchestrator-side. It is already computed there and shipped per-token (`advantage`).
- The two sides are tied by the term **name**. The orchestrator ships each enabled term's
  per-token `(mask, weight)`; the trainer looks the core up by name and applies it.

The **DPPO trust-region mask** (`probs_diff` thresholds) is *internal* to the `rl` core
(it zeroes the per-token loss for violators) — it is **not** a user-facing filter.

## 4. Pointers + presets (resolution model)

Every slot value is **either a built-in key or a dotted import path**, plus a `kwargs` dict —
resolved exactly like the existing convention (`import_object(path)` then
`functools.partial(fn, **kwargs)`, as in `orchestrator/envs.py` and `orchestrator/advantage.py`).

Built-in registry (resolve by key to in-repo functions):

- **cores**: `rl` → `default_loss_fn` (DPPO+KL, today's `main`), `sft` → masked NLL,
  (`opd` stays a separate per-sample path — see §8).
- **filters**: `completion` (the sampled-completion / `loss_mask` tokens),
  `role` (kwargs: `roles: list[str]`, `tools: set[str] | None`).
- **weights**: `grpo` (the shipped per-token advantage, optionally `× adv_tau`),
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
- **Semantic** (presets only): e.g. `rl` filter must be `completion`-like; reject obviously
  broken pairings. Custom pointers are **caller-beware** — we expose every option and validate
  structure, but do not police the semantics of user functions.

## 5. Config surface

Default — nothing to write (`losses` defaults to the `rl` preset):

```python
losses = ["rl"]          # == main, exactly
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
# orchestrator / env config
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

Common surface stays tiny: `["rl"]` or a one-line preset. The three-slot machinery only
appears when deliberately going custom.

## 6. Overlap semantics

- Terms may overlap; **gradients sum**. There is no automatic cross-term exclusion.
- "Echo replaces RL on the completion" (today's behavior) is expressed by **omitting `rl`**:
  `enabled_losses = ["echo"]`. Deactivating RL is trivial.
- A token trained by two terms simply receives both gradients and counts toward both terms'
  denominators (§9). This is intended ("do both and add the losses").

## 7. Per-env kwarg overrides

Global term defaults, overridable per-env. Merge is per-slot shallow: term's global `kwargs`
provide defaults, the per-env override patches individual keys. **The cost depends on which
side the slot runs on:**

- **Orchestrator-side kwargs (filters, weight — incl. `alpha`, and `adv_tau`): free.**
  The orchestrator already resolves per-env config and bakes the *result* into the shipped
  per-token `(mask, weight)`. `adv_tau` folds into the advantage value (a scalar multiply)
  before shipping. This is exactly how per-env `alpha` already works for echo today.
- **Trainer-side core kwargs that touch trainer-only quantities (`kl_tau`,
  `dppo_mask_low/high`): require a per-sample field.** They scale quantities that only exist
  after the forward (`(trainer_lp − inference_lp)²`, `exp(trainer_lp) − exp(inference_lp)`),
  so they cannot be baked orchestrator-side. To make them per-env, the orchestrator resolves
  `global ⊕ per-env` and **stamps the resolved core kwargs onto the sample**; the trainer reads
  them per sample (cheap via msgspec `omit_defaults` — nothing ships when unset).

Rule of thumb: *a core kwarg is per-env-free if it folds into a shipped per-token quantity
(`adv_tau` → advantage); otherwise it ships per sample.* We implement the per-sample
resolved-kwargs field (it is the env-sampler-aligned shape), defaulting to the global value.

## 8. Relationship to `training_mode` (sft/opd entrypoints)

`training_mode` (`sft`/`opd`/`rl`, stamped per sample by the orchestrator) is a **separate
axis** — it routes mixed datasets (SFT demos vs RL rollouts) to a path. We **keep it as-is**.
The `losses` list governs composition *within the `rl` path*. `opd` and `sft` per-sample modes
stay their own functions. (The `sft` *core* in a loss term — masked NLL — is the objective
form; it is not the same thing as a sample being `training_mode="sft"`.)

## 9. Normalization

Generalize the single global `loss_scale` to **one scale per term** = the global (dp_cp)
count of that term's mask tokens (all-reduced, `max(·, 1)`), so each term is a true per-token
mean over the global batch and terms do not dilute each other. The FSDP per-rank
`fsdp_gradient_divide_factor` undo after the micro-batch loop is unchanged.

(`main` today: `compute_loss(..., loss_scale)` then `scaled_loss = total_loss / loss_scale`.
New: each term divides by its own scale inside `compute_loss`, then terms sum. With
`losses = ["rl"]`, the rl scale equals today's `loss_scale` ⇒ identical result.)

## 10. Wire format (transport/types.py)

Generalize echo's single `echo_alpha` to per-term data:

- `rl` term reuses existing fields (`loss_mask`, `advantage`, `inference_logprobs`) — no new
  per-token data for the default path.
- Each **additional** enabled term ships per-token `(mask, weight)`. Concretely a
  `term_weights: dict[str, list[float | None]]` keyed by term name (`None` = ineligible),
  which is the natural generalization of `echo_alpha` (echo was exactly "the echo term's
  per-token weight, `None` where ineligible"). The boolean mask is `weight is not None`.
- Optional `term_loss_kwargs: dict[str, dict]` for per-sample resolved core kwargs (§7),
  omitted when equal to the global default.
- `enabled_losses` need not ship if derivable from term presence; ship the names for clarity.

All `omit_defaults` (msgspec) so the default path adds nothing to the wire.

## 11. File-by-file changes (against `main`)

### Config (`packages/prime-rl-configs/src/prime_rl/configs/`)

- **`trainer.py`** — add the term registry. New `LossTermConfig` (`name`, `loss`, `filters`,
  `weight`), discriminated sub-configs for each slot (`RLCore`/`SFTCore`/`CustomCore`;
  `CompletionFilter`/`RoleFilter`/`CustomFilter`; `GRPOWeight`/`ConstantWeight`/`CustomWeight`).
  Add `losses: list[LossTermConfig | str]` to `TrainerConfig` (default `["rl"]`); the `str`
  form is a preset name. Keep `DefaultLossConfig`/`CustomLossConfig` as the `rl`/`custom`
  *core* configs (rename/move under the term, preserving fields `kl_tau`, `dppo_mask_*`,
  `adv_tau`, `import_path`, `kwargs`). Preset-expansion + validators here.
- **`orchestrator.py`** — per-env `enabled_losses: list[str]` + `loss_overrides: dict[str, dict]`.
  **Remove `EchoConfig`/`RoleEchoConfig`/`EchoFilterConfig`** (echo is now a preset). The
  per-env filter/weight definitions either live here or are read from the shared `losses`
  list (see §12).
- **`rl.py`** (`RLConfig`) — wire the shared `losses` definition so both `trainer` and
  `orchestrator` sub-configs can resolve their slots (see §12 for placement decision).

### Transport (`src/prime_rl/transport/types.py`)

- Add `term_weights` (+ optional `term_loss_kwargs`, `enabled_losses`) to `TrainingSample` and
  the per-token `term_weights` to `MicroBatch`; all `omit_defaults`. Remove `echo_alpha`.

### Orchestrator (`src/prime_rl/orchestrator/`)

- **`losses.py`** (new; generalizes the echo-branch `echo.py`) — bind each enabled term's
  filters + weight per env (`import_object` + `functools.partial`), run them per rollout to
  produce per-token `(mask, weight)`; AND-compose filters; validate filter return shape (port
  `apply_echo_filter`'s checks: `list[list[bool]]`, outer == #steps, inner == prompt+completion,
  plain bools). Built-in `completion`/`role` filters and `grpo`/`constant` weights live here.
- **`envs.py`** — replace the single `echo_filter_fn` binding with per-term bound
  filters/weights from `enabled_losses` (+ `loss_overrides`).
- **`train_sink.py`** — replace the `build_echo_annotations(...)` call with the per-term
  builder; stamp `term_weights` (+ resolved `term_loss_kwargs`) onto each `TrainingSample`.
- **`trajectories.py`** — generalize the `echo_alpha` extension logic to extend per-term
  weights across multi-step trajectories.
- **`advantage.py`** — unchanged mechanism; the `grpo` weight reads its output. `adv_tau`
  baking (§7) applied here or at stamp time.

### Trainer (`src/prime_rl/trainer/`)

- **`rl/loss.py`** — `default_loss_fn` stays **exactly `main`** (DPPO+KL over its `loss_mask`,
  knows nothing about terms). Add `sft`/echo core = masked NLL `−(weight · logprob)[mask].sum()/scale`.
  Add a built-in **core registry** `{"rl": default_loss_fn, "sft": ...}` + custom import.
  Rework `compute_loss` to iterate enabled terms, apply each core over its `(mask, weight,
  scale, kwargs)`, **sum**, and aggregate per-term metrics. Keep `LossInputs`/`LossOutputs`.
- **`batch.py`** — `prepare_sample`: build per-term per-token `(mask, weight)` from the shipped
  `term_weights` (generalizing the current single echo path). Propagate through
  `packed_samples_into_micro_bs`, `pad_micro_batch`, `_make_dummy_batch` (parallel lists).
- **`rl/packer.py`** — generalize the `echo_alpha` length assertions to per-term weights.
- **`rl/train.py`** — compute **per-term scales** (generalize the single all-reduced
  `loss_scale`); move per-term `term_weights` to CUDA + `.split(response_lengths)`; pass to
  `compute_loss`; keep the **single** `loss.backward()` and the FSDP-undo loop unchanged.
- **`rl/token_export.py`** — export per-term masks/weights instead of the single echo mask.

## 12. Open decision: where the term *definition* physically lives

The core runs trainer-side; filters/weights run orchestrator-side. Two ways to avoid drift:

- **(A) Single shared `losses` list at `RLConfig` level**, passed to both processes; each reads
  the slots it executes. Cleanest single-source-of-truth. Requires both entrypoints to accept
  the shared section. **(recommended)**
- **(B) Cores in `TrainerConfig.losses`, filters/weights + per-env selection in
  `OrchestratorConfig`, tied by name.** Mirrors today (echo config is orchestrator-side, CE
  core is trainer-side) but two places to keep in sync.

Recommendation: **(A)** — define once, select per-env by name. Confirm with the launcher how
`RLConfig` distributes sub-config to the two processes before committing to it.

## 13. Testing (conservative, per AGENTS.md)

- **Golden: `losses=["rl"]` is bit-identical to `main`** on a fixed input — the contract for
  "default unchanged." (Primary regression guard.)
- Term composition sums correctly; overlapping masks double-train; per-term scales correct.
- Filter AND-composition; `completion`/`role` built-ins select the right tokens.
- Only pure-logic units (loss cores, mask/weight builders, preset expansion, validators).
  No new framework-glue tests.

## 14. Suggested PR phases (keep default green throughout)

1. **Term abstraction + registry + `compute_loss` over a list**, with `losses=["rl"]` only.
   No behavior change; the golden test guards it. (loss.py, train.py scales, configs.)
2. **`sft` core + `echo` preset** (role filter + constant weight) — feature parity with the
   echo branch, now as a preset. (orchestrator `losses.py`, wire format, batch.py.)
3. **Per-env `enabled_losses` + overrides** (incl. per-sample resolved core kwargs).
4. **Custom pointers + validation polish.**

Draft PR. Reference `feat/per-role-echo` (#2677) for the orchestrator-side filter mechanics,
per-token weight plumbing, and tests as a porting source.

## 15. Loose ends / decisions for Sebastian

- Confirm **§12** placement (shared `RLConfig` list vs trainer+orchestrator tied by name).
- Confirm we ship **per-sample resolved core kwargs** in phase 3 (vs deferring per-env `kl_tau`).
- Fate of **#2677** (`feat/per-role-echo`): close in favor of this, or land losses then migrate
  its configs. (The custom-echo-loss experiment is stashed on that branch: `stash@{1}`.)
- `adv_tau` placement: baked into advantage orchestrator-side (per-env-free) vs applied in the
  `rl` core. Plan assumes baked.
