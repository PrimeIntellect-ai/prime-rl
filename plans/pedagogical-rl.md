# Pedagogical RL + the unified reference-logprobs mechanism — design note

Status: **design, not implemented.** Extends the advantage-function pipeline in `plans/losses.md`
(advantage_fn → core → hooks → reduce → λ-sum). No code yet.

## Goal

Support **Pedagogical RL** (https://noahziems.com/pedagogical-rl). A *self-teacher* (sees the gold
answer) is trained to produce trajectories that are both correct **and easy for a frozen student to
imitate**; the student then assimilates them. Two rewards/losses:

- **Teacher (RL):** $r_{\mathrm{ped}} = R(x,c,\tau)\cdot G_{\mathrm{spike}}^{\theta_S}(\tau\mid x)$ — task
  reward × a learnability penalty scored under the **frozen student**, where
  $d_t=\log\frac{\pi_{\theta_S}(a_t^{\max})}{\pi_{\theta_S}(\tau_t)}$ (surprise of the teacher's token to the student).
- **Student (assimilation):** $\mathcal{L}_{\mathrm{assim}}=\frac{1}{\sum_t w_t}\sum_t w_t\,\ell_t$ with
  $w_t=\sigma(\kappa(\log\pi_{\theta_S}(\tau_t)-\gamma))$ — an OPD/imitation loss gated by the student's
  own surprisal.

## Decomposition: two separate runs (freeze one, train the other)

| Run | Trains | Frozen | Loss |
|---|---|---|---|
| **Phase 1** | the pedagogical teacher | the student | RL (GRPO) on $r_{\mathrm{ped}}$ |
| **Phase 2** | the student | the teacher | OPD imitation + surprisal gate |

The teacher↔student **alternation** (curriculum) lives *above* the loss layer (multi-run
orchestration) and is **out of scope** here — this note covers what each single run needs.

## The keystone: a unified reference-model logprobs feed

Both phases need **another (frozen) model's per-token logprobs over the trajectory, computed before
the loss**. That already exists in single-model form — OPD's `teacher` + `compute_teacher_logprobs`
prefill + the `teacher_logprobs` wire field. We **generalize that one mechanism** (and make it serve
OPD, pedagogical RL, and any future distillation) rather than adding a parallel surface.

### Config interface

Rename `orchestrator.teacher` → **`orchestrator.reference`** (the model we score trajectories under;
**no alias** — clean rename, parallels the `student` field). Hang a `logprobs` spec off it:

```toml
[orchestrator.reference]          # OPD teacher / pedagogical frozen student / any scored-against model
name     = "Qwen/Qwen3-32B"       # RolloutModelConfig (model + client) — unchanged
logprobs = { top_k = 8 }          # NEW: top-k (token_id, logprob) per position; default 1
```

- `logprobs.top_k` is the only new knob. It lives on the **reference model** (a scoring/prefill setting
  computed once); consumers read ≤ what's shipped, so set it to the max anyone needs.
- `top_k = 1` ≈ today (the argmax) + the always-shipped **sampled-token** logprob. `top_k = 8` ships the
  reference's top-8 per position → richer OPD distillation.
- The block is extensible — the existing reference scoring temperature (today's `teacher_tau`) folds in
  next to `top_k`.

### Decouple the reference from `training_mode` (validator change)

Today (`orchestrator.py:835-838`): rl-mode **forbids** a teacher; opd/sft **require** one. Pedagogical
Phase 1 is rl-mode but needs the reference. New rule:

- a `reference` is **allowed in any `training_mode`**;
- **required** for `training_mode ∈ {opd, sft}`;
- optional in `rl` (the user wires it when a loss term consumes reference logprobs — e.g. the
  $G_{\mathrm{spike}}$ advantage). Drop the "rl forbids teacher" check; keep the opd/sft requirement.

### Wire + exposure

`teacher_logprobs: list[float]` (sampled-token logprob) → generalize to **sampled-token logprob +
per-token top-k**, renamed to the reference:

```python
reference_logprobs:      list[float] | None        # reference's logprob of the sampled (trajectory) token (= today)
reference_topk_ids:      list[list[int]] | None     # per token: top-k token ids   (k = logprobs.top_k)
reference_topk_logprobs: list[list[float]] | None   # per token: top-k logprobs
```

Expose all of it to **every** loss-side fn: `LossInputs` (cores), `RenderHints` (advantage_fns), and a
hook's ctx — not just the opd core. `top_k = 1` collapses the wire back to ~today. Prefill
(`compute_teacher_logprobs`) gains a top-k path.

## Consumers — three seams, one feed (none of these is "the same" as another)

| Use | Reads reference logprobs at | Seam |
|---|---|---|
| **OPD** (Phase 2 base) | the **core** (top-k distillation target) | a *core* |
| **Phase 1** $G_{\mathrm{spike}}$ | a **custom advantage_fn** (frozen student's argmax + sampled-token logprob → $d_t$ → reward shaping) | an *advantage* |
| **Phase 2 gate** $w_t$ | the **live forward** of the model being trained | a *hook* (or snapshot → advantage) |

Notes:
- $G_{\mathrm{spike}}$ **cannot** be a post-loss hook: it's inside the reward *before* GRPO centering
  (group-relative), which is orchestrator-side. It's a custom Layer-1 advantage_fn. It needs the
  reference's **argmax** logprob (covered by `top_k ≥ 1`) plus the sampled-token logprob.
- The Phase-2 gate $w_t$ depends on the **student being trained** → live `trainer_logprobs` → a **hook**
  (the one genuine hook use case). The $1/\sum_t w_t$ normalization is a **custom `reduce`** (the per-term
  `reduce` axis already shipped).
- **Snapshot escape hatch:** the paper recomputes $w_t$ "under the updated student" *between* iterations.
  If a per-iteration snapshot is acceptable, the orchestrator scores the current student (another
  reference read) and ships $w_t$ as a per-token advantage → **no hook needed at all**. Hooks are only
  required for in-step *live* gating.

## Build order

1. **Rename + config:** `teacher` → `reference`, add `logprobs = { top_k }`, loosen the
   training_mode validator (allow-any / require-opd-sft).
2. **Wire + prefill:** generalize `teacher_logprobs` → `reference_logprobs` + top-k; top-k path in
   `compute_teacher_logprobs`; expose to `LossInputs` + `RenderHints` (+ hook ctx).
3. **OPD top-k:** make the opd core consume the top-k distribution (the "make OPD work + richer" step).
4. **Pedagogical Phase 1:** a custom $G_{\mathrm{spike}}$ advantage_fn reading the reference (frozen
   student) from `RenderHints`; reference = the frozen student checkpoint, `top_k = 1`.
5. **Phase 2 gate:** either (a) snapshot → custom advantage + custom reduce (no new machinery), or
   (b) **build the hook stage** (`plans/losses.md` stage 3) for a live gate. Decide per appetite.

## Decisions still open

- **Snapshot vs live gate** (Phase 2) — determines whether the hook stage gets built at all.
- Exact `reference_logprobs` wire shape (parallel lists vs a small struct) — msgspec `omit_defaults`,
  nothing ships when `top_k = 1` and unused.
- Confirm the OPD core's top-k distillation form (forward-KL over top-k vs the current sampled-token NLL).

## Out of scope

- The teacher↔student alternation / curriculum (multi-run orchestration).
- `training_mode` removal (the plan's longer-term direction; not needed for this).
