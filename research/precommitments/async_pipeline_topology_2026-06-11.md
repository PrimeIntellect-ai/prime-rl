# Precommitment: async-RL pipeline smoothing & topology balance (GH200 16-node)
Date: 2026-06-11
Status: LOCKED (2026-06-11 ~19:55 UTC, Joan)

## Research Question
Is the dispatch gate (TARGET_LAG=1), not capacity, the cause of the train/inference oscillation in prime-rl's async debate training — and does the 4-train-node shape deliver its predicted trainer speedup?

## Background & Motivation
Today's balance analysis (tmp/infbench-20260611/balance/summary.md) found a ~7-step boom-bust cycle: the lag-1 gate floods rollout permits, the ~800 s episode latency returns them as a wave, the orchestrator ships 5–7 batches fast, the gate closes, the inference fleet idles (<5% of capacity for 11–25 min), then the trainer stalls ~800 s when the buffer dries. Inference idle totals 41–46% of wall time against 2.1× capacity. The mechanism is code-confirmed (orchestrator.py:986–998 — bang-bang gate, no proportional control).

Prior art pre-answers the qualitative question: PipelineRL (arXiv:2509.19128) and AReaL (arXiv:2505.24298) document this flood/famine pathology under step-synchronized gating and remove it with deeper/streaming pipelines; INTELLECT-2 ran two-step async in this codebase's own lineage. Static deeper lag is the weakest fix in that literature (in-flight updates and interruptible rollouts dominate). This experiment is therefore **local calibration**: one observation of the never-seen smooth regime on this stack, this protocol, this cluster — before deploying a lag change to real runs — plus a measured 4-node trainer point. Deployment-grade quality evidence is explicitly out of scope and pre-registered as a follow-up (see Quality Follow-up).

## Conditions

| ID | Shape | target_lag | max_inflight | Nodes | Notes |
|----|-------|-----------|--------------|-------|-------|
| B (context) | 3t13i / 3t12i | 1 | 1424 | 16/15 | historical (a1154/a1345/a1551); crashed survivors, pre-target_lag-commit, daytime. Context only, NOT a primary comparator. |
| E0 | 3t13i | 3 | 1424 | 16 | single-knob gate treatment (permits held at B's value) |
| E2 | 4t11i | 3 | 1280 | 15 | deployment candidate; permits capacity-bounded (11×128=1408; 1280 leaves queue headroom — declared deviation, rationale: permits>capacity inflates TTFT, a confound) |

E1 (4t11i/lag-1) is **dropped** per review: H2 (trainer scaling) is measured directly from E2's trainer busy time, which is gate-independent. Run order: E0 → E2, fixed. max_steps=14 per condition (step 12 must exist in logs; the drain consumes the final step).

Registered auxiliary observation (zero cost): v9/a1810 (lag-1, permits 1024, running now) — **prediction: cycle shortens to ~5 steps with the same ~800 s stall.** Its post-mortem data tests the permits axis of the period arithmetic.

v9 disposition (pre-declared): v9 is killed when E0 is ready to launch; it has already served its purposes (leak attribution, eval replication). Expected host-OOM ~20:25 may resolve this naturally.

## Hypotheses

Pinned baseline values (computed from balance CSVs with the formulas below): cadence 358 / 444 / 428 s; trainer busy median 238 / 262 / 265 s; inference idle-fraction 41–46%; ≥1 drought per 7 steps.

### H1: gate-smoothing (E0)
- **Mechanism claim:** the oscillation disappears when the pipeline is ≥3 batches deep.
- **Confirm:** cadence ≤ 310 s AND drought_count = 0 AND inf_idle_frac < 15%.
- **Falsify:** drought_count ≥ 1 OR inf_idle_frac ≥ 30%.
- **Ambiguous:** otherwise (e.g., cadence 310–360 with no droughts → partial smoothing; report as such).
- Prediction: cadence ≈ 280–300 s (T_train ≈ 261 s + eval amortization; Little floor L/lag = 800/3 ≈ 267 s — thresholds assume L does NOT shrink; if L shrinks under smooth dispatch, results land below prediction, which only strengthens confirmation).

### H2: 4-node trainer scaling (measured inside E2)
- **Confirm:** median trainer busy ≤ 210 s (≥0.85 scaling efficiency from the 238–265 s 3-node band).
- **Falsify:** median trainer busy ≥ 230 s.
- Time-based on purpose: token-count keys are unreliable (perf/throughput logged NaN all day); busy time at fixed bs512 is the unambiguous quantity.

### H3: combined operating point (E2)
- **Confirm:** cadence ≤ 300 s AND inf_util ≥ 55%.
- **Falsify:** cadence ≥ 360 s.
- Prediction: ≈ 280–300 s — bounded below by max(T_train ≈ 196–210 s, L/3 ≈ 267 s) + eval share. The Little floor, not trainer time, is binding at this shape; pushing below ~270 s would require lag ≥ 4 or shorter episodes (out of scope, recorded for follow-up).

Null model (observable as written): E0 cadence ≥ 360 s with drought_count ≥ 1 and idle ≥ 30% — the oscillation persists despite lag-3 → gate-depth is not the cause; attention moves to episode-latency variance and PREFER_EVAL serialization.

## Metrics (all formulas pinned; two analysts must get identical numbers)

| Metric | Formula | Role |
|---|---|---|
| cadence | (t_ship(12) − t_ship(2)) / 10, t_ship(N) = wandb `_timestamp` of orchestrator step-N row (eval and drought steps INCLUDED — they are the phenomenon and the deployment cost) | **Primary (H1, H3)** |
| drought_count | #{steps 3–12, excluding eval-trigger steps 5,10: orchestrator `time/step` ≥ 600 s} | **Primary (H1)** |
| inf_idle_frac | fraction of `inference/agg` samples with running/(128·R) < 0.05, sampled over [t_ship(2), t_ship(12)); R = replica count; known 10 s tick cadence; NaN ticks dropped | **Primary (H1)** |
| inf_util | mean running/(128·R), same window | **Co-primary (H3)** |
| trainer_busy | median over trainer steps 3–12 of (`time/step` − `time/wait_for_batch`) | **Primary (H2)** |
| mean_wall, p90_wall | mean and p90 of orchestrator `time/step`, steps 3–12 all | Secondary (full distribution reported as a table — N≤10 numbers per condition, no bootstrap) |
| cancelled_frac | dispatcher cancelled ÷ dispatched, train env, full condition | Secondary (staleness price) |
| off_policy_dist | per-step shipped Max Off-Policy: median, max | Secondary (recorded) |
| mismatch_kl | trainer per-step | Secondary (recorded) |
| L_proxy | t_first_batch_ship − t_pool_ready (startup); drought durations (steady) | Secondary (floor accounting) |
| judge_scoring_p95 | env scoring-latency p95 per condition | Sensitivity flag: if cross-condition ratio > 1.5×, the affected comparison gets a stated confidence downgrade — never experiment-invalid |

Eval-shadow sensitivity (pre-declared, secondary): recompute cadence excluding steps whose ship time falls within 1100 s after an eval trigger; report alongside primary. The v1 claim that eval interference is "identical across conditions" was wrong (the shadow covers more steps at faster paces) and is withdrawn; H3's thresholds knowingly include eval coexistence — that IS the deployment quantity.

## Design & Stopping
- 14 max_steps; analysis window steps 3–12; fresh policy per condition (no carry-over).
- Per-condition wall cap **150 min** (BOTEC below includes the ~28 min step-0 warmup the v1 cap forgot). A cap-truncated condition with ≥6 analyzable non-eval steps is **VALID** and analyzed on its truncated window (a slow condition is a result, not an anomaly); INVALID is reserved for crashes/startup failures.
- ≤2 startup attempts per condition; if E0 cannot start twice, the experiment is abandoned (report as infrastructure failure).
- Hard end: 60 min before allocation expiry (04:07 UTC) regardless of state.
- No interim metric influences continuation (conditions run to their cap/completion regardless of early numbers). E2 runs whether E0 confirms or falsifies (it tests a different decision).

## Confound Controls

| Confound | Risk | Control | Residual |
|---|---|---|---|
| Historical baseline (time-of-day, code drift, crashed survivors) | High | B demoted to context; H1/H3 decided against pinned absolute thresholds + within-experiment structure (droughts, idle), not against B | Threshold placement inherits B's measurement era |
| Two-knob bundle | — | **Eliminated**: permits held at 1424 (E0); E2's 1280 is a declared capacity bound | E2 permits differ from E0 (1280 vs 1424); both ≥ 2.5 batches, above the lag-3 pipeline requirement |
| Eval-wave × pace interaction | Med | Eval included in primary cadence; pre-declared shadow-excluded sensitivity metric | Eval cost share genuinely differs by pace — part of the measured quantity, stated in claims |
| Age-drop selection (lag-3 keeps stragglers B would drop) | Med | cancelled_frac recorded per condition; direction noted (lengthens treated batches — conservative for H1) | Uncorrected in cadence |
| Decode-length drift (+25%/10 steps) | Med | Identical step window; fresh policy per condition | Smoke samples the short-completion regime; long-run balance shifts inference-ward (stated in claims) |
| Diurnal API/Lustre drift | Med | judge_scoring_p95 flag at 1.5×; conditions back-to-back | Sub-1.5× drift can move cadence ~30–40 s; acknowledged |
| Trainer 4-node interconnect placement | Low-Med | Same train-node set within E2; busy-time metric independent of gate | Single placement sampled |
| Crash hazard | High | Today's serial killers fixed (timeouts, retry semantics, table schema, --mem irrelevant in-alloc at 12–14 steps); window shortened to ≥6-step validity; v9 surviving past step 15 would further confirm | Residual per-condition crash risk est. ~20–30% |
| v9 interference | — | Hard gate: killed before E0 launch | None |

## Compute Budget (revised per accountant)
- E0: ~10 min startup + ~28 min warmup/step-0 + ~11 steps incl. 2 eval waves ≈ **120–140 min**. E2 same ≈ 130 min + 10 min re-carve. Total ≈ **4.5–5.5 h** incl. one retry allowance, against ~7.5 h post-v9-kill window. Headroom ~30%.
- External: judge ≈ $8–12 (verified $0.10/M prompt), graders ≈ $4. Ceiling $100.
- GPU·h: ~300–390, sunk allocation.

## Success Criteria
- **Positive:** H1 confirmed and H3 confirmed → deploy target_lag=3 for future *throughput* purposes at 4t11i/4t12i, conditional on the registered quality follow-up.
- **Negative:** H1 falsified (droughts persist at lag-3) → gate-depth rejected as the primary cause; investigate episode-latency variance and PREFER_EVAL serialization next.
- **Ambiguous:** partial smoothing (cadence improves, droughts nonzero) → lag-3 insufficient depth; test lag ≥ 4 or permits interaction next.
- **Invalid:** a condition crashes twice before 6 analyzable steps.
Claims are scoped to: this debate protocol (2-turn simultaneous), this eval cadence (interval 5, 440 rollouts), 15–16 GH200 nodes, early-training length regime. 4t12i (model-predicted optimum) remains untested. Lag-3 is a stopgap relative to streaming/in-flight approaches in the literature.

## Quality Follow-up (pre-registered now, decided later)
Adopting lag-3 for real runs additionally requires, on the first lag-3 full run vs the queued lag-1 simul-sbatch run (same config otherwise): (a) single-agent GT-acc curve non-inferior (Δ ≥ −0.05 at matched eval steps 10–50); (b) median mismatch_kl ≤ 2× the lag-1 run at matched steps; (c) cancelled_frac ≤ 5%. These criteria are fixed today, before any throughput result exists.

## Analysis Plan
Descriptive. Per condition: the full step table (≤10 walls), cadence, drought_count, idle/util, trainer_busy table, staleness/cancellation columns. Verdicts strictly by the pre-specified thresholds. Plots: (1) running/capacity timeline per condition (with pre-registered scalar: max idle-spell duration); (2) step-wall strip plot with thresholds; (3) trainer busy per step; (4) off-policy & mismatch-KL traces. v9 post-mortem cycle check reported alongside.

## Pre-registered Amendments
- v2.1 (2026-06-11, at lock, pre-data): E0 runs 3t12i — 15 nodes, nid010193 benched (its documented wedge record: v5 12-min load, v7 EngineDeadError; ~30% failure odds over a 2.3h condition). R=12 in idle/util formulas; the 12-replica fleet is inside the baseline band (a1551). E2 unchanged (already 11 replicas without nid010193).
- v2 (2026-06-11, pre-lock): full redesign after adversarial review — primary metric median→cadence (median auto-confirmed under bimodality: baseline medians 158–161 s vs thresholds 240–290 s); E1 dropped (H2 from E2 busy time); permits unbundled from lag; max_steps 12→14 (step-12 logging); cap 110→150 min with ≥6-step validity (null was censoring to INVALID); v9 disposition declared; judge-p95 clause single-consequence; eval-identical claim withdrawn; Little-floor stated; quality follow-up criteria registered; prior-art scoping added.
