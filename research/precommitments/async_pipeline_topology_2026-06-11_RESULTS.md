# Topology experiment results — 2026-06-11 (locked prereg: research/precommitments/async_pipeline_topology_2026-06-11.md)

## Verdicts (locked formulas)
| Metric | E0 (3t12i lag3 p1424) | E2 (4t11i lag3 p1280) | Baseline (lag-1) |
|---|---|---|---|
| cadence (t12-t2)/10 | 330.3s (ambiguous) | 242.2s CONFIRM | 358-444s |
| drought_count | 1 (step 8: 1525s) FALSIFY | 0 | >=1/cycle |
| inf_idle_frac | 0.330 FALSIFY | 0.000 | 0.41-0.46 |
| inf_util | 0.415 | 0.757 CONFIRM | 0.35-0.38 |
| trainer busy med | 273.9s | 184.1s CONFIRM | 238-265s |

H1 FALSIFIED (eval-displacement drought + idle>=30% at 3t). H2 CONFIRMED (1.49x busy speedup on
1.33x nodes — superlinear). H3 CONFIRMED (242s, util .76, zero droughts/cancellations/idle).

## Step walls (3-12)
E0: 355 134 143 121 240 [1525] 134 (97) 218 338   ((..)=eval-trigger steps 5,10 excluded from droughts)
E2: 267 140 (285) 261 256 175 301 (180) 288 269

## Trainer busy (3-12)
E0: 297 234 274 232 340 226 227 323 312  | E2: 155 178 187 184 175 173 293 184 186

## Mechanism
Gate oscillation: eliminated in BOTH conditions (1 warmup pause each; off-policy rode 2-8, 0 cancels).
Residual pathology: eval-wave displacement (440 rollouts, PREFER_EVAL), phase/depth-dependent —
droughts the pipeline only when shallow (E0 step 8); fully absorbed at E2's depth/trainer pace.
Registered in-run prediction "second drought at E0 steps 11-12" FAILED (wave absorbed by primed
pipeline) — first evidence depth absorbs eval waves.
v9 permits-axis aux check: cycle ~5-6 steps pre-death, qualitatively consistent, low confidence.

## Deployment recommendation (per locked tree)
target_lag=3 + 4-train-node shape (4t12i on 16 healthy nodes) for future runs, CONTINGENT on the
pre-registered quality follow-up: GT-acc delta >= -0.05, mismatch-KL <= 2x lag-1, cancelled <= 5%,
vs the queued lag-1 simul-sbatch run at matched steps. Criteria frozen pre-result.
Claims scoped: 2-turn simultaneous debate protocol, eval interval 5/440 rollouts, GH200 15-16 nodes,
early-training length regime. Lag-3 is a stopgap vs streaming approaches (PipelineRL/AReaL).

## Bonus findings
- In-alloc 449G cgroup ceiling killed E0 post-window (trainer-master OOM, same as v6-v8) — sbatch
  --mem=0 template is the fix for all future runs; in-alloc smokes must fit ~85min on the master.
- E2's trainer superlinearity: 4-node FSDP relieves the per-rank memory pressure that throttled 3-node.
