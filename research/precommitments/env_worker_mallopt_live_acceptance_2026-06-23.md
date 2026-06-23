# Preregistration — env_worker mallopt(M_MMAP_THRESHOLD) LIVE acceptance (2026-06-23)

**Status:** locked before the next real debate run. The synthetic A/B (request+response gap-closed) PASSED — CONTROL ratchets RSS +98.7 MB/1k & arena `sys_cur` to 10.6 GB; FIX holds RSS flat (slope −26 MB/1k), `sys_cur` ~1.2 GB; throughput delta p50 +1.0% / p95 +1.4% / CPU −0.4 pts. But the harness ran load-bound on a shared head node (~1 core, ~3.4 s/rollout) — it **cannot** measure production magnitude (real workers parallelize 512 threads, real payload mix, real co-located vLLM on the same node under `orchestrator_on_inference`). This file pre-registers the LIVE pass/fail so the result can't be rationalized post-hoc.

**Patch under test:** `deps/verifiers/.../serve/server/env_worker.py:440-456`, `EnvWorker.run_worker` calls `ctypes.CDLL("libc.so.6").mallopt(-3, 1<<20)` (M_MMAP_THRESHOLD = 1 MiB) before uvloop/executor/`asyncio.run`, `rc != 1 → raise`. Orchestrator side already trims per shipped batch (`orchestrator.py:633 trim_process_memory` via `utils.py:118`).

**Baseline (the thing that OOM'd):** orch-tree `kids` term ramped **3 → 167 GB over 27 steps** (~6 GB/step monotone) → host-OOM before run finished.

---

## Primary signal — (a) RSS flatness of the orch-tree `kids` term

**Field:** gauge `mem/orch_children_gb` (`orchestrator.py:972`, key in `_MEM_GAUGE_KEYS` :133). This is recursive-children RSS = the env-worker pool = the term that ratcheted. Logged on the pipeline tick.

**How to read it from `outputs/.../logs/orchestrator.log`:** the console fragment is appended after `||` on the periodic pipeline line (`orchestrator.py:944`, format :980-985):
```
... || mem node 71% (608/856G, avail 248G) | orch-tree 174G (proc 7G + 512 kids 167G) | other 427G
```
The `kids` number (here `167G`) is `mem/orch_children_gb` rounded; `orch-tree` is `mem/orch_tree_gb` (proc+kids). Extract the series:
```
grep "orch-tree" outputs/<run>/logs/orchestrator.log \
  | sed -E 's/.*([0-9]+) kids ([0-9]+)G.*/\2/'   # children GB per tick
```
Prefer the gauge over the rounded console int when wandb/jsonl is available: `mem/orch_children_gb` (and `mem/orch_tree_gb`, `mem/node_percent`, `mem/node_available_gb`).

**PASS predicate (pre-registered):** over the first ~50 steps,
- least-squares slope of `mem/orch_children_gb` vs step **≤ 0.5 GB/step** (vs baseline ~6 GB/step; ≥12× attenuation), AND
- `mem/orch_children_gb` at step 50 **< 3×** its value at step 5 (no multiplicative ratchet), AND
- end-of-window `mem/orch_children_gb` **< 40 GB** (baseline crossed 40 G before step ~7; FIX must stay well under the OOM trajectory).

Read `mem/orch_tree_gb` and `mem/node_other_gb` jointly: a flat `kids` with rising `tree` would mean the leak migrated to the main proc (would still fail (c)); rising `node_other` is the co-located vLLM, not this patch (don't attribute).

## Throughput signal — (b) trainer step cadence within ~10%

**Field:** `time/step` (`train.py:662`), console-mirrored on the `Step N | <time> | Loss ...` success line (`train.py:619`, `logger.success`). Also `perf/throughput` (tokens/s, :633) as a cross-check.

**How to read from `outputs/.../logs/trainer.log`:**
```
grep -E "^.*Step [0-9]+ \|" outputs/<run>/logs/trainer.log   # human-readable step times
# or the time/step gauge from wandb/jsonl for exact p50/p95
```
Compute p50/p95 of `time/step` over steps 5–50 (drop steps 0–4 = warmup/compile/first weight-broadcast).

**Baseline:** p50/p95 `time/step` from the pre-fix debate run at the **same** (model, t·i topology, lag, batch, seq_len, node count). If no clean pre-fix baseline at matching config exists, the prior gemma4 50-step science runs (5175049-52) are the reference — state which, do not mix configs.

**PASS predicate:** `|p50_fix − p50_base| / p50_base ≤ 0.10` AND `|p95_fix − p95_base| / p95_base ≤ 0.10`. (Synthetic prior: +1.0% p50 / +1.4% p95 → expect well inside 10%; the gate exists to catch a *production*-specific regression the head-node-bound harness couldn't see.)

## Hard constraint — (c) no host OOM through 50 steps

Run reaches step 50 with **no** OOM-kill / SIGKILL of any env worker, orchestrator, or vLLM engine on the orchestrator node; `mem/node_percent` never hits 100 and `mem/node_available_gb` never crosses below ~20 G (the danger band on the 856 G node). A run that OOMs before step 50 is an automatic FAIL regardless of (a)/(b).

## (d) Kill / re-open rule — RSS ratchets anyway

**Abort trigger (monitor, do not auto-act — flag to Joan per CLAUDE.md "Monitoring ≠ Acting"):** if `mem/orch_children_gb` shows a **monotone rise ≥ 0.5 GB/step sustained over any 10 consecutive ticks** OR crosses **80 GB** at any point before step 50 → the synthetic harness missed a production-only allocation path. Action:
1. Snapshot: dump the full `orch-tree` series + `mem/node_*` + `perf/throughput` from the run's logs.
2. Stop the run **only after** showing Joan current step / next checkpoint / what's lost (Destructive-Operations gate).
3. **Re-open the investigation** (don't re-tune `1<<20` blindly): the likely production-only suspects are (i) a payload class above 1 MiB that *still* lands in an arena because it's allocated outside the worker's threshold scope (e.g. in vLLM/torch C-ext, not glibc malloc), (ii) main-proc leak (`mem/orch_proc_gb` rising) the trim doesn't catch, or (iii) #76's "bulk tensors off the message bus" being the real fix. Record which.

A FAIL on (a) but PASS on (c) at step 50 still re-opens — flat-enough-to-not-OOM-in-50 ≠ flat; longer runs would OOM.

---

## Does the CPU-validate throughput delta predict production latency risk?

**No measurable risk, with one honest caveat.**

- The harness delta is **+1.0% p50 / +1.4% p95 latency, −1.1% throughput, −0.4 pts CPU** — the cost of routing >1 MiB allocs through `mmap`/`munmap` (extra syscalls + page-fault-on-touch + VMA churn). That is the *mechanism's* intrinsic cost and it transfers to production roughly 1:1 because it's per-allocation, not per-core: each rollout still packs one ~3.4 MB routed_experts response that now takes the mmap path regardless of how many threads run.
- The harness was **load-bound at ~1 core** (CPU ~103%), so its *absolute* 3.4 s/rollout is not a production figure — but that cuts the right way for risk: under ~1 core the mmap syscall/fault overhead is **maximally** exposed (no spare cores to hide it), and it still only cost ~1%. On a dedicated worke node with the 512-thread executor actually parallelizing, the same syscalls overlap with compute across cores, so the relative hit should be **≤** the synthetic 1%, not worse. The one place production could differ: **page-fault storms under genuinely concurrent munmap/mmap** (kernel mmap_sem / per-mm VMA-tree contention) at 128+ simultaneous inflight on a NUMA GH200 — the harness's 128 inflight didn't truly run in parallel, so this contention was under-tested. That is exactly why (b) is a *live* gate at 10% and not assumed-passed: if page-fault contention bites, it shows up as a `time/step` p95 regression, and the gate catches it.
- Net: the throughput delta predicts **negligible** production latency risk (point estimate <1%, ceiling 10% bounded by gate (b)); the residual unknown is parallel-munmap kernel contention, which the live p95 gate is designed to surface.

**Pre-registered expectation:** all four pass (slope ≤0.5 GB/step, p50/p95 within 10%, no OOM, no abort-trigger). Confidence the patch holds in production: **80%**. The 20% is dominated by the parallel-munmap-contention unknown above, not by the leak re-appearing (the arena-placement mechanism is well-identified).
