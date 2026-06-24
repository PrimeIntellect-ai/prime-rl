# RESULTS — EP-vs-TP4 decode bench (Qwen3.5-35B-A3B debate, Isambard GH200)

**Prereg:** `ep_decode_2node_2026-06-24.md` (locked before results, commit 73ec6d293). Run 2026-06-24, autonomous, single bench node nid010225, LoRA debate-r64 (`broadcasts/step_17`), real 4t12i debate prompt token-ids, prod sampling (temp1/top_p0.95/top_k20), all windows firewall-gated (pure-decode <0.05, zero-preempt, waiting-stable, CoV<0.15, no request failures).

## HEADLINE VERDICT: STAY TP4 (EP not worth adopting for the fleet)

EP's per-token comm reclaim is **real** but its KV-replication penalty overwhelms it once both arms use the obvious levers (fp8-KV + adequate `max_num_seqs`). TP4 wins the deployable throughput at **both** contexts when caps are co-optimized; EP's only lead (8k at the prod cap=256) is a **cap artifact** that evaporates when TP4's cap is raised.

## The two questions, separated

**(1) iso-BATCH — per-token comm efficiency** (same batch B, LoRA-on):
| ctx | EP /GPU | TP4 /GPU | EP edge |
|---|---|---|---|
| 8k (B=144) | 1538 | 1327 | **+15.9%** |
| 32k (B=56) | 560 | 536 | **+4.6%** |
→ EP's comm reclaim is **real** (1-GPU engines skip TP's all-reduce), but **shrinks at long context** (decode is less comm-dominated at 32k). Confirms the 35%-TP-comm profile.

**(2) iso-SATURATION — what a saturated fleet replica delivers/GPU** (each at its own KV-clean max, LoRA-on):
| /GPU | EP-bf16 | EP+fp8 | TP4-bf16 | TP4+fp8 (cap256) | TP4+fp8 (cap512) |
|---|---|---|---|---|---|
| 8k | 1538 @144 | 2145 @268 | 1982 @256 | 1977 @256 | **2285 @384** |
| 32k | 560 @56 | 790 @96 | 833 @110 | **812 @150** | — |

- **bf16:** TP4 wins both (EP KV-starved: batch 144 vs 256 @8k, 56 vs 110 @32k — EP replicates non-expert weights → 2× less node KV: 504,627 vs 4,076,637 tok).
- **fp8 @prod cap 256:** EP+fp8 wins 8k (2145 vs 1977, +8.5%) — fp8 unlocks EP's deep per-engine batch (144→268) while TP4 stays cap-throttled (KV only 37%, fp8 useless to it). TP4 wins 32k (812 vs 790).
- **fp8 @raised cap 512 (the decider):** TP4+fp8 8k jumps to **2285 @384 > EP+fp8 2145** (+6.5%). TP4's 8k engine is compute-starved (profile SM-issue 16%) so it scales with the deeper batch fp8 affords (7.69M tok KV, 234×@32k). EP's 8k win was purely the 256-cap throttling TP4.

**Net:** co-optimized (fp8 + adequate cap), TP4 wins 8k (+6.5%) and 32k (+3%). EP leads only iso-batch (per-token), never deployable saturation.

## Hypothesis outcomes vs locked predictions
- **H_throughput** (pred 50%: EP +8–20% iso-sat 8k) → **REFUTED.** EP loses iso-saturation everywhere co-optimized. The +16% existed only at iso-batch (a per-token metric, not the fleet number).
- **H_KV_confound** (pred: EP gains are KV/DRAM artifacts) → **CONFIRMED.** fp8-KV (the relief) is exactly what EP needs to be competitive; it's a KV-capacity story, not a comm win. fp8 helps the KV-limited arm (EP), not the cap-limited arm (TP4 @256).
- **H_correctness** (EP-vs-TP4 routing) → **leg-2 PASS-leaning:** 23.6% of (token,layer) flip but 98% single-marginal-expert near-tie flips (mean 1.14), no mod-64 structural signature (8.5%≈chance) → benign reduction-order drift, EP IDs in correct global index space. **OWED:** the dominant trainer-leg + logit-margin sub-diagnostics (need gate logits, not captured).
- **H_expert_skew** → per-rank straggler modest (max/mean 1.35) but per-expert hotness high (8–12×) → EPLB-ADDRESSABLE-but-LoRA-blocked (quantize⊥LoRA) → §7b future-work, trigger MET but moot given the throughput verdict.

## Decision-table cell (§6): PASS-leaning × iso-sat-loss → **NO-GO on EP throughput. Stay TP4.**

## Caveats / owed (do not over-claim)
- **Both arms are sync/unfused LOWER BOUNDS** on this HW: EP has no async DeepEP (x86/RDMA-only); TP4 has no async-TP (SequenceParallel hard-disabled at hidden_size=2048) and no AR-fusion (fuse_allreduce_rms hardcoded-False, Slingshot deadlock). A real optimized fleet could shift absolute numbers; the *direction* (TP4 ≥ EP on deployable throughput) is robust because EP's deficit is structural (KV replication), not a missing kernel.
- **Single-launch deltas**, not the prereg's ≥3-launch variance protocol. Within-window CoVs tiny (0.005–0.04) but cross-launch SD unmeasured; +6.5%/+3% are directional, not variance-bounded.
- Trainer-leg correctness firewall (dominant gate) NOT run — EP-vs-TP4 leg + skew only.
- Front-end (api_server_count=1) saturation not stress-tested at fleet scale (would further penalize EP's 4-engines-per-front-end).
- Open-loop single-node: production step-makespan / closed-loop scheduling not measured; this is decode-physics only.

## Data: tmp/ep_decode/results/*.json ; captures tmp/ep_decode/captures/*.npz ; harness tmp/ep_decode/*.py ; full trail tmp/ep_decode/SYNC_LOG.md
