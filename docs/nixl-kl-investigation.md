# NIXL vs NCCL+FP8 KL mismatch investigation

Scratchpad / lab notebook. Append-only per iteration.

## Goal

Drive NIXL broadcast's step-wise `Mismatch KL` into a **bounded**
regime that matches the NCCL+FP8 (kernel-format) baseline behavior.

- **Acceptance (10 steps):** `Mismatch KL < 0.005` **strictly** across
  the first 10 steps. Per-step absolute numbers won't line up with the
  NCCL baseline due to per-run non-determinism; what matters is that
  KL does **not grow** and stays bounded.
- **Confirmation (20 steps):** extend a clean 10-step run to 20. Some
  slack allowed — isolated spikes up to ~0.007 are OK as long as KL
  stays bounded and doesn't trend upward. Sustained drift or a spike
  much above that = fail, go back to iterating.

The NCCL+FP8 baseline run is in flight (user). Record its numbers
when ready — mainly to confirm the "bounded <0.005" regime is
achievable at all on this config.

## How to iterate

Each cycle takes ~20 min. Budget is tight — be surgical.

### Monitoring a run

Runs sometimes hang or crash and the user may not be around to
notice. **I'm responsible for detecting and restarting.** Minimum
health loop:

1. `squeue -u $USER` — job still in `R` state?
2. `ls -la /beegfs/outputs/nixl-broadcast/logs/trainer/node_0.log`
   — file mtime advancing?
3. Grep latest KL entries — step counter still moving?

**Hang signature:** log mtime frozen for >3 min during the training
loop, or SPG barrier log lines but no subsequent `Step N` line for
>5 min. **Crash signature:** `Traceback`, `NIXL_ERR_`,
`remote index out of range`, `SIZE MISMATCH`.

**Restart protocol:** `scancel <jobid>` → fix the config / code if
needed → `source .env && uv run rl @ prod.toml` again. Note the
restart + reason in the investigation log.

Use `ScheduleWakeup` at ~10-15 min intervals during an active
iteration to poll without burning cache on empty checks.

### Commit discipline

Every experiment gets its own commit so any judgment call I made can
be reverted independently:

- One commit per code change being tested (hypothesis / fix).
- Commit **before** launching the run, so the SHA is the exact thing
  that ran. Use `git log -1` to record the SHA in the iteration log.
- Commit message format: `Exp: <hypothesis> — <what changed>`.
- Record the KL series + verdict in the iteration log below, along
  with the commit SHA. That way the user can `git revert <sha>` on
  any iteration where they disagree with my read of the trend.
- When in doubt about whether a trend is bounded vs. growing, err on
  the side of failing the iteration and flagging it explicitly in
  the log — false acceptance is worse than an extra iteration.

**Run NIXL broadcast (candidate under test):**

```bash
# On the nixl-weight-transfer branch
git checkout nixl-weight-transfer
# In prod.toml: weight_broadcast.type = "nixl" and comment out
#   quantize_in_weight_transfer (NIXL always quantizes).
source .env
uv run rl @ prod.toml
```

**Run NCCL+FP8 baseline (reference):**

```bash
# On main
git checkout main
# In prod.toml: weight_broadcast.type = "nccl" and
#   quantize_in_weight_transfer = true (uncomment).
source .env
uv run rl @ prod.toml
```

## Reading Mismatch KL from logs

Per-step summary lines land in the trainer rank-0 log. The output dir
is wiped between runs (`clean_output_dir = true` in `prod.toml`), so
extract KL values *before* the next run starts:

```bash
# Latest running / just-finished job:
ls -lat /beegfs/outputs/nixl-broadcast/logs/trainer/node_0.log
# Per-step line pattern:
# SUCCESS Step N | Time: Xs | Loss: ... | Entropy: ... | Mismatch KL: Y | ...
```

If the log has ANSI color codes, strip with
`sed 's/\x1b\[[0-9;]*m//g'` before grepping. The slurm stdout
`job_NNN.log` also captures this.

## Current NIXL state — last run before investigation started

- NIXL branch `nixl-weight-transfer` head: `f23d68b52` (FP8 scale
  floor = 1e-12, HSDP support, EP partition assertion, post-refactor).
- NCCL baseline branch `main` head: `6bf14b80f` (vf bump).

Run submitted via `uv run rl @ prod.toml` on NIXL branch at
`f23d68b52`. Output dir was wiped before I could re-check; values
below are from a pre-wipe scrape.

| Step | Mismatch KL |
|---:|---:|
| 0 | 0.0022 |
| 1 | 0.0016 |
| 2 | 0.0027 |
| 3 | 0.0033 |
| 4 | 0.0033 |
| 5 | 0.0038 |
| 6 | 0.0060 |
| 7 | 0.0050 |
| 8 | 0.0062 |
| 9 | 0.0059 |
| 10 | 0.0082 |
| 11 | 0.0120 |
| 12 | 0.0227 |
| 13 | 0.0110 |
| 14 | 0.0142 |
| 15 | 0.0354 |
| 16 | 0.0198 |

**Trend:** drift from ~0.002 at step 0 to ~0.02-0.03 by step 15-16,
with at least one spike (step 15 = 0.035). Not a one-shot bug — looks
like accumulating error across pushes.

## Suspect list (hypotheses to check against baseline)

Ordered by my current priority:

1. **Scale-buffer layout / offset bug at inference-side narrow.** The
   refactor moved from one flat `non_expert_layout` dict to per-slot
   `LayoutEntry`s. For FP8-quantized fused specs (e.g. fused_qkv_a)
   the weight goes into offset regions via `narrow(0, offset_rows,
   rows)`, and so does the scale via `scale_offset_rows`. The
   `_infer_nixl_trainer_ws` / `dp_shard_cp` change might also
   re-index scales differently from before. Worth verifying the
   scale buffer is written at the right byte offset end-to-end.

2. **DTensor `to_local()` vs `full_tensor()` for per-shard FP8.**
   `_resolve_source` returns `to_local()` when slot_rows != src_rows.
   For FP8-quantized per-shard slots, quantizing only the local
   shard is bit-exact only if the shard boundary is 128-row-aligned
   AND if the source's tensor-parallel sharding matches the FSDP
   shard axis we're reading. If a param is actually replicated (not
   sharded), `to_local()` returns the local copy but `src.shape[0]
   == slot_rows` may be False due to DTensor's reported shape — need
   to verify per-shard dispatch gate is doing the right thing when
   a source is Replicate() rather than Shard(0).

3. **Post-refactor chunked write ordering.** Previously all writes
   went into the write table in a specific order. Now
   `slot.build_writes(peers)` iterates per-slot. If we accidentally
   wrote `weight` and `weight_scale_inv` to the wrong peer (e.g.
   weight to prefill-0, scale to prefill-1 when they correspond to
   different global experts), inference would see a
   weight/scale mismatch.

4. **HSDP shard-rank mismatch.** My recent HSDP change moved
   `my_rank` from global `dist.get_rank()` to
   `dp_shard_cp.get_local_rank()`. On `dp_replicate=1` these are
   identical, but there may be a subtle bug if the mesh's
   dp_shard_cp rank ordering differs from `dist.get_rank() %
   (dp_shard × cp)` on the primary replica. Worth assertion-checking.

5. **Expert slot mapping: local expert idx → remote chunk idx.** The
   routing computes `remote_chunk_idx = peer_experts.index(global_id)`.
   If the peer's `expert_map[moe_prefix]` ordering on inference
   differs from what the trainer expects, experts end up in the
   wrong slots on inference. This used to work in the old NCCL+FP8
   path because the whole expert table was shipped atomically with
   the routing info; in NIXL we key per-write on expert_map.

6. **Shared expert naming / shape handling.** Old code had a
   `sw.ndim == 3` squeeze path for legacy checkpoints. New code
   doesn't. If the model ever produces 3D shared expert params
   (shouldn't, but worth confirming on GLM-5.1 state_dict), the new
   path would shape-mismatch silently.

7. **Scale dtype mismatch.** FP8 kernel writes fp32 scale. Inference
   may expect `torch.float32` or `torch.float32` with column-major
   layout. If we write row-major and vLLM reads column-major, the
   saved scales look transposed.

## Instrumentation ideas (low risk, high information)

Before chasing hypotheses, add a one-shot numeric check per push:

- Trainer side: for a canonical layer (e.g. layer 3
  `self_attn.fused_qkv_a_proj`), compute `weight.float().sum()` +
  `scale.sum()` after `convert()` but before post_write. Log it.
- Inference side: after `update_weights_from_path`, for the same
  layer, read the vLLM param + scale buffer, compute the same sums.
  Log it.
- If trainer sum != inference sum for any slot, it's a write-table
  bug (hypothesis 1, 3, or 5). If they match but KL still diverges,
  it's a quantization / dequantization parity bug (hypothesis 2 or
  7).

Also useful: run NIXL with `flush_every=1` (instead of 100). If a
larger batch of pending writes is what corrupts ordering, per-write
drain will make the bug go away. Slow but definitive.

## Investigation log

### Iteration 0 — NCCL+FP8 baseline captured

Branch `main` @ `6bf14b80f`. `prod.toml`: `weight_broadcast.type =
"nccl"` + `quantize_in_weight_transfer = true`. Job 5647, 11 steps
observed before cancel:

| Step | KL | Step | KL |
|---:|---:|---:|---:|
| 0 | 0.0004 | 6 | 0.0011 |
| 1 | 0.0022 | 7 | 0.0016 |
| 2 | 0.0010 | 8 | 0.0000 |
| 3 | 0.0015 | 9 | 0.0007 |
| 4 | 0.0015 | 10 | 0.0016 |
| 5 | 0.0010 | | |

Max KL: 0.0022 (step 1). All 11 steps under 0.003, well inside the
<0.005 bound. The bounded regime is reachable on this config.

**Reference NIXL run (previous, same SHA `f23d68b52`):** 0.002 → 0.035
over 17 steps. Fails the bound at step 11+. That's what we're fixing.

### Iteration 1 — reproduce NIXL drift on current SHA (baseline)

SHA: `81be8e763` (investigation doc commit on top of `f23d68b52`; no
numerical code changes). `prod.toml`: NIXL, wandb
`nixl-iter1-repro`. Job 5648.

| Step | KL | | Step | KL |
|---:|---:|---|---:|---:|
| 0 | 0.0026 | | 5 | 0.0027 |
| 1 | 0.0016 | | 6 | **0.0064** |
| 2 | 0.0032 | | 7 | **0.0051** |
| 3 | 0.0006 | | 8 | **0.0062** |
| 4 | 0.0046 | | 9 | **0.0075** |

**Verdict:** FAIL. KL breaks 0.005 at step 6 and stays elevated.
Matches prior run qualitatively (steady slow upward drift). Drift is
deterministic, not noise. Good diagnostic baseline to work from.

Cancelled at step 9.

### Iteration 2 — end-to-end signature diagnostic

Hypothesis: transport isn't faithful across steps. Add per-step
signature logs on both trainer and inference sides for a canonical
anchor slot. If signatures match every step but KL still grows, the
transport is fine and the drift is in quantization/dequantization
consistency. If signatures diverge, it's a transport/addressing bug.

- Anchor: `model.layers.3.input_layernorm.weight` — small, 1D
  bf16, non-quantized, GatheredSlot (full tensor on every trainer
  rank; rank 0 writes once per inference peer round-robin).
- Trainer rank-0 log after `slot.convert()` (before post_write):
  `[nixl SIG trainer] key=... sum=... shape=...`
- Inference (all workers) log after SPG barrier:
  `[nixl SIG inference] key=... sum=... shape=...`
- Expected: trainer step N's sum should equal every inference
  worker's sum for that step. Drift = write bug.
- `prod.toml` unchanged (still NIXL). Wandb name
  `nixl-iter2-sigdiag`.

Job 5650, 10 steps observed:

| Step | KL |
|---:|---:|
| 0 | 0.0024 |
| 1 | 0.0015 |
| 2 | 0.0014 |
| 3 | 0.0038 |
| 4 | 0.0037 |
| 5 | 0.0021 |
| 6 | **0.0067** |
| 7 | 0.0032 |
| 8 | **0.0057** |
| 9 | 0.0034 |

**Verdict on KL:** FAIL — breaks 0.005 at step 6. Consistent with
iter1 (different numbers, same shape).

**SIG comparison, `model.layers.3.input_layernorm.weight`:**

| Step | Trainer sum | All 32 inference workers |
|---:|---:|---:|
| 0 | 249.99527359 | 249.99527359 ✓ |
| 1 | 249.99526978 | 249.99526978 ✓ |
| 2 | 249.99527359 | 249.99527359 ✓ |
| 3 | 249.99527359 | 249.99527359 ✓ |
| ... | all match | all match |

**Key result:** transport is bit-exact for the bf16 GatheredSlot. KL
drift does NOT come from this slot type. Rules out a broad transport
bug. Problem is in FP8-quantized slots or ExpertSlots (or in how
inference uses them). Next: expand anchors to F (fp8 gather) and E
(expert) to localize.

### Iteration 3 — expand signature diagnostic to FP8 + expert slots

SHA will be added post-commit. Adds anchors:
- **F** `model.layers.3.self_attn.kv_b_proj.weight` — FP8 gathered
  (rows=28672, fsdp_total=64 → per-shard rows=448, fails 128-align
  gate → GatheredSlot fp8).
- **E** `model.layers.3.mlp.experts.w13_weight[0]` — ExpertSlot fp8
  grouped, expert 0 specifically. Trainer rank 0 owns experts
  [0..4); inference worker that owns global expert 0 (via its
  `expert_map`) logs the local index.

Wandb name `nixl-iter3-sigdiag-3anchors`.

Job 5653, 11 steps observed (KL drifts: breaks 0.005 at steps 4, 6,
8, 10 — matches iter1 drift pattern).

**Trainer SIG (rank 0, anchor stepwise):**
- G: `249.99527359 → 249.99527359 → ... → 249.99527740` (bf16 noise).
- F w_bytes: `2300248320 → 2300245929 → 2300253387 → ...` (changes).
- E w_bytes: `3793815079 → 3793813458 → 3793823409 → ...` (changes).

**Inference SIG (all workers matched):**

| Step | Anchor | Trainer | Inference | Match |
|---:|---|---:|---:|---|
| 0 | G w | 249.99527359 | 249.99527359 | ✓ |
| 0 | F w_bytes | 2300248320 | 2300248320 | ✓ |
| 0 | F scale | 0.10161374 | **0.00000000** | ✗ |
| 0 | E w_bytes | 3793815079 | 3793815079 | ✓ |
| 0 | E scale | 0.27224183 | **0.00000000** | ✗ |

**Interpretation:** FP8 weight bytes match. Scales read as 0 on
inference — almost certainly a diagnostic bug: vLLM stores
`weight_scale_inv` as a `Parameter`, not a `Buffer`, so my
`named_bufs[...]` lookup returns nothing and I print 0. Need to
check both `named_parameters()` and `named_buffers()`.

### Iteration 4 — fix scale lookup on inference diagnostic

Same anchors, but the inference-side `_lookup()` helper now checks
both `named_parameters` and `named_buffers`, and logs the location
(`loc=param/buf/MISSING`) so we can tell.

Wandb name `nixl-iter4-sigdiag-scalefix`.

Job 5657. Key inference-side SIG observations (all 32 workers, step 0):

| Anchor | loc | Trainer | Inference | Match |
|---|---|---:|---:|---|
| G | param | 249.99527359 | 249.99527359 | ✓ |
| F | param/param | w=2300248861 s=0.10161349 | w=2300248861 s=0.10161349 | ✓ |
| E | param/param | w=3793830601 s=0.27224205 | w=3793830601 s=0.27224205 | ✓ |

Both FP8 weight AND scale match bit-exact on all three slot types.
Transport is fully verified. KL drift is NOT a transport bug. It's
elsewhere.

**KL series (iter4), same deterministic pattern:**
0.0021, 0.0031, 0.0006, 0.0012, 0.0049, 0.0038, 0.0021, 0.0045, **0.0050** → fails bound.

**Analysis:** vLLM's `Fp8LinearMethod.process_weights_after_loading`
calls `maybe_post_process_fp8_weight_block` when
`use_deep_gemm=True`, which runs `transform_sf_into_required_layout`
to permute `weight_scale_inv` into a DeepGemm-specific layout. This
happens ONCE at initial model load. After that, both NCCL+FP8 and
NIXL paths `copy_` raw blockwise scales over the transformed scales
— DG then reads the raw values in the transformed layout → wrong
GEMM output.

The NCCL baseline shows KL ~0.001 while NIXL shows ~0.002-0.03; if
this hypothesis is correct, NCCL is also slightly broken but just
less bad for reasons unclear. The cleanest way to falsify is to
disable DeepGemm entirely and see if NIXL becomes bounded.

### Iteration 5 — disable `use_deep_gemm` to falsify DG layout hypothesis

Single change in `prod.toml`: `inference.use_deep_gemm = false`.
All SIG diagnostics stay. Wandb name `nixl-iter5-no-deep-gemm`.

If NIXL KL stays bounded <0.005 → DG layout mismatch is the bug,
and the fix is either (a) re-run the DG layout transform after
writes, or (b) keep DG off, or (c) have trainer write scales in
DG layout.
If NIXL KL still drifts → problem is elsewhere.

Job 5660. KL series (step 0-9):
0.0020, 0.0013, 0.0025, 0.0037, **0.0113**, **0.0088**, **0.0157**, **0.0063**, **0.0194**, **0.0084**.

**FALSIFIED.** Disabling DG made KL **worse**, not better. So the issue
isn't that DG reads raw scales wrong — it's something else. Possible:
when DG is off, vLLM falls back to a non-DG GEMM path that has its
*own* scale layout expectation, and that one is also clobbered by
raw scales (or something unrelated breaks). Too noisy a test to be
useful alone.

### Iteration 6 — shape + stride diagnostic

Narrow hypothesis: trainer writes scale bytes in row-major order but
inference's `weight_scale_inv` (after DG's
`maybe_post_process_fp8_weight_block` at initial model load) is in a
permuted / column-major / TMA-aligned layout. Byte count matches so
NIXL accepts the write, but values end up in wrong positions (sum is
permutation-invariant, so my SIG sum check doesn't catch it).

Add `.shape` and `.stride()` to both trainer and inference SIG logs
for all three anchors. A discrepancy = silent layout corruption.

DG re-enabled (`use_deep_gemm = true`). Wandb name
`nixl-iter6-shape-stride`.

Results (step 0 only, cancelled early):

| Anchor | Trainer | Inference | Match |
|---|---|---|---|
| G | shape=(6144,) stride=(1,) sum=249.99527359 | shape=(6144,) stride=(1,) sum=249.99527359 | ✓ |
| F w | shape=(28672, 512) stride=(512, 1) bytes=2300251362 | same | ✓ |
| F s | shape=(224, 4) stride=(4, 1) sum=0.10161371 | shape=(224, 4) stride=(4, 1) sum=0.10161371 | ✓ |
| E w | shape=(4096, 6144) stride=(6144, 1) bytes=3793825267 | shape=(16, 4096, 6144) (per-expert slice identical) | ✓ |
| E s | shape=(32, 48) stride=(48, 1) sum=0.27224193 | same | ✓ |

**Layouts match bit-exact.** No transpose, no permutation, no TMA-aligned
reshape. My layout-mismatch theory is falsified. DG on Hopper with
disable_ue8m0=True must keep the scale in raw row-major blockwise
layout, or the transform is a no-op for these shapes.

KL step 0 = 0.0013 — still drifting (but only 1 step observed before
cancel). Pattern matches prior iterations qualitatively.

### Iteration 7 — multi-source fused-region sum check

Hypothesis: single-source transport is proven faithful. Multi-source
fused specs (e.g. `fused_qkv_a_proj`) have two trainer slots written
to different offsets in one inference tensor. If the offset math is
off (even by a few rows), the sub-ranges would be swapped/shifted
but sum-over-full-tensor stays correct.

Check: on inference, slice `fused_qkv_a_proj.weight[0:2048]` and
`[2048:2624]` and log each sum. Trainer logs per-source slot sum.
Matching pairs confirm fused routing; mismatched pairs reveal the
offset bug.

Also widen expert anchors to global experts 0-3 (trainer rank 0's
owned set), so we cross-check the per-expert remote_idx mapping
across multiple entries.

Wandb name `nixl-iter7-fused-region`.

**Step 0 SIGs (from job 5665):**

| Anchor | Trainer | Inference | Match |
|---|---|---|---|
| F_q  | w=1956669286 s=0.09529844 | w=1956669286 s=0.09529844 | ✓ |
| F_kv | w=533551617  s=0.09167859 | w=533551617  s=0.09167859 | ✓ |
| E[0] | w=3793822673 s=0.27224198 | w=3793822673 s=0.27224198 | ✓ |
| E[1] | w=4007950214 s=0.28839970 | w=4007950214 s=0.28839970 | ✓ |
| E[2] | w=3353069296 s=0.29298384 | w=3353069296 s=0.29298384 | ✓ |
| E[3] | w=4060886279 s=0.25664634 | w=4060886279 s=0.25664634 | ✓ |

Fused multi-source offsets and multi-expert routing both match
bit-exact. Transport is **100% verified** for all slot types and
routing modes I've tested in layer 3.

**Breakthrough hypothesis: non-layer tensors aren't transported.**
Listing `conversion_specs()` output shows coverage is **per-layer only**:

- Per-layer: layernorms, attention projections, indexer, MoE bits.
- Missing: `model.embed_tokens.weight`, `model.norm.weight`,
  `lm_head.weight`.

In the NCCL+FP8 baseline, `filter_state_dict_by_layers` yields
`layer_idx=-1` containing exactly these non-layer tensors, and they
get broadcast. In NIXL, `TransportPlan.__init__` iterates only
`range(num_hidden_layers)` and never picks them up. They stay at
whatever inference loaded from disk; trainer's gradient updates them
locally; KL drifts as the two copies diverge.

### Iteration 8 — transport non-layer tensors (embed, norm, lm_head)

Code changes:
- Base model: new `non_layer_conversion_specs()` hook, default `()`.
- `GlmMoeDsaForCausalLM`: override to return specs for embed, norm,
  and (conditionally) lm_head.
- `ConversionSpec.build_slots` + slot `from_spec`: `_join()` helper
  so empty prefix → plain source name (no leading dot).
- `TransportPlan.__init__`: after per-layer slots, append slots from
  `model.non_layer_conversion_specs()` at prefix=`""`.

If this is the bug, KL should drop to NCCL-baseline-like level
(<0.005) across all steps. Wandb name `nixl-iter8-non-layer`.

_(append iterations below as they run)_
