# TTT Cheat-Sheet

One-pager for the cluster session. Built from the post-mortem
(`docs/test-time-training-lora.md`), the implementation plan
(`docs/ttt-implementation-plan.md`), and the probe specs
(`docs/ttt-probes.md`). When something looks broken or a config feels
wrong, start here.

## Cluster facts (as of 2026-05-20)

- **GPUs**: H200 (140 GiB), 8 per node, 2-4 nodes per run.
- **Storage**: `/beegfs` (140T total, ~25T free — getting tight; reaper
  cadence matters).
- **HF cache**: `/beegfs/huggingface/hub/` — `Qwen3-4B-Instruct-2507`
  already pinned there.
- **vLLM**: `0.21.0` pinned (`vllm-0.21.0+cu129` wheel).
- **Broadcast**: `nccl` in production.
- **Production memory**: peak 126.6 GiB on the 140 GiB card before
  `optim_cpu_offload=true` was added today. TTT adds learner-side
  forward+backward; budget accordingly.
- **Trainer LoRA defaults**: rank=16, alpha=32,
  `target_modules = [q,k,v,o,gate,up,down]_proj + experts + fc1/2_latent_proj`.
- **The gate_up_proj failure is fixed upstream** (`b2ba40b5e`, PR #2482,
  on `main`). The remaining live risk is reusing the same
  `lora_int_id` — see "Things that already burned us" below.
- **Multi-LoRA infra is production-hardened** via hosted training; the
  question for TTT is specifically the chunked-snapshot churn pattern,
  not whether multi-LoRA works.

## Pre-flight (before any run)

- [ ] Code branch: `feat/ttt-online-lora` (or later) — never `main`.
- [ ] Configs branch: feature branch in research-configs.
- [ ] `git submodule status` shows the expected SHAs for `deps/renderers`
      and `deps/verifiers`; both should be on the `sebastian/...`
      feature branches.
- [ ] `uv sync --all-extras` ran cleanly. uv.lock is the cluster's
      problem, not the laptop's.
- [ ] Cluster checkout has the pushed feature-branch commits, not stale.

## Required config invariants

If any of these are wrong, the run is invalid:

- [ ] `orchestrator.use_renderer = true`.
- [ ] `orchestrator.use_token_client = false`.
- [ ] `window_seq_len > max_completion_tokens + headroom`.
      The prior bug had `window_seq_len = 8192, max_completion_tokens =
      8192`, leaving the prompt budget at *one token*. Look for
      `prompt_tokens=1` in `/prepare_turn` logs as the smoking gun.
- [ ] `max_total_completion_tokens` carries the long-trajectory budget.
- [ ] `vllm.enable_lora = true`.
- [ ] `vllm.max_lora_rank >= experimental.ttt.lora.rank`.
- [ ] `vllm.max_loras` sized roughly to concurrent generation requests,
      not concurrent sessions.
- [ ] `experimental.ttt.adapter_dir` is on shared storage visible to
      learner, vLLM, and trainer. Probe writes from all three at boot.

Recommended starting values (from the post-mortem, sized for Qwen3-4B
on H200; pairs with production Forth config shape):

```toml
[experimental.ttt]
enabled = true
window_seq_len = 8192
max_completion_tokens = 2048   # or 4096; both leave real prompt budget
max_total_completion_tokens = 32768
update_every_tokens = 1024
adapter_dir = "/beegfs/sebastian/ttt-adapters"

[experimental.ttt.lora]
rank = 8
alpha = 16
```

`update_every_tokens = 1024` with `total_seq_len = 32768` means up to
32 chunk snapshots per rollout. With production oversampling=4 and
batch=128, that's ~16k snapshots in flight at steady state — the
reaper has to keep up.

## Things that already burned us

### Prompt budget collapse

Symptom: `prompt_tokens=1` in learner `/prepare_turn` logs. Run starts
but is meaningless.

Fix: never set `max_completion_tokens == window_seq_len`. The renderer
prompt budget is `window_seq_len - max_completion_tokens - headroom`,
so they cannot be equal. The validator should fail loudly.

### Global tool filters poisoning eval

Symptom: tool-output training silently disabled or wrong on eval envs
that don't share the train env's tool list.

Fix: tool-name filters are **per-env**, on `TrainEnvConfig.sft.tool_names`.
No global filter exists. If a "global" knob appears, that's the bug.

### Tool-output mask dropped in transit

Symptom: trainer reports zero tool-output loss; SFT advantage overlay
never fires.

Fix: `sft_mask` (and on the verifiers side, `prompt_attribution` +
`prompt_message_tool_names`) must survive response-token parsing,
trajectory interleaving, packing, padding, and the dummy-batch path.
This is already done on `feat/sft-on-tool-outputs`; if a regression
appears, suspect a code path that constructs `MicroBatch` /
`TrainingSample` without copying the new fields.

### TTT session leaks

Symptom: learner memory grows without bound across runs; reaper
sweeps adapter dirs that should still be live.

Fix: session id == `trajectory_id`, propagated through rollout state.
On rollout failure or reschedule, orchestrator MUST call
`/abort_session`. On normal completion, `/finish_session`. The
learner's heartbeat reaper handles stragglers but should be a backstop,
not the primary cleanup path.

### vLLM adapter id reuse poisoning worker state

Symptom (per `daniel/gptoss-lora-nan-repro` HANDOVER): "repeated LoRA
reload/update may poison or fail to refresh vLLM worker-side adapter
state, especially when the same adapter id is reused." Hosted issue
was gpt-oss-20b MoE, but the underlying mechanism is the
LoRA-id-reuse refresh path inside vLLM.

Fix: **every chunk snapshot gets a fresh `lora_int_id` and a fresh
`lora_name`.** Never reuse. Stale ids are evicted by vLLM's LRU
cache. If you find a code path that reuses an id "to save a load,"
that's the bug.

### vLLM dies on dynamic base weight update with active adapters
(historical)

Symptom: errors near `layers.0.mlp.gate_up_proj.weight` during a
mid-rollout policy update.

**Status**: fixed upstream in PR #2482 (`b2ba40b5e` on `main`). The
layerwise alias-buffer reload patch is already in our branch via the
ancestor chain. If this error still surfaces in a TTT run, the
patched path is regressing — check the diff against `main`.

Phase A still queues base weight updates between rollouts, but as a
conservatism choice (one less moving part during smoke), not because
the bug remains.

### Trace double-count

Symptom: same completion span trained twice, NLL metrics inconsistent
across runs.

Fix Phase A: one TTT sample per rollout. Don't split spans across
microbatches until span-splitting is unit-tested.

### Stale tests from the split-LoRA design

If you find references to `Phi_p`, `Phi_c`, `ttt_final_prompt_adapter`,
`new_prompt_ids`, or `token_role` — those are from the abandoned
two-LoRA design. Delete or rewrite, don't reanimate.

## Renderer integration

- TTT requires the renderer path. Token-client path doesn't have the
  exact-token-ids contract TTT needs.
- The renderer must:
  - strip TTT control keys before forwarding sampling args to vLLM (no
    leaks into `completion_params`)
  - reject missing exact token ids when `require_exact_token_ids=true`
  - apply prompt windowing before generation
  - preserve TTT trace metadata in rollout output
- Train and eval sessions are isolated. Eval rollout sessions are
  always fresh; train sessions are never reused for eval.

## Adapter lifecycle

Hard rules:

1. Active session LoRA lives in learner GPU memory while the request
   is being processed.
2. Session LoRA + optimizer state offloaded to CPU between requests
   (Phase B+).
3. Per-turn adapter snapshots materialized to `adapter_dir` for vLLM
   and trainer.
4. vLLM unloads adapter after the generation request that used it
   returns.
5. Trainer loads frozen adapter snapshots only for replay. LoRA params
   are `requires_grad=False` in the trainer.
6. Trainer evicts adapter slot after backward.
7. Rank 0 deletes consumed adapter dirs **only** after replay and
   optimizer step complete successfully, with refcount tracking from
   vLLM as well.

**Never** delete an adapter before trainer replay has consumed it.
**Never** unload a vLLM adapter before its generation request has
returned.

## What we already know works in this codebase

- Multi-LoRA per-token routing on the trainer side
  (`MultiLoRALinear`, `_grouped_mm`, `lora_num_tokens`, `scaling_factors`).
- `MultiRunManager` session lifecycle, including discovery,
  synchronization, eviction, creation/deletion hooks.
- vLLM dynamic LoRA loading via patched `_load_adapter` and the
  `POST /load_lora_adapter` endpoint.
- Per-rollout `lora_name` on the scheduler (currently used as one
  global name; small extension to per-task).
- `trajectory_id` propagating through verifiers rollout state.

Reuse these. The TTT changes are mostly glue, not new infrastructure.

## First cluster tasks (in order)

1. **Smoke `feat/sft-on-tool-outputs`** — *effectively already done*.
   Three runs are live on `/beegfs/outputs/forth-lang-qwen-{rl,cmb-code-ct,cmb-both-ct}-r1`
   under `wandb://sft-process-thesis`. The `cmb-both-ct-a0.5` run is
   pulling ahead on `forth-lang-test/pass@1` by step 200 and tool-call
   frequencies (`run_code_calls`, `submit_code_calls`) are diverging
   from baseline, which validates the SFT path behaviorally. No
   separate smoke run needed.
2. **Probes** — see `docs/ttt-probes.md` for the four probe specs.
   Order: snapshot churn → learner forward+backward → multi-LoRA
   forward overhead → adapter write throughput. The first two are
   single-GPU and small; can run during a 10-minute compute gap.
3. **Phase A skeleton**: config schema, learner service, orchestrator
   client, transport types, trainer-side replay path. Lean on the
   implementation plan for file-by-file scope.
4. **One Forth + one retrieval smoke TTT run** — pair with the
   existing `qwen-rl.toml` so the only delta is TTT-enabled.

If the probes show snapshot churn > 200ms p99 or learner
forward+backward > the between-chunk interval, halt and revisit
Phase A scope before writing code.

## When in doubt

- Re-read the post-mortem
  (`docs/test-time-training-lora.md`).
- Re-read the implementation plan
  (`docs/ttt-implementation-plan.md`).
- The plan has more architectural detail; the post-mortem has more
  historical context on what failed and why.
- If something contradicts between them, the implementation plan wins
  (it's later and explicitly accounts for existing prime-rl
  infrastructure).
