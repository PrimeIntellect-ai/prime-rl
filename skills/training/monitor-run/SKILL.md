---
name: monitor-run
description: Monitor an ongoing prime-rl training run — find the output directory, tail logs, check key metrics, inspect SLURM jobs, and restart safely. Use when asked to check on a run, debug training, or investigate performance.
---

# Monitor a run

## Runbook

### On launch

1. Find the run dir and read the resolved configs at `{run_dir}/configs/latest/resolved/` (start with `rl.json`, or `orchestrator.json` on local runs). Read the launch command from `{run_dir}/configs/latest/command.txt`. The launch TOML is copied verbatim to `{run_dir}/configs/latest/rl.toml`. The run dir is `{output_dir}/{run_name}` — `run.name` auto-generates as `<envs>--<model>--<short-id>`, so if you only know the output dir, pick the most recently modified subdirectory (`ls -t {output_dir} | head -1`) or read `run.name` from the launch command.
2. Confirm all processes are alive and the run is making progress.
3. Write the initial summary into `{run_dir}/STATUS.md`.

### Recurring check-ins

Default cadence: **1 hour** (researcher can override). At each check-in:

1. Confirm processes are alive.
2. Grep logs for errors/warnings; note current step and key metrics.
3. **Append** an entry to `{run_dir}/STATUS.md` (never overwrite):

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation, eval scores, env-specific metrics.
**Stability**: entropy, mismatch_kl, grad_norm — flag spikes.
**Performance**: trainer vs orchestrator step time, env lag, inference pressure.

**Notes**: anything unusual (errors, restarts, hangs). Omit if nothing notable.
```

In W&B, each project auto-gets an **"overview" saved view** (train / eval / stability / performance sections) on its first run — use it for a quick check instead of the auto-generated default workspace.

### Restarting a run

**Never restart unless the researcher explicitly asked.** Confirm the exact restart command and the conditions that warrant one.

**Never** run kill or launch commands yourself. Hand the researcher the exact command and let them run it; after a restart, verify all processes are back up and progress resumed before the next check-in.

---

## Reference

### Where to find things

- `{run_dir}/configs/latest/` — the current attempt's command, launch TOML, and `resolved/` JSON files. Each launch stays under `configs/attempt_<n>/`.
- `{run_dir}/logs/latest/` — the current attempt's logs (each launch gets `logs/attempt_<n>/`; resumes never overwrite earlier attempts). See below.
- `{run_dir}/monitors/file/` — the metrics, and the traces with the annotations about them (see Episodes below).

### Dashboard

`uv run dashboard [output_dir ...]` (default `outputs/`, or `$PRL_OUTPUT_DIR` if set; several dirs can be tracked at
once) serves a local web dashboard at `http://localhost:7788` with four views per run:
metrics (the W&B overview sections, read from `metrics.jsonl`), per-attempt config
files, a rollout trace viewer with per-token overlays (advantage, trainer logprob,
entropy, KL mismatch, stable/loss/content masks), and merged component logs. It only reads the run dirs — safe to run against a live run.
`--port`/`--host` pick the bind address; a taken port automatically bumps to the next
free one, so several dashboards run side by side without coordination. GPU deps live
behind the `gpu` extra, so `uv sync --extra dashboard && uv run dashboard` works
without the training stack (e.g. on a head node).

**Daemon (auto-start)**: launchers auto-start one dashboard per host per user and a
live one absorbs each new run's output dir automatically — see the `dashboard` skill
for discovery, kill/restart commands, and `--isolated`. The short version: the live
port can differ from 7788 (a taken port bumps), so read the discovery file:

```bash
cat ~/.cache/prime-rl/dashboard/daemon.json   # {"pid": ..., "url": "http://localhost:<actual port>"}
ps aux | grep PRL::Dashboard                  # the daemon's process title
```

Verify liveness with `curl -sf <url>/api/runs` and hand the researcher the `url`.

### Logs

For checkpoint-cache inspection, use the path printed by the trainer's weight
load log. A standalone helper's default Hub cache can differ from the cache
selected by the training entrypoint. Compare original and converted weights
under the actual load path before concluding that a conversion is missing.
The same applies to offline dataset helpers: match the run's `HF_HOME` or
`HF_DATASETS_CACHE` before interpreting an offline cache miss as missing data.

```
{run_dir}/logs/latest/
├── trainer.log                # rank 0 stdout
├── orchestrator.log           # orchestrator stdout
├── evals.log                  # SFT online-eval evals stdout
├── inference.log              # vLLM stdout
├── trainer/
│   ├── node_*.log             # per-node (multi-node only)
│   └── torchrun/              # per-rank stdout/stderr
├── inference/
│   ├── node_*.log             # per-node (multi-node only)
│   └── router.log             # the single global router (multi-node only; single-node logs it in inference.log)
└── envs/{train,eval}/{env_name}.log    # one log file per env
```

SLURM batch logs are under `{run_dir}/launcher/logs/*job_*.log`.

Usually tailing `trainer.log`, `orchestrator.log`, and `inference.log` is enough. Drop into per-node or per-rank logs only when debugging. All logs are loguru with `HH:mm:ss  LEVEL  message`; levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`.

Scan for problems:

```bash
grep -E "WARNING|ERROR" {run_dir}/logs/latest/{trainer,orchestrator,evals,inference}.log
grep -E "WARNING|ERROR" {run_dir}/logs/latest/envs/{train,eval}/*.log
```

### Metrics

All metrics print to the console log (and W&B when configured).

**Progress** — orchestrator log. Rollout metrics mirror the episode/trace hierarchy, at two levels:

- `{scope}/{subset}/<metric>/<stat>` — episode-level facts only: the token/turn/branch counts, summed over an episode's traces.
- `{scope}/{subset}/<agent>/<metric>/<stat>` — every trace-level metric (reward, truncation, errors, timing, env metrics, curriculum admission, eval scores), keyed by agent name so seats never mix. Flat over that agent's traces: one sample is one trace, so an in-episode fan-out like n solvers contributes n samples.

`scope` is `train/agg` (all train envs) or `train/<env>` (`eval/<env>` for eval); `subset` is `all` (every rollout) or `effective` (admitted, clean, and trainable). Single-agent envs have one agent — usually `agent` — and one trace per episode, so both levels agree; multi-agent envs name each seat (`proposer`, `solver`, `judge`, …).

| Metric | Description |
|--------|-------------|
| `train/agg/effective/<agent>/reward/mean` | mean training reward for that agent (per env: `train/<env>/effective/<agent>/reward/mean`) |
| `train/agg/effective/num_total_tokens/mean` | avg tokens per episode, summed over its agents (also `num_input_tokens`, `num_output_tokens`) |
| `train/agg/effective/num_turns/mean` | avg turns per episode, summed over its agents |
| `train/<env>/effective/<agent>/num_turns/mean` | avg turns for that agent alone (also token counts, `num_branches`) |
| `train/agg/effective/<agent>/is_truncated/mean` | fraction of that agent's rollouts truncated |
| `train/agg/all/<agent>/has_error/mean` | fraction of that agent's rollouts errored (per-type under `train/agg/all/<agent>/error/<type>`; also `dispatcher/errored/{train,eval}`) |
| `train/agg/all/<agent>/is_trainable/mean` | fraction carrying a training signal — 0.0 for a frozen seat like a judge |
| `train/agg/all/<agent>/is_admitted/mean` | fraction accepted by the source curriculum; per-source counters and custom policy metrics live under `curriculum/<env>/` |
| `train/<env>/effective/<agent>/metrics/<name>/mean` | env-specific metrics for that agent (e.g. pass rate) |
| `train/<env>/effective/<agent>/timing/agent/model/mean` | model vs harness share of that agent's phase |
| `eval/<env>/effective/<agent>/{avg@k,pass@k}` | eval scores for that agent, when configured |

**Stability** — trainer log:

| Metric | Description |
|--------|-------------|
| `mismatch_kl/{all,env}/{mean,std,max}` | KL between trainer and (old) inference policy over trainable tokens |
| `entropy/{all,env}/{mean,std,max}` | policy entropy over trainable tokens |
| `is_masked/mean` | fraction of tokens masked by the IPO trust region |
| `optim/grad_norm` | spikes may precede divergence |

**Performance** — trainer and orchestrator step independently, so comparing step times shows who's waiting on whom.

| Source | Metric | Description |
|--------|--------|-------------|
| trainer | `time/step` | total trainer step |
| trainer | `time/wait_for_batch` | **high → orchestrator is bottleneck** |
| trainer | `time/forward_backward`, `time/broadcast_weights`, `time/save_ckpt` | phase timings |
| trainer | `perf/throughput`, `perf/mfu` | tokens/s and MFU % |
| orchestrator | `time/step`, `time/save_ckpt` | phase timings |
| orchestrator | `time/wait_for_policy` | **high → trainer is bottleneck** |
| orchestrator | `dispatcher/off_policy/{mean,max}`, `dispatcher/inflight/{train,eval}`, `dispatcher/queued/eval` | dispatcher / async state |
| orchestrator | `off_policy/{mean,max}`, `off_policy/{in_flight,in_queue}/{mean,max}`, `off_policy/dropped` | per-step staleness of trained rollouts |
| env server | event loop lag (min/mean/p90/p99/max), active task distribution | periodic |

The trainer warns when batch wait time exceeds active trainer time. Add inference nodes when this warning persists. The orchestrator warns when policy wait time exceeds active orchestrator time. Add trainer nodes when this warning persists. The orchestrator also warns when it discards more than half of an episode window and reports stale, errored, and no-signal counts.

`orchestrator.constant_trainer_batch_size` defaults to `true`. It keeps each rollout batch at `orchestrator.batch_size` effective episodes. Set it to `false` for faster collection with variable trainer batch sizes.

For live vLLM stats, query Prometheus directly:

```bash
curl -s http://localhost:8100/metrics | grep -E "num_requests|gpu_cache_usage"  # engine port (8000 is the router)
# vllm:num_requests_running, vllm:num_requests_waiting, vllm:gpu_cache_usage_perc (→1.0 = KV cache saturated)
```

### Episodes

```
{run_dir}/monitors/file/metrics.jsonl                              # every metric row, tagged by producer
{run_dir}/monitors/file/traces/stream/00000.jsonl.zst              # every episode, appended as it arrives: sealed chunks ...
{run_dir}/monitors/file/traces/stream/00001.jsonl                  # ... and the live one, plain text
{run_dir}/monitors/file/traces/stream.index.jsonl                  # one compact row per episode, with its chunk and byte offset
{run_dir}/monitors/file/traces/annotations/{producer}/00000.jsonl  # trace updates: orch ship-time facts, trainer per-token streams
{run_dir}/monitors/file/traces/annotations/{producer}.index.jsonl  # each update's scalars and where its record sits
```

Everything the file monitor dumps lives under `monitors/file/`; nothing is written
there when the monitor is off. A step can have several metric rows: mismatch
diagnostics and optimizer statistics may arrive separately. Join the relevant
keys by step and producer instead of assuming one row contains every metric.
Some infrastructure rows have `step=null`; select the desired metric keys or
exclude those rows before comparing numeric step values.
The traces and everything written about them sit under
`traces/`. Each stream is a directory of numbered chunks — the writer rolls to a new
chunk at `monitors.file.chunk_bytes` (5 GiB) and, with `monitors.file.compress` (on),
seals the full one with zstd in the background; a finished run seals its live chunk
too. Sealed chunks use seekable frames, so a seek still costs one frame, and
`zstd -dcf` streams them together with the plain live chunk. Every index is named for
the stream it indexes and sits beside it. Those indexes are what keep reading a run
cheap: a consumer browses them instead of the streams, and seeks by the chunk and
offset they carry to read a single episode or its token streams. Both are derived, so
deleting them only costs a reader the work of rebuilding what it needs. The stream
holds native `vf.Episode` records (training tensors excluded; per-token floats rounded
to `monitors.file.float_decimals`, 4 by default), one line per episode in arrival order, whatever kind of work it did —
including trace-less failures, curriculum-rejected work, and work that never enters a
batch, so it is crash-durable. Each record carries its provenance: `env` (`id` plus the
orchestrator's `name`), full `task`, `group` (`id`), and `run`.

A trace has several steps, so each is stamped as its own event rather than implied by
where the record sits. The file monitor stamps `info.kind` and `info.dispatch`/
`info.arrival` (`{step, time}` each) as an episode lands; the ship-time annotation adds
`info.effective` and `info.ship` — the orchestrator step whose batch shipped the
cohort, or for eval the step that triggered the evaluation. Staleness is
`ship.step - dispatch.step`. Only `effective` ties to a step; `all` is the whole stream.

For RL evaluations, inspect `run.work.policy.start` and `.end` before attributing
an accuracy score to one checkpoint. The dispatcher can schedule training once
all evaluation requests have been dispatched, while evaluations are still in
flight. The evaluation success line and `eval/<env>/policy_version` report the
minimum starting version; they do not establish that every response stayed on
that version. Group evaluation records by `run.work.step`, check every policy
span, and report mixed spans explicitly. The strict training mismatch audit
covers consumed training traces separately.

Everything learned after arrival is an append-only trace update keyed by `trace_id`,
one file per producer so each has a single writer: the orchestrator records cohort
membership, the scalar advantage and per-branch advantage streams; the trainer records
its recomputed per-token logprobs and entropies. Readers fold the updates onto the
stream records, newest winning.

```bash
wc -l {run_dir}/monitors/file/traces/stream.index.jsonl
zstd -dcf {run_dir}/monitors/file/traces/stream/* | jq '.traces[].rewards'
zstd -dcf {run_dir}/monitors/file/traces/stream/* | jq 'select(.ok | not) | {id, env: .env.id, errors}'
jq '{trace_id, info}' {run_dir}/monitors/file/traces/annotations/orch.index.jsonl
```

The batches consumed by the trainer are shipped over ZMQ by default, so nothing binary is written. With `rollout_transport.type = "filesystem"` they land at `{run_dir}/batches/step_{n}/rank_<rank>.bin` (one packed micro-batch file per trainer DP rank).

### Common failure modes

For numerical train/inference alignment, enable
`trainer.model.debug.mismatch_diagnostics` and use synchronized rollouts
(`orchestrator.max_off_policy_steps = 0`). Read `logprob_bit_mismatch/all/mean`,
`logprob_abs_error/all/max`, `logprob_nonfinite/all/mean`, and
`mismatch_k3_stable/all/mean` alongside the standard KL. A rounded zero KL does
not prove equal logprobs. Require nonempty sampled-token coverage and zero
bit mismatches/non-finite pairs; set both
`trainer.monitors.file.float_decimals = "None"` and
`orchestrator.monitors.file.float_decimals = "None"` for exact trace inspection
(the shared file-monitor block currently only propagates `path`). See
`configs/experiments/mismatch/README.md` for
the frozen-weight and two-node experiment workflow.

A few warnings are normal. Escalate when errors are persistent, growing, or hit a large fraction of rollouts.

- **Env workers**: exceptions in env code, timeouts, sandbox errors, OOM kills (most common source — runs user code).
- **Orchestrator**: empty/errored rollout spikes, weight-broadcast failures, checkpoint errors.
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients.
- **Final broadcast**: if the trainer waits for `.receiver_ready` at the last
  step, check whether the orchestrator still consumes dispatcher results.
  Cancellation events use the bounded result queue; waiting for weights inside
  batch finalization can deadlock that queue. The orchestrator must stop train
  scheduling, drain results, then wait for the final broadcast when idle and
  re-check for any final eval work before exiting.
- **Inference**: NCCL/CUDA errors, OOM, request timeouts.
- **Batch-invariant serving and NCCL weight broadcast**: vLLM 0.28's
  `VLLM_BATCH_INVARIANT=1` overrides NCCL protocol, algorithm, channel, and
  transport settings. Apply those same NCCL settings to the trainer before
  communicator creation; asymmetric settings can fail the communicator warmup
  with `Message truncated` and leave the other ranks blocked. See the shared
  environment table in `configs/experiments/mismatch/vllm-batch-invariant.toml`.
- **Synchronized mismatch audits**: `max_off_policy_steps=0` checks each live
  episode's policy span before shipment and rejects a span that differs from
  `step - 1`. Trainer trace annotations record the training step and incoming
  policy version. Run `uv run python tools/audit_mismatch_policy.py <run_dir>`
  after a run; it requires nonempty trainer coverage and joins generation,
  shipment, and trainer records. Missing trainer version records fail the audit.
  For an exact-zero arm, `tools/audit_mismatch_zero.py <run_dir>` additionally
  compares raw FP32 trace bits and requires all configured steps. It reports
  actual sampled-token and long-position coverage, and fails incomplete runs
  even if their completed steps have zero mismatch.
  Prefill replay must run while the matching frozen server is alive. For short
  queued runs, the experiment helper `outputs/mismatch/watch_prefill.py` accepts
  positional job ID, run directory, and replay limit. It waits for the first
  trainer metric, then uses `srun --jobid ... --overlap` inside that allocation;
  it does not allocate another node. It rejects nonzero learning rates and
  checks finite, bitwise equality across trainer, prefill, and decode. Watch
  its log for completion; a terminal job before replay is a failed check.
  The dispatch gate also uses zero lead in this mode: after shipping a batch,
  new work waits for its updated policy instead of generating stale samples
  that the train sink would immediately drop.
- **Frozen diagnostics with uniform rewards**: default zero-advantage pruning
  can starve a scoring run on easy tasks. Set
  `orchestrator.filter_zero_advantages=false` to retain those sampled tokens
  for measurement; keep `trainer.optim.lr=0`. Record this setting when comparing
  metrics, since filtering changes the measured population.
- **Dense operator alignment probes**: run
  `uv run python tools/probe_mismatch_ops.py <output.json>` on an allocated idle
  GPU. vLLM 0.28's activation modules require `set_current_vllm_config` even
  when calling their CUDA forward directly. The optional trainer
  `model.debug.inference_swiglu` path uses that CUDA operation with an eager
  backward; full-graph compilation is incompatible with its graph break.
  The `model.debug.dense_alignment` experiment additionally requires eager
  trainer execution and the matching `dense-alignment.toml` inference overlay.
  That overlay disables `inference.enable_fp32_lm_head` so serving uses the
  shared BF16 projection instead of the default FP32-output projection that
  bypasses the linear module. Log-softmax still computes in FP32.
  Verify the attention version in worker logs: vLLM 0.28 downgrades FA4 to FA2
  when `VLLM_BATCH_INVARIANT=1`, even on Hopper. The dense experiment overrides
  that selector only on SM90 with Qwen3's 128-dimensional heads and pins one
  split. Its paged-cache probe must pass; a requested FA4 config alone is not
  proof that serving executed FA4.
  Compose `fp32-head.toml` after the dense overlay to preserve FP32 head
  accumulator outputs in both engines. Validate it with the probe's
  `--fp32-head-only` option. On vLLM's V2 model runner, patching the older
  `Sampler.compute_logprobs` alone does not affect selected-token logprobs:
  `vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs` is a separate
  reduction, also used by prompt-logprob scoring. Tiny residual errors with
  rounded-zero KL require checking this path explicitly.
  Validate its gradients and attention prefill/decode agreement before a full
  run. The vLLM-vendored FA4 function may return a tuple even without an explicit
  LSE request; extract its output tensor before numerical comparisons. Include
  actual long positions in RoPE and attention probes, rather than inferring
  coverage from the configured context limit.
- **MoE alignment probes**: the EP1/TP1 Qwen3 experiment uses a shared router,
  expert GEMMs, and explicit expert-ID-ordered FP32 summation. Validate with
  `tools/probe_mismatch_moe.py`, including empty-expert gradients and batch
  shapes. Triton 3.7.1 compiled a borrowed GEMM's `tl.dot` as TF32 even with
  `default_dot_input_precision='ieee'` in its launch metadata. Check the actual
  IR and a high-precision reference; the shared router kernel explicitly passes
  `input_precision='ieee'` to `tl.dot`. Its Python-only source is registered in
  the prime-kernels checkout and must be available to both trainer and serving.
  A frozen run can pass while learning fails after the first update if FP32
  router weights are downcast on the wire. In the aligned MoE path, preserve
  `mlp.router.gate.weight` through `keep_in_fp32_for_weight_transfer` as well as
  preserving FP32 router compute. Source-checkpoint precision alone does not
  describe updated router values. Recheck after real optimizer updates.
  With both MoE alignment and mismatch diagnostics enabled, the trainer also
  snapshots one router's local shards before backward and compares them after
  the optimizer completes. `optim/router_probe_changed_elements` counts changed
  finite FP32 elements across ranks; `optim/router_probe_nonfinite_elements`
  must stay zero. The startup log names the sampled parameter. A positive count
  proves that router changed; zero does not prove every model parameter stayed
  fixed. This is useful with full CPU offload, where gradient clipping and its
  gradient-norm metric are disabled. The zero audit reports these update steps
  separately from logprob equality and requires complete, finite probe records
  when they are present. For a positive-learning-rate run with probe records,
  the audit also requires a scored policy after at least one observed update.
  A frozen run is exempt; a learning run with all-zero gradients must not pass
  this learning gate just because its unchanged-policy logprobs match.
  The GLM variant additionally needs sigmoid routing with selection bias,
  partial RoPE, and shared-expert addition. Validate CPU-offloaded weight views
  directly when using UVA; successful GPU-resident GEMMs alone do not validate
  that path. Record actual model fit separately from small operator probes.
  `tools/probe_mismatch_ops.py <output.json> --moe-layouts` exercises 32-query/4-KV
  and 96-query/8-KV head layouts, covering Qwen3 MoE and GLM attention. It compares
  prefill, decode, and paged decode at lengths257,1024,8192 and checks backward
  against an FP32 reference. Inspect the forward bit-mismatch fields as well as
  the exit status; backward comparisons use numerical tolerances.
- **Pinned serving-weight budgets**: vLLM's CPU offload budget counts logical
  tensor bytes. PyTorch's pinned allocator can reserve more: GLM's2.75GiB and
  1.375GiB expert tensors allocate6GiB together under power-of-two rounding.
  On the tested PyTorch2.13 runtime, `pinned_max_round_threshold_mb:1` in
  `PYTORCH_ALLOC_CONF` makes these allocations exact-size; set the deprecated
  `PYTORCH_CUDA_ALLOC_CONF` consistently if the launcher exports it too. Use
  `tools/probe_mismatch_pinned_memory.py <output.json>` before a large offload
  launch and compare requested versus allocated bytes. Initialize CUDA before
  reading `torch.cuda.memory.host_memory_stats()`; otherwise it returns an empty
  dictionary even after a C++ pinned allocation. Monitor host `MemAvailable` and
  `Shmem`, not only GPU memory or `AnonPages`, while serving weights load.
  The large GLM experiment launcher also runs `outputs/mismatch/host_memory_guard.py`
  on each allocated node. It checks available host RAM each second and cancels
  its own job below the experiment's64GiB reserve. This is a host-exhaustion stop,
  separate from the priority watchdog and numerical audit; inspect its per-node
  logs if a run ends unexpectedly. The reserve is for these1.5TiB cluster nodes.
  Inspect the aggregate SLURM batch log as well as `inference.log`: a single
  external-LB replica can fail while others keep serving. Resolve its path from
  the launcher's `#SBATCH --output` directive; generated runs use
  `launcher/logs/job_<job_id>.log`. A glob over `launcher/*.log` only finds the
  yield log and misses this aggregate output. GLM learning job314
  hit CUDA allocation failure on its first decode at the default0.9 GPU-memory
  utilization, before its separate priority cancellation. The GLM alignment
  overlay reserves more GPU headroom with0.75 utilization and an8192-token
  context cap. Job321 completed twenty reverse-text training steps with a passing
  exact-zero updated-policy audit under these settings. Job322 also completed
  twenty single-turn GSM8K steps, and job323 completed three frozen long-input
  steps with exact sampled logprobs through sequence length4,137 and a passing
  live trainer/prefill/decode replay. These measurements validate the tested
  batches; validate longer or differently packed workloads separately.
- **Priority-yield watchdogs**: when a run is configured to yield to eligible
  pending jobs, inspect dependencies as well as pending reasons. After a SLURM
  dependency update, an unsatisfied job can briefly show reason `None` or `Resources` before
  `Dependency`. `squeue --format='%i|%r|%E'` exposes the unfulfilled dependency
  during that transition. It is not yet eligible demand for GPU resources.
  If the researcher permits using idle GPUs, inspect pending jobs' explicit
  excluded nodes (`%x`) against this allocation (`SLURM_JOB_NODELIST`). A job
  excluding every node in the allocation cannot consume these GPUs. Recheck
  exclusions each poll because scheduling constraints can change. If node
  expansion fails, keep the conservative yield behavior. Do not ignore a job
  merely because its current reason is `BadConstraints`.
- **Missing cached compiler artifacts**: a vLLM startup traceback for a missing
  `.cubin` or `.ttir` under a shared `torch_compile_cache` can come from stale
  cache metadata. Retry with fresh `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
  and `TRITON_CACHE_DIR` directories under node-local `/tmp`, unique per job and
  external-LB serving port. Set them in each inference process's launch
  environment; do not clear another run's shared cache. Preserve compilation
  settings when comparing numerical baselines.
  Apply the same isolation to trainer ranks that invoke Triton kernels. A
  private inference cache does not isolate the trainer's default
  `~/.triton/cache`. Set each trainer's cache paths before importing torch/model
  code, using its job ID and `LOCAL_RANK`; an experiment wrapper passed as the
  torchrun script can do this after torchrun assigns the rank.
  Avoid periodic `faulthandler.dump_traceback_later` in the borrowed Python 3.12
  runtime: an experiment's ranks received SIGSEGV while the timed dump was
  printing Torch/checkpoint frames. Keep cache isolation separate from stack
  instrumentation, and use normal logs and process state for routine monitoring.
- **Subprocess harnesses stuck before inference**: if rollouts are in flight but
  vLLM has generated no tokens, inspect subprocess CPU use and wait channels
  without printing argv (harness argv contains temporary API secrets). Many
  Python processes in `open_last_lookups`/`do_renameat2` can indicate shared
  filesystem import/bytecode contention. A node-local `UV_CACHE_DIR` puts PEP
  723 harness environments on local storage; set `PYTHONDONTWRITEBYTECODE=1`
  and cap `orchestrator.concurrency` during the smoke. Export the cache location
  on each node before env servers start; keep the installed project runtime
  separate and unchanged. Verify that rollouts actually complete after retrying.

### Process tree

All processes use `setproctitle` so they're visible in `ps`/`htop`/`pstree`:

```
PRL::Launcher
├── PRL::Inference          (vLLM server, GPU 0)
├── PRL::EnvServer          (verifiers' ZMQ env server, run in-process; one per train/eval source)
│   └── Verifiers::EnvWorker0..N
├── PRL::Orchestrator       (CPU-only; connects to each env server)
├── torchrun
│   └── PRL::Trainer        (GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are on separate nodes — use `srun` or `ssh` to inspect them.
