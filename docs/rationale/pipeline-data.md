# Pipeline data plumbing in prime-rl: what was built and what was not

Four items from the pipeline needs list (P1 training part, P3, the index follow-up from
flow's rollups, P6) were checked against prime-rl (`feat/episode-cancel` c3ef3b1f),
verifiers (`feat/pipeline-integration` c17b296f, upstream `origin/main` 6e69354f), flow
(`feat/pipeline-observe` 29de8a1) and the pipelines that would use them. Two are built,
two are not.

| Item | Verdict | Where |
| --- | --- | --- |
| P1 `observe` on the training path | Not built | - |
| P3 pass@k and early-stop stats as data | Not built (exists, or is pipeline policy) | - |
| Trace labels + per-model usage in the index row | Built | verifiers `trace.info["labels"]`, prime-rl `index.py` |
| P6 trace to SFT rows | Built (the generic core only) | prime-rl `monitors/file/traces/sft.py` |

## P1: `Taskset.observe` on the training path - not built

**Claim.** The eval runner calls `observe`/`on_failed`/`on_cancelled` exactly once per
dispatched episode (e71df869, c3ef3b1f); training does not.

**Evidence.**
- prime-rl e26cb4fe (2026-10-06, "Keep training as is: stream only eval envs' infinite
  tasksets") already decided that streaming tasksets are an eval feature. Training's
  `TrainSource` hands each env's tasks to a `Curriculum`, which samples them with
  replacement (`curriculum/base.py`, `sampler.observe(group)`): one task runs many times
  across steps. `observe`'s contract is "exactly once per dispatched task", and flow's
  `FlowTaskset.observe` commits a job attempt (attempt fence) on it. Calling it from
  training would commit the same attempt many times; that is a different contract, not
  a small wiring change.
- No pipeline runs a flow taskset in training. Hillclimb runs every stage as an eval
  (`configs/generate.toml` `[run]`); the tasks it produces go to training as a finite,
  exported set. Flow's `ItemTaskset` (`query.py`) is the training-side bridge: a finite
  taskset over accepted item versions. Its `observe` only writes optional facts
  (`record`, default none), so training without `observe` loses nothing today.
- The redesign spec (flow `docs/redesign.md`, Layer 2) asked for the training call
  "only if it is a small, contained change; otherwise note it as follow-up".

**Decision.** Not built. The trigger to revisit: a curriculum that must learn from a
pipeline's live outcomes during RL (e.g. retire a task once the policy solves it). That
needs a training-side contract (`observe` per group, many per task), designed with the
curriculum, not the eval contract reused.

## P3: pass@k and early-stop stats as data - not built

**Claim** (unification plan): "pass@k exists only in `dashboard/static/app.js`."

**Evidence.**
- The claim is wrong for prime-rl. `orchestrator/metrics.py` `EvalMetrics.to_wandb`
  writes `eval/<env>/effective/<agent>/pass@k` and `pass^k` (unbiased estimator,
  `orchestrator/utils.py` `compute_pass_metrics`) with `avg@k` for every eval epoch, to
  wandb and to `monitors/file/metrics.jsonl` (`docs/training.md`: "`eval/{env}/{avg@k,pass@k}`").
  The dashboard's `passAtK` is a per-task view of the same estimator over the index's
  `group` and `reward` columns, which any reader can recompute.
- The measuring pipelines do not measure through prime-rl groups. Hillclimb's solve
  groups are flow fan-outs (`hillclimb/pipeline/measure.py`: "one child job per rollout,
  each its own episode"); prime-rl sees `group_size` single-episode tasks. Early stop is
  `Pipeline.stop_early` with hillclimb's lane rule (`tasks/pool.py` `lane_decided`), and
  the join already records `stopped_early`, `clean_count` and `pass_at_1` per group as
  data in the run store. Repoforge's hypergeometric rule is its own policy.
- The math that is shared (the unbiased pass@k from n and c) is a few lines and lives
  in prime-rl already; the decision rules (bands, lanes, hypergeometric) are pipeline
  policy, and a shared measure block is flow's F12, out of scope here.

**Decision.** Nothing to build in prime-rl. The needs list entry should be corrected.

## Trace labels and per-model usage in the index row - built

**Problem.** prime-rl's file monitor writes one index row per episode
(`monitors/file/traces/index.py`): ids, tokens, cost, duration, timing. It has no model
and no trace labels. Flow's rollups (`prime_flow/pipeline/rollups.py`) need the stage
(`info.flow = {job, stage, attempt}`, set by `FlowEnv.trace_labels`) and per-model
calls/tokens/cost, so they seek into the stream and parse whole records (pyzstd for
sealed chunks) for every episode in the window. Flow's own rationale
(`docs/rationale/observability.md`, "prime-rl: add the episode's trace labels ... and a
per-model usage summary to the index row") names this as the lower-layer fix.

**Why it needs verifiers.** `Env.trace_labels(task)` merges into each trace's `info`, the
same dict harnesses and envs write into (a real hillclimb record has `harbor_tests`,
`hillclimb_panel` beside `flow`, some tens of KB). The record does not say which keys were
labels, so prime-rl cannot copy "the labels" without copying all of `info`, and a
prime-rl list of keys to copy would be per-pipeline config for what the env already
knows. The env is the only place that knows, so verifiers puts them under one reserved
key of each trace's `info`, `info["labels"]` (every trace of an episode carries the same
labels); the trace stays the one record of them.

**Index row additions** (in `summarize_episode`, so a stream without an index derives
the same rows):
- `labels`: the first trace's `info["labels"]` (`{}` for a record without it).
- `models`: per requested model, `{calls, errors, prompt_tokens, cached_input_tokens,
  completion_tokens, cost}`, summed over every trace's calls (verifiers' `Usage` names;
  `prompt_tokens` excludes cache reads). `cost` stays null unless some call reported one.
  A call with no model counts under `"?"`.

Per-call error messages (429 causes) stay in the record: they are only needed for
episodes that failed, and their size is unbounded.

**Not built.** No index schema version, no rewrite of old indexes (a reader treats a
missing key as empty), no dashboard view of either column.

## P6: trace to SFT rows - built (the generic core)

**Existing exporters.**
- data-flywheel (`prime-envs-private/environments/data-flywheel/data_flywheel/export.py`
  `sft`, `rows`, `wire`, `Pictures`, `tools_of`): one row per context window of a rho
  ledger turn (`rho.traces.windows`), messages via `message_to_wire` with text parts
  flattened, images written once per sha256 beside the rows, tools in the OpenAI
  envelope, and flywheel-specific `meta` (seat, turn, outcome).
- sft-gen (`/data/repos/sft-gen/sft_gen/export.py`): rows are rewritten conversations
  (replay candidates, receipts, hygiene), not stored branches. Generic parts:
  `chat_message` (= `message_to_wire` + text flatten) and `tool_spec` (no envelope).
  It also re-implements reading a run's streams across attempts (`read_records`).
- worldsims (`worldsims_runtime/trajectory/sft.py`) exports from its own trajectory
  model, not `vf.Trace`; not a consumer.
- prime-rl's SFT trainer reads `{messages, tools}` rows (`trainer/sft/data.py`); prime-rl
  and verifiers have no trace-to-row converter. RL already trains one sample per
  trainable `Trace.branches` entry.

**The generic core.** Three pieces repeat: a stored message in the wire form prime-rl's
SFT loader reads (text-only content as one string), the trace's tools in the OpenAI
envelope, and image data URLs written out as files. Over a prime-rl stream there is a
fourth: every episode of a run, across resume attempts, one row per trainable branch.
Selection (which episodes are good), windowing by a harness ledger, and per-pipeline
`meta` stay in the pipelines.

**Built** (`prime_rl/monitors/file/traces/sft.py`, `python -m prime_rl.monitors.file.traces.sft`):
- `sft_message(message, images=None)` and `sft_tools(tools)`: the per-message and tools
  conversion both exporters wrote.
- `Images`: content-addressed image files beside the rows (flywheel's `Pictures`).
- `trace_rows(episode, trace, images=None)`: one row per trainable branch,
  `{messages, tools, meta}` with `meta = {episode, trace, branch, env, task, agent,
  reward, labels}`.
- `export(run_dirs, dest, ...)`: every episode of the runs (all attempts, newest record
  per episode id, via `eval/resume.py`), filtered by ok (default), kind, env, agent, and
  minimum reward, written atomically.

**Not built.** No dedup across near-identical branches, no token-length filter (the SFT
loader already truncates to `seq_len`), no rewrite/quality gates, no HF upload. A
non-sampled assistant message on a branch is exported as an assistant message (the SFT
loss mask is by role); the row keeps the branch whole rather than guessing.
