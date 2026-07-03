# Replay

The replay tasksets turn a run's saved rollouts back into training tasks. Every RL run writes its train rollouts to `<output_dir>/rollouts/step_N/train_rollouts.jsonl` (one WireTrace per line, stamped with `info.prime_rl` metadata); a replay taskset indexes those files as a buffer and serves tasks derived from them — resume a conversation from a compaction point or a tool call, or ask the model to re-check a finished attempt. A replay env is a normal `[[orchestrator.train.env]]` entry mixed into training with a `ratio`, over either a previous run's rollouts (offline) or the current run's own (online, `buffer_dir = "self"`).

The machinery is split like verifiers' `harbor`/`textarena` tasksets: the shared base (buffer, trace surgery, scoring delegation) lives in verifiers as `verifiers.v1.tasksets.replay`, and each derivation is a thin subclass package under `environments/`:

| Taskset id | Derived task |
|---|---|
| `replay-continue-v1` | Resume a source rollout mid-way — from a context-compaction point (`anchor = "compaction"`: the seed is the post-compaction prompt the original harness restarted from; one task per compaction) or right after a tool result (`anchor = "tool-call"`: one deterministically sampled resumable point per source; only points where every issued call has its result). |
| `replay-recheck-v1` | The finished conversation plus an appended check-your-work user turn (`instruction`); one task per rollout, seeded from its final state. |

Every derivation is scored by the original taskset: `inner` must reproduce the source run's taskset config, and the inner rewards land on the trace with their original names and weights. The replay base also delegates tools, setup, and finalize to the inner taskset, so tool-using tasks replay with their tools intact. Each drawn source spawns a fresh GRPO group.

## Table of Contents

- [Offline vs Online Buffers](#offline-vs-online-buffers)
- [Configuration](#configuration)
- [Metrics](#metrics)
- [Writing a New Derivation](#writing-a-new-derivation)
- [Constraints and Caveats](#constraints-and-caveats)

## Offline vs Online Buffers

**Offline** (the default): point `buffer_dir` at a finished run's rollouts. The buffer is indexed once at env-server startup — steps are scanned newest-first up to the buffer's capacity (the newest few thousand candidates are retained) — and each task index maps deterministically to one candidate. GRPO group members dispatched as independent rollouts therefore still bind the same source rollout.

**Online** (`buffer_dir = "self"`, which implies `online = true`): the buffer is the run's own growing rollout dir. The orchestrator resolves the `"self"` sentinel to `<output_dir>/rollouts` before the env servers spawn. Four consequences:

- **Barrier semantics.** The jsonl is written non-atomically, so an online buffer only reads steps whose sibling `train_rollouts.bin` exists — the orchestrator writes and closes the jsonl strictly before the atomic rename that creates the `.bin`, so the barrier file marks the jsonl complete. This means online replay requires the filesystem rollout transport (a run with a non-filesystem transport never writes the `.bin`). Offline buffers skip the barrier: finished runs' files are complete by definition.
- **Group dispatch.** Every request samples a fresh source rollout, so the whole GRPO group must share one draw — the taskset forces whole-group dispatch (`REQUIRES_GROUP_ROLLOUTS`), and all group members arrive at the env server as a single request.
- **Startup.** Until the run has produced its first replayable step, replay requests wait briefly, then fail (with a logged warning) so their dispatch permits free up for the source envs — the orchestrator retries replay groups naturally, and they start succeeding once the first step's barrier appears. The buffer rescans for new steps during training, and its capacity doubles as recency eviction: the newest candidates are retained, keeping the buffer on recent policy behavior.
- **Choose your sources.** A `"self"` buffer sees every train env's saved rollouts — including the replay envs' own. By default (`source_envs` unset) replay-derived records are skipped: rechecking your own rechecks is a feedback loop unless chosen deliberately. Listing env names replays exactly those — and naming a replay env is the deliberate opt-in for chained derivations (a recheck env sourcing another recheck env is depth-2 self-correction; the env topology *is* the depth control, and scoring/tools/provisioning always resolve to the innermost original task).

## Configuration

The derivation packages are installed via the `envs` extra. A replay env's taskset config is flat — the base fields plus the package's own:

```toml
[orchestrator]
batch_size = 128
group_size = 16

# Fresh rollouts (3/4 of the batch) — also what fills the buffer.
[[orchestrator.train.env]]
name = "reverse-text"
ratio = 3.0
taskset = { id = "reverse-text-v1" }
harness = { id = "null", runtime = { type = "subprocess" } }

# Recheck tasks over this run's own rollouts (1/4). Ratios are all-or-none.
[[orchestrator.train.env]]
name = "replay-recheck"
ratio = 1.0
harness = { id = "null", runtime = { type = "subprocess" } }

[orchestrator.train.env.taskset]
id = "replay-recheck-v1"
buffer_dir = "self"
source_envs = ["reverse-text"]

[orchestrator.train.env.taskset.inner]
id = "reverse-text-v1"
```

Base fields (every replay taskset, from `verifiers.v1.tasksets.replay`):

| Field | Default | What it does |
|---|---|---|
| `buffer_dir` | *(required)* | The saved-rollout dir to replay: a run's `rollouts` dir (or the run dir containing it). Under prime-rl the literal `"self"` resolves to this run's own rollout dir (an online buffer over the run's freshly written rollouts). |
| `inner` | *(required)* | The original taskset's config — it scores the derived rollouts and provides their tools, so it must reproduce the source run's taskset config. |
| `source_envs` | `None` | Which envs' rollouts to replay, by their stamped name (`info.prime_rl.env_name`). Unset: every env except replay envs. An explicit list replays exactly those — naming a replay env opts into chained derivations (recheck a recheck). With a list set, records without the stamp never match. |
| `allow_container` | `false` | Allow sources whose task ran in a container image. The container state the transcript references is gone — a fresh container is provisioned from the same image, so the model resumes in a reset world. |
| `online` | *(inferred)* | Set automatically for `buffer_dir = "self"`. Set it yourself only to treat another still-running run's dir as a growing buffer. |

Per-package fields: `replay-continue-v1` adds `anchor = "compaction" | "tool-call"`; `replay-recheck-v1` adds `instruction` (the appended check-your-work turn, with a built-in default).

Runnable debug configs live at `configs/debug/replay/` (`online_recheck.toml`, `offline_recheck.toml`).

## Metrics

Every replay rollout logs `replay/source_reward` (the reward the source rollout received) and `replay/source_step` (the trainer step that wrote it) alongside the inner taskset's own rewards and metrics.

## Writing a New Derivation

A derivation is a thin package over the base, exactly like `swebench-pro-v1` over verifiers' `harbor` taskset: subclass `ReplayTaskset` (binding your narrowed config in the generic) and implement two hooks — `record_anchors` (the resume points one saved rollout offers; each becomes one task) and `build_prompt` (the seeded conversation for one anchor). The base owns the buffer, online semantics, lazy binding, lineage, and inner-taskset delegation; `verifiers.v1.tasksets.replay.surgery` provides the graph enumerators (`compaction_forks`, `tool_call_anchors`, `recheck_seed`, ...). See `environments/replay_recheck_v1` for the ~30-line reference. Register the package in the root `pyproject.toml` (`envs` extra + `[tool.uv.sources]`) and `uv sync`.

## Constraints and Caveats

- **One derivation per env entry.** Env-level settings (harness runtime, token budgets, `ratio`) are per-env — mix derivations with multiple env entries.
- **Harness: match the source.** Replay resumes conversations — run the replay env under the same harness the source env used, so the resumed rollout has the same scaffold and harness-side tools as the original. The harness must support message-prompt seeding (`SUPPORTS_MESSAGE_PROMPT`; the built-in `default` and `null` harnesses do — string-prompt CLI harnesses can't be seeded with a prior conversation).
- **Replay envs need explicit ratios.** Without ratios, envs are weighted by task count — an online replay env advertises a single virtual task and would effectively never be drawn. Setting `ratio` on the replay env forces ratios on every train env (the all-or-none rule), which is the intended, explicit state.
- **Token budgets default from `seq_len`.** Every v1 train env's `max_total_tokens` defaults to the run's `seq_len` at config-resolve time, so replayed seeds stop instead of producing samples that can't fit training. Override `max_total_tokens` (or set it to `"None"`) per env entry to opt out.
- **Long continue tasks age off-policy.** A continue task resumes deep into a long trajectory and keeps going, so its rollout can span many trainer steps and be discarded by `orchestrator.max_off_policy_steps` (default 8). If a continue env shows sustained `errored_rollouts`, raise `max_off_policy_steps` or shorten the tasks.
- **`source_envs` needs the stamp.** The filter matches `info.prime_rl.env_name`, which prime-rl's rollout writer stamps into every saved record. Records without the stamp (rollout files from other producers) never match an explicit list — leave `source_envs` unset to replay them.
- **`allow_container` is off for a reason.** The trace records the conversation, not the container: for a containerized source task, replay provisions a fresh container from the same image, so any filesystem state the transcript references is gone and the model resumes in a reset world — especially acute for `anchor = "tool-call"`, which resumes mid-trajectory. Enable it only when the task's image is self-contained enough for that to be fair — and give the replay env's harness a container runtime (docker/prime): with a subprocess runtime every imaged request fails at episode construction.
- **Inner-taskset limits.** Replay delegates scoring to `inner` but cannot delegate group scoring (`@group_reward` tasksets), custom `State` classes, or user simulators — those are rejected loudly at env construction. Shared-placement inner toolsets are also unsupported, but fail per-rollout (the serving layer inspects toolsets on a stub task and never starts them).
