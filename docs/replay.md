# Replay

The `replay-v1` environment turns a run's saved rollouts back into training tasks. Every RL run writes its train rollouts to `<output_dir>/rollouts/step_N/train_rollouts.jsonl` (one WireTrace per line, stamped with `info.prime_rl` metadata); the replay taskset indexes those files as a buffer and serves derived tasks from them — resume a conversation from its compaction point, ask the model to re-check a finished attempt, or have it judge whether an attempt was correct. A replay env is a normal `[[orchestrator.train.env]]` entry mixed into training with a `ratio`, over either a previous run's rollouts (offline) or the current run's own (online, `buffer_dir = "self"`).

## Table of Contents

- [The Three Derivations](#the-three-derivations)
- [Offline vs Online Buffers](#offline-vs-online-buffers)
- [Configuration](#configuration)
  - [Field Reference](#field-reference)
  - [Example](#example)
- [Metrics](#metrics)
- [Constraints and Caveats](#constraints-and-caveats)

## The Three Derivations

Each replay env serves exactly one derivation, chosen by `derivation.type`; mix derivations (and weight them) with multiple env entries.

| `derivation.type` | Seed | Scored by |
|---|---|---|
| `continue` | The post-compaction prompt (system + summary user message) recovered from the trace's message graph — the model resumes the task from a context-compaction point. | The original (`inner`) taskset. |
| `recheck` | The finished conversation plus an appended check-your-work user turn (`instruction`). | The original (`inner`) taskset. |
| `judge` | A rendered transcript of the rollout and the question "was this correct?" (`instruction`). | Self-contained: reward is `judge_correct` — 1.0 when the model's `VERDICT: CORRECT\|INCORRECT` line matches whether the source rollout's reward exceeded `success_threshold`, 0.0 otherwise (including unparseable verdicts). |

For `continue` and `recheck`, the rollout is scored by the original taskset: `inner` must reproduce the source run's taskset config, and the inner rewards land on the trace with their original names and weights. The replay taskset also delegates tools, setup, and finalize to the inner taskset, so tool-using tasks replay with their tools intact. `judge` is self-contained — its derivation config has no `inner`.

`continue` produces one task per compaction point in the source rollout, so only rollouts that actually compacted are candidates. `recheck` and `judge` produce one task per rollout.

## Offline vs Online Buffers

**Offline** (the default): point `buffer_dir` at a finished run's rollouts. The buffer is indexed once at env-server startup — steps are scanned newest-first up to the buffer's capacity (the newest few thousand candidates are retained) — and each task index maps deterministically to one candidate. GRPO group members dispatched as independent rollouts therefore still bind the same source rollout.

**Online** (`buffer_dir = "self"`, which implies `online = true`): the buffer is the run's own growing rollout dir. The orchestrator resolves the `"self"` sentinel to `<output_dir>/rollouts` before the env servers spawn. Four consequences:

- **Barrier semantics.** The jsonl is written non-atomically, so an online buffer only reads steps whose sibling `train_rollouts.bin` exists — the orchestrator writes and closes the jsonl strictly before the atomic rename that creates the `.bin`, so the barrier file marks the jsonl complete. This means online replay requires the filesystem rollout transport (a run with a non-filesystem transport never writes the `.bin`). Offline buffers skip the barrier: finished runs' files are complete by definition.
- **Group dispatch.** Every request samples a fresh source rollout, so the whole GRPO group must share one draw — the taskset forces whole-group dispatch (`REQUIRES_GROUP_ROLLOUTS`), and all group members arrive at the env server as a single request.
- **Startup.** Until the run has produced its first replayable step, replay requests wait briefly, then fail (with a logged warning) so their dispatch permits free up for the source envs — the orchestrator retries replay groups naturally, and they start succeeding once the first step's barrier appears. The buffer rescans for new steps during training, and its capacity doubles as recency eviction: the newest candidates are retained, keeping the buffer on recent policy behavior.
- **Choose your sources.** A `"self"` buffer sees every train env's saved rollouts — including the replay envs' own. By default (`source_envs` unset) replay-derived records are skipped: a judge judging its own judge rollouts is a feedback loop, not a task. Listing env names replays exactly those — and naming a replay env is the deliberate opt-in for chained derivations (a recheck env sourcing another recheck env is depth-2 self-correction; the env topology *is* the depth control, and scoring/tools/provisioning always resolve to the innermost original task).

## Configuration

The env is registered as `replay-v1` (installed via the `envs` extra). Configure it as a v1 taskset on a train env entry:

```toml
[orchestrator.train.env.taskset]
id = "replay-v1"
buffer_dir = "self"        # implies online
# Judge only the fresh env's rollouts. (Unset, source_envs replays every env except
# replay envs; listing a replay env by name opts into chained derivations.)
source_envs = ["reverse-text"]

[orchestrator.train.env.taskset.derivation]
type = "judge"
```

### Field Reference

Core fields (every replay env):

| Field | Default | What it does |
|---|---|---|
| `buffer_dir` | *(required)* | The saved-rollout dir to replay: a run's `rollouts` dir (or the run dir containing it). Under prime-rl the literal `"self"` resolves to this run's own rollout dir (an online buffer over the run's freshly written rollouts). |
| `derivation` | *(required)* | The derived task this env serves: `{ type = "continue" \| "recheck" \| "judge", ... }` with the per-derivation fields below. One derivation per env entry. |
| `source_envs` | `None` | Which envs' rollouts to replay, by their stamped name (`info.prime_rl.env_name`). Unset: every env except replay envs. An explicit list replays exactly those — naming a replay env opts into chained derivations (recheck a recheck). With a list set, records without the stamp never match. |
| `online` | *(inferred)* | Set automatically for `buffer_dir = "self"`. Set it yourself only to treat another still-running run's dir as a growing buffer. |

Per-derivation fields:

| Field | On | Default | What it does |
|---|---|---|---|
| `inner` | `continue`, `recheck` | *(required)* | The original taskset's config — it scores the derived rollouts and provides their tools, so it must reproduce the source run's taskset config. |
| `allow_container` | `continue`, `recheck` | `false` | Allow sources whose task ran in a container image. The container state the transcript references is gone — a fresh container is provisioned from the same image, so the model resumes in a reset world. |
| `instruction` | `recheck`, `judge` | *(built-in prompt)* | Recheck: the user turn appended to the finished conversation. Judge: the prompt template; `{transcript}` is replaced with the rendered rollout. |
| `success_threshold` | `judge` | `0.5` | The source rollout counts as correct when its reward exceeds this. |
| `max_transcript_chars` | `judge` | `60000` | Total transcript budget, sized to the model's context; over it, middle messages are elided (the task statement and the trailing conversation are kept). Single messages are capped at 1/20 of it. |

### Example

A run that spends 3/4 of each batch on fresh reverse-text rollouts and 1/4 judging its own recent rollouts. Ratios are all-or-none across train envs: once any env sets one, every env must.

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

# Judge tasks over this run's own rollouts (1/4).
[[orchestrator.train.env]]
name = "replay-judge"
ratio = 1.0
harness = { id = "null", runtime = { type = "subprocess" } }
# Judge prompts embed a rendered transcript, so give the env input headroom.
max_input_tokens = 1536
max_total_tokens = 2048

[orchestrator.train.env.taskset]
id = "replay-v1"
buffer_dir = "self"
source_envs = ["reverse-text"]

[orchestrator.train.env.taskset.derivation]
type = "judge"
max_transcript_chars = 4000
```

For an offline continue/recheck env, point `buffer_dir` at the prior run and set the derivation's `inner`:

```toml
[orchestrator.train.env.taskset]
id = "replay-v1"
buffer_dir = "outputs/prior_run/rollouts"

[orchestrator.train.env.taskset.derivation]
type = "recheck"

[orchestrator.train.env.taskset.derivation.inner]
id = "reverse-text-v1"
```

Runnable debug configs live at `configs/debug/replay/` (`online_judge.toml`, `offline_recheck.toml`).

## Metrics

Every replay rollout logs `replay/source_reward` (the reward the source rollout received) and `replay/source_step` (the trainer step that wrote it). Judge rollouts additionally log `replay/judge_parseable` — the fraction of replies with a parseable `VERDICT:` line; watch it early, an unparseable verdict scores 0.

## Constraints and Caveats

- **One derivation — and one judge — per env entry.** The derivation is a taskset field, and env-level settings (harness runtime, token budgets, `ratio`) are per-env, so judge belongs on its own `[[orchestrator.train.env]]` entry rather than sharing one with continue/recheck.
- **Harness: match the source for continue/recheck; `null` for judge.** Continue and recheck resume a conversation — run them under the same harness the source env used, so the resumed rollout has the same scaffold and harness-side tools as the original. The harness must support message-prompt seeding (`SUPPORTS_MESSAGE_PROMPT`; the built-in `default` and `null` harnesses do — string-prompt CLI harnesses can't be seeded with a prior conversation). Judge is a tool-less verdict task: use the `null` chat-loop harness on a subprocess runtime.
- **Replay envs need explicit ratios.** Without ratios, envs are weighted by task count — an online replay env advertises a single virtual task and would effectively never be drawn. Setting `ratio` on the replay env forces ratios on every train env (the all-or-none rule), which is the intended, explicit state.
- **Set explicit token budgets on continue/recheck envs.** Continue seeds carry a compaction summary and recheck seeds carry the whole source conversation, so replay prompts are far longer than a fresh task's. Set `max_input_tokens` / `max_total_tokens` on the replay env entry so seeds fit `seq_len` instead of truncating mid-rollout.
- **Long continue tasks age off-policy.** A continue task resumes deep into a long trajectory and keeps going, so its rollout can span many trainer steps and be discarded by `orchestrator.max_off_policy_steps` (default 8). If a continue env shows sustained `errored_rollouts`, raise `max_off_policy_steps` or shorten the tasks.
- **`source_envs` needs the stamp.** The filter matches `info.prime_rl.env_name`, which prime-rl's rollout writer stamps into every saved record. Records without the stamp (rollout files from other producers) never match an explicit list — leave `source_envs` unset to replay them.
- **`allow_container` is off for a reason.** The trace records the conversation, not the container: for a containerized source task, continue/recheck provisions a fresh container from the same image, so any filesystem state the transcript references is gone and the model resumes in a reset world. Enable it only when the task's image is self-contained enough for that to be fair — and give the replay env's harness a container runtime (docker/prime): with a subprocess runtime every imaged request fails at episode construction.
- **Uniform-verdict judge groups train nothing.** When every rollout in a judge group gives the same verdict, the group's rewards are uniform, group-relative advantages are zero, and the default `zero_advantage` filter drops the group — expected GRPO behavior, and the built-in 1:1 label balancing keeps a constant-verdict policy at chance so the surviving groups carry signal. A high filtered fraction on the judge env means verdicts have collapsed, not that the env is broken.
- **Inner-taskset limits.** Replay delegates scoring to `inner` but cannot delegate group scoring (`@group_reward` tasksets), custom `State` classes, or user simulators — those are rejected loudly at env construction. Shared-placement inner toolsets are also unsupported, but fail per-rollout (the serving layer inspects toolsets on a stub task and never starts them).
