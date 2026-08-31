---
name: dashboard
description: Find, start, use, and stop the local run dashboard for metrics, configs, traces, logs, and reports. Use when asked for its URL, to watch or inspect a run, to control the open dashboard, or to create a cited dashboard report explicitly requested by the user.
---

# Run dashboard

`uv sync --extra dashboard && uv run dashboard [output_dir ...]` (default
`outputs/`, or `$PRL_OUTPUT_DIR` if set) serves a web UI at `http://localhost:7788`. It only reads run
dirs — safe against live runs — and installs anywhere (cluster head node,
laptop against a mounted outputs dir): GPU dependencies live behind the
`gpu` extra.

The Config and Logs views default to `latest (attempt <n>)`. Select an attempt
to inspect the immutable config or log files from an earlier launch. The Config
view shows a copyable launch command above the launch TOML or resolved JSON.

Every dashboard instance serves the dirs it was started with **plus** every dir
in the per-user registry (`~/.cache/prime-rl/dashboard/dirs.json`, re-read
live). Launchers (`rl`, `sft`) register their output dir on every
start and, in interactive sessions, auto-start a dashboard only when none is
live — an already-running one absorbs the new dir automatically, whatever port
it is on. `--no-dashboard` opts a run out; non-interactive launches (CI, nohup)
register their dir but never spawn.

## Isolated mode

`--isolated` serves only the given dirs: no registry read or write, no
discovery claim, and launchers ignore the instance. Use it for focused views
(demos, debugging one run dir) or to keep a scratch dir out of the registry.

## Finding the live dashboard

The live port can differ from 7788 (a taken port bumps to the next free one),
so read the discovery file:

```bash
cat ~/.cache/prime-rl/dashboard/daemon.json   # {"pid": ..., "url": "http://localhost:<actual port>"}
curl -sf $(jq -r .url ~/.cache/prime-rl/dashboard/daemon.json)/api/runs > /dev/null && echo live
ps aux | grep PRL::Dashboard             # process title
```

Hand the researcher the `url` from `daemon.json`. Launcher logs also print it:
startup ends with a `Dashboard · <url>` banner. The auto-started instance logs
to `~/.cache/prime-rl/dashboard/daemon.log`.

## Stopping / restarting

```bash
kill $(jq -r .pid ~/.cache/prime-rl/dashboard/daemon.json)   # the discovered instance
pkill -f PRL::Dashboard                                 # every dashboard on the host
```

A clean exit releases `daemon.json`; a stale file from a dead process is taken
over by the next start. Killing a dashboard never affects runs (it only reads),
and killing a run never takes the dashboard down (it runs in its own session).
Restart by launching any run, or directly: `uv run dashboard`.

## Point the open dashboard

Use `POST /api/view` to show relevant run data in every connected dashboard
tab:

```bash
curl -sS -X POST $(jq -r .url ~/.cache/prime-rl/dashboard/daemon.json)/api/view \
  -H 'content-type: application/json' -d '{
    "run": "demo-rl", "tab": "traces",
    "step": 0, "kind": "train", "subset": "effective",
    "episode": "ep-s00-reverse-text-0",
    "highlight": [{"node": 3, "quote": "hint: reverse the words", "reason": "tool result the policy conditioned on"}]
  }'
```

`run` is required; other fields are optional and leave unspecified UI state
unchanged. For trace evidence, supply `step`, `kind`, and `subset` together and
address the episode by stable `id`. Use optional `trace` and `branch` indices
for multi-agent traces and `highlight` entries shaped as `{node, quote,
reason, field?}`. On `409`, tell the user to open the returned `url`; the stored
command applies when the tab connects.

The trace viewer's token-signal selector colors recorded tokens by advantage,
logprob, trainer logprob, entropy, mismatch KL, IPO stable mask, loss mask, or
content membership. Every episode is on disk once, in
`<run>/traces/step_<n>/{train,eval}.jsonl`; post-hoc facts arrive as
trace-update records in the sibling `annotations/{orchestrator,trainer}.jsonl`
(effective membership, advantages, trained step; trainer logprobs and
entropies) and are folded onto the traces at read time, so the overlays work in
both the `all` and `effective` views once the trainer has processed the step.

## Write a report only when asked

Create a report only when the user explicitly asks for one. Otherwise answer
normally; use `/api/view` when showing trace evidence would help.

Write requested reports to `<run>/reports/<slug>.md`, then POST `{"run": ...,
"tab": "report", "report": "<slug>"}`. Use Markdown with a frontmatter
`title` and one-line JSON citation definitions:

```markdown
---
title: Why does reward dip at step 4?
---

The dip is provider errors, not policy regression [^err].

[^err]: {"step": 4, "kind": "train", "subset": "all", "episode": "ep-...", "node": 0, "quote": "engine overloaded", "note": "The failed call that emptied this step's batch."}
```

The frontmatter `title` is rendered as the report H1; do not repeat it with a
Markdown `#` heading. The inline renderer supports only HTTP(S) and anchor
links. Relative Markdown links render as literal text, so identify local source
files with inline code paths or link to a supported HTTP endpoint.

Each citation requires `step`, `kind`, `subset`, `episode`, `quote`, and `note`.
Use the top-level rollout record `id` returned by the dashboard episode-list
API—not a nested `traces[*].id`, and never `line`. Copy a short, distinctive
quote exactly; matching is case-sensitive and whitespace-insensitive. Keep
`note` to 1–2 sentences explaining why the quote supports the claim.

Cite major empirical, comparative, and diagnostic conclusions, but not routine
explanation. Use exact trace citations for trajectory-level claims. For
aggregate statistics, identify the run and source file; do not imply that one
example trajectory proves the aggregate.

Use adjacent markers (`[^a] [^b]`) only when one claim genuinely depends on
distinct passages, such as a comparison or corroboration. Use one citation
when one passage is sufficient.

Optional fields are `run`, `trace`, `branch`, `node`, `field`, `prefix`, and
`suffix`. Use `field: "content"` or `"reasoning"` only to disambiguate message
parts. Use verbatim adjacent `prefix`/`suffix` only when a quote repeats.
Ambiguous or mismatched citations remain broken and do not navigate.

Use Markdown; raw HTML is escaped.

Before handoff, reload the report and verify through the dashboard API that
every referenced citation resolves uniquely to its episode, node, and quote.
Confirm zero broken citations, one rendered title, and no unsupported relative
links.
