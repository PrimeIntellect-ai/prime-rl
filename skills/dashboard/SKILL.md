---
name: dashboard
description: Find, start, use, and stop the local run dashboard — the web UI for run output dirs (metrics, configs, traces, logs, reports). Use when asked for the dashboard URL, when a run needs watching in the browser, when a dashboard must be restarted or killed, or when answering questions about a run's traces — write a report with citations and point the user's open tab at the evidence via POST /api/view.
---

# Run dashboard

`uv sync --extra dashboard && uv run dashboard [output_dir ...]` (default
`outputs/`, or `$PRL_OUTPUT_DIR` if set) serves a web UI at `http://localhost:7788`. It only reads run
dirs — safe against live runs — and installs anywhere (cluster head node,
laptop against a mounted outputs dir): GPU dependencies live behind the
`gpu` extra.

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

## Pointing the open dashboard (view commands)

When you answer a question about a run, don't just describe an episode — show
it. `POST /api/view` with an on-disk address navigates every connected browser
tab there:

```bash
curl -sS -X POST $(jq -r .url ~/.cache/prime-rl/dashboard/daemon.json)/api/view \
  -H 'content-type: application/json' -d '{
    "run": "demo-rl", "tab": "traces",
    "step": 0, "kind": "train", "subset": "effective",
    "episode": "ep-s00-reverse-text-0",
    "highlight": [{"node": 3, "quote": "hint: reverse the words", "reason": "tool result the policy conditioned on"}]
  }'
```

Fields (all optional except `run`; missing fields leave that part of the UI
alone):

| field | meaning |
|---|---|
| `run` | run id as `/api/runs` lists it |
| `tab` | `metrics` / `config` / `traces` / `logs` / `report` |
| `step`, `kind`, `subset` | `rollouts/step_N/{train,eval}/{all,effective}` (eval-only runs: `0/eval/all`) |
| `episode` | episode `id` from the traces file (`line` is a positional fallback) |
| `trace`, `branch` | multi-agent seat index and branch leaf (`-1` = concatenated) |
| `report` | report file under `<run>/reports/` to open on the report tab |
| `highlight` | list of `{node, quote, reason}`: node index into `trace.nodes`, verbatim quote to mark, optional callout text |

The server validates the address against the filesystem (unknown run/episode →
404, so you cannot point at nothing). `409` means the command was stored but no
tab is connected — tell the user to open the `url` from the response body and
the command applies when they do. `GET /api/view` returns the last command.
Errors and broken episodes usually live only in the `all` subset (`effective`
drops them).

## Writing reports (the report tab)

For a question that deserves a written answer, put it in
`<run>/reports/<slug>.md` and POST `{"run": ..., "tab": "report", "report":
"<slug>"}`. The dashboard renders it live (poll-based — appending while you
write streams to the reader). Format: markdown with a `title:` frontmatter
line, citing evidence with `[^id]` markers defined anywhere in the file as one
JSON object per line:

```markdown
---
title: Why does reward dip at step 4?
---

The dip is provider errors, not policy regression [^err].

[^err]: {"step": 4, "kind": "train", "subset": "all", "episode": "ep-...", "node": 0, "quote": "engine overloaded", "note": "the failed call"}
```

A citation is a view command plus a verbatim quote: same address fields as
`/api/view`, plus `quote` (copied **exactly** from the trace — message content
and `reasoning_content` both work as sources — the dashboard
re-checks every quote against the files and renders the chip green only when it
matches; a paraphrase shows the reader a red "broken" chip) and an optional
`note` shown as a callout. Clicking a chip peeks the cited node inline;
"open in traces" jumps to the full trace with the quote highlighted. Supported
markdown: headings, lists, tables, fenced code, blockquotes, bold/italic/code,
links. Raw HTML is escaped, not rendered.
