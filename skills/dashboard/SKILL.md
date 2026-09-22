---
name: dashboard
description: Find, start, use, and stop the local run dashboard for metrics, configs, traces, logs, and reports. Use when asked for its URL, to watch or inspect a run, to control the open dashboard, or to create a cited dashboard report explicitly requested by the user.
---

# Run dashboard

`uv sync --extra dashboard && uv run dashboard [output_dir ...]` (default `outputs/`, or `$PRL_OUTPUT_DIR`) serves a web UI at `http://localhost:7788`. It only reads run dirs, so it is safe against live runs and installs anywhere: GPU dependencies live behind the `gpu` extra.

What is not obvious from the UI:

- Overview and Metrics read the same `monitors/file/metrics.jsonl`. **Overview** is the curated view (the sections of the W&B overview: train, eval, stability, inference, performance). **Metrics** is every key in the file, one pane per key, sectioned by parent path (`Flat`) or as a tree of sections along the key path (`Nested`), with a regex filter; time-keyed rows (`inference/`, `concurrency/`, `dispatcher/`) chart against wall time.
- Config and Logs default to `latest (attempt <n>)`; select an attempt to inspect an earlier launch. The Config view shows a copyable launch command.
- Trace viewer views: **Transcript**, **Timeline** (wall-clock Gantt of prefix branches), **Replay** (the branch as a terminal session; response text is paced evenly across the recorded model-call span and labelled inferred; default 8×; **Skip inference** collapses model calls but keeps tool delays; **T** toggles thinking, never advertise Ctrl+T) and **Semantic** (`semantic_parents` as a causal graph of model calls; width reflects peak agent concurrency; unplaceable fragments show as unlinked). Traces without semantic parents disable Semantic.
- Live rollouts appear in the traces table with a pulsing dot and a phase badge, and open in the viewer, which follows them (pinned to the newest turn until you scroll up; the request in flight's uncommitted messages render dimmed as `awaiting model`). They come from `monitors/file/traces/live/*.jsonl`, polled once a second.
- For an eval run the overview tab shows one env at a time (filter → env, `include errors`): a block progress bar (one cell per expected episode, from `monitors/file/plan.json`), summary tiles (avg@k, pass@k, error and truncation rate coloured >10% yellow / >50% red, turns / branches, episode time, tokens, cost), env metrics as dot plots or beeswarms (hover a dot for its episode, click to open it; hover the background for the quantiles), and usage / timing composition panes (icicle of the mean split over per-episode strips; `other` is wall time no phase accounts for; hover a part to dim the rest; 40 strips by default, `show all` for every one).
- `view on platform` in the top bar links to the run on the Prime platform when `--monitors.prime` was on (`monitors/prime/run.json`); an eval's evaluation opens at epoch start, so the link is live from the first episode.
- An eval reads `completed` once its stream is sealed by a clean exit; a crashed or interrupted one reads `stopped` after its files go quiet.

Every dashboard instance serves the dirs it was started with plus every dir in the per-user registry (`~/.cache/prime-rl/dashboard/dirs.json`). Launchers register their output dir on every start and, in interactive sessions, auto-start a dashboard only when none is live; `--no-dashboard` opts a run out, non-interactive launches never spawn one. `--isolated` serves only the given dirs and ignores the registry.

## Finding, stopping, restarting

The port can differ from 7788 (a taken port bumps to the next free one), so read the discovery file and hand the researcher its `url`; launcher logs also end with a `Dashboard · <url>` banner.

```bash
cat ~/.cache/prime-rl/dashboard/daemon.json                   # {"pid": ..., "url": "http://localhost:<port>"}
kill $(jq -r .pid ~/.cache/prime-rl/dashboard/daemon.json)   # stop; the next launch (or `uv run dashboard`) restarts it
```

Killing a dashboard never affects runs and killing a run never takes the dashboard down. The daemon serves the code of the checkout that started it (`readlink /proc/<pid>/cwd`): after switching branches or worktrees, kill it so the next launch restarts it on the current code. Its log is `~/.cache/prime-rl/dashboard/daemon.log`.

## Point the open dashboard

`POST /api/view` shows run data in every connected tab. `run` is required; the other fields leave unspecified state alone. For trace evidence pass `step`, `kind` and `subset` together and address the episode by its stable `id` (`subset: "effective"` narrows the traces tab to the cohort shipped at that step); `highlight` entries are `{node, quote, reason, field?}`. On `409`, tell the user to open the returned `url`.

```bash
curl -sS -X POST $(jq -r .url ~/.cache/prime-rl/dashboard/daemon.json)/api/view \
  -H 'content-type: application/json' -d '{"run": "demo-rl", "tab": "traces", "step": 0, "kind": "train", "subset": "effective",
    "episode": "ep-s00-reverse-text-0", "highlight": [{"node": 3, "quote": "hint: reverse the words", "reason": "the tool result the policy conditioned on"}]}'
```

## Reports, only when asked

Write a requested report to `<run>/reports/<slug>.md`, then POST `{"run": ..., "tab": "report", "report": "<slug>"}`. Markdown with a frontmatter `title` (rendered as the H1, do not repeat it) and one-line JSON citations:

```markdown
The dip is provider errors, not policy regression [^err].

[^err]: {"step": 4, "kind": "train", "subset": "all", "episode": "ep-...", "node": 0, "quote": "engine overloaded", "note": "The failed call that emptied this step's batch."}
```

A citation needs `step`, `kind`, `subset`, `episode` (the top-level record `id`, never `line` or a `traces[*].id`), an exact short `quote` (case-sensitive, whitespace-insensitive) and a one-sentence `note`; optional `trace`, `branch`, `node`, `field`, `prefix`, `suffix` disambiguate. Cite empirical and diagnostic claims, not routine explanation; one citation per passage. Only HTTP(S) and anchor links render. Before handing over, verify through the API that every citation resolves.
