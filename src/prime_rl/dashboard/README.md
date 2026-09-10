# Dashboard

`uv run dashboard [output_dir ...]` — a local web dashboard for run output
directories (metrics, resolved configs, rollout traces, cited reports,
component logs) at
http://localhost:7788. The trace viewer includes transcript, synchronized
agent timeline, and terminal replay views; timeline activities open the
corresponding transcript call. Needs the `dashboard` extra. Every instance also serves
the dirs registered by launchers in `~/.cache/prime-rl/dashboard/dirs.json`
(one dashboard per host per user); `--isolated` serves only the given dirs and
skips the registry. See `skills/dashboard/SKILL.md` for discovery,
kill/restart commands, and the local view-command/report contract.

The Config and Logs views keep each launch attempt available. The Config view
shows a copyable command above the launch TOML or resolved JSON. Both views
open at `latest (attempt <n>)` and let you select an earlier attempt.

GPU dependencies live behind the `gpu` extra, so this works anywhere — a
cluster head node, a laptop against a mounted outputs dir:

```bash
uv sync --extra dashboard && uv run dashboard [output_dir ...]
```

The trace viewer's **Messages** mode keeps structured `message.content` and
`trace.tools` visibly separate. **Rendered** decodes each selected branch's
recorded post-renderer `token_ids` as one sequence, retaining special tokens;
it never reconstructs a chat template. The recorded IDs remain the source of
truth, and the viewer reports when IDs, the renderer model, or its tokenizer
are unavailable. Text, advantage, logprob, mask, and content signals apply to
both views; Rendered with Text uses the exact full-sequence decode.

**Replay** plays the selected trace branch as a terminal session. Model
responses, tool commands, and tool results follow their recorded wall-clock
spans, with play/pause, seeking, restart, and speed controls. Providers persist
the full model-call span rather than timestamps for individual tokens, so text
is revealed evenly across that measured span and the UI labels the token cadence
as inferred. Replay defaults to 8× for long coding-agent waits while preserving
their relative timing; 1× restores exact wall time, and speeds up to 32× are
available. In-flight calls show measured progress and recorded output usage even
when the response contains only a tool call. Events from older traces without
timing stay in order and are marked untimed. Prompt context committed at response
time appears at the linked model call's start rather than creating a false blank
delay. **Skip inference** makes every model response immediate while retaining
the recorded delays between tool commands and their outputs. Recorded model
thinking is visible by default and can be toggled from the controls or with
**T** while Replay is open. **Top** (or **Home**) stops following the newest
line and scrolls to the beginning; **Live** (or **End**) resumes following output.

This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests.
The integration suite covers it end to end (`tests/integration/dashboard_smoke.py`
runs after every integration test).
