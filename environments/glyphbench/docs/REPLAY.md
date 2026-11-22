# Trajectory replay

`glyphbench replay` (also `gb replay`) shows saved observations alongside the
agent's generated reasoning, parsed actions, rewards, and optional memory.
It accepts Verifiers evaluation `results.jsonl` files and Pro-harness
`transcript_*.jsonl` files, or directory trees containing them.

![Replay UI from the paper's Miniatari BankHeist example.](assets/replay-ui.png)

## Quickstart

```bash
# List the model, task, and seed for every saved rollout.
uv run glyphbench replay path/to/runs --list

# Step through matching episodes.
uv run glyphbench replay path/to/runs \
  --env glyphbench/miniatari-bankheist-v0 --pause

# Play continuously with a longer delay between turns.
uv run glyphbench replay path/to/runs --delay 0.4
```

A single file can be passed in place of `path/to/runs`. When output is not a
terminal, replay uses a plain text view.

## Filters and playback

| Option | Effect |
|---|---|
| `--env`, `--suite`, `--model`, `--seed` | Filter by task ID, suite, model ID, or integer seed. Each option is repeatable. |
| `--episode N` | Play only the Nth matching rollout, counted from zero. |
| `--list` | Print the matching rollout index without playback. |
| `--pause` | Advance interactively. |
| `--delay N` | Seconds between turns in continuous playback; default `0.15`. |
| `--reasoning-lines N` | Limit the displayed reasoning; `0` disables this additional clipping. Panel space is still limited by terminal size. |

Different filter types combine with AND. Repeated values within a filter
combine with OR: `--suite minigrid --suite minihack --seed 42` selects seed 42
from either suite.

## Pause-mode keys

| Key | Action |
|---|---|
| `→` | Next turn. |
| `←` | Previous turn. |
| `q` | End this rollout and advance to the next match. |
| `s` | Open the full system prompt. |
| `r` | Open the current turn's full reasoning. |
| `a` | Open the parsed action and raw action response. |
| `l` | Open the full glyph legend. |
| `m` | Open previous and updated memory. |

Full panels open in `$PAGER`, which defaults to `less -R`. With the default
pager, press `q` to return to replay, `/` to search, and `g` or `G` to jump to
the beginning or end.

## Reading the panels

The header identifies the model, task, seed, and rollout return. The grid
shows the observation for the displayed decision. Separate panels show the
turn budget, HUD, legend, action, environment feedback, and available memory.
Long text is clipped to fit; use the pager keys for the full contents.

When per-turn reward metadata is available, feedback displays the actual step
reward. Older result files may have less detail. Parse and truncation metadata
produce warning indicators:

| Indicator | Meaning |
|---|---|
| `[forfeit]` | The action could not be parsed after any retries; the configured forfeit policy applied. |
| `[trunc-action]` | The action response reached its output-token limit. |
| `[trunc-memory]` | The memory response reached its output-token limit. |
| `[mem-parse-fail]` | Memory extraction failed and the previous memory was retained. |

See the [failure diagnostics](llm-agent-failure-modes.md) for the distinction
between output truncation, game timeouts, and context limits.

Replay uses the evaluation action parser and, when the environment is
available, its canonical action vocabulary. Malformed responses remain parse
failures rather than being silently repaired for display. The reasoning panel
supports both explicit `<think>` tags and responses whose chat template
supplied the opening tag.

For standalone trajectory files and glyph-based GIF export, see
[`scripts/replay_trajectory.py`](../scripts/README.md#demo--replay). Prime-RL
v1 token traces are inspected with `scripts/rl/audit_prime_rl_trace.py`.
