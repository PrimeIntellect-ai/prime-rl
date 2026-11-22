#!/usr/bin/env python
"""Demo all GlyphBench envs with the same TUI layout as ``gb replay``.

Random-agent rollouts are rendered in the rich panel layout used by the
replay command — header bar, collapsible system prompt panel, grid +
side panels — so the demo and the replay share one visual language.

Usage:
    uv run python scripts/demo_all_envs.py
    uv run python scripts/demo_all_envs.py --suite minigrid
    uv run python scripts/demo_all_envs.py --env glyphbench/atari-pong-v0 --pause
    uv run python scripts/demo_all_envs.py --list

Pause-mode hotkeys (mirrors the replay):

    →  / any other key   advance one step (step env if past history end)
    ←                    move backward through the in-episode history
    s                    open full system prompt in $PAGER
    l                    open full per-turn legend in $PAGER
    a                    open full action list in $PAGER (demo extra)
    q  /  n              quit current env, advance to the next one
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time
from collections import deque

import glyphbench  # noqa: F401  — registers all envs

# Reuse the exact helpers gb replay uses so the look matches.
from glyphbench.cli import (
    _clip_to_lines,
    _extract_block,
    _extract_grid,
    _read_one_key,
    _restore_terminal_sane,
    _scale_grid,
    _show_in_pager,
)
from glyphbench.core import all_glyphbench_env_ids, make_env

# ---------------------------------------------------------------------------
# Frame builder (mirrors _render_rollout_rich's per-turn layout, minus the
# memory + reasoning panels which don't apply to a random agent).
# ---------------------------------------------------------------------------


def _build_frame(
    *,
    console_height: int,
    env_id: str,
    seed: int,
    t_idx: int,
    n_turns_so_far: int,
    is_history_view: bool,
    sys_text: str,
    sys_cap: int,
    obs_text: str,
    action_name: str | None,
    reward: float,
    total_reward: float,
    terminated: bool,
    truncated: bool,
    recent_actions: list[str],
):
    """Construct the per-step rich Layout. Keeps the gb-replay grammar:

    LEFT:   grid (top, sized to content) + recent-actions panel (fills rest)
    RIGHT:  step, HUD, legend, action, env feedback
    """
    from rich.console import Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    line_width_cap = 200
    legend_cap = max(4, console_height // 5)
    hud_cap = max(2, console_height // 8)
    actions_cap = max(4, console_height // 4)

    grid = _extract_grid(obs_text) or "(no [Grid] block)"
    grid = _scale_grid(grid, 2, 1)
    hud = _extract_block(obs_text, "[HUD]") or ""
    legend = _extract_block(obs_text, "[Legend]") or ""
    message = _extract_block(obs_text, "[Message]") or ""

    # Header bar (same shape as replay's, minus model_id + parse flags).
    hdr = Table.grid(expand=True, padding=(0, 1))
    hdr.add_column(justify="left", ratio=2)
    hdr.add_column(justify="right", ratio=1)
    left_pieces = [
        Text.assemble(
            ("demo", "bold cyan"), " · ", (env_id, "bold white"),
        ),
    ]
    if is_history_view:
        left_pieces.append(Text(" "))
        left_pieces.append(Text(" history view ", style="bold black on grey70"))
    right_pieces = Text.assemble(
        ("turn ", "dim"), (f"{t_idx}/{n_turns_so_far}", "bold magenta"),
        "   ", ("seed=", "dim"), (str(seed), "yellow"),
        "   ", ("episodic reward=", "dim"),
        (f"{total_reward:+.3f}", "bold green"),
    )
    hdr.add_row(Group(*left_pieces), right_pieces)

    rollout_state: list = []
    if terminated:
        rollout_state.append(Text(" terminated ", style="bold black on green"))
    if truncated:
        rollout_state.append(Text(" truncated ", style="bold black on red"))
    if rollout_state:
        sub = Table.grid(expand=False, padding=(0, 1))
        for _s in rollout_state:
            sub.add_column()
        sub.add_row(*rollout_state)
        header_block: object = Group(hdr, sub)
    else:
        header_block = hdr

    # ---- LEFT column: grid panel + recent-actions panel ----
    grid_panel = Panel(
        Text(grid, style="bright_white"),
        title="grid", title_align="left",
        border_style="bright_white", padding=(0, 1),
    )
    grid_panel_height = grid.count("\n") + 1 + 2  # +2 for borders

    if recent_actions:
        actions_text = "\n".join(reversed(recent_actions[-actions_cap:]))
        actions_text = _clip_to_lines(
            actions_text, actions_cap, mode="head",
            max_line_width=line_width_cap,
        )
        actions_panel = Panel(
            Text(actions_text, style="grey70", overflow="fold"),
            title="recent actions (newest first)", title_align="left",
            border_style="grey50", padding=(0, 1),
        )
    else:
        actions_panel = Panel(
            Text("(no actions yet)", style="dim grey50"),
            title="recent actions", title_align="left",
            border_style="grey50", padding=(0, 1),
        )

    # ---- RIGHT column entries ----
    right_entries: list[tuple[object, int]] = []

    # Split HUD into a dedicated `step` panel + residual.
    step_text = ""
    hud_residual = hud.strip() if hud else ""
    if hud_residual:
        m_step = re.search(r"Step:\s*\d+\s*/\s*\d+", hud_residual)
        if m_step:
            step_text = m_step.group(0)
            hud_residual = (
                hud_residual[: m_step.start()] + hud_residual[m_step.end():]
            )
            hud_residual = re.sub(r"\s{2,}", "    ", hud_residual)
            hud_residual = hud_residual.strip(" \t\n,;|")
    if step_text:
        right_entries.append((
            Panel(Text(step_text, style="bold yellow"),
                  title="step", title_align="left",
                  border_style="yellow", padding=(0, 1)),
            3,
        ))
    if hud_residual:
        hud_lines = max(2, min(hud_cap, hud_residual.count("\n") + 1))
        hud_txt = _clip_to_lines(hud_residual, hud_lines, mode="tail",
                                 max_line_width=line_width_cap)
        right_entries.append((
            Panel(Text(hud_txt, style="cyan", overflow="fold"),
                  title="HUD", title_align="left",
                  border_style="cyan", padding=(0, 1)),
            hud_lines + 2,
        ))
    if legend:
        legend_txt = _clip_to_lines(
            legend.strip(), legend_cap, mode="head",
            max_line_width=line_width_cap,
        )
        right_entries.append((
            Panel(Text(legend_txt, style="bright_cyan", overflow="fold"),
                  title="legend", title_align="left",
                  border_style="bright_cyan", padding=(0, 1)),
            legend_cap + 2,
        ))
    action_color = "green" if action_name else "grey50"
    right_entries.append((
        Panel(
            Text(action_name or "(none — initial state)",
                 style=f"bold {action_color}"),
            title="action", title_align="left",
            border_style=action_color, padding=(0, 1),
        ),
        3,
    ))
    if action_name is not None or message:
        feedback_lines: list = []
        if action_name is not None:
            feedback_lines.append(
                Text.assemble(("reward: ", "bold"),
                              (f"{reward:+.3f}", "yellow"))
            )
        if message:
            feedback_lines.append(
                Text.assemble(("message: ", "bold"),
                              (message.strip(), "magenta"))
            )
        right_entries.append((
            Panel(Group(*feedback_lines), title="env feedback",
                  title_align="left", border_style="yellow", padding=(0, 1)),
            len(feedback_lines) + 2,
        ))

    # ---- compose ----
    body = Layout(name="body")
    body.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    body["left"].split_column(
        Layout(grid_panel, name="grid", size=grid_panel_height),
        Layout(actions_panel, name="actions"),
    )
    body["right"].split_column(
        *[Layout(panel, size=size) for panel, size in right_entries]
    )

    root = Layout(name="root")
    sub_layouts = [
        Layout(Panel(header_block, border_style="cyan", padding=(0, 1)),
               name="hdr", size=4 + (1 if rollout_state else 0)),
    ]
    if sys_text:
        clipped_sys = _clip_to_lines(
            sys_text, sys_cap, mode="head",
            max_line_width=line_width_cap,
        )
        sys_panel = Panel(
            Text(clipped_sys, style="dim", overflow="fold"),
            title="system prompt", title_align="left",
            border_style="grey39", padding=(0, 1),
        )
        sub_layouts.append(Layout(sys_panel, name="sys", size=sys_cap + 2))
    sub_layouts.append(body)
    root.split_column(*sub_layouts)
    return root


# ---------------------------------------------------------------------------
# Per-env driver
# ---------------------------------------------------------------------------


def _run_env(
    env_id: str,
    *,
    seed: int,
    delay: float,
    pause: bool,
    actions_history_len: int = 32,
) -> tuple[float, str]:
    """Run one env with a uniform-random agent. Returns (return, exit_reason).

    ``exit_reason`` is one of: "terminated", "truncated", "quit_user",
    "error".
    """
    from rich.console import Console
    from rich.live import Live

    console = Console()
    is_tty = console.is_terminal

    env = make_env(env_id)
    obs_text, info = env.reset(int(seed))

    sys_text = ""
    try:
        sys_text = env.system_prompt().rstrip()
    except Exception:
        sys_text = ""

    console_h = console.size.height or 40
    sys_cap = max(4, console_h // 8)

    action_names = env.action_spec.names
    history: list[dict] = [{
        "obs_text": obs_text,
        "action_name": None,
        "reward": 0.0,
        "total_reward": 0.0,
        "terminated": False,
        "truncated": False,
    }]
    recent_actions: deque[str] = deque(maxlen=actions_history_len)

    def step_one() -> dict:
        last = history[-1]
        if last["terminated"] or last["truncated"]:
            return last
        action_idx = int(env.rng.integers(0, env.action_spec.n))
        action_name = action_names[action_idx]
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        total = last["total_reward"] + float(reward)
        entry = {
            "obs_text": next_obs,
            "action_name": action_name,
            "reward": float(reward),
            "total_reward": total,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        history.append(entry)
        recent_actions.append(action_name)
        return entry

    def make_layout(t_idx: int):
        entry = history[t_idx - 1]
        return _build_frame(
            console_height=console_h,
            env_id=env_id,
            seed=seed,
            t_idx=t_idx,
            n_turns_so_far=len(history),
            is_history_view=(t_idx < len(history)),
            sys_text=sys_text,
            sys_cap=sys_cap,
            obs_text=entry["obs_text"],
            action_name=entry["action_name"],
            reward=entry["reward"],
            total_reward=entry["total_reward"],
            terminated=entry["terminated"],
            truncated=entry["truncated"],
            recent_actions=list(recent_actions),
        )

    exit_reason = "terminated"
    try:
        with Live(
            make_layout(1), console=console, auto_refresh=False,
            screen=is_tty, transient=False, redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            if pause and is_tty:
                t_idx = 1
                while True:
                    live.update(make_layout(t_idx), refresh=True)
                    ch = _read_one_key()
                    pager_text: str | None = None
                    if ch in ("q", "n"):
                        exit_reason = "quit_user"
                        break
                    elif ch == "s":
                        pager_text = sys_text or "(no system prompt)"
                    elif ch == "l":
                        full_legend = (
                            _extract_block(history[t_idx - 1]["obs_text"],
                                           "[Legend]") or ""
                        ).strip()
                        pager_text = full_legend or "(no legend in this turn)"
                    elif ch == "a":
                        pager_text = env.action_spec.render_for_prompt()
                    if pager_text is not None:
                        live.stop()
                        _show_in_pager(pager_text)
                        live.start()
                        live.update(make_layout(t_idx), refresh=True)
                        continue
                    # Navigation. ← previous, → / any other = next.
                    if ch == "\x1b[D":
                        t_idx = max(1, t_idx - 1)
                    elif ch == "\x1b[C":
                        if t_idx == len(history):
                            entry = step_one()
                            if entry is history[-1]:
                                t_idx = len(history)
                        else:
                            t_idx = min(len(history), t_idx + 1)
                    else:
                        # Default: next.
                        if t_idx == len(history):
                            entry = history[-1]
                            if entry["terminated"] or entry["truncated"]:
                                exit_reason = "terminated" if entry["terminated"] else "truncated"
                                break
                            step_one()
                            t_idx = len(history)
                        else:
                            t_idx = min(len(history), t_idx + 1)
            else:
                # Continuous mode — drive until terminal.
                t_idx = 1
                live.update(make_layout(t_idx), refresh=True)
                while True:
                    last = history[-1]
                    if last["terminated"]:
                        exit_reason = "terminated"
                        break
                    if last["truncated"]:
                        exit_reason = "truncated"
                        break
                    step_one()
                    t_idx = len(history)
                    live.update(make_layout(t_idx), refresh=True)
                    if delay > 0:
                        time.sleep(delay)
                # Hold the final frame so the viewer sees the outcome.
                if is_tty:
                    time.sleep(min(2.0, max(0.5, delay * 8)))
    finally:
        with contextlib.suppress(Exception):
            env.close()
        _restore_terminal_sane()

    return history[-1]["total_reward"], exit_reason


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Demo every GlyphBench env with a uniform-random agent, "
            "rendered in the gb-replay panel layout."
        )
    )
    parser.add_argument(
        "--suite", type=str,
        help="Filter env ids by suite (substring match: minigrid, atari, …).",
    )
    parser.add_argument(
        "--env", type=str,
        help="Run a single env (full id, e.g. glyphbench/craftaxfull-v0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed used for env.reset() (default: 42).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.05,
        help="Seconds between steps in continuous mode (default: 0.05).",
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Pause for a keypress between steps (s/l/a pager hotkeys, ←/→ navigate, q/n advance env).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the matched env ids and exit (no demo).",
    )
    args = parser.parse_args()

    if args.env:
        env_ids = [args.env]
    else:
        env_ids = all_glyphbench_env_ids()
        env_ids = [e for e in env_ids if "dummy" not in e]
        if args.suite:
            env_ids = [e for e in env_ids if args.suite in e]

    if not env_ids:
        print("(no envs matched the filter)", file=sys.stderr)
        return 1

    if args.list:
        for eid in env_ids:
            print(eid)
        return 0

    print(f"Demoing {len(env_ids)} env(s) with seed {args.seed}.\n")

    try:
        for i, env_id in enumerate(env_ids, 1):
            print(f"[{i}/{len(env_ids)}] {env_id}")
            try:
                ret, reason = _run_env(
                    env_id, seed=args.seed,
                    delay=args.delay, pause=args.pause,
                )
                print(f"  return={ret:+.3f}  exit={reason}")
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                return 130
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
            if not args.pause and i < len(env_ids):
                time.sleep(0.3)
    finally:
        _restore_terminal_sane()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
