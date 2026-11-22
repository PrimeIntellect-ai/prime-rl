#!/usr/bin/env python3
"""Interactive curses-based player for any glyphbench environment.

Play any environment with keyboard controls. The action menu shows
numbered actions; press the corresponding number key to act. Use [ and ]
to browse action pages when an environment has more than ten actions.

Usage:
    uv run python scripts/play_interactive.py glyphbench/craftaxfull-v0
    uv run python scripts/play_interactive.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import curses

from terminal_colors import char_attr, init_colors

import glyphbench  # noqa: F401 — trigger env registration
from glyphbench.core import ActionSpec, BaseGlyphEnv, make_env


def _parse_obs(obs: str) -> dict[str, str]:
    """Parse a rendered observation into sections."""
    sections: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []

    for line in obs.split("\n"):
        if line.startswith("[") and line.endswith("]"):
            if current_key:
                sections[current_key] = "\n".join(current_lines)
            current_key = line[1:-1]
            current_lines = []
        else:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines)
    return sections


def _draw_grid(win: curses.window, grid_str: str, y0: int, x0: int) -> None:
    """Draw the grid with per-character coloring."""
    max_y, max_x = win.getmaxyx()
    for row_idx, line in enumerate(grid_str.split("\n")):
        y = y0 + row_idx
        if y >= max_y - 1:
            break
        for col_idx, ch in enumerate(line):
            x = x0 + col_idx
            if x >= max_x - 1:
                break
            with contextlib.suppress(curses.error):
                win.addch(y, x, ch, char_attr(ch))


def _draw_text(
    win: curses.window,
    text: str,
    y0: int,
    x0: int,
    attr: int = 0,
    max_width: int = 0,
) -> int:
    """Draw wrapped text, return number of lines used."""
    max_y, max_x = win.getmaxyx()
    if max_width <= 0:
        max_width = max_x - x0 - 1
    if max_width <= 0:
        return 0
    lines_used = 0
    for line in text.split("\n"):
        if not line:
            lines_used += 1
            continue
        while line:
            chunk = line[:max_width]
            line = line[max_width:]
            y = y0 + lines_used
            if y >= max_y - 1:
                return lines_used
            with contextlib.suppress(curses.error):
                win.addnstr(y, x0, chunk, max_width, attr)
            lines_used += 1
    return lines_used


# Common keyboard shortcuts for movement actions
_KEY_TO_ACTION: dict[int, tuple[str, ...]] = {
    curses.KEY_UP: ("MOVE_UP", "UP", "MOVE_N", "MOVE_FORWARD"),
    curses.KEY_DOWN: ("MOVE_DOWN", "DOWN", "MOVE_S"),
    curses.KEY_LEFT: ("MOVE_LEFT", "LEFT", "MOVE_W", "TURN_LEFT"),
    curses.KEY_RIGHT: ("MOVE_RIGHT", "RIGHT", "MOVE_E", "TURN_RIGHT"),
    ord("w"): ("MOVE_UP", "UP", "MOVE_N", "MOVE_FORWARD"),
    ord("s"): ("MOVE_DOWN", "DOWN", "MOVE_S"),
    ord("a"): ("MOVE_LEFT", "LEFT", "MOVE_W", "TURN_LEFT"),
    ord("d"): ("MOVE_RIGHT", "RIGHT", "MOVE_E", "TURN_RIGHT"),
    # MiniGrid: forward/left/right/toggle/pickup/drop
    ord("f"): ("MOVE_FORWARD",),
    ord("t"): ("TOGGLE",),
    ord("p"): ("PICKUP",),
    ord("x"): ("DROP",),
    # MiniGrid turn
    ord(","): ("TURN_LEFT", "LEFT"),
    ord("."): ("TURN_RIGHT", "RIGHT"),
    # General
    ord(" "): ("NOOP",),
    ord("e"): ("DO", "INTERACT"),
    # Minihack directional
    ord("h"): ("MOVE_W",),
    ord("j"): ("MOVE_S",),
    ord("k"): ("MOVE_N",),
    ord("l"): ("MOVE_E",),
}


def _action_for_key(key: int, spec: ActionSpec, page: int) -> int | None:
    if ord("A") <= key <= ord("Z"):
        key += ord("a") - ord("A")
    for name in _KEY_TO_ACTION.get(key, ()):
        with contextlib.suppress(KeyError):
            return spec.index_of(name)
    if ord("0") <= key <= ord("9"):
        index = page * 10 + key - ord("0")
        if index < spec.n:
            return index
    return None


def _play(stdscr: curses.window, args: argparse.Namespace, env: BaseGlyphEnv) -> None:

    # Curses setup
    init_colors()
    with contextlib.suppress(curses.error):
        curses.curs_set(0)
    stdscr.keypad(True)

    # Env setup
    action_names = env.action_spec.names
    n_actions = env.action_spec.n
    action_page = 0
    n_pages = (n_actions + 9) // 10

    obs, info = env.reset(args.seed)
    total_reward = 0.0
    step_num = 0
    last_action = "(reset)"
    last_reward = 0.0
    done = False

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        if max_y < 10 or max_x < 40:
            with contextlib.suppress(curses.error):
                stdscr.addnstr(0, 0, "Resize terminal; q quits", max(0, max_x - 1))
            stdscr.refresh()
            if stdscr.getch() in (ord("q"), ord("Q")):
                break
            continue

        sections = _parse_obs(obs)
        grid_str = sections.get("Grid", "")
        grid_lines = grid_str.split("\n")
        grid_h = len(grid_lines)
        grid_w = max((len(line) for line in grid_lines), default=0)

        # Layout: grid left, info right
        panel_x = min(grid_w + 3, max_x * 2 // 3)
        panel_w = max_x - panel_x - 1

        # Title bar
        title = f" {args.env_id} | seed={args.seed} | step={step_num} "
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                0, 0, title.center(max_x - 1),
                curses.A_REVERSE | curses.A_BOLD,
            )

        # Grid
        _draw_grid(stdscr, grid_str, 2, 1)

        # Right panel: HUD
        panel_y = 2
        hud = sections.get("HUD", "")
        if hud and panel_w > 10:
            with contextlib.suppress(curses.error):
                stdscr.addstr(panel_y, panel_x, " HUD ", curses.A_REVERSE)
            panel_y += 1
            panel_y += _draw_text(
                stdscr, hud.strip(), panel_y, panel_x,
                max_width=panel_w,
            )
            panel_y += 1

        # Right panel: Legend
        legend = sections.get("Legend", "")
        if legend and panel_w > 10 and panel_y < max_y - 5:
            with contextlib.suppress(curses.error):
                stdscr.addstr(panel_y, panel_x, " Legend ", curses.A_REVERSE)
            panel_y += 1
            for legend_line in legend.strip().split("\n"):
                if panel_y >= max_y - min(n_actions, 10) - 4:
                    break
                if len(legend_line) >= 1:
                    sym = legend_line[0]
                    try:
                        stdscr.addch(panel_y, panel_x, sym, char_attr(sym))
                        stdscr.addnstr(
                            panel_y, panel_x + 1,
                            legend_line[1:min(len(legend_line), panel_w)],
                            panel_w - 1,
                        )
                    except curses.error:
                        pass
                panel_y += 1

        # Action + reward info below grid
        info_y = max(grid_h + 3, 2)
        action_line = f" Last: {last_action}"
        reward_line = f"R={last_reward:+.2f}  Total={total_reward:.2f}"
        try:
            stdscr.addstr(
                info_y, 1, action_line,
                curses.color_pair(3) | curses.A_BOLD,
            )
            rattr = (
                curses.color_pair(2) | curses.A_BOLD if last_reward > 0
                else curses.color_pair(1) | curses.A_BOLD if last_reward < 0
                else curses.A_DIM
            )
            r_x = max(len(action_line) + 3, max_x - len(reward_line) - 2)
            stdscr.addstr(info_y, r_x, reward_line, rattr)
        except curses.error:
            pass

        # Message
        message = sections.get("Message", "").strip()
        if message:
            with contextlib.suppress(curses.error):
                stdscr.addnstr(
                    info_y + 1, 1, f" {message}",
                    max_x - 2, curses.color_pair(3),
                )

        # Action menu below grid/info
        menu_y = info_y + 3
        if menu_y < max_y - 2 and not done:
            with contextlib.suppress(curses.error):
                stdscr.addstr(
                    menu_y, 1, f" Actions {action_page + 1}/{n_pages} ([/]=page) ",
                    curses.A_REVERSE,
                )
            menu_y += 1
            for idx in range(action_page * 10, min((action_page + 1) * 10, n_actions)):
                if menu_y >= max_y - 1:
                    break
                label = f" {idx % 10}: {action_names[idx]}"
                with contextlib.suppress(curses.error):
                    stdscr.addnstr(
                        menu_y, 1, label, max_x - 2,
                        curses.A_BOLD,
                    )
                menu_y += 1

        # Status bar
        if done:
            status = " DONE | q=quit r=reset "
            attr = curses.A_REVERSE | curses.color_pair(1)
        else:
            status = (
                " arrows/wasd=move  0-9=action  [/]=page  "
                "space=noop  e=DO  q=quit  r=reset "
            )
            attr = curses.A_REVERSE
        with contextlib.suppress(curses.error):
            stdscr.addstr(max_y - 1, 0, status.ljust(max_x - 1), attr)

        stdscr.refresh()

        # Wait for input
        key = stdscr.getch()
        if key == ord("q") or key == ord("Q"):
            break
        if key == ord("r") or key == ord("R"):
            obs, info = env.reset(args.seed)
            total_reward = 0.0
            step_num = 0
            last_action = "(reset)"
            last_reward = 0.0
            done = False
            continue

        if done:
            continue

        if key in (ord("["), curses.KEY_PPAGE):
            action_page = (action_page - 1) % n_pages
            continue
        if key in (ord("]"), curses.KEY_NPAGE):
            action_page = (action_page + 1) % n_pages
            continue

        action_idx = _action_for_key(key, env.action_spec, action_page)
        if action_idx is None:
            continue

        last_action = action_names[action_idx]
        obs, reward, terminated, truncated, info = env.step(action_idx)
        last_reward = reward
        total_reward += reward
        step_num += 1

        if terminated or truncated:
            done = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive player for glyphbench envs")
    parser.add_argument("env_id", help="Env ID")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, help="Override the environment's turn limit")
    args = parser.parse_args()
    if args.max_turns is not None and args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    kwargs = {} if args.max_turns is None else {"max_turns": args.max_turns}
    with contextlib.closing(make_env(args.env_id, **kwargs)) as env:
        curses.wrapper(_play, args, env)


if __name__ == "__main__":
    main()
