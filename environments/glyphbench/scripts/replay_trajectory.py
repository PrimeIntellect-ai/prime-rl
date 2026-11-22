#!/usr/bin/env python
"""Replay trajectory files with canonical glyph + color terminal rendering.

Usage:
    uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl
    uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --delay 0.2
    uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --gif output.gif
    uv run python scripts/replay_trajectory.py path/to/trajectories/  # replay all in dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI color mapping by symbol category
# ---------------------------------------------------------------------------
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"

SYMBOL_COLORS = {
    # Player
    "^": "\033[1;32m",   # bold green
    "v": "\033[1;32m",
    "<": "\033[1;32m",
    ">": "\033[1;32m",
    "@": "\033[1;32m",
    # Walls
    "#": "\033[90m",     # dark gray
    "█": "\033[90m",
    # Floor
    ".": "\033[38;5;237m",  # very dark gray
    "·": "\033[38;5;237m",
    " ": "",
    # Goal
    "G": "\033[1;33m",   # bold yellow
    "★": "\033[1;33m",
    "*": "\033[1;33m",
    # Danger
    "X": "\033[1;31m",   # bold red
    "!": "\033[1;31m",
    # Items
    "K": "\033[1;35m",   # bold magenta (keys)
    "D": "\033[33m",     # yellow (doors)
    "▣": "\033[33m",
    "+": "\033[1;36m",   # bold cyan
    # Water
    "~": "\033[34m",     # blue
    "≈": "\033[34m",
    # Default
    "o": "\033[37m",
    "O": "\033[37m",
}

GLYPH_RGB: dict[str, tuple[int, int, int]] = {
    "#": (80, 80, 80),
    "█": (80, 80, 80),
    ".": (50, 50, 50),
    "·": (50, 50, 50),
    "^": (0, 220, 0), "v": (0, 220, 0), "<": (0, 220, 0), ">": (0, 220, 0),
    "↑": (0, 220, 0), "↓": (0, 220, 0), "←": (0, 220, 0), "→": (0, 220, 0),
    "@": (0, 220, 0),
    "G": (255, 220, 0), "*": (255, 220, 0),
    "★": (255, 220, 0),
    "X": (255, 50, 50), "!": (255, 50, 50),
    "K": (200, 50, 200),
    "D": (200, 180, 0),
    "▣": (200, 180, 0),
    "~": (50, 100, 255),
    "≈": (50, 100, 255),
}

DEFAULT_COLOR = "\033[37m"  # white


def colorize_char(ch: str) -> str:
    """Color a single canonical glyph without changing the glyph itself."""
    color = SYMBOL_COLORS.get(ch, DEFAULT_COLOR)
    if color:
        return f"{color}{ch}{ANSI_RESET}"
    return ch


def render_colored(text: str) -> str:
    """Apply color rendering to an observation string.

    Only colors characters inside the [Grid] section.
    Leaves [Legend], [HUD], and other sections as-is but with subtle coloring.
    The glyphs themselves are never remapped; GIFs and terminal replay must
    match the canonical observation text exactly.
    """
    lines = text.split("\n")
    output = []
    in_grid = False

    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            in_grid = line == "[Grid]"
            output.append(f"{ANSI_BOLD}\033[36m{line}{ANSI_RESET}")
        elif in_grid and line and not line.startswith("["):
            colored = "".join(colorize_char(ch) for ch in line)
            output.append(colored)
        elif line.startswith("["):
            output.append(f"{ANSI_BOLD}\033[36m{line}{ANSI_RESET}")
            in_grid = False
        else:
            output.append(f"\033[37m{line}{ANSI_RESET}")

    return "\n".join(output)


def _step_memory(step: dict) -> str:
    """Return stored memory for either legacy step logs or verifiers extras."""
    extras = step.get("extras")
    extras_memory = (
        extras.get("glyphbench_memory") if isinstance(extras, dict) else None
    )
    candidates = [
        step.get("memory"),
        step.get("stored_memory"),
        step.get("glyphbench_memory"),
        extras_memory,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, dict):
            for key in ("stored_memory", "parsed_memory", "memory"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


CURSOR_HOME = "\033[H"
ERASE_TO_END = "\033[J"
CLEAR_SCREEN = "\033[2J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory(path: Path) -> list[dict]:
    """Load a .jsonl trajectory file."""
    steps = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


# ---------------------------------------------------------------------------
# Terminal replay
# ---------------------------------------------------------------------------

def _step_frame(step: dict, path: Path) -> str:
    turn = step["turn"]
    action = step["action_name"]
    reward = step["reward"]
    cum_reward = step["cumulative_reward"]
    terminated = step["terminated"]
    truncated = step["truncated"]
    parse_failed = step.get("parse_failed", False)

    status = ""
    if terminated:
        status = " \033[1;32m[TERMINATED]\033[0m"
    elif truncated:
        status = " \033[1;33m[TRUNCATED]\033[0m"

    lines = [
        f"\033[1m{'=' * 70}\033[0m",
        (
            f"\033[1m{path.name}\033[0m  "
            f"Turn {turn}  "
            f"Action: \033[1;36m{action}\033[0m  "
            f"Reward: {reward:+.2f}  "
            f"Total: {cum_reward:+.2f}"
            f"{status}"
        ),
    ]
    if parse_failed:
        lines.append("\033[1;31m  [PARSE FAILED - fell back to NOOP]\033[0m")
    observation = step["observation"]
    memory = _step_memory(step)
    if memory:
        observation = f"{observation}\n\n[Memory]\n{memory}"

    lines.extend([
        f"\033[1m{'=' * 70}\033[0m",
        "",
        render_colored(observation),
        "",
    ])
    return "\n".join(lines)


def replay_terminal(steps: list[dict], path: Path, delay: float = 0.1) -> None:
    """Replay a trajectory in the terminal with color.

    Flicker-free: we assemble the entire frame, then emit it in one write
    with cursor-home + erase-below. That avoids the blanked-screen window
    `os.system("clear")` leaves between frames.
    """
    sys.stdout.write(CLEAR_SCREEN + HIDE_CURSOR)
    sys.stdout.flush()
    try:
        for i, step in enumerate(steps):
            frame = _step_frame(step, path)
            sys.stdout.write(CURSOR_HOME + frame + ERASE_TO_END)
            sys.stdout.flush()
            if i < len(steps) - 1:
                time.sleep(delay)
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    print(f"\n\033[1mEpisode finished: {len(steps)} steps, return={steps[-1]['cumulative_reward']:+.3f}\033[0m")


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def export_gif(
    steps: list[dict],
    output_path: Path,
    font_size: int = 14,
    width: int = 800,
    grid_only: bool = False,
) -> None:
    """Export a trajectory as a GIF of rendered terminal frames.

    Requires Pillow (pip install Pillow).

    If ``grid_only`` is True, only the [Grid] block is rendered (no header,
    legend, HUD, or memory). The output is a tight crop around the grid only.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow required for GIF export. Install with: uv add Pillow")
        sys.exit(1)

    # Try to load a monospace font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    char_w = font.getbbox("M")[2]
    line_h = font_size + 4

    frames: list[Image.Image] = []

    for step in steps:
        obs = step["observation"]
        memory = _step_memory(step)
        if memory and not grid_only:
            obs = f"{obs}\n\n[Memory]\n{memory}"
        turn = step["turn"]
        action = step["action_name"]
        cum_reward = step["cumulative_reward"]

        if grid_only:
            # Extract only the [Grid] block (no header, no Legend/HUD/Memory).
            grid_lines: list[str] = []
            in_grid = False
            for line in obs.split("\n"):
                if line.startswith("[Grid]"):
                    in_grid = True
                    continue
                if in_grid:
                    if line.startswith("["):
                        break
                    grid_lines.append(line)
            # Drop trailing blank lines.
            while grid_lines and not grid_lines[-1].strip():
                grid_lines.pop()
            if not grid_lines:
                continue

            max_cols = max(len(line) for line in grid_lines)
            img_w = max_cols * char_w + 20
            img_h = len(grid_lines) * line_h + 20

            img = Image.new("RGB", (img_w, img_h), (15, 15, 15))
            draw = ImageDraw.Draw(img)
            for row, line in enumerate(grid_lines):
                y = 10 + row * line_h
                for col, ch in enumerate(line):
                    x = 10 + col * char_w
                    color = GLYPH_RGB.get(ch, (200, 200, 200))
                    draw.text((x, y), ch, font=font, fill=color)
            frames.append(img)
            continue

        # Build header + observation lines
        header = f"Turn {turn}  Action: {action}  Return: {cum_reward:+.2f}"
        obs_lines = obs.split("\n")
        all_lines = [header, "=" * 60] + obs_lines

        max_cols = max(len(line) for line in all_lines) if all_lines else 40
        img_w = max(width, max_cols * char_w + 20)
        img_h = len(all_lines) * line_h + 20

        img = Image.new("RGB", (img_w, img_h), (15, 15, 15))
        draw = ImageDraw.Draw(img)

        in_grid = False
        for row, line in enumerate(all_lines):
            y = 10 + row * line_h
            if line.startswith("[Grid]"):
                in_grid = True
                draw.text((10, y), line, font=font, fill=(0, 200, 200))
                continue
            elif line.startswith("["):
                in_grid = line == "[Grid]"
                draw.text((10, y), line, font=font, fill=(0, 200, 200))
                continue

            if row < 2:
                draw.text((10, y), line, font=font, fill=(200, 200, 200))
            elif in_grid:
                for col, ch in enumerate(line):
                    x = 10 + col * char_w
                    color = GLYPH_RGB.get(ch, (200, 200, 200))
                    draw.text((x, y), ch, font=font, fill=color)
            else:
                draw.text((10, y), line, font=font, fill=(180, 180, 180))

        frames.append(img)

    if frames:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            str(output_path),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        print(f"GIF saved: {output_path} ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay GlyphBench trajectories with color")
    parser.add_argument("path", type=Path, help="Trajectory .jsonl file or directory")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between frames")
    parser.add_argument("--gif", type=Path, help="Export as GIF instead of terminal replay")
    parser.add_argument("--font-size", type=int, default=14, help="Font size for GIF")
    args = parser.parse_args()

    if args.path.is_dir():
        files = sorted(args.path.glob("*.jsonl"))
        if not files:
            print(f"No .jsonl files in {args.path}")
            sys.exit(1)
    else:
        files = [args.path]

    for f in files:
        steps = load_trajectory(f)
        if not steps:
            print(f"Empty trajectory: {f}")
            continue

        if args.gif:
            gif_path = args.gif if len(files) == 1 else args.gif.parent / f"{f.stem}.gif"
            export_gif(steps, gif_path, font_size=args.font_size)
        else:
            replay_terminal(steps, f, delay=args.delay)
            if len(files) > 1:
                input("\nPress Enter for next trajectory...")


if __name__ == "__main__":
    main()
