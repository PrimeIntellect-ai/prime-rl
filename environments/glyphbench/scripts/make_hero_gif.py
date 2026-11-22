#!/usr/bin/env python
"""Render the README hero GIF: a smooth, looping montage of a few cool envs.

For each env in HERO_ENVS we roll out a seeded random agent, render only the
[Grid] block (no header / Legend / HUD / Memory), normalize every frame onto a
shared dark canvas, hold the clip for ~`--hold` seconds, then cross-fade
("shaded" dissolve) into the next env. The final clip cross-fades back into the
first so the GIF loops seamlessly.

Usage:
    uv run --extra assets python scripts/make_hero_gif.py --output gifs/glyphbench__readme-hero.gif
    uv run --extra assets python scripts/make_hero_gif.py --hold 2.0 --fps 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from replay_trajectory import GLYPH_RGB

import glyphbench  # noqa: F401
from glyphbench.core import make_env

# A couple of visually distinct, "cool" envs spanning every suite. Envs that
# fail to load (missing optional deps) are skipped gracefully, so heavy suites
# (craftaxfull, nethack) are included on a best-effort basis.
HERO_ENVS: list[str] = [
    "glyphbench/minigrid-multiroom-n4-s5-v0",
    "glyphbench/minihack-corridor-r3-v0",
    "glyphbench/atari-mspacman-v0",
    "glyphbench/classics-snake-medium-v0",
    "glyphbench/craftax-choptrees-v0",
    "glyphbench/procgen-maze-v0",
    "glyphbench/miniatari-spaceinvaders-v0",
    "glyphbench/classics-tetris-v0",
    "glyphbench/craftaxfull-v0",
    "glyphbench/nethack-full-v0",
]

BG = (15, 15, 15)


def grid_lines(obs: str) -> list[str]:
    """Extract just the [Grid] block from an observation string."""
    lines: list[str] = []
    in_grid = False
    for line in obs.split("\n"):
        if line.startswith("[Grid]"):
            in_grid = True
            continue
        if in_grid:
            if line.startswith("["):
                break
            lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def rollout_grids(env_id: str, seed: int, frames_wanted: int) -> list[list[str]]:
    """Seeded random rollout -> list of [Grid] blocks (one per step)."""
    env = make_env(env_id)
    obs, _ = env.reset(seed)
    grids: list[list[str]] = []
    g = grid_lines(obs)
    if g:
        grids.append(g)
    steps = 0
    # Roll out long enough to fill the hold window; cap so dead envs terminate.
    while steps < frames_wanted * 3:
        action = int(env.rng.integers(0, env.action_spec.n))
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        g = grid_lines(obs)
        if g:
            grids.append(g)
        if terminated or truncated:
            break
    env.close()
    return grids


def render_grid(grid: list[str], font, char_w: int, line_h: int):
    """Render a single [Grid] block to a tight RGB image."""
    from PIL import Image, ImageDraw

    max_cols = max((len(line) for line in grid), default=1)
    img = Image.new("RGB", (max_cols * char_w + 16, len(grid) * line_h + 16), BG)
    draw = ImageDraw.Draw(img)
    for row, line in enumerate(grid):
        y = 8 + row * line_h
        for col, ch in enumerate(line):
            color = GLYPH_RGB.get(ch, (200, 200, 200))
            draw.text((8 + col * char_w, y), ch, font=font, fill=color)
    return img


def fit_onto_canvas(img, canvas_w: int, canvas_h: int):
    """Scale `img` to fit inside the canvas (preserving aspect) and center it."""
    from PIL import Image

    box_w, box_h = canvas_w - 24, canvas_h - 24
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    canvas.paste(resized, ((canvas_w - new_size[0]) // 2, (canvas_h - new_size[1]) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("gifs/glyphbench__readme-hero.gif"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds per env clip")
    parser.add_argument("--transition", type=float, default=0.6, help="Cross-fade seconds")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--envs", nargs="*", default=None, help="Override HERO_ENVS")
    args = parser.parse_args()

    try:
        from PIL import Image, ImageFont
    except ImportError:
        sys.exit("ERROR: Pillow required. Install with: uv add --extra assets Pillow")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", args.font_size)
    except OSError:
        font = ImageFont.load_default()
    char_w = font.getbbox("M")[2]
    line_h = args.font_size + 4

    hold_frames = max(1, round(args.hold * args.fps))
    trans_frames = max(1, round(args.transition * args.fps))
    env_ids = args.envs if args.envs else HERO_ENVS

    # Build one normalized clip (list of canvas frames) per env.
    clips: list[list] = []
    for env_id in env_ids:
        try:
            grids = rollout_grids(env_id, args.seed, hold_frames)
        except Exception as exc:  # missing optional deps, etc.
            print(f"skip {env_id}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if not grids:
            print(f"skip {env_id}: empty grid")
            continue
        canvases = [
            fit_onto_canvas(render_grid(g, font, char_w, line_h), args.width, args.height)
            for g in grids
        ]
        # Stretch/sample the clip to exactly `hold_frames` (loop short clips,
        # subsample long ones) so every env holds for the same duration.
        clip = [canvases[round(i * (len(canvases) - 1) / max(1, hold_frames - 1))]
                for i in range(hold_frames)] if len(canvases) > 1 else canvases * hold_frames
        clips.append(clip)
        print(f"ok   {env_id}: {len(grids)} grids -> {hold_frames}f clip")

    if not clips:
        sys.exit("No clips rendered.")

    # Assemble: hold clip, then cross-fade into the next (wrapping to the first).
    frames: list = []
    n = len(clips)
    for i in range(n):
        frames.extend(clips[i])
        nxt = clips[(i + 1) % n]
        a, b = clips[i][-1], nxt[0]
        for t in range(1, trans_frames + 1):
            frames.append(Image.blend(a, b, t / (trans_frames + 1)))

    duration = round(1000 / args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(args.output),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    kb = args.output.stat().st_size / 1024
    print(f"\nHero GIF saved: {args.output}  ({len(frames)} frames, {len(clips)} envs, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
