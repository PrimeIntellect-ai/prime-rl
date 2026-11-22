"""Native Craftax pixel rendering and replay GIF helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_native_craftax_frame(game: object) -> Any | None:
    """Render the live fork state with Craftax's native pixel renderer."""

    state = getattr(game, "_state", None)
    if state is None:
        return None
    try:
        import numpy as np
        from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
        from craftax.craftax.renderer import render_craftax_pixels

        pixels = np.asarray(render_craftax_pixels(state, BLOCK_PIXEL_SIZE_IMG))
        return np.clip(pixels, 0, 255).astype("uint8")
    except Exception:
        return None


def write_gif(frames: list[Any], path: str | Path, *, fps: int = 8) -> None:
    """Write an animated GIF using a stable per-frame duration."""

    if not frames:
        raise ValueError("cannot write a GIF without frames")
    import imageio.v2 as imageio

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000.0 / max(1, int(fps))))
    imageio.mimsave(target, frames, duration=duration_ms, loop=0)


__all__ = ["render_native_craftax_frame", "write_gif"]
