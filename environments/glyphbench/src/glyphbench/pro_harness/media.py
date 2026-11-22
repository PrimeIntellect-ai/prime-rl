"""Streaming native-pixel episode recording for long-horizon harnesses."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


def frame_to_png_data_url(frame: Any) -> str:
    """Encode one native RGB frame for a Responses API ``input_image``.

    PNG is lossless, keeping the observation identical to the environment's
    native renderer rather than routing the model through the MP4 recording.
    """
    import imageio.v3 as imageio

    payload = imageio.imwrite("<bytes>", frame, extension=".png")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class PixelVideoRecorder:
    """Append HxWx3 frames to a fragmented MP4 without retaining them.

    The fragmented container remains inspectable after periodic disk flushes
    and bounds memory independently of the native episode horizon.
    """

    def __init__(
        self,
        path: Path,
        *,
        fps: int = 8,
        max_frames: int | None = None,
    ) -> None:
        self.path = path
        self.fps = max(1, int(fps))
        self.max_frames = max_frames
        self.frames = 0
        self.error: str | None = None
        self._writer: Any | None = None

    def append(self, frame: Any) -> None:
        if self.error is not None:
            return
        if self.max_frames is not None and self.frames >= self.max_frames:
            return
        try:
            if self._writer is None:
                import imageio.v2 as imageio

                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = imageio.get_writer(
                    self.path,
                    fps=self.fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=None,
                    output_params=["-movflags", "+frag_keyframe+empty_moov"],
                )
            self._writer.append_data(frame)
            self.frames += 1
        except Exception as exc:  # noqa: BLE001 - media must not abort an eval
            self.error = f"{type(exc).__name__}: {exc}"
            self.close()

    def close(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"

    @property
    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def __enter__(self) -> PixelVideoRecorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
