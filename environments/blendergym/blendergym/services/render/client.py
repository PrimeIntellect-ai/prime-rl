"""RenderClient: synchronous HTTP client for the Render Service.

Used by env workers to call the Render Service and reconstruct RenderResult.
Assembles stdout/stderr into blender.log format matching one-shot run_blender.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from ...render import RenderResult

logger = logging.getLogger(__name__)


class RenderClient:
    """Synchronous Render Service client for use in env worker processes."""

    def __init__(
        self,
        base_url: str = "http://localhost:8420",
        timeout_s: int = 600,
    ):
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s + 10)
        self._timeout_s = timeout_s

    def render(self, *, blend_file, code, output_dir) -> RenderResult:
        """Call Render Service, return RenderResult.

        stdout/stderr are assembled into blender.log format (=== STDOUT === /
        === STDERR ===) to maintain observability parity with one-shot
        run_blender().
        """
        try:
            resp = self._client.post(
                "/render",
                json={
                    "blend_file": str(blend_file),
                    "code": str(code),
                    "output_dir": str(output_dir),
                    "timeout": self._timeout_s,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            log_parts = []
            if data.get("stdout"):
                log_parts.append(f"=== STDOUT ===\n{data['stdout']}")
            if data.get("stderr"):
                log_parts.append(f"=== STDERR ===\n{data['stderr']}")
            combined_log = "\n".join(log_parts)

            return RenderResult(
                success=data["success"],
                image_paths=[Path(p) for p in data.get("image_paths", [])],
                stderr=combined_log,
                duration_s=data.get("duration_s", 0),
                timed_out=data.get("timed_out", False),
                gpu_id=data.get("gpu_id"),
                code_path=Path(data["code_path"]) if data.get("code_path") else None,
            )
        except httpx.HTTPError as e:
            logger.error("Render service error: %s", e)
            return RenderResult(
                success=False,
                image_paths=[],
                stderr=str(e),
                duration_s=0,
                timed_out=False,
            )

    def close(self) -> None:
        self._client.close()
