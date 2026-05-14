"""ScoreClient: async HTTP client for the Score Service.

Used by rubric.py to call the Score Service for CLIP similarity scoring.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class ScoreClient:
    """Async Score Service client for use in rubric."""

    def __init__(
        self,
        base_url: str = "http://localhost:8421",
        timeout_s: int = 30,
    ):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def score(self, image_a: str, image_b: str) -> float:
        """Call Score Service, return CLIP cosine similarity. Returns 0.0 on error."""
        try:
            resp = await self._client.post(
                "/score",
                json={"image_a": image_a, "image_b": image_b},
            )
            resp.raise_for_status()
            return resp.json()["similarity"]
        except httpx.HTTPError as e:
            logger.error("Score service error: %s", e)
            return 0.0

    async def close(self) -> None:
        await self._client.aclose()
