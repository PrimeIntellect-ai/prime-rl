import time
from pathlib import Path

from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.orchestrator.clients import AdminPlane
from prime_rl.utils.logger import format_time, get_logger


class DraftLoader(Reconfigurable):
    """Keeps every policy engine on the configured draft and drafting state. A change applies at the next
    policy weight sync, while the engines are paused. Drafts only change speed, never the sampled
    distribution, so a failed load restores the last good draft and the run continues; ``take_rejected``
    hands the failed draft to the orchestrator, which reverts the config. A failed switch retries at the
    next weight sync."""

    def __init__(self, admin_plane: AdminPlane, served: Path, draft: Path, enabled: bool = True):
        self.admin_plane = admin_plane
        self.served: Path | None = served
        """The draft every engine serves; None when a failed load left them uncertain."""
        self.good = served
        self.draft = draft
        self.rejected: Path | None = None
        self.loads = 0
        self.failures = 0
        self.load_time = 0.0
        self.enabled = enabled
        self.served_enabled: bool | None = True
        """Whether every engine drafts (engines launch drafting); None after a failed switch."""

    def reconfigure(self, update: ConfigUpdate) -> None:
        self.draft = update.config.speculative_decoding.draft
        self.enabled = update.config.speculative_decoding.enabled

    async def sync(self) -> None:
        if self.draft != self.served:
            await self.swap(self.draft)
        if self.enabled != self.served_enabled:
            await self.switch(self.enabled)

    async def swap(self, draft: Path) -> None:
        if await self.load(draft):
            self.good = draft
            return
        self.draft = self.good
        if draft != self.good:
            self.rejected = draft
            await self.load(self.good)

    async def load(self, draft: Path) -> bool:
        start = time.perf_counter()
        try:
            await self.admin_plane.update_draft_weights(draft)
            digests = set(await self.admin_plane.draft_digests())
            if len(digests) != 1:
                raise RuntimeError(f"engines report {len(digests)} different draft fingerprints")
        except Exception as error:
            self.served = None
            self.failures += 1
            get_logger().error(f"Failed to load draft {draft} on every engine: {error!r}")
            return False
        self.served = draft
        self.loads += 1
        self.load_time = time.perf_counter() - start
        get_logger().success(f"Loaded draft {draft} on every engine in {format_time(self.load_time)}")
        return True

    async def switch(self, enabled: bool) -> None:
        state = "on" if enabled else "off"
        try:
            await self.admin_plane.set_speculative_decoding(enabled)
        except Exception as error:
            self.served_enabled = None
            get_logger().error(f"Failed to turn drafting {state} on every engine, retrying at the next sync: {error!r}")
            return
        self.served_enabled = enabled
        get_logger().success(f"Turned drafting {state} on every engine")

    def take_rejected(self) -> Path | None:
        rejected, self.rejected = self.rejected, None
        return rejected

    def gauges(self) -> dict[str, float]:
        return {
            "speculator/draft_loads": float(self.loads),
            "speculator/draft_load_failures": float(self.failures),
            "speculator/draft_load_time": self.load_time,
            "speculator/enabled": float(self.served_enabled is True),
        }
