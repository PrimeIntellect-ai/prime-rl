import time
from pathlib import Path

from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.orchestrator.clients import AdminPlane
from prime_rl.utils.logger import format_time, get_logger


class DraftLoader(Reconfigurable):
    """Keeps every policy engine on the configured draft and draft tokens per step. A change applies at the
    next policy weight sync, while the engines are paused. Drafts only change speed, never the sampled
    distribution, so a failed load restores the last good draft and the run continues; ``take_rejected``
    hands the failed draft to the orchestrator, which reverts the config. A failed switch retries at the
    next weight sync."""

    def __init__(
        self,
        admin_plane: AdminPlane,
        served: Path,
        draft: Path,
        num_speculative_tokens: int | None = None,
        launch_num_speculative_tokens: int | None = None,
    ):
        self.admin_plane = admin_plane
        self.served: Path | None = served
        """The draft every engine serves; None when a failed load left them uncertain."""
        self.good = served
        self.draft = draft
        self.rejected: Path | None = None
        self.loads = 0
        self.failures = 0
        self.load_time = 0.0
        self.num_speculative_tokens = num_speculative_tokens
        self.served_num_speculative_tokens: int | None = None
        """Draft tokens per step every engine runs; None for the launch count."""
        self.switch_failed = False
        self.launch_num_speculative_tokens = launch_num_speculative_tokens

    def reconfigure(self, update: ConfigUpdate) -> None:
        self.draft = update.config.speculative_decoding.draft
        self.num_speculative_tokens = update.config.speculative_decoding.num_speculative_tokens

    async def sync(self) -> None:
        if self.draft != self.served:
            await self.swap(self.draft)
        if self.switch_failed or self.num_speculative_tokens != self.served_num_speculative_tokens:
            await self.switch(self.num_speculative_tokens)

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

    async def switch(self, num: int | None) -> None:
        count = "the launch count of" if num is None else str(num)
        start = time.perf_counter()
        try:
            await self.admin_plane.set_num_speculative_tokens(num)
        except Exception as error:
            self.switch_failed = True
            get_logger().error(
                f"Failed to set {count} draft tokens on every engine, retrying at the next sync: {error!r}"
            )
            return
        self.switch_failed = False
        self.served_num_speculative_tokens = num
        get_logger().success(
            f"Set {count} draft tokens per step on every engine in {format_time(time.perf_counter() - start)}"
        )

    def take_rejected(self) -> Path | None:
        rejected, self.rejected = self.rejected, None
        return rejected

    def gauges(self) -> dict[str, float]:
        gauges = {
            "speculator/draft_loads": float(self.loads),
            "speculator/draft_load_failures": float(self.failures),
            "speculator/draft_load_time": self.load_time,
        }
        served = self.served_num_speculative_tokens
        count = self.launch_num_speculative_tokens if served is None else served
        if count is not None and not self.switch_failed:
            gauges["speculator/num_speculative_tokens"] = float(count)
        return gauges
