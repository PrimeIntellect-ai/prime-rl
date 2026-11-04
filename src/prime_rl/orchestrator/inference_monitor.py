"""InferenceMonitor: probes every policy inference server so the orchestrator keeps training when one
fails. A server that stops answering is lost: it leaves weight updates, and an update that needs every
server is stopped. A server that answers again rejoins at the next weight update."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from prime_rl.configs.shared import InferenceMonitorConfig
from prime_rl.orchestrator.clients import AdminPlane, Engine
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.logger import get_logger


class InferenceMonitor:
    def __init__(
        self, admin_plane: AdminPlane, config: InferenceMonitorConfig, served_policy: Callable[[], int]
    ) -> None:
        self.admin_plane = admin_plane
        self.config = config
        self.served_policy = served_policy
        """The policy inference serves, passed with every probe: a server behind it stops serving."""
        self.task: asyncio.Task | None = None
        # A failed admin call waits until this monitor could have judged its server. A server this
        # monitor could not reach for as long stops taking requests by itself (its lease lapses).
        self.lease = config.lost_after * (config.probe_interval + config.probe_timeout)
        admin_plane.loss_verdict_s = self.lease

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name="inference-monitor")

    async def stop(self) -> None:
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def run(self) -> None:
        while True:
            await asyncio.gather(*(self.probe(engine) for engine in self.admin_plane.engines))
            await asyncio.sleep(self.config.probe_interval)

    async def probe(self, engine: Engine) -> None:
        try:
            response = await engine.client.get(
                "/ready",
                params={"policy": self.served_policy(), "lease": self.lease},
                timeout=self.config.probe_timeout,
            )
            ready = response.json()
            instance_id = ready["instance_id"]
        except Exception as error:
            if engine.status == "lost":
                return
            engine.failed_probes += 1
            get_logger().debug(f"Probe {engine.failed_probes} of inference server {engine.url} failed: {error!r}")
            if engine.failed_probes >= self.config.lost_after:
                self.admin_plane.mark_lost(engine, f"{engine.failed_probes} probes failed")
            return
        if ready.get("engine_dead"):
            # Not a missed probe: the server cannot recover in place, so its updates stop at once.
            self.admin_plane.mark_lost(engine, "its engine core died")
            return
        engine.failed_probes = 0
        if engine.instance_id is None:
            engine.instance_id = instance_id
        replaced = instance_id != engine.instance_id
        if replaced:
            engine.instance_id = instance_id
            self.admin_plane.mark_lost(engine, "replaced")
        if engine.status != "lost":
            return
        # Neither serves a stale policy: a replacement starts without one, and a server that came back
        # stopped taking requests on this probe if it fell behind. It joins the next update.
        self.admin_plane.mark_joining(engine)
