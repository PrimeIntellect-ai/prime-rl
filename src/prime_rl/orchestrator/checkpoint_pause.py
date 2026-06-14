from __future__ import annotations

import asyncio
from pathlib import Path

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.checkpoint_pause import (
    PAUSED,
    RELEASE,
    RESUMED,
    get_pending_pause_requests,
    read_marker,
    write_marker,
)
from prime_rl.utils.client import InferencePool, pause_inference, resume_inference
from prime_rl.utils.logger import get_logger


class CheckpointPauseWatcher:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        inference: InferencePool,
        update_lock: asyncio.Lock,
        poll_interval: float = 0.5,
    ) -> None:
        self.config = config
        # Trainer writes pause markers under the shared experiment root; the
        # orchestrator output_dir is the per-run child (e.g. run_default).
        self.output_dir = config.output_dir.parent
        self.inference = inference
        self.update_lock = update_lock
        self.poll_interval = poll_interval
        self.task: asyncio.Task | None = None
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                requests = get_pending_pause_requests(self.output_dir)
                if requests:
                    step, step_dir, request_id = requests[0]
                    await self.handle_request(step, step_dir, request_id)
                    continue
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def handle_request(self, step: int, step_dir: Path, request_id: str) -> None:
        async with self.update_lock:
            await pause_inference(
                self.inference.admin_clients,
                reason=f"Pausing inference for trainer checkpoint at step {step}",
            )
            write_marker(step_dir, PAUSED, request_id)
            try:
                await self.wait_for_release(step_dir, request_id)
            finally:
                await resume_inference(self.inference.admin_clients)
                write_marker(step_dir, RESUMED, request_id)
                get_logger().info(f"Resumed inference after trainer checkpoint at step {step}")

    async def wait_for_release(self, step_dir: Path, request_id: str) -> None:
        release_path = step_dir / RELEASE
        while not self.stopped.is_set():
            if read_marker(release_path) == request_id:
                return
            await asyncio.sleep(self.poll_interval)
