"""Engines that pull their weights from ModelExpress (``weight_broadcast.pull``): the engine moves itself onto each
policy version the trainer publishes, instead of waiting for the orchestrator to pause it and push one.

The puller polls ModelExpress for the newest READY version, published after the one it serves. While the engines
serve, every worker stages that version on a background thread; once all have, the engines pause with the running
requests kept, install it, label the tokens sampled from then on with its step and resume. Versions published in
between are skipped. The orchestrator only reads the version each engine serves (``GET /weight_version``).

On several nodes every DP rank is its own server. The ranks of an expert-parallel DP group share every forward pass,
so they switch together: the group's first rank pulls for all of them, through their admin routes. The pause is
vLLM's data-parallel pause, which stops every rank of the group at the same step, so no forward pass mixes versions.
Groups switch independently, each as soon as it is ready.

A failed stage leaves the live weights untouched: the engines keep serving and the puller tries the newest version
again. A failed install may leave them partly written, so the engines stay paused and retry with the newest version
until one installs; past ``timeout`` the server exits rather than serve mixed weights.
"""

import asyncio
import os
import random
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from modelexpress_rl import ModelExpressControlClient, WeightVersion, WeightVersionState
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger("vllm.inference.vllm.weight_pull")

POLL_INTERVAL_S = 0.2
STATUS_INTERVAL_S = 0.05
MAX_RETRY_INTERVAL_S = 5.0


@dataclass(frozen=True)
class PullConfig:
    """What a pulling server needs: the ModelExpress server, the model whose versions it follows, the staging
    settings of every worker it moves, and how long a paused engine retries a failed install."""

    host: str
    port: int
    model_name: str
    staging_buffer_bytes: int | None
    staging_buffers_count: int
    timeout: int

    @property
    def server_url(self) -> str:
        return f"{self.host}:{self.port}"


def mesh_key(version: WeightVersion) -> str:
    """The trainer a version comes from: workers record how one trainer's tensors map onto theirs."""
    return f"{version.trainer_mesh_id}:{version.trainer_mesh_generation}"


class Engines(Protocol):
    """The engines a puller moves onto each version, together."""

    async def ready(self) -> None:
        """Wait until every engine serves."""

    async def init(self, payload: dict) -> None:
        """Connect every worker to ModelExpress (``init_broadcaster``)."""

    async def can_stage_live(self, mesh: str) -> bool:
        """Whether every worker can stage a version of trainer ``mesh`` while the engines serve."""

    async def stage(self, version_uid: str, live: bool) -> None:
        """Stage a version on every worker, while serving if ``live``. Raises if any worker failed."""

    async def discard(self) -> None:
        """Release what every worker staged."""

    async def pause(self) -> None:
        """Pause the engines at the next step, keeping the running requests."""

    async def apply(self, version_uid: str, mesh: str, step: int) -> None:
        """Install the staged version on every worker of the paused engines and label the tokens sampled from then
        on with ``step``."""

    async def resume(self) -> None: ...

    async def served(self) -> tuple[int | None, str | None]:
        """The step label and the ModelExpress version every engine serves; None where they disagree, or before an
        install."""


async def on_every_worker(client: EngineClient, method: str, *args) -> list:
    """Every worker's result of ``method``, from every engine of this server. vLLM's ``collective_rpc`` reaches every
    engine but returns only the first engine's results."""
    per_engine = await client.engine_core.call_utility_all_async("collective_rpc", method, None, args, None)
    return [result for results in per_engine for result in results]


class LocalEngine:
    """The engines of this server, reached in process: one, or every data-parallel engine of an internal-LB server."""

    def __init__(self, client: EngineClient):
        self.client = client

    async def ready(self) -> None:
        pass

    async def init(self, payload: dict) -> None:
        # Reaches every engine of the server
        await self.client.collective_rpc("init_broadcaster", kwargs=payload)

    async def can_stage_live(self, mesh: str) -> bool:
        return all(await on_every_worker(self.client, "modelexpress_can_stage_live", mesh))

    async def stage(self, version_uid: str, live: bool) -> None:
        await on_every_worker(self.client, "modelexpress_stage", version_uid, live)
        if not live:
            return
        while True:
            statuses = await on_every_worker(self.client, "modelexpress_stage_status")
            if "staging" not in statuses:
                break
            await asyncio.sleep(STATUS_INTERVAL_S)
        if failed := [status for status in statuses if status != "staged"]:
            raise RuntimeError(f"{len(failed)} of {len(statuses)} workers did not stage {version_uid}: {failed[0]}")

    async def discard(self) -> None:
        await on_every_worker(self.client, "modelexpress_discard")

    async def pause(self) -> None:
        await self.client.pause_generation(mode="keep", clear_cache=False)

    async def apply(self, version_uid: str, mesh: str, step: int) -> None:
        await on_every_worker(self.client, "modelexpress_apply", version_uid, mesh)
        await self.client.update_weight_version(str(step))

    async def resume(self) -> None:
        await self.client.resume_generation()

    async def served(self) -> tuple[int | None, str | None]:
        try:
            label = await self.client.get_weight_version()
        except RuntimeError:
            # vLLM requires every engine to report the same label: they disagree during an install
            label = None
        uids = set(await on_every_worker(self.client, "modelexpress_served"))
        version = int(label) if label is not None and label.isdigit() else None
        return version, uids.pop() if len(uids) == 1 else None


class EngineGroup:
    """The ranks of one DP group (``weight_broadcast.group``), this server's own included, reached through their
    admin routes."""

    def __init__(self, admin_urls: list[str], admin_token_var: str):
        from prime_rl.configs.shared import VLLMClientConfig
        from prime_rl.orchestrator.clients import AdminPlane

        # The engines' client config, as the orchestrator's: their API key, and the admin token to send
        config = VLLMClientConfig(base_url=admin_urls[0], admin_base_url=admin_urls, admin_token_var=admin_token_var)
        self.admin_plane = AdminPlane(config)

    @property
    def clients(self):
        return self.admin_plane.clients

    async def _post_all(self, path: str, json: dict | None = None, timeout: float = 3600) -> list[dict]:
        async def post(client) -> dict:
            response = await client.post(path, json=json, timeout=timeout)
            if response.is_error:
                raise RuntimeError(f"{client.base_url}{path} failed ({response.status_code}): {response.text[:2000]}")
            return response.json()

        results = await asyncio.gather(*map(post, self.clients), return_exceptions=True)
        if errors := [result for result in results if isinstance(result, BaseException)]:
            raise errors[0]
        return results

    async def ready(self) -> None:
        from prime_rl.orchestrator.clients import check_health

        await check_health(self.clients)

    async def init(self, payload: dict) -> None:
        await self._post_all("/init_broadcaster", json=payload)

    async def can_stage_live(self, mesh: str) -> bool:
        results = await self._post_all("/modelexpress/can_stage_live", json={"mesh": mesh})
        return all(result["live"] for result in results)

    async def stage(self, version_uid: str, live: bool) -> None:
        await self._post_all("/modelexpress/stage", json={"version_uid": version_uid, "live": live})

    async def discard(self) -> None:
        await self._post_all("/modelexpress/discard")

    async def pause(self) -> None:
        await self._post_all("/pause")

    async def apply(self, version_uid: str, mesh: str, step: int) -> None:
        await self._post_all("/modelexpress/apply", json={"version_uid": version_uid, "mesh": mesh, "step": step})

    async def resume(self) -> None:
        await self._post_all("/resume")

    async def served(self) -> tuple[int | None, str | None]:
        async def get(client) -> tuple[int | None, str | None]:
            response = await client.get("/weight_version", params={"uid": "true"}, timeout=60)
            response.raise_for_status()
            return response.json()["version"], response.json()["uid"]

        served = set(await asyncio.gather(*map(get, self.clients)))
        return served.pop() if len(served) == 1 else (None, None)


class WeightPuller:
    def __init__(self, engines: Engines, config: PullConfig):
        self.engines = engines
        self.config = config
        self.served: WeightVersion | None = None
        """The version the engines serve, None until they installed one."""
        self.switches = 0
        self.last_pause_s = 0.0
        self.lock = asyncio.Lock()
        """Held through every switch. Anything else that moves these engines (a checkpoint swap) takes it too."""
        self._control: ModelExpressControlClient | None = None

    async def run(self) -> None:
        logger.info(f"Pulling policy weights of {self.config.model_name} from ModelExpress at {self.config.server_url}")
        await self.engines.ready()
        payload = {
            "host": self.config.host,
            "port": self.config.port,
            "timeout": self.config.timeout,
            "session_id": uuid.uuid4().hex,
            "staging_buffer_bytes": self.config.staging_buffer_bytes,
            "staging_buffers_count": self.config.staging_buffers_count,
            "model_name": self.config.model_name,
        }
        while True:
            # The ModelExpress server may come up after the engines
            try:
                await self.engines.init(payload)
                break
            except Exception as error:
                logger.warning(f"Connecting the engines to ModelExpress failed; retrying: {error!r}")
                await asyncio.sleep(MAX_RETRY_INTERVAL_S)
        self._control = ModelExpressControlClient.connect(server_url=self.config.server_url)
        retry_interval = POLL_INTERVAL_S
        while True:
            try:
                target = await self.newest(after=self.served)
                if target is not None:
                    await self.switch(target)
                    retry_interval = POLL_INTERVAL_S
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning(f"Pulling policy weights failed, the engines keep serving; retrying: {error!r}")
                await asyncio.sleep(random.uniform(retry_interval / 2, retry_interval))
                retry_interval = min(2 * retry_interval, MAX_RETRY_INTERVAL_S)
                continue
            await asyncio.sleep(POLL_INTERVAL_S)

    async def newest(self, after: WeightVersion | None) -> WeightVersion | None:
        """The newest READY version, if it was published after ``after``. Publication order decides, not the step:
        a trainer restarted from a checkpoint publishes lower steps than the one it replaced."""
        assert self._control is not None
        versions = await asyncio.to_thread(self._control.list_weight_versions, model_name=self.config.model_name)
        ready = [
            version
            for version in versions
            if version.state is WeightVersionState.READY and version.version_number is not None
        ]
        newest = max(ready, key=lambda version: version.created_at_unix_ms, default=None)
        if newest is None or (after is not None and newest.created_at_unix_ms <= after.created_at_unix_ms):
            return None
        return newest

    async def switch(self, target: WeightVersion) -> None:
        """Move the engines onto ``target``: stage it while they serve, then install it in a short pause."""
        async with self.lock:
            await self._switch(target)

    async def _switch(self, target: WeightVersion) -> None:
        start = time.perf_counter()
        live = await self.engines.can_stage_live(mesh_key(target))
        if live:
            try:
                await self.engines.stage(target.version_id, live=True)
            except BaseException:
                await self.engines.discard()
                raise
        staged = time.perf_counter()
        try:
            await self.engines.pause()
        except BaseException:
            # Nothing was installed: whatever paused may serve again
            await self.engines.discard()
            await self.engines.resume()
            raise
        paused = time.perf_counter()
        deadline = time.monotonic() + self.config.timeout
        # Whether an install started, so that the weights may be partly written
        written = False
        while True:
            try:
                if not live:
                    try:
                        await self.engines.stage(target.version_id, live=False)
                    except Exception:
                        if written:
                            raise
                        # Nothing was installed: the engines serve their weights again and the puller retries later
                        await self.engines.discard()
                        await self.engines.resume()
                        raise
                written = True
                await self.engines.apply(target.version_id, mesh_key(target), target.version_number)
                break
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not written:
                    raise
                # The weights may be partly written: stay paused, and install the newest version from scratch
                logger.exception(f"Installing policy v{target.version_number} failed; retrying while paused")
                try:
                    await self.engines.discard()
                except Exception as discard_error:
                    logger.warning(f"Discarding the staged version failed: {discard_error!r}")
                if time.monotonic() > deadline:
                    exit_server(f"No policy version installed within {self.config.timeout}s after {error!r}")
                    # Stay paused until the server is gone
                    await asyncio.Event().wait()
                await asyncio.sleep(1.0)
                try:
                    target = await self.newest(after=None) or target
                except Exception as list_error:
                    logger.warning(f"Listing policy versions failed; retrying v{target.version_number}: {list_error!r}")
                live = False
        installed = time.perf_counter()
        await self.engines.resume()
        resumed = time.perf_counter()
        self.served = target
        self.switches += 1
        self.last_pause_s = resumed - paused
        logger.info(
            f"Pulled policy v{target.version_number} ({target.version_id}): "
            f"{'staged while serving' if live else 'staged while paused'} in {staged - start:.3f}s, "
            f"reached a pause in {paused - staged:.3f}s, paused {resumed - paused:.3f}s "
            f"(install {installed - paused:.3f}s, resume {resumed - installed:.3f}s)"
        )


def exit_server(reason: str) -> None:
    """Stop the server: an engine that cannot install a version must not resume with partly written weights. Under
    API server scale-out vLLM stops every API server and engine only when one exits in error, so exit hard if the
    graceful shutdown has not finished in time."""
    logger.error(f"{reason}; shutting down the inference server")
    asyncio.get_running_loop().call_later(30, os._exit, 1)
    os.kill(os.getpid(), signal.SIGTERM)
