"""Keeps an engine on the newest policy version its trainer published to ModelExpress.

The engine installs the newest published version before its API server answers a request, so it never samples
with weights no version describes and plain vLLM health is its readiness. A background task then installs each
newer version in place, from the trainer or from a peer engine that already has it, and labels the tokens sampled
from then on with it. Only this engine pauses for the copy; the others keep serving.
"""

import asyncio
import os
import signal
import socket
import time

from modelexpress_rl import ModelExpressControlClient, WeightVersion, WeightVersionState
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

# vLLM only configures loggers under its own namespace.
logger = init_logger(f"vllm.{__name__}")

POLL_INTERVAL_S = 0.5
UNVERSIONED = "default"
"""vLLM's weight version before anything is installed."""


async def follow_published_versions(
    engine_client: EngineClient, server: str, staging_buffer_bytes: int | None, staging_buffers_count: int
) -> asyncio.Task | None:
    """Install the newest published version, then follow newer ones in a background task, which is
    returned for the caller to hold. Under API server scale-out only the first API server moves weights;
    the others wait for its first install and return None."""
    if engine_client.vllm_config.parallel_config._api_process_rank != 0:
        await _wait_for_first_install(engine_client)
        return None

    model_name = engine_client.vllm_config.model_config.model
    control = ModelExpressControlClient.connect(server_url=server)
    host, port = server.rsplit(":", 1)
    session_id = f"{socket.gethostname()}-{os.getpid()}"
    await engine_client.collective_rpc(
        "init_broadcaster",
        args=(host, int(port), 0, 1, None, session_id, staging_buffer_bytes, staging_buffers_count),
    )

    serving = await _install_newest(engine_client, control, model_name, after=None)
    task = asyncio.create_task(_follow(engine_client, control, model_name, serving))
    task.add_done_callback(_exit_on_failure)
    return task


async def _follow(
    engine_client: EngineClient, control: ModelExpressControlClient, model_name: str, serving: int
) -> None:
    while True:
        serving = await _install_newest(engine_client, control, model_name, after=serving)


def _exit_on_failure(task: asyncio.Task) -> None:
    if task.cancelled() or task.exception() is None:
        return
    # An engine that stops following must not keep serving an ever older policy.
    logger.error("Following published policy versions failed; shutting down", exc_info=task.exception())
    os.kill(os.getpid(), signal.SIGTERM)


async def _install(engine_client: EngineClient, version: WeightVersion) -> None:
    """Stage the version on every rank, then apply it in place on all of them between model steps; if any rank
    cannot stage it yet, every rank lets go of it so no rank installs alone and no source stays held."""
    start = time.perf_counter()
    errors = await engine_client.collective_rpc("stage_weights_from_modelexpress", args=(version.version_id,))
    if error := next((error for error in errors if error), None):
        await engine_client.collective_rpc("discard_staged_modelexpress")
        raise RuntimeError(error)
    staged = time.perf_counter()
    await engine_client.collective_rpc("update_weights_from_modelexpress", args=(version.version_id,))
    await engine_client.update_weight_version(str(version.version_number))
    logger.info(
        "Installed policy v%d: staged in %.2fs, copied in place in %.2fs",
        version.version_number,
        staged - start,
        time.perf_counter() - staged,
    )


async def _install_newest(
    engine_client: EngineClient, control: ModelExpressControlClient, model_name: str, after: int | None
) -> int:
    """Install the newest published version above ``after`` that a source can serve, and return its number.
    Versions are tried newest first: the newest may have no source yet, because no peer has installed it or the
    trainer is busy or gone, and an older one a peer already holds keeps the engine on a recent policy instead of
    none. Every token records its version, so the orchestrator accounts for the staleness. While no version can
    be installed, waits and tries again."""
    reported = set()
    while True:
        ready = await _ready_versions(control, model_name)
        candidates = [version for version in ready if after is None or version.version_number > after]
        for version in sorted(candidates, key=lambda version: version.version_number, reverse=True):
            try:
                await _install(engine_client, version)
            except Exception as error:
                if version.version_id not in reported:
                    logger.info("Policy v%d not installable yet: %s", version.version_number, error)
                    reported.add(version.version_id)
                continue
            return version.version_number
        await asyncio.sleep(POLL_INTERVAL_S)


async def _ready_versions(control: ModelExpressControlClient, model_name: str) -> list[WeightVersion]:
    versions = await asyncio.to_thread(control.list_weight_versions, model_name=model_name)
    return [version for version in versions if version.state is WeightVersionState.READY]


async def _wait_for_first_install(engine_client: EngineClient) -> None:
    while await engine_client.get_weight_version() == UNVERSIONED:
        await asyncio.sleep(POLL_INTERVAL_S)
