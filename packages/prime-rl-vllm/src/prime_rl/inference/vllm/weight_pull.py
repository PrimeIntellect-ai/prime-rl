"""Keeps an engine on the newest policy version its trainer published to ModelExpress.

The engine installs the newest published version before its API server answers a request, so it never samples
with weights no version describes and plain vLLM health is its readiness. A background task then installs each
newer version in place, from the trainer or from a peer engine that already has it. Each install pauses only
this engine, between model steps, and labels the tokens sampled from then on with the new version; the other
engines keep serving.

Every call reaches every worker of every engine the server runs; how the model is split across them is
vLLM's and ModelExpress's business. Staging a version only prepares it and touches no live weight, so a failed
stage is retried. Applying writes the live weights, so a failed apply leaves them half written and takes the
whole server down.
"""

import asyncio
import os
import random
import socket
import time

from modelexpress_rl import ModelExpressControlClient, WeightVersion, WeightVersionState
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

# vLLM only configures loggers under its own namespace.
logger = init_logger(f"vllm.{__name__}")

POLL_INTERVAL_S = 0.5
MAX_RETRY_INTERVAL_S = 8.0
UNVERSIONED = "default"
"""vLLM's weight version before anything is installed (``vllm/v1/engine/core.py``)."""


async def follow_published_versions(
    engine_client: EngineClient, server: str, staging_buffer_bytes: int, staging_buffers_count: int
) -> asyncio.Task | None:
    """Install the newest published version, then follow newer ones in a background task, which is
    returned for the caller to hold. Under API server scale-out only the first API server moves weights;
    the others wait for its first install and return None."""
    # vLLM's own renderer reads the API server rank this way.
    if engine_client.vllm_config.parallel_config._api_process_rank != 0:
        await _wait_for_first_install(engine_client)
        return None

    model_name = engine_client.vllm_config.model_config.model
    control = ModelExpressControlClient.connect(server_url=server)
    await engine_client.collective_rpc(
        "init_modelexpress_pull",
        args=(server, f"{socket.gethostname()}-{os.getpid()}", staging_buffer_bytes, staging_buffers_count),
    )

    serving = await _install_newest(engine_client, control, model_name, serving=None)
    task = asyncio.create_task(_follow(engine_client, control, model_name, serving))
    task.add_done_callback(_exit_on_failure)
    return task


async def _apply(engine_client: EngineClient, version: WeightVersion) -> None:
    """Install the staged version in place. Generation pauses for the copy, so no step samples from weights
    its version label does not describe (vLLM changes weight versions only while paused). The pause also drops
    the KV cache, so no token attends to keys and values the previous weights computed: the cache salt follows
    the orchestrator's policy version, which an engine installing on its own schedule does not match, and
    requests running now recompute their prefix on resume. A failed install leaves generation paused: the
    weights may be half written, and the server is about to exit."""
    start = time.perf_counter()
    await engine_client.pause_generation(mode="keep", clear_cache=True)
    await engine_client.collective_rpc("update_weights_from_modelexpress", args=(version.version_id,))
    await engine_client.update_weight_version(str(version.version_number))
    await engine_client.resume_generation()
    logger.info("Installed policy v%d in place in %.2fs", version.version_number, time.perf_counter() - start)


def _exit_on_failure(task: asyncio.Task) -> None:
    if task.cancelled() or task.exception() is None:
        return
    # A server that stops following must not keep serving an ever older policy or half-written weights. Under
    # API server scale-out vLLM stops every API server and engine only when one exits in error; a clean exit
    # would leave the others serving.
    logger.error("Following published policy versions failed; shutting down", exc_info=task.exception())
    os._exit(1)


async def _follow(
    engine_client: EngineClient, control: ModelExpressControlClient, model_name: str, serving: WeightVersion
) -> None:
    while True:
        serving = await _install_newest(engine_client, control, model_name, serving=serving)


async def _install_newest(
    engine_client: EngineClient, control: ModelExpressControlClient, model_name: str, serving: WeightVersion | None
) -> WeightVersion:
    """Install the newest published version newer than ``serving`` that a source can serve, and return it.

    Versions are tried newest first. The newest may have no free source yet, because no peer has installed it
    or the trainer is busy, and an older one a peer already holds keeps the engine on a recent policy instead
    of none; every token records its version, so the orchestrator accounts for the staleness. Rounds in which
    nothing can be installed back off with jitter, so waiting engines neither poll in lockstep nor keep
    interrupting their own serving."""
    retry_interval = POLL_INTERVAL_S
    reported = set()
    while True:
        for version in await _newer_versions(control, model_name, serving):
            if error := await _stage(engine_client, version):
                if version.version_id not in reported:
                    logger.info("Policy v%d not installable yet: %s", version.version_number, error)
                    reported.add(version.version_id)
                continue
            await _apply(engine_client, version)
            return version
        await asyncio.sleep(random.uniform(retry_interval / 2, retry_interval))
        retry_interval = min(retry_interval * 2, MAX_RETRY_INTERVAL_S)


async def _newer_versions(
    control: ModelExpressControlClient, model_name: str, serving: WeightVersion | None
) -> list[WeightVersion]:
    """READY versions published after ``serving``, newest first. Publication order, not the version number,
    decides: a trainer restarted from a checkpoint publishes lower numbers than the one it replaced. An
    unreachable server yields none, so the engine keeps serving and asks again."""
    try:
        versions = await asyncio.to_thread(control.list_weight_versions, model_name=model_name)
    except Exception as error:
        logger.warning("Listing published policy versions failed; retrying: %s", error)
        return []
    newer = [
        version
        for version in versions
        if version.state is WeightVersionState.READY
        and (serving is None or version.created_at_unix_ms > serving.created_at_unix_ms)
    ]
    return sorted(newer, key=lambda version: version.created_at_unix_ms, reverse=True)


async def _on_every_worker(engine_client: EngineClient, method: str, args: tuple = ()) -> list:
    """Every worker's result of ``method``, from every engine the server runs. ``collective_rpc`` reaches every
    engine but returns only the first engine's results, so this asks each engine the way vLLM's own
    ``compute_weight_checksums`` does."""
    per_engine = await engine_client.engine_core.call_utility_all_async("collective_rpc", method, None, args, None)
    return [result for results in per_engine for result in results]


async def _stage(engine_client: EngineClient, version: WeightVersion) -> str | None:
    """Stage ``version`` on every worker, or return why some worker could not, having every worker let go of it
    so none installs alone and no source stays held."""
    errors = await _on_every_worker(engine_client, "stage_weights_from_modelexpress", args=(version.version_id,))
    if error := next((error for error in errors if error), None):
        await _on_every_worker(engine_client, "discard_staged_modelexpress")
        return error
    return None


async def _wait_for_first_install(engine_client: EngineClient) -> None:
    while True:
        try:
            if await engine_client.get_weight_version() != UNVERSIONED:
                return
        except RuntimeError:
            # vLLM requires every engine to report the same version, and they disagree while the first
            # install's label reaches them one by one.
            pass
        await asyncio.sleep(POLL_INTERVAL_S)
