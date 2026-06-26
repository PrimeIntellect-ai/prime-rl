import asyncio
import ctypes
import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

import orjson
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import InterceptHandler, get_logger, setup_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the student renderer and inference pool, returning ``(renderer, inference_pool)``.

    Training is renderer-only: RL/OPD roll out through the env server's renderer client
    (token-in/out), and SFT — which rolls out against a chat-completions teacher that returns
    no tokens — re-renders the conversation with this renderer to backfill them. The renderer
    is built here from the (always-set) ``config.renderer`` and also supplies the multimodal
    token-type-id map. The eval client is plain chat-completions (eval traces aren't trained)."""
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name
    renderer = create_renderer(tokenizer, config.renderer)
    get_logger().info("Using renderer rollout client")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=config.renderer,
        pool_size=config.pool_size,
    )
    return renderer, inference_pool


def save_rollouts(rollouts: list[dict], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts (Trace dicts, already JSON-serializable) to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "wb") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            f.write(orjson.dumps(row, default=str, option=opts))


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def setup_env_server_logging(log_level: str, json_logging: bool = False) -> None:
    """Configure logging for an env-server process: prime-rl's logger + routing v1's stdlib
    logs through it. Passed to verifiers' ``serve_env`` so it runs in the broker and in every
    spawned worker — fresh ``spawn`` processes that otherwise have no handlers and would drop
    their per-rollout logs."""
    setup_logger(log_level, json_logging=json_logging)
    intercept_vf_logging(logger="verifiers.v1", level=log_level)


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().info(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def trim_process_memory() -> None:
    """Return freed heap pages to the OS on glibc systems."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as exc:
        get_logger().debug(f"malloc_trim(0) failed: {exc!r}")


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill."""
    import httpx
    from openai import AsyncOpenAI
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    async def _compute_single(client_config: vf.ClientConfig, sample: TrainingSample) -> list[float]:
        client = AsyncOpenAI(
            base_url=client_config.base_url,
            api_key=os.environ.get(client_config.api_key_var, "EMPTY"),
            default_headers=client_config.headers or None,
        )

        # Two escape hatches from ``AsyncOpenAI.post``:
        #   1. URL — ``/inference/v1/generate`` is mounted at server root, not
        #      under ``/v1``. Pass an absolute URL so the SDK's
        #      ``_prepare_url`` skips the base-url merge (it short-circuits
        #      when the path passes ``httpx.URL.is_relative_url`` as False).
        #   2. Parse — vLLM's ``GenerateResponse`` is a plain
        #      ``pydantic.BaseModel`` and the SDK's parse layer rejects any
        #      ``cast_to`` that doesn't subclass ``openai.BaseModel``. Use
        #      ``cast_to=httpx.Response`` so the SDK still builds the request
        #      (preserving ``auth_headers``, retries, timeouts, idempotency
        #      keys) and just hands us the raw response to validate ourselves.
        base = str(client.base_url).rstrip("/").removesuffix("/v1")
        http_response = await client.post(
            f"{base}/inference/v1/generate",
            cast_to=httpx.Response,
            body={
                "model": model_name,
                "token_ids": list(sample.token_ids),
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "prompt_logprobs": 1,
                },
            },
        )
        response = GenerateResponse.model_validate_json(http_response.content)
        # ``prompt_logprobs[i]`` is a ``{token_id: Logprob}`` dict for tokens
        # the engine could score, or ``None`` for the leading token which has
        # no preceding context. Flatten to ``list[float]`` with 0.0 in the
        # unscored slot.
        flat: list[float] = []
        for entry in response.prompt_logprobs or []:
            if not entry:
                flat.append(0.0)
                continue
            first = next(iter(entry.values()))
            lp = first.logprob if hasattr(first, "logprob") else first.get("logprob")
            flat.append(float(lp) if lp is not None else 0.0)
        return flat

    return await asyncio.gather(*[_compute_single(client, sample) for client, sample in zip(cycle(clients), samples)])


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
