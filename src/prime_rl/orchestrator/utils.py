import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import verifiers as vf
from rich.console import Console
from rich.table import Table
from verifiers.utils.client_utils import setup_openai_client
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import InterceptHandler, format_time, get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the student inference pool + matching renderer. Returns
    ``(renderer | None, inference_pool)``; ``renderer`` is ``None`` on the
    MITO path (``config.renderer is None``)."""
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name

    if config.renderer is not None:
        renderer = create_renderer(tokenizer, config.renderer)
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=config.renderer,
            pool_size=config.pool_size,
        )
        get_logger().info("Using direct renderer rollout client")
        return renderer, inference_pool

    get_logger().info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="openai_chat_completions",
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


def get_model_completion_len(output: vf.RolloutOutput) -> int:
    """Sum of model-generated completion tokens across all turns (excludes
    environment-injected tokens between turns)."""
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"] if step.get("tokens"))


def get_tool_response_len(output: vf.RolloutOutput) -> int:
    """Total tool-response tokens consumed across the whole rollout, read from a
    harness-emitted metric (e.g. RLM's `rlm_total_tool_response_tokens`, deduped
    across turns/branches/sub-RLMs). Returns 0 when no such metric is present."""
    metrics = output.get("metrics") or {}
    for key, value in metrics.items():
        if key.endswith("total_tool_response_tokens") and isinstance(value, (int, float)):
            return int(value)
    return 0


def save_rollouts(rollouts: list[vf.RolloutOutput], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts to a JSONL file using verifiers serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "wb") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            f.write(orjson.dumps(row, default=make_serializable, option=opts))


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().info(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted step time values.
    First N rows show the per-step values, and the last row shows the mean,
    std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


def _flatten_prompt_logprobs(raw: list[Any] | None) -> list[float]:
    """Shared flattener used by both transports.

    ``prompt_logprobs[i]`` is a ``{token_id: Logprob}`` dict for tokens the
    engine could score, or ``None`` for the leading token which has no
    preceding context. Flatten to ``list[float]`` with 0.0 in the unscored
    slot. Accepts both vLLM's typed ``Logprob`` objects and dynamo's
    ``PromptLogprobEntry`` dict shape (`{logprob, rank?, decoded_token?}`).
    """
    flat: list[float] = []
    for entry in raw or []:
        if not entry:
            flat.append(0.0)
            continue
        first = next(iter(entry.values()))
        lp = first.logprob if hasattr(first, "logprob") else first.get("logprob")
        flat.append(float(lp) if lp is not None else 0.0)
    return flat


async def _compute_teacher_logprobs_vllm(
    client_config: vf.ClientConfig, model_name: str, sample: TrainingSample
) -> list[float]:
    """Legacy path: prime-rl's vLLM sidecar ``/inference/v1/generate``."""
    import httpx
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    client = setup_openai_client(client_config)
    # Two escape hatches from ``AsyncOpenAI.post``:
    #   1. URL — ``/inference/v1/generate`` is mounted at server root, not
    #      under ``/v1``. Pass an absolute URL so the SDK's ``_prepare_url``
    #      skips the base-url merge.
    #   2. Parse — vLLM's ``GenerateResponse`` isn't an ``openai.BaseModel``.
    #      Use ``cast_to=httpx.Response`` and validate the body ourselves.
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    http_response = await client.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model_name,
            "token_ids": list(sample.prompt_ids) + list(sample.completion_ids),
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": 1,
            },
        },
    )
    response = GenerateResponse.model_validate_json(http_response.content)
    return _flatten_prompt_logprobs(response.prompt_logprobs)


async def _compute_teacher_logprobs_dynamo(
    client_config: vf.ClientConfig, model_name: str, sample: TrainingSample
) -> list[float]:
    """rl-sdk-2 path: dynamo via ``/v1/chat/completions`` with nvext envelope.

    Wire shape (per plan.md A1+A3 — already implemented on rl-sdk-2):
      - top-level ``prompt_logprobs: 1`` (CommonExt sampling param)
      - ``nvext.token_data`` carries pre-tokenized prompt
      - ``nvext.extra_fields = ["prompt_logprobs"]`` opts into the response
        field; dynamo emits ``response.nvext.prompt_logprobs`` shaped as
        ``[None | {token_id: {logprob, rank?, decoded_token?}}]``, which the
        shared flattener consumes unchanged.

    Required engine adapter support: vLLM worker must populate
    ``LLMEngineOutput.prompt_logprobs`` when ``SamplingParams.prompt_logprobs``
    is set. Without that (A10), the response payload is None.
    """
    client = setup_openai_client(client_config)
    token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
    body = {
        "model": model_name,
        "messages": [],
        "max_completion_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "prompt_logprobs": 1,
        "nvext": {
            "token_data": token_ids,
            "extra_fields": ["prompt_logprobs"],
        },
    }
    # Dynamo's response is just a standard chat-completion JSON with an extra
    # ``nvext`` field. Use ``cast_to=httpx.Response`` so we can read the raw
    # body and pluck ``nvext.prompt_logprobs`` ourselves — the OpenAI SDK
    # response models drop unknown fields.
    import httpx as _httpx

    http_response = await client.post(
        "/chat/completions",
        cast_to=_httpx.Response,
        body=body,
    )
    payload = http_response.json()
    nvext_resp = (payload or {}).get("nvext") or {}
    raw = nvext_resp.get("prompt_logprobs")
    return _flatten_prompt_logprobs(raw)


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill.

    Dispatches to the vLLM-sidecar or dynamo-nvext path based on the
    per-client ``renderer_transport``:

      - ``prime_vllm_generate``  (default): POST ``/inference/v1/generate``
      - ``dynamo_chat_nvext``               : POST ``/v1/chat/completions`` with nvext

    Both flatten to ``list[float]`` via the shared helper.
    """

    async def _compute_single(client_config: vf.ClientConfig, sample: TrainingSample) -> list[float]:
        if client_config.renderer_transport == "dynamo_chat_nvext":
            return await _compute_teacher_logprobs_dynamo(client_config, model_name, sample)
        return await _compute_teacher_logprobs_vllm(client_config, model_name, sample)

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
