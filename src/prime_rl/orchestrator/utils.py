import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
from rich.console import Console
from rich.table import Table
from verifiers.utils.client_utils import setup_openai_client

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import (
    format_time,
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)


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


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill."""
    import httpx
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    async def _compute_single(client_config: vf.ClientConfig, sample: TrainingSample) -> list[float]:
        client = setup_openai_client(client_config)

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


def setup_external_rollout_model(config: OrchestratorConfig, logger) -> tuple[Any, str, bool]:
    """Resolve rollout client/model and whether student policy updates are enabled.

    - rl/opd: student generates rollouts, policy updates always enabled.
    - sft: teacher generates rollouts; policy updates enabled iff the student
      inference client is configured (non-empty base_url).
    """
    if config.training_mode in ("rl", "opd"):
        return config.student.client, config.student.model.name, True

    assert config.teacher is not None  # validated by validate_training_mode
    rollout_client_config = config.teacher.client
    rollout_model_name = config.teacher.model.name
    enable_policy_updates = "base_url" in config.student.client.model_fields_set
    logger.info(
        f"Using teacher rollout model (base_url={', '.join(rollout_client_config.base_url)}, model={rollout_model_name})"
    )
    return rollout_client_config, rollout_model_name, enable_policy_updates
