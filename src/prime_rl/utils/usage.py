import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Annotated, Any

import httpx
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pydantic_config import BaseConfig

if TYPE_CHECKING:
    import verifiers as vf

__all__ = ["UsageConfig", "UsageReporter", "init_usage_reporter", "report_inference_usage"]

_reporter: "UsageReporter | None" = None


class UsageConfig(BaseConfig):
    base_url: Annotated[str, Field(description="Base URL for the usage API.")]
    api_key_var: Annotated[str, Field(description="Environment variable containing the API key.")] = "PRIME_API_KEY"
    timeout: Annotated[int, Field(description="HTTP request timeout in seconds.")] = 10


class UsageReporter:
    """Fire-and-forget token usage reporter."""

    def __init__(self, config: UsageConfig | None = None):
        self._executor: ThreadPoolExecutor | None = None
        self._client: httpx.Client | None = None
        self._base_url: str = ""
        self._api_key: str = ""

        if not config:
            return

        api_key = os.getenv(config.api_key_var)
        if not api_key:
            get_logger().debug(f"UsageReporter disabled: {config.api_key_var} not set")
            return

        self._base_url = config.base_url
        self._api_key = api_key
        self._client = httpx.Client(timeout=config.timeout)
        self._executor = ThreadPoolExecutor(max_workers=2)
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._client:
            self._client.close()

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=1, max=60), reraise=True)
    def _post_with_retry(self, payload: dict[str, Any]) -> None:
        resp = self._client.post(f"{self._base_url}/usage", json=payload, headers={"x-api-key": self._api_key})
        if resp.status_code != 409:
            resp.raise_for_status()

    def _post(self, payload: dict[str, Any]) -> None:
        try:
            self._post_with_retry(payload)
        except Exception as e:
            get_logger().warning(f"Usage report failed: {e}")

    def report_training(self, run_id: str, step: int, tokens: int) -> None:
        if self._executor:
            self._executor.submit(
                self._post, {"run_id": run_id, "step": step, "usage_type": "training", "tokens": tokens}
            )

    def report_inference(self, run_id: str, step: int, input_tokens: int, output_tokens: int) -> None:
        if self._executor:
            self._executor.submit(
                self._post,
                {
                    "run_id": run_id,
                    "step": step,
                    "usage_type": "inference",
                    "tokens": input_tokens + output_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )


def init_usage_reporter(config: UsageConfig | None) -> None:
    """Initialize the global usage reporter. Call once at startup."""
    global _reporter
    _reporter = UsageReporter(config)


def _get_inference_tokens(state: "vf.State") -> tuple[int, int]:
    """Total tokens processed by vLLM across all trajectory steps."""
    total_input = 0
    total_output = 0
    for step in state["trajectory"]:
        if step["tokens"]:
            total_input += len(step["tokens"]["prompt_ids"])
            total_output += len(step["tokens"]["completion_ids"])
    return total_input, total_output


def report_inference_usage(step: int, rollouts: list["vf.State"]) -> None:
    """Report inference token usage. No-op if reporter not initialized or RUN_ID not set."""
    run_id = os.getenv("RUN_ID")
    if not run_id or not _reporter:
        return
    input_tokens, output_tokens = 0, 0
    for rollout in rollouts:
        inp, out = _get_inference_tokens(rollout)
        input_tokens += inp
        output_tokens += out
    _reporter.report_inference(run_id, step, input_tokens, output_tokens)
