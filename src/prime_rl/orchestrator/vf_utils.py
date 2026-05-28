import logging
from pathlib import Path

import orjson
import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.utils.logger import InterceptHandler


def get_model_completion_len(output: vf.RolloutOutput) -> int:
    """Sum of model-generated completion tokens across all turns (excludes
    environment-injected tokens between turns)."""
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"] if step.get("tokens"))


def get_num_turns(output: vf.RolloutOutput) -> int:
    """Number of turns (trajectory steps) in a rollout."""
    return len(output["trajectory"])


def get_tool_response_len(output: vf.RolloutOutput) -> int:
    """
    Total tool-response tokens consumed across the whole rollout.

    Read from a harness-emitted metric (e.g. RLM's `rlm_total_tool_response_tokens`,
    deduped across turns/branches/sub-RLMs). Returns 0 if no harness metric is
    present, which makes this a no-op for envs without tool-response accounting.
    """
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
