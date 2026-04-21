import json
import logging
from pathlib import Path

import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.utils.logger import InterceptHandler


# TODO: remove once usage is tracked by verifiers
def get_prompt_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of prompt tokens from vf.RolloutOutput. Defined as the
    number of prompt ids from the first trajectory step. If raw tokens are not
    available, falls back to checking the usage of the first response.
    """
    if not output["trajectory"]:
        return 0
    first_step = output["trajectory"][0]
    if first_step["tokens"] is not None:
        return len(first_step["tokens"]["prompt_ids"])
    first_step_response = first_step["response"]
    return (first_step_response.get("usage") or {}).get("prompt_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_seq_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of tokens from vf.RolloutOutput. Defined as the sum of prompt
    and completion tokens from the last trajectory step. If raw tokens are not
    available, falls back to checking the usage of the last response.
    """
    if not output["trajectory"]:
        return 0
    last_step = output["trajectory"][-1]
    if last_step["tokens"] is not None:
        return len(last_step["tokens"]["prompt_ids"]) + len(last_step["tokens"]["completion_ids"])
    last_step_response = last_step["response"]
    return (last_step_response.get("usage") or {}).get("total_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_completion_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of completion tokens from vf.RolloutOutput. Defined as
    the difference between the total number of tokens and the number of prompt
    tokens.
    """
    return get_seq_len(output) - get_prompt_len(output)


def get_model_completion_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of model-generated completion tokens across all turns.
    Unlike get_completion_len, this excludes environment responses injected
    between turns in multi-turn rollouts.
    """
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"] if step.get("tokens"))


def save_rollouts(rollouts: list[vf.RolloutOutput], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts to a JSONL file using verifiers serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            json.dump(row, f, default=make_serializable)
            f.write("\n")


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
