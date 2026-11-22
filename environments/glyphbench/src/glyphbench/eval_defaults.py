"""Canonical public defaults for comparable GlyphBench evaluations."""

from __future__ import annotations

import argparse
import json
import shlex
from copy import deepcopy
from typing import Any

DEFAULT_PROVIDER = "vllm"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_HARNESS = "default"

DEFAULT_EPISODES_PER_ENV = 3
DEFAULT_ROLLOUTS_PER_EXAMPLE = 1
DEFAULT_SEED = 42
DEFAULT_N_FRAMES = 0
DEFAULT_MAX_TURNS: int | None = None

DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_OUTPUT_TOKENS = DEFAULT_MAX_TOKENS
DEFAULT_MEMORY_UPDATE_MAX_TOKENS = 4096
# Model/training capacity, deliberately separate from GlyphBench environment
# configuration. Native games own step horizons; the inference/training stack
# owns the token context window.
DEFAULT_CONTEXT_WINDOW_TOKENS = 65536
DEFAULT_MAX_MODEL_LEN = DEFAULT_CONTEXT_WINDOW_TOKENS
DEFAULT_MAX_NUM_BATCHED_TOKENS = DEFAULT_MAX_MODEL_LEN
DEFAULT_MAX_CONCURRENT: int | None = None
DEFAULT_MAX_NUM_SEQS: int | None = None
DEFAULT_TEMPERATURE = 1.0

DEFAULT_EXCLUDE_SUITES = ("atari", "craftaxfull", "nethack", "agentick")
CRAFTAX_SUITE_TASK_PREFIX = "glyphbench/craftax-"
CRAFTAX_SUITE_MAX_CONCURRENT = 1
CRAFTAX_SUITE_MAX_NUM_SEQS = 1
ARCHIVAL_FULL_TASK_IDS = (
    "glyphbench/craftaxfull-v0",
    "glyphbench/nethack-full-v0",
)
ARCHIVAL_FULL_MAX_TOKENS = 32768
ARCHIVAL_FULL_MAX_OUTPUT_TOKENS = ARCHIVAL_FULL_MAX_TOKENS
ARCHIVAL_FULL_MEMORY_UPDATE_MAX_TOKENS = 32768
ARCHIVAL_FULL_MAX_MODEL_LEN = 196608
ARCHIVAL_FULL_MAX_NUM_BATCHED_TOKENS = 32768
ARCHIVAL_FULL_MAX_CONCURRENT = 1
ARCHIVAL_FULL_MAX_NUM_SEQS = 1
ARCHIVAL_FULL_ENV_REQUEST_TIMEOUT_SECONDS = 172800

DEFAULT_SAMPLING_ARGS: dict[str, Any] = {
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "extra_body": {
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
}


def sampling_args() -> dict[str, Any]:
    """Return a mutable copy of the canonical sampling args."""

    return deepcopy(DEFAULT_SAMPLING_ARGS)


def sampling_args_json() -> str:
    """Return the compact JSON string passed to ``prime eval --sampling-args``."""

    return json.dumps(DEFAULT_SAMPLING_ARGS, separators=(",", ":"))


def exclude_suites_json() -> str:
    """Return the compact JSON suite exclusion list used by the default sweep."""

    return json.dumps(list(DEFAULT_EXCLUDE_SUITES), separators=(",", ":"))


def eval_settings_for_task_ids(task_ids: str | list[str] | tuple[str, ...]) -> dict[str, int | None]:
    """Return token/context defaults for a concrete task selection."""

    ids = (task_ids,) if isinstance(task_ids, str) else tuple(task_ids)
    settings = {
        "max_tokens": DEFAULT_MAX_TOKENS,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "memory_update_max_tokens": DEFAULT_MEMORY_UPDATE_MAX_TOKENS,
        "context_window_tokens": DEFAULT_CONTEXT_WINDOW_TOKENS,
        "max_model_len": DEFAULT_MAX_MODEL_LEN,
        "max_num_batched_tokens": DEFAULT_MAX_NUM_BATCHED_TOKENS,
        "max_concurrent": DEFAULT_MAX_CONCURRENT,
        "max_num_seqs": DEFAULT_MAX_NUM_SEQS,
        "env_request_timeout_seconds": None,
    }
    if any(task_id.startswith(CRAFTAX_SUITE_TASK_PREFIX) for task_id in ids):
        settings.update(
            {
                "max_concurrent": CRAFTAX_SUITE_MAX_CONCURRENT,
                "max_num_seqs": CRAFTAX_SUITE_MAX_NUM_SEQS,
            }
        )
    if any(task_id in ARCHIVAL_FULL_TASK_IDS for task_id in ids):
        settings.update(
            {
                "max_tokens": ARCHIVAL_FULL_MAX_TOKENS,
                "max_output_tokens": ARCHIVAL_FULL_MAX_OUTPUT_TOKENS,
                "memory_update_max_tokens": ARCHIVAL_FULL_MEMORY_UPDATE_MAX_TOKENS,
                "max_model_len": ARCHIVAL_FULL_MAX_MODEL_LEN,
                "max_num_batched_tokens": ARCHIVAL_FULL_MAX_NUM_BATCHED_TOKENS,
                "max_concurrent": ARCHIVAL_FULL_MAX_CONCURRENT,
                "max_num_seqs": ARCHIVAL_FULL_MAX_NUM_SEQS,
                "env_request_timeout_seconds": ARCHIVAL_FULL_ENV_REQUEST_TIMEOUT_SECONDS,
            }
        )
    return settings


def protocol_dict() -> dict[str, Any]:
    """Return the comparable eval protocol as a JSON-serializable mapping."""

    return {
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "harness": DEFAULT_HARNESS,
        "episodes_per_env": DEFAULT_EPISODES_PER_ENV,
        "rollouts_per_example": DEFAULT_ROLLOUTS_PER_EXAMPLE,
        "seed": DEFAULT_SEED,
        "n_frames": DEFAULT_N_FRAMES,
        "max_turns": DEFAULT_MAX_TURNS,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "memory_update_max_tokens": DEFAULT_MEMORY_UPDATE_MAX_TOKENS,
        "context_window_tokens": DEFAULT_CONTEXT_WINDOW_TOKENS,
        "max_model_len": DEFAULT_MAX_MODEL_LEN,
        "max_num_batched_tokens": DEFAULT_MAX_NUM_BATCHED_TOKENS,
        "max_concurrent": DEFAULT_MAX_CONCURRENT,
        "max_num_seqs": DEFAULT_MAX_NUM_SEQS,
        "temperature": DEFAULT_TEMPERATURE,
        "sampling_args": sampling_args(),
        "exclude_suites": list(DEFAULT_EXCLUDE_SUITES),
        "craftax_suite_task_prefix": CRAFTAX_SUITE_TASK_PREFIX,
        "craftax_suite_max_concurrent": CRAFTAX_SUITE_MAX_CONCURRENT,
        "craftax_suite_max_num_seqs": CRAFTAX_SUITE_MAX_NUM_SEQS,
        "archival_full_task_ids": list(ARCHIVAL_FULL_TASK_IDS),
        "archival_full_max_tokens": ARCHIVAL_FULL_MAX_TOKENS,
        "archival_full_max_output_tokens": ARCHIVAL_FULL_MAX_OUTPUT_TOKENS,
        "archival_full_memory_update_max_tokens": ARCHIVAL_FULL_MEMORY_UPDATE_MAX_TOKENS,
        "archival_full_max_model_len": ARCHIVAL_FULL_MAX_MODEL_LEN,
        "archival_full_max_num_batched_tokens": ARCHIVAL_FULL_MAX_NUM_BATCHED_TOKENS,
        "archival_full_max_concurrent": ARCHIVAL_FULL_MAX_CONCURRENT,
        "archival_full_max_num_seqs": ARCHIVAL_FULL_MAX_NUM_SEQS,
        "archival_full_env_request_timeout_seconds": ARCHIVAL_FULL_ENV_REQUEST_TIMEOUT_SECONDS,
    }


def shell_assignments() -> str:
    """Return shell assignments consumed by the eval bash entrypoints."""

    values = {
        "GLYPHBENCH_EVAL_MODEL": DEFAULT_MODEL,
        "GLYPHBENCH_EVAL_EPISODES": DEFAULT_EPISODES_PER_ENV,
        "GLYPHBENCH_EVAL_ROLLOUTS_PER_EXAMPLE": DEFAULT_ROLLOUTS_PER_EXAMPLE,
        "GLYPHBENCH_EVAL_SEED": DEFAULT_SEED,
        "GLYPHBENCH_EVAL_N_FRAMES": DEFAULT_N_FRAMES,
        "GLYPHBENCH_EVAL_MAX_TOKENS": DEFAULT_MAX_TOKENS,
        "GLYPHBENCH_EVAL_MEMORY_UPDATE_MAX_TOKENS": DEFAULT_MEMORY_UPDATE_MAX_TOKENS,
        "GLYPHBENCH_EVAL_MAX_MODEL_LEN": DEFAULT_MAX_MODEL_LEN,
        "GLYPHBENCH_EVAL_TEMPERATURE": DEFAULT_TEMPERATURE,
        "GLYPHBENCH_EVAL_SAMPLING_ARGS": sampling_args_json(),
        "GLYPHBENCH_EVAL_EXCLUDE_SUITES": exclude_suites_json(),
    }
    return "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "format",
        nargs="?",
        choices=("json", "shell", "sampling-args"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args(argv)

    if args.format == "shell":
        print(shell_assignments())
    elif args.format == "sampling-args":
        print(sampling_args_json())
    else:
        print(json.dumps(protocol_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
