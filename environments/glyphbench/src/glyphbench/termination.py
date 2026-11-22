"""Canonical rollout-stop classification shared by training and evaluation."""

from __future__ import annotations

# ``context_length`` is the Verifiers v1 provider-capacity stop and
# ``prompt_too_long`` is the legacy adapter's name for the same event.
# Classification follows the
# framework stop, not the last successful token count: the refused next call
# can occur while the last realized sequence is still below the capacity.
CONTEXT_LIMIT_STOP_CONDITIONS = frozenset(
    {"context_length", "prompt_too_long"}
)


def is_context_limit_stop(stop_condition: object) -> bool:
    """Return whether a framework stop represents a context-capacity hit."""

    return (
        isinstance(stop_condition, str)
        and stop_condition in CONTEXT_LIMIT_STOP_CONDITIONS
    )


__all__ = ["CONTEXT_LIMIT_STOP_CONDITIONS", "is_context_limit_stop"]
