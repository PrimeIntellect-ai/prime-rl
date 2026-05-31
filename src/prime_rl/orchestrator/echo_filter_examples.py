"""Example echo filters — smoke tests and copy-paste reference.

These filters are intentionally trivial. They exist so:

  1. The :class:`EchoFilterConfig` plumbing has CPU-runnable smoke
     coverage end-to-end without needing users to wire up their own
     module on PYTHONPATH before the very first test run.
  2. Users writing real filters have a working template to start from
     (signature, return shape, the "narrow only, never add" pattern).

Both filters are deterministic given their input rollout — they touch
no global state, no randomness, no wall-clock — satisfying the
determinism contract from :class:`EchoFilterConfig`.

To reference these from a TOML config:

.. code-block:: toml

    [orchestrator.train.env.echo.filter]
    import_path = "prime_rl.orchestrator.echo_filter_examples.identity"
"""

from typing import Any


def identity(rollout: dict[str, Any]) -> list[list[bool]]:
    """All-True mask for every position in every step.

    Semantic no-op — composed with any role baseline, this is exactly
    equivalent to ``filter=None`` (every position the role gate set
    to a float stays at that float). Useful for smoke-testing the
    filter plumbing without changing actual training behavior.

    Args:
        rollout: The full ``vf.RolloutOutput`` dict as returned by the
            env server.

    Returns:
        Per-step masks. Outer length equals ``len(rollout["trajectory"])``;
        each inner mask has length
        ``len(step.tokens.prompt_ids) + len(step.tokens.completion_ids)``,
        with every element set to ``True``.
    """
    return [
        [True] * (
            len(step["tokens"]["prompt_ids"])
            + len(step["tokens"]["completion_ids"])
        )
        for step in rollout["trajectory"]
    ]


def drop_completion(rollout: dict[str, Any]) -> list[list[bool]]:
    """Keep prompt-side echo, drop completion-side echo.

    Returns ``True`` for every prompt position and ``False`` for every
    completion position in each step. Combined with
    :class:`AssistantRoleEchoConfig`, this narrows the assistant-side
    completion echo away while preserving any prompt-side echo (tool,
    user, system, or prompt-side assistant turns from earlier in the
    conversation).

    Useful as a smoke filter that produces a *visible* effect on the
    resulting ``echo_alpha`` array — easier to verify in logs than the
    no-op :func:`identity`.

    Args:
        rollout: The full ``vf.RolloutOutput`` dict.

    Returns:
        Per-step masks. Each step's mask is
        ``[True] * prompt_len + [False] * completion_len``.
    """
    masks: list[list[bool]] = []
    for step in rollout["trajectory"]:
        prompt_len = len(step["tokens"]["prompt_ids"])
        completion_len = len(step["tokens"]["completion_ids"])
        masks.append([True] * prompt_len + [False] * completion_len)
    return masks
