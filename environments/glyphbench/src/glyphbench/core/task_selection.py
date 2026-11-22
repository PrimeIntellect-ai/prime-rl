"""Filter the env registry by suite name and/or fnmatch task patterns.

Used by load_environment, eval scripts, and prime-rl training configs to
build the list of task_ids to operate on. Pure function — does not import
or instantiate any env.
"""
from __future__ import annotations

import fnmatch

from glyphbench.core.registry import all_glyphbench_env_ids


def _suite_of(env_id: str) -> str:
    """Extract the suite token from an env_id.

    The convention is ``glyphbench/<suite>-<rest>-v0``; suite is the first
    hyphen-separated segment after the package prefix.
    """
    name = env_id.split("/", 1)[1] if "/" in env_id else env_id
    return name.split("-", 1)[0]


def _matches_any_pattern(env_id: str, patterns: list[str]) -> bool:
    return any(p == env_id or fnmatch.fnmatchcase(env_id, p) for p in patterns)


def list_task_ids(
    include_suites: list[str] | None = None,
    exclude_suites: list[str] | None = None,
    include_tasks: list[str] | None = None,
    exclude_tasks: list[str] | None = None,
) -> list[str]:
    """Filter the registered env ids.

    Rules:
      - ``None`` (default) for any kwarg means no constraint of that kind.
      - ``include_*`` lists are whitelists: at least one must match for the id
        to pass. If both ``include_suites`` and ``include_tasks`` are given,
        either may match (OR, not AND).
      - ``exclude_*`` lists are blacklists: any match removes the id.
      - Excludes always win over includes.
      - ``__dummy`` envs are excluded by default and only appear if explicitly
        named by exact env_id in ``include_tasks``.
      - Returns a sorted list.
    """
    suites_in = include_suites or []
    suites_ex = exclude_suites or []
    tasks_in = include_tasks or []
    tasks_ex = exclude_tasks or []
    has_include = bool(suites_in or tasks_in)

    out: list[str] = []
    for env_id in all_glyphbench_env_ids():
        suite = _suite_of(env_id)
        # Exclude __dummy unless explicitly named in include_tasks.
        if "__dummy" in env_id and not any(env_id == t for t in tasks_in):
            continue
        if has_include:
            ok = (suite in suites_in) or _matches_any_pattern(env_id, tasks_in)
            if not ok:
                continue
        if suite in suites_ex:
            continue
        if _matches_any_pattern(env_id, tasks_ex):
            continue
        out.append(env_id)
    return sorted(out)
