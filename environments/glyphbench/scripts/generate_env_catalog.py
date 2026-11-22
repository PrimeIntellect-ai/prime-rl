#!/usr/bin/env python3
"""Generate public environment catalog from the registry."""

from contextlib import closing
from pathlib import Path

import glyphbench  # noqa: F401
from glyphbench.core import make_env
from glyphbench.core.task_selection import list_task_ids


def main() -> None:
    envs = sorted(list_task_ids())

    suites: dict[str, list[str]] = {}
    for eid in envs:
        suite = eid.split("/")[1].split("-")[0]
        suites.setdefault(suite, []).append(eid)

    lines = [
        "# GlyphBench Environment Catalog",
        "",
        "Generated from the current registry.",
        "Optional adapters may be absent when their dependencies are not installed.",
        "",
    ]

    for suite_name, suite_envs in sorted(suites.items()):
        lines.append(f"## {suite_name.title()}")
        lines.append("")
        lines.append("| Env ID | Actions |")
        lines.append("|--------|---------|")
        for eid in suite_envs:
            try:
                with closing(make_env(eid)) as env:
                    n_actions = env.action_spec.n
                lines.append(f"| `{eid}` | {n_actions} |")
            except ImportError:
                lines.append(f"| `{eid}` | ? |")
        lines.append("")

    output = "\n".join(lines)
    destination = Path(__file__).resolve().parents[1] / "docs" / "ENVIRONMENTS.md"
    destination.write_text(output, encoding="utf-8")
    print("Generated docs/ENVIRONMENTS.md from the current registry")


if __name__ == "__main__":
    main()
