"""Per-dataset task overrides, read from a JSON file shipped next to the task dirs.

The taskset code knows only this schema; which tasks a dataset holds back, and why,
lives in the file:

```json
{
  "version": 1,
  "holdback_groups": {
    "<group>": {
      "optional": true,
      "description": "why this group is held back",
      "tasks": {"<task>": "<reason>"}
    }
  },
  "agent_users": {"<task>": "<user>"},
  "gold_as_root": {"<task>": "<reason>"},
  "home_artifact_excludes": {"<task>": {"<artifact source>": ["<exclude>"]}}
}
```

- `holdback_groups`: tasks that do not load by default. An `optional` group loads when
  the taskset config names it in `include_groups`; any other group never loads.
- `agent_users`: the agent user for tasks whose image names it differently from the
  config's `agent_user` (applies only when the task declares no `[agent].user`).
- `gold_as_root`: tasks whose gold check runs `solution/solve.sh` as root.
- `home_artifact_excludes`: extra `exclude` entries per declared artifact source, for
  tasks whose agent home is an artifact (leave out what the harness installs there).

Every task named here must be a task dir, and a task sits in at most one group.
"""

from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OVERRIDES_FILE = "overrides.json"


class HoldbackGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optional: bool
    """Whether `include_groups` can bring this group back."""
    description: str = ""
    tasks: dict[str, str]
    """Task name -> why it is held back."""


class Overrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    holdback_groups: dict[str, HoldbackGroup] = Field(default_factory=dict)
    agent_users: dict[str, str] = Field(default_factory=dict)
    gold_as_root: dict[str, str] = Field(default_factory=dict)
    home_artifact_excludes: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @classmethod
    def read(cls, path: Path | None) -> "Overrides":
        if path is None:
            return cls()
        return cls.model_validate_json(path.read_bytes())

    def check(self, task_names: Collection[str], include_groups: Collection[str]) -> None:
        """Raise on an include that names no optional group, a task named twice across
        groups, or a task with no task dir."""
        if unknown := set(include_groups) - set(self.holdback_groups):
            raise ValueError(
                f"include_groups names unknown hold-back groups {sorted(unknown)}; "
                f"the overrides define {sorted(self.holdback_groups)}"
            )
        if fixed := sorted(name for name in include_groups if not self.holdback_groups[name].optional):
            raise ValueError(f"hold-back groups {fixed} are not optional and cannot be included")
        seen: dict[str, str] = {}
        for group, entry in self.holdback_groups.items():
            for task in entry.tasks:
                if task in seen:
                    raise ValueError(f"task {task!r} is in hold-back groups {seen[task]!r} and {group!r}")
                seen[task] = group
        named = set(seen) | set(self.agent_users) | set(self.gold_as_root) | set(self.home_artifact_excludes)
        if missing := named - set(task_names):
            listed = sorted(missing)
            raise ValueError(f"the overrides name {len(listed)} task(s) with no task dir: {listed[:10]}")

    def held_back(self, include_groups: Collection[str]) -> dict[str, str]:
        """Task name -> reason, for every task held back with `include_groups` included."""
        held: dict[str, str] = {}
        for group, entry in self.holdback_groups.items():
            if not (entry.optional and group in include_groups):
                held.update(entry.tasks)
        return held
